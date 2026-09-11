"""
Comprehensive test suite for Phase 1: Structured QA Judge + Multi-Check Quality Pipeline.
Covers:
- Strict Objective Evidence Priority (LLM cannot override objective failure)
- Multi-check pipeline (AST, Pytest, Lint, Typecheck, Security)
- Optional tool graceful fallback (SKIPPED if tool not installed)
- Static AST security analysis (eval, exec, shell=True, leaked secrets)
- Calibrated confidence and regression risk scoring
- Graph integration (qa_node, qa_router, revision telemetry)
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.developer.models import FilePatch
from backend.graph.nodes import qa_node, qa_router, revision_node
from backend.graph.state import AgentState
from backend.qa.judge import StructuredQAJudge
from backend.qa.pipeline import QualityPipeline
from backend.revision.models import RevisionAttempt, RevisionHistory
from backend.sandbox.models import TestExecutionResult
from backend.schemas.developer import DeveloperResult
from backend.schemas.planning import ExecutionPlan, PlanStep
from backend.schemas.qa import FailureCategory, QAIssue, QualityCheck, QualityCheckStatus, QAResult


# ---------------------------------------------------------------------------
# Pipeline Unit Tests
# ---------------------------------------------------------------------------


def test_quality_pipeline_ast_check_valid(tmp_path):
    """Verify AST check passes for valid Python code."""
    file_path = tmp_path / "sample.py"
    file_path.write_text("def existing(): pass\n", encoding="utf-8")

    patch = FilePatch(
        file_path="sample.py",
        original_code_snippet="def existing(): pass\n",
        updated_code_snippet="def add(a, b):\n    return a + b\n",
        explanation="Test valid code",
    )
    result = QualityPipeline.check_ast(str(tmp_path), [patch])
    assert result.status == QualityCheckStatus.PASS.value
    assert result.name == "ast"
    assert result.duration_ms >= 0


def test_quality_pipeline_ast_check_syntax_error(tmp_path):
    """Verify AST check fails for syntax error."""
    file_path = tmp_path / "broken.py"
    file_path.write_text("def existing(): pass\n", encoding="utf-8")

    patch = FilePatch(
        file_path="broken.py",
        original_code_snippet="def existing(): pass\n",
        updated_code_snippet="def broken_syntax(:\n    return\n",
        explanation="Test broken code",
    )
    result = QualityPipeline.check_ast(str(tmp_path), [patch])
    assert result.status == QualityCheckStatus.FAIL.value
    assert "SyntaxError" in (result.stderr_summary or "") or "validation failed" in (result.reason or "")

    # .py syntax errors must still be categorized as an AST/Python syntax
    # failure - the QA categorization fix must not weaken this.
    qa_result = StructuredQAJudge.evaluate([result])
    assert qa_result.status == "FAIL"
    assert qa_result.failure_category == FailureCategory.AST_FAILURE.value
    assert "python" in qa_result.summary.lower() and "syntax" in qa_result.summary.lower()


def test_quality_pipeline_ast_check_readme_snippet_mismatch_not_python_syntax(tmp_path):
    """
    README.md with a mismatched anchor snippet must still FAIL (patch
    pre-flight/anchor validation stays mandatory for every file type), but
    must be reported and categorized as a patch pre-flight failure, never
    as a Python syntax/AST error - README.md is never valid Python.
    """
    file_path = tmp_path / "README.md"
    file_path.write_text("# Project\n\nSome existing content.\n", encoding="utf-8")

    patch = FilePatch(
        file_path="README.md",
        original_code_snippet="## Section That Does Not Exist\n",
        updated_code_snippet="## New Section\n",
        explanation="Add a new section",
    )
    result = QualityPipeline.check_ast(str(tmp_path), [patch])
    assert result.status == QualityCheckStatus.FAIL.value
    assert "Target original snippet not found" in result.stderr_summary
    # The reason must not *claim* a Python syntax/AST problem - it may
    # honestly clarify that it is NOT one, so check the framing, not just
    # for the substring "python" (which also appears in that negation).
    assert "AST syntax validation failed" not in (result.reason or "")
    assert "pre-flight" in (result.reason or "").lower()

    qa_result = StructuredQAJudge.evaluate([result])
    assert qa_result.status == "FAIL"
    assert qa_result.failure_category != FailureCategory.AST_FAILURE.value
    assert "introduces python syntax or ast parsing errors" not in qa_result.summary.lower()
    assert "target" in qa_result.summary.lower() or "pre-flight" in qa_result.summary.lower()


def test_quality_pipeline_ast_check_readme_valid_patch_skips_python_parser(tmp_path, monkeypatch):
    """A valid README.md patch must pass pre-flight validation without the
    Python AST parser ever being invoked on Markdown content."""
    import backend.developer.patcher as patcher_module

    file_path = tmp_path / "README.md"
    file_path.write_text("# Project\n\nSome existing content.\n", encoding="utf-8")

    parse_calls = []
    original_validate = patcher_module.validate_python_syntax

    def spy_validate(code):
        parse_calls.append(code)
        return original_validate(code)

    monkeypatch.setattr(patcher_module, "validate_python_syntax", spy_validate)

    patch = FilePatch(
        file_path="README.md",
        original_code_snippet="Some existing content.\n",
        updated_code_snippet="Some updated content.\n",
        explanation="Update content",
    )
    result = QualityPipeline.check_ast(str(tmp_path), [patch])
    assert result.status == QualityCheckStatus.PASS.value
    assert parse_calls == []


def test_quality_pipeline_ast_check_mixed_readme_and_python(tmp_path):
    """
    Mixed changeset: README.md anchor validation and .py AST validation are
    both mandatory and independently reported - one passing/being skipped
    must never hide or weaken the other.
    """
    readme_path = tmp_path / "README.md"
    readme_path.write_text("# Project\n\nExisting.\n", encoding="utf-8")
    py_path = tmp_path / "broken.py"
    py_path.write_text("def existing(): pass\n", encoding="utf-8")

    readme_patch = FilePatch(
        file_path="README.md",
        original_code_snippet="## Missing Section\n",  # mismatched anchor
        updated_code_snippet="## New Section\n",
        explanation="Add section",
    )
    py_patch = FilePatch(
        file_path="broken.py",
        original_code_snippet="def existing(): pass\n",
        updated_code_snippet="def broken_syntax(:\n    return\n",
        explanation="Broken",
    )

    result = QualityPipeline.check_ast(str(tmp_path), [readme_patch, py_patch])
    assert result.status == QualityCheckStatus.FAIL.value
    assert "README.md" in result.stderr_summary
    assert "Target original snippet not found" in result.stderr_summary
    assert "broken.py" in result.stderr_summary
    assert "SyntaxError" in result.stderr_summary

    qa_result = StructuredQAJudge.evaluate([result])
    assert qa_result.status == "FAIL"
    # A real Python syntax problem is present alongside the README failure -
    # it must not be downgraded, hidden, or miscategorized.
    assert qa_result.failure_category == FailureCategory.AST_FAILURE.value


def test_quality_pipeline_security_check_clean(tmp_path):
    """Verify static security check passes for safe code."""
    patch = FilePatch(
        file_path="safe_calc.py",
        original_code_snippet="",
        updated_code_snippet="def multiply(x, y):\n    return x * y\n",
        explanation="Safe math function",
    )
    result = QualityPipeline.check_security(str(tmp_path), [patch])
    assert result.status == QualityCheckStatus.PASS.value
    assert result.name == "security"


def test_quality_pipeline_security_check_dangerous_eval(tmp_path):
    """Verify static security check flags eval/exec calls."""
    patch = FilePatch(
        file_path="dangerous.py",
        original_code_snippet="",
        updated_code_snippet="def run_code(user_input):\n    return eval(user_input)\n",
        explanation="Uses eval",
    )
    result = QualityPipeline.check_security(str(tmp_path), [patch])
    assert result.status == QualityCheckStatus.FAIL.value
    assert "eval" in (result.stderr_summary or "")


def test_quality_pipeline_security_check_shell_true(tmp_path):
    """Verify static security check flags shell=True in subprocess."""
    patch = FilePatch(
        file_path="shell_injection.py",
        original_code_snippet="",
        updated_code_snippet="import subprocess\ndef execute(cmd):\n    subprocess.run(cmd, shell=True)\n",
        explanation="Subprocess with shell",
    )
    result = QualityPipeline.check_security(str(tmp_path), [patch])
    assert result.status == QualityCheckStatus.FAIL.value
    assert "shell=True" in (result.stderr_summary or "")


def test_quality_pipeline_security_check_hardcoded_token(tmp_path):
    """Verify static security check flags detected secret patterns."""
    patch = FilePatch(
        file_path="config.py",
        original_code_snippet="",
        updated_code_snippet="SECRET_KEY = 'ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'\n",
        explanation="Leaked token",
    )
    result = QualityPipeline.check_security(str(tmp_path), [patch])
    assert result.status == QualityCheckStatus.FAIL.value
    assert "credential or token" in (result.stderr_summary or "")


def test_quality_pipeline_optional_tool_graceful(tmp_path):
    """Verify missing lint or typecheck tools return SKIPPED without failing."""
    with patch("shutil.which", return_value=None):
        lint_res = QualityPipeline.check_lint(str(tmp_path), ["sample.py"])
        assert lint_res.status == QualityCheckStatus.SKIPPED.value
        assert "NOT_AVAILABLE" in (lint_res.reason or "")

        type_res = QualityPipeline.check_typecheck(str(tmp_path), ["sample.py"])
        assert type_res.status == QualityCheckStatus.SKIPPED.value
        assert "NOT_AVAILABLE" in (type_res.reason or "")


def test_quality_pipeline_run_all_success(tmp_path):
    """Verify run_all runs checks and aggregates results."""
    mock_test_res = TestExecutionResult(
        success=True,
        exit_code=0,
        passed_count=3,
        failed_count=0,
        stdout="3 passed in 0.05s",
        stderr="",
        duration_seconds=0.05,
    )

    with patch.object(QualityPipeline, "check_pytest", return_value=(
        QualityCheck(name="pytest", status=QualityCheckStatus.PASS.value, exit_code=0),
        mock_test_res
    )):
        patch_item = FilePatch(
            file_path="clean.py",
            original_code_snippet="",
            updated_code_snippet="def compute():\n    return 42\n",
            explanation="Clean code",
        )
        checks, test_res = QualityPipeline.run_all(str(tmp_path), [patch_item])
        assert len(checks) >= 3  # AST, pytest, security, + optional tools
        assert all(c.status in [QualityCheckStatus.PASS.value, QualityCheckStatus.SKIPPED.value] for c in checks)
        assert test_res is not None
        assert test_res.success is True


# ---------------------------------------------------------------------------
# Structured QA Judge Unit Tests
# ---------------------------------------------------------------------------


def test_judge_all_checks_pass():
    """When all objective checks pass, QA Judge returns PASS with high confidence."""
    checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="pytest", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="security", status=QualityCheckStatus.PASS.value),
    ]

    qa_result = StructuredQAJudge.evaluate(checks)
    assert qa_result.status == "PASS"
    assert qa_result.confidence >= 0.85
    assert qa_result.regression_risk == "LOW"
    assert qa_result.failure_category is None


def test_judge_objective_priority_overrides_llm():
    """
    CRITICAL INVARIANT:
    Even if LLM or external agent is optimistic, an objective test failure
    MUST produce FAIL and TEST_FAILURE category.
    """
    checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(
            name="pytest",
            status=QualityCheckStatus.FAIL.value,
            stderr_summary="AssertionError: 4 != 5",
            reason="AssertionError: 4 != 5",
        ),
        QualityCheck(name="security", status=QualityCheckStatus.PASS.value),
    ]

    # Simulate optimistic LLM review that thought everything looked fine
    optimistic_llm_result = QAResult(
        status="PASS",
        confidence=0.99,
        summary="Code changes look clean and well written.",
    )

    qa_result = StructuredQAJudge.evaluate(checks, llm_qa_result=optimistic_llm_result)
    assert qa_result.status == "FAIL"
    assert qa_result.failure_category == FailureCategory.TEST_FAILURE.value
    assert qa_result.confidence <= 0.40
    assert "pytest" in qa_result.summary.lower() or "assertionerror" in qa_result.summary.lower()


def test_judge_ast_failure():
    """AST failure causes immediate FAIL and AST_FAILURE category."""
    checks = [
        QualityCheck(
            name="ast",
            status=QualityCheckStatus.FAIL.value,
            stderr_summary="SyntaxError: invalid syntax",
            reason="Syntax error on line 3",
        ),
        QualityCheck(name="pytest", status=QualityCheckStatus.SKIPPED.value),
        QualityCheck(name="security", status=QualityCheckStatus.SKIPPED.value),
    ]

    qa_result = StructuredQAJudge.evaluate(checks)
    assert qa_result.status == "FAIL"
    assert qa_result.failure_category == FailureCategory.AST_FAILURE.value
    assert qa_result.regression_risk in ["HIGH", "CRITICAL"]


def test_judge_security_failure():
    """Security failure triggers CRITICAL regression risk and SECURITY_FAILURE category."""
    checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(
            name="security",
            status=QualityCheckStatus.FAIL.value,
            stderr_summary="Dangerous builtin 'eval()' called in script.py.",
            reason="Dangerous builtin 'eval()' called in script.py.",
        ),
        QualityCheck(name="pytest", status=QualityCheckStatus.PASS.value),
    ]

    qa_result = StructuredQAJudge.evaluate(checks)
    assert qa_result.status == "FAIL"
    assert qa_result.failure_category == FailureCategory.SECURITY_FAILURE.value
    assert qa_result.regression_risk == "CRITICAL"
    assert qa_result.confidence <= 0.30


def test_judge_confidence_bounds():
    """Confidence must strictly satisfy 0.0 <= confidence <= 1.0."""
    # Worst case: multiple failures
    bad_checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.FAIL.value),
        QualityCheck(name="pytest", status=QualityCheckStatus.FAIL.value),
        QualityCheck(name="security", status=QualityCheckStatus.FAIL.value),
        QualityCheck(name="lint", status=QualityCheckStatus.FAIL.value),
        QualityCheck(name="typecheck", status=QualityCheckStatus.FAIL.value),
    ]
    qa_bad = StructuredQAJudge.evaluate(bad_checks)
    assert 0.0 <= qa_bad.confidence <= 1.0

    # Best case: all passed
    good_checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="pytest", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="security", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="lint", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="typecheck", status=QualityCheckStatus.PASS.value),
    ]
    qa_good = StructuredQAJudge.evaluate(good_checks)
    assert 0.0 <= qa_good.confidence <= 1.0
    assert qa_good.confidence >= 0.85


# ---------------------------------------------------------------------------
# LLM advisory FAIL must never leave a contradictory "all passed" summary
# (run_3ab2193a5e3a: objective checks all PASS/SKIPPED, LLM advisory returned
# status="FAIL" with no summary/issues, and the resulting QAResult kept the
# "All required quality and sandbox test checks passed." placeholder text
# next to status="FAIL".)
# ---------------------------------------------------------------------------


def test_judge_objective_pass_llm_pass_keeps_passed_summary():
    """Baseline, unchanged: objective checks pass and the LLM advisory also
    passes -> the normal 'all passed' summary is used, status PASS."""
    checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="pytest", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="security", status=QualityCheckStatus.PASS.value),
    ]
    llm_result = QAResult(status="PASS", confidence=0.9, summary="Looks good.")

    qa_result = StructuredQAJudge.evaluate(checks, llm_qa_result=llm_result)
    assert qa_result.status == "PASS"
    assert "passed" in qa_result.summary.lower()


def test_judge_llm_fail_without_explanation_produces_honest_summary():
    """
    Objective checks all pass, but the LLM advisory review returns FAIL with
    no summary and no issues. The final summary must never claim everything
    passed next to a FAIL status - it must clearly say the LLM advisory
    review failed without an explanation.
    """
    checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="pytest", status=QualityCheckStatus.SKIPPED.value),
        QualityCheck(name="security", status=QualityCheckStatus.PASS.value),
    ]
    llm_result = QAResult(status="FAIL", confidence=0.5, summary="", issues=[])

    qa_result = StructuredQAJudge.evaluate(checks, llm_qa_result=llm_result)
    assert qa_result.status == "FAIL"
    assert qa_result.confidence <= 0.40
    assert "passed" not in qa_result.summary.lower()
    assert "fail" in qa_result.summary.lower()
    assert "explanation" in qa_result.summary.lower() or "without" in qa_result.summary.lower()


def test_judge_llm_fail_with_meaningful_summary_is_preserved():
    """
    When the LLM advisory review returns FAIL WITH a real summary, that
    explanation must be used/preserved in the final summary - not replaced
    by the generic 'failed without an explanation' fallback.
    """
    checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="pytest", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="security", status=QualityCheckStatus.PASS.value),
    ]
    llm_result = QAResult(
        status="FAIL",
        confidence=0.3,
        summary="The change does not address the user's request.",
    )

    qa_result = StructuredQAJudge.evaluate(checks, llm_qa_result=llm_result)
    assert qa_result.status == "FAIL"
    assert "does not address the user's request" in qa_result.summary
    assert "passed" not in qa_result.summary.lower()


def test_judge_llm_fail_with_issues_but_no_summary_avoids_passed_wording():
    """
    LLM advisory FAIL with issues but no summary text: the issues are real
    information, but the final summary must still never claim everything
    passed next to a FAIL status.
    """
    checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="pytest", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="security", status=QualityCheckStatus.PASS.value),
    ]
    llm_result = QAResult(
        status="FAIL",
        confidence=0.5,
        summary="",
        issues=[QAIssue(file_path="app.py", issue="Missing null check", severity="MEDIUM")],
    )

    qa_result = StructuredQAJudge.evaluate(checks, llm_qa_result=llm_result)
    assert qa_result.status == "FAIL"
    assert "passed" not in qa_result.summary.lower()


# ---------------------------------------------------------------------------
# LangGraph Node & Telemetry Integration Tests
# ---------------------------------------------------------------------------


def test_qa_node_with_passing_checks():
    """Verify qa_node executes QualityPipeline and updates state with QAResult."""
    state: AgentState = {
        "user_message": "Fix bug",
        "project_id": "test_proj",
        "plan": ExecutionPlan(
            goal="Fix bug",
            steps=[PlanStep(step_number=1, action="Fix it", agent="developer")],
            success_criteria="Done"
        ),
        "developer_result": DeveloperResult(summary="Patch generated", changes=[], requires_testing=True, notes=[]),
        "generated_patches": [
            FilePatch(
                file_path="sample.py",
                original_code_snippet="def add(a, b): return a\n",
                updated_code_snippet="def add(a, b): return a + b\n",
                explanation="Fix addition",
            )
        ],
    }

    mock_checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="pytest", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="security", status=QualityCheckStatus.PASS.value),
    ]
    mock_test_res = TestExecutionResult(
        success=True,
        exit_code=0,
        passed_count=1,
        failed_count=0,
        stdout="1 passed",
        stderr="",
        duration_seconds=0.1,
    )

    with patch("backend.graph.nodes.review_code_changes") as mock_review, \
         patch("backend.qa.pipeline.QualityPipeline.run_all", return_value=(mock_checks, mock_test_res)):
        mock_review.return_value = QAResult(status="PASS", summary="Approved")

        res = qa_node(state)

        assert "qa_result" in res
        assert res["qa_result"].status == "PASS"
        assert res["test_result"].success is True


def test_qa_node_with_failing_checks():
    """Verify qa_node properly handles test failure."""
    state: AgentState = {
        "user_message": "Fix bug",
        "project_id": "test_proj",
        "plan": ExecutionPlan(
            goal="Fix bug",
            steps=[PlanStep(step_number=1, action="Fix it", agent="developer")],
            success_criteria="Done"
        ),
        "developer_result": DeveloperResult(summary="Patch generated", changes=[], requires_testing=True, notes=[]),
        "generated_patches": [
            FilePatch(
                file_path="sample.py",
                original_code_snippet="def add(a, b): return a\n",
                updated_code_snippet="def add(a, b): return a - b\n",
                explanation="Broken addition",
            )
        ],
    }

    mock_checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="pytest", status=QualityCheckStatus.FAIL.value, stderr_summary="AssertionError: 1 != 2"),
        QualityCheck(name="security", status=QualityCheckStatus.PASS.value),
    ]
    mock_test_res = TestExecutionResult(
        success=False,
        exit_code=1,
        passed_count=0,
        failed_count=1,
        stdout="",
        stderr="AssertionError: 1 != 2",
        duration_seconds=0.1,
        error_summary="AssertionError: 1 != 2",
    )

    with patch("backend.graph.nodes.review_code_changes") as mock_review, \
         patch("backend.qa.pipeline.QualityPipeline.run_all", return_value=(mock_checks, mock_test_res)):
        mock_review.return_value = QAResult(status="PASS", summary="Looks good to LLM")

        res = qa_node(state)

        assert "qa_result" in res
        assert res["qa_result"].status == "FAIL"
        assert res["test_result"].success is False
        assert res["qa_result"].failure_category == FailureCategory.TEST_FAILURE.value


def test_qa_router_deterministic_decisions():
    """Verify qa_router accurately routes to pass, fail, or max_retries."""
    # Scenario 1: Passing QA
    state_pass: AgentState = {
        "user_message": "Fix bug",
        "qa_result": QAResult(status="PASS"),
        "revision_count": 0,
    }
    assert qa_router(state_pass) == "pass"

    # Scenario 2: Failing QA with retries remaining
    state_fail: AgentState = {
        "user_message": "Fix bug",
        "qa_result": QAResult(status="FAIL"),
        "revision_count": 1,
    }
    assert qa_router(state_fail) == "fail"

    # Scenario 3: Failing QA at MAX_REVISIONS
    state_max: AgentState = {
        "user_message": "Fix bug",
        "qa_result": QAResult(status="FAIL"),
        "revision_count": 3,  # MAX_REVISIONS is 3
    }
    assert qa_router(state_max) == "max_retries"


def test_revision_node_records_structured_telemetry():
    """Verify revision_node captures failed_check, failure_category, and patch hashes."""
    failed_qa = QAResult(
        status="FAIL",
        failure_category=FailureCategory.TEST_FAILURE.value,
        summary="Test failed: test_calc.py",
        checks=[
            QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
            QualityCheck(name="pytest", status=QualityCheckStatus.FAIL.value, stderr_summary="AssertionError: 2 != 3"),
        ],
    )
    old_patch = FilePatch(
        file_path="calc.py",
        original_code_snippet="def f(): return 1\n",
        updated_code_snippet="def f(): return 2\n",
        explanation="Attempt 1",
    )
    revised_patch = FilePatch(
        file_path="calc.py",
        original_code_snippet="def f(): return 2\n",
        updated_code_snippet="def f(): return 3\n",
        explanation="Attempt 2 with fix",
    )

    state: AgentState = {
        "user_message": "Fix issue",
        "project_id": "test_proj",
        "plan": ExecutionPlan(
            goal="Fix issue",
            steps=[PlanStep(step_number=1, action="Revise code", agent="developer")],
            success_criteria="Done"
        ),
        "developer_result": DeveloperResult(summary="Initial", changes=[], requires_testing=True, notes=[]),
        "generated_patches": [old_patch],
        "qa_result": failed_qa,
        "revision_count": 0,
    }

    with patch("backend.agents.developer.revise_code_changes") as mock_revise, \
         patch("backend.agents.revision.generate_revision_patches", return_value=[revised_patch]):
        mock_revise.return_value = DeveloperResult(
            summary="Fixed return value to 3",
            changes=[],
            requires_testing=True,
            notes=[],
        )

        res = revision_node(state)

        assert res["revision_count"] == 1
        assert res["revision_history"] is not None
        attempts = res["revision_history"].attempts
        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt.attempt_number == 1
        assert attempt.failed_check == "pytest"
        assert attempt.failure_category == FailureCategory.TEST_FAILURE.value
        assert attempt.previous_patch_hash is not None
        assert attempt.new_patch_hash is not None
        assert attempt.duration_seconds >= 0.0
