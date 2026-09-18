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


class TestPytestNoTestsCollectedClassification:
    """Regression coverage for the confirmed AWS E2E finding: pytest exit
    code 5 ("no tests collected") was silently reported as an unqualified
    QualityCheckStatus.PASS, indistinguishable from a repository whose
    tests genuinely ran and passed. The project architecture intentionally
    permits repositories without a test suite (e.g. a fresh documentation
    repo like king-man1905/agentic-ai-test-repo), so this must not be
    treated as a FAILURE either - but it must be explicitly classified, not
    silently folded into PASS, consistent with the SKIPPED semantics
    already used elsewhere in this same pipeline (check_lint/check_typecheck
    when the tool itself isn't installed)."""

    def test_check_pytest_real_no_tests_collected_is_skipped_not_pass(self, tmp_path):
        """A. Real pytest execution (no mocking) against a repository with
        no test files at all must produce exit_code 5, and check_pytest
        must classify it as SKIPPED with an explicit reason - never PASS."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "README.md").write_text("# no tests here\n", encoding="utf-8")

        check, test_result = QualityPipeline.check_pytest(str(repo), timeout=30.0)

        assert test_result is not None
        assert test_result.exit_code == 5
        assert check.status == QualityCheckStatus.SKIPPED.value
        assert check.exit_code == 5
        assert "no tests collected" in (check.reason or "").lower()
        assert "not treated as a failure" in (check.reason or "").lower()

    def test_check_pytest_real_genuine_pass_is_still_pass(self, tmp_path):
        """B. A repository whose tests genuinely run and pass must still
        report PASS (with exit_code 0), unaffected by the exit-code-5
        classification change."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test_ok.py").write_text(
            "def test_ok():\n    assert 1 == 1\n", encoding="utf-8"
        )

        check, test_result = QualityPipeline.check_pytest(str(repo), timeout=30.0)

        assert test_result is not None
        assert test_result.exit_code == 0
        assert check.status == QualityCheckStatus.PASS.value
        assert check.reason is None

    def test_check_pytest_real_genuine_failure_is_still_fail(self, tmp_path):
        """C. A repository whose tests genuinely fail must still report
        FAIL (with the real, non-5 exit code), unaffected by the
        exit-code-5 classification change."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test_bad.py").write_text(
            "def test_bad():\n    assert 1 == 2\n", encoding="utf-8"
        )

        check, test_result = QualityPipeline.check_pytest(str(repo), timeout=30.0)

        assert test_result is not None
        assert test_result.exit_code not in (0, 5)
        assert check.status == QualityCheckStatus.FAIL.value
        assert check.reason

    def test_no_tests_collected_does_not_block_overall_qa_pass(self, tmp_path):
        """D. StructuredQAJudge must not treat a SKIPPED pytest check (no
        tests collected) as a blocking failure - a documentation-only
        change to a test-less repository can still reach an overall PASS,
        exactly as the architecture intends for repositories without a
        test suite. SKIPPED checks are already non-blocking (only a small
        confidence reduction) via the same code path lint/typecheck use."""
        no_tests_check = QualityCheck(
            name="pytest",
            status=QualityCheckStatus.SKIPPED.value,
            exit_code=5,
            reason="No tests collected (pytest exit code 5) - repository has no test suite; not treated as a failure.",
        )
        ast_check = QualityCheck(name="ast", status=QualityCheckStatus.PASS.value, exit_code=0)
        security_check = QualityCheck(name="security", status=QualityCheckStatus.PASS.value, exit_code=0)

        result = StructuredQAJudge.evaluate(checks=[ast_check, no_tests_check, security_check])

        assert result.status == "PASS"
        # A SKIPPED check must not silently read as "tests were verified" -
        # the objective evidence (exit_code=5, 0 items collected) is
        # preserved and inspectable on the check itself.
        pytest_result_check = next(c for c in result.checks if c.name == "pytest")
        assert pytest_result_check.status == QualityCheckStatus.SKIPPED.value
        assert pytest_result_check.exit_code == 5

    def test_no_tests_collected_is_distinguishable_from_genuine_pass_in_summary_data(self):
        """E. The objective evidence (exit_code + status) must make it
        possible to tell "no tests ran" apart from "tests ran and passed"
        purely from the QAResult - the exact gap the AWS E2E run exposed
        (QA reported PASS with no visible distinction from a real test
        pass)."""
        genuine_pass_check = QualityCheck(name="pytest", status=QualityCheckStatus.PASS.value, exit_code=0)
        no_tests_check = QualityCheck(
            name="pytest", status=QualityCheckStatus.SKIPPED.value, exit_code=5,
            reason="No tests collected (pytest exit code 5) - repository has no test suite; not treated as a failure.",
        )

        assert genuine_pass_check.status != no_tests_check.status
        assert genuine_pass_check.exit_code != no_tests_check.exit_code


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


# ============================================================================
# P0-2: SECURITY GATE MUST BLOCK EXECUTION, EXECUTION MUST BE ISOLATED
# ============================================================================

class TestSecurityGateBlocksExecution:
    def test_dangerous_patch_blocks_pytest_lint_typecheck_and_they_never_execute(self, tmp_path):
        """
        A patch whose content trips check_security must prevent
        check_pytest/check_lint/check_typecheck from running at all - not
        just be reported as a failure alongside them. Proven concretely: a
        real test file on disk that would create a sentinel file if pytest
        ever actually collected and ran it must never create that sentinel.
        """
        sentinel = tmp_path / "PYTEST_ACTUALLY_RAN.marker"
        dangerous_code = (
            "import os\n"
            f"os.system({str(sentinel)!r} and 'echo pwned > ' + {str(sentinel)!r})\n"
            "\n"
            "def test_noop():\n"
            "    assert True\n"
        )
        (tmp_path / "test_evil.py").write_text(dangerous_code, encoding="utf-8")

        patch = FilePatch(
            file_path="test_evil.py",
            original_code_snippet="",
            updated_code_snippet=dangerous_code,
            explanation="malicious patch containing os.system",
        )

        checks, test_result = QualityPipeline.run_all(str(tmp_path), [patch])
        by_name = {c.name: c for c in checks}

        assert by_name["security"].status == QualityCheckStatus.FAIL.value
        assert by_name["pytest"].status == QualityCheckStatus.SKIPPED.value
        assert "BLOCKED_BY_SECURITY_GATE" in (by_name["pytest"].reason or "")
        assert by_name["lint"].status == QualityCheckStatus.SKIPPED.value
        assert "BLOCKED_BY_SECURITY_GATE" in (by_name["lint"].reason or "")
        assert by_name["typecheck"].status == QualityCheckStatus.SKIPPED.value
        assert "BLOCKED_BY_SECURITY_GATE" in (by_name["typecheck"].reason or "")
        assert test_result is None

        # The concrete proof: the dangerous os.system() call was never
        # actually executed by pytest, because pytest was never run.
        assert not sentinel.exists()


class TestExecutionIsolatedFromPersistentWorkspace:
    def test_pytest_execution_cannot_modify_persistent_repo_state(self, tmp_path):
        """
        A safe (security-scan-passing) test file that writes a marker via
        plain file I/O when pytest actually collects and runs it must have
        that marker appear ONLY in the disposable execution copy, never in
        repo_path itself - proving execution happened under a different
        workspace path, not the persistent one that diff/commit/push trust.
        """
        repo = tmp_path / "repo"
        repo.mkdir()

        marker_name = "PYTEST_EXECUTION_MARKER.txt"
        safe_code = (
            "import os\n"
            "\n"
            "def test_writes_marker():\n"
            f"    with open(os.path.join(os.path.dirname(__file__), {marker_name!r}), 'w') as f:\n"
            "        f.write('pytest ran here')\n"
            "    assert True\n"
        )
        (repo / "test_writes_marker.py").write_text(safe_code, encoding="utf-8")

        patch = FilePatch(
            file_path="test_writes_marker.py",
            original_code_snippet="",
            updated_code_snippet=safe_code,
            explanation="legitimate test that writes a marker file when it runs",
        )

        checks, test_result = QualityPipeline.run_all(str(repo), [patch])
        by_name = {c.name: c for c in checks}

        assert by_name["security"].status == QualityCheckStatus.PASS.value
        assert by_name["pytest"].status == QualityCheckStatus.PASS.value
        assert test_result is not None
        assert test_result.success is True

        # The marker must never land in the real, persistent repo - only
        # (if anywhere, since isolated_workspace cleans up on exit) in a
        # disposable copy that no longer exists once run_all returns.
        assert not (repo / marker_name).exists()


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
