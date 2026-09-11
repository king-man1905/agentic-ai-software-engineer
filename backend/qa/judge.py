from typing import List, Optional

from backend.developer.models import FilePatch
from backend.sandbox.models import TestExecutionResult
from backend.schemas.qa import (
    FailureCategory,
    QAIssue,
    QAResult,
    QualityCheck,
    QualityCheckStatus,
)


class StructuredQAJudge:
    """
    Evidence-driven Quality Assurance Judge.
    Evaluates objective sandbox verification checks first and enforces strict priority:
    LLM reasoning or optimism can NEVER override objective test or syntax failures.
    """

    @classmethod
    def evaluate(
        cls,
        checks: List[QualityCheck],
        test_result: Optional[TestExecutionResult] = None,
        patches: Optional[List[FilePatch]] = None,
        llm_qa_result: Optional[QAResult] = None,
    ) -> QAResult:
        """
        Synthesizes individual QualityCheck results into an immutable QAResult
        with calibrated confidence and regression risk.
        """
        check_map = {c.name: c for c in checks}
        issues: List[QAIssue] = []
        test_cases: List[str] = []

        # ---------------------------------------------------------------------
        # 1. Objective Evidence Evaluation (Strict Priority)
        # ---------------------------------------------------------------------
        ast_check = check_map.get("ast")
        pytest_check = check_map.get("pytest")
        security_check = check_map.get("security")
        lint_check = check_map.get("lint")
        type_check = check_map.get("typecheck")

        failed_checks: List[QualityCheck] = [
            c for c in checks if c.status == QualityCheckStatus.FAIL.value
        ]

        # Rule A: Security check failure is critical and blocking
        if security_check and security_check.status == QualityCheckStatus.FAIL.value:
            issues.append(
                QAIssue(
                    file_path="security",
                    issue=security_check.reason or security_check.stderr_summary or "Security violation detected.",
                    severity="HIGH",
                )
            )
            return QAResult(
                status="FAIL",
                confidence=0.1,
                regression_risk="CRITICAL",
                checks=checks,
                failure_category=FailureCategory.SECURITY_FAILURE.value,
                issues=issues,
                summary=f"Quality Gate FAILED: Security analysis identified blocking vulnerabilities.\n{security_check.stderr_summary}",
            )

        # Rule B: AST/patch pre-flight validation failure. check_ast's
        # underlying SafePatcher.apply_patch enforces two different things
        # behind one "ast" stage: the anchor/snippet must exist in the
        # target file (every file type, mandatory), and - only for .py
        # files - the patched content must be valid Python. `category`
        # (set by check_ast) says which actually failed, so a non-Python
        # file's snippet mismatch is never reported as a Python/AST error.
        if ast_check and ast_check.status == QualityCheckStatus.FAIL.value:
            issues.append(
                QAIssue(
                    file_path="ast",
                    issue=ast_check.stderr_summary or "AST pre-flight syntax error.",
                    severity="HIGH",
                )
            )
            check_category = getattr(ast_check, "category", None)
            if check_category == FailureCategory.PATCH_APPLICATION_FAILURE.value:
                result_category = FailureCategory.PATCH_APPLICATION_FAILURE.value
                summary_prefix = "Quality Gate FAILED: Proposed patch failed pre-flight validation (target snippet not found in source file)."
            elif check_category == "MIXED":
                result_category = FailureCategory.AST_FAILURE.value
                summary_prefix = (
                    "Quality Gate FAILED: Proposed patch failed pre-flight validation on one or more "
                    "files, and introduces Python syntax or AST parsing errors on at least one .py file."
                )
            else:
                result_category = FailureCategory.AST_FAILURE.value
                summary_prefix = "Quality Gate FAILED: Proposed patch introduces Python syntax or AST parsing errors."
            return QAResult(
                status="FAIL",
                confidence=0.15,
                regression_risk="HIGH",
                checks=checks,
                failure_category=result_category,
                issues=issues,
                summary=f"{summary_prefix}\n{ast_check.stderr_summary}",
            )

        # Rule C: Pytest execution failure (LLM can NEVER override this)
        if pytest_check and pytest_check.status == QualityCheckStatus.FAIL.value:
            err_text = pytest_check.stderr_summary or pytest_check.reason or "Test suite execution failed."
            is_timeout = "timeout" in err_text.lower()
            category = FailureCategory.TIMEOUT.value if is_timeout else FailureCategory.TEST_FAILURE.value

            issues.append(
                QAIssue(
                    file_path="tests",
                    issue=err_text[:300],
                    severity="HIGH",
                )
            )
            return QAResult(
                status="FAIL",
                confidence=0.25,
                regression_risk="HIGH",
                checks=checks,
                failure_category=category,
                issues=issues,
                summary=f"Quality Gate FAILED: Sandbox pytest execution failed.\n{err_text}",
            )

        # Rule D: Mandatory Linter failure (if configured and failed)
        if lint_check and lint_check.status == QualityCheckStatus.FAIL.value:
            issues.append(
                QAIssue(
                    file_path="linter",
                    issue=lint_check.stderr_summary or "Style or lint violations detected.",
                    severity="MEDIUM",
                )
            )
            return QAResult(
                status="FAIL",
                confidence=0.45,
                regression_risk="MEDIUM",
                checks=checks,
                failure_category=FailureCategory.LINT_FAILURE.value,
                issues=issues,
                summary=f"Quality Gate FAILED: Linter reported code quality violations.\n{lint_check.stderr_summary}",
            )

        # Rule E: Type check failure (if configured and failed)
        if type_check and type_check.status == QualityCheckStatus.FAIL.value:
            issues.append(
                QAIssue(
                    file_path="typecheck",
                    issue=type_check.stderr_summary or "Type checking errors detected.",
                    severity="MEDIUM",
                )
            )
            return QAResult(
                status="FAIL",
                confidence=0.5,
                regression_risk="MEDIUM",
                checks=checks,
                failure_category=FailureCategory.TYPECHECK_FAILURE.value,
                issues=issues,
                summary=f"Quality Gate FAILED: Type checker reported type errors.\n{type_check.stderr_summary}",
            )

        # ---------------------------------------------------------------------
        # 2. All Objective Checks Passed (or Skipped)
        # ---------------------------------------------------------------------
        confidence = 0.95
        regression_risk = "LOW"

        # If pytest collected and verified actual tests, boost confidence
        if test_result and test_result.passed_count > 0:
            confidence = min(0.99, 0.92 + min(test_result.passed_count * 0.01, 0.07))

        # If optional checks were skipped, adjust confidence slightly
        skipped_count = sum(1 for c in checks if c.status == QualityCheckStatus.SKIPPED.value)
        if skipped_count > 0:
            confidence = max(0.85, confidence - (skipped_count * 0.03))

        # Assess regression risk based on changeset size
        if patches:
            total_lines = sum(
                len((p.updated_code_snippet or "").splitlines()) for p in patches
            )
            if len(patches) > 4 or total_lines > 150:
                regression_risk = "MEDIUM"

        # ---------------------------------------------------------------------
        # 3. Integrate LLM Review (Supplementary Reasoning)
        # ---------------------------------------------------------------------
        summary = "All required quality and sandbox test checks passed."
        failure_category = None
        if llm_qa_result:
            if llm_qa_result.issues:
                # Merge advisory issues
                for issue in llm_qa_result.issues:
                    issues.append(issue)
                    if issue.severity.upper() == "HIGH":
                        regression_risk = "HIGH"
                        confidence = max(0.40, confidence - 0.3)

            if llm_qa_result.test_cases:
                test_cases.extend(llm_qa_result.test_cases)

            if llm_qa_result.summary:
                summary = f"{summary} LLM Review: {llm_qa_result.summary}"

            llm_status = (getattr(llm_qa_result, "status", None) or "").upper()
            if llm_status == "FAIL" or any(i.severity.upper() == "HIGH" for i in issues):
                status = "FAIL"
                failure_category = getattr(llm_qa_result, "failure_category", None) or FailureCategory.UNKNOWN_FAILURE.value
                confidence = min(confidence, 0.40)
            else:
                status = "PASS"
        else:
            status = "PASS"

        return QAResult(
            status=status,
            confidence=round(confidence, 2),
            regression_risk=regression_risk,
            checks=checks,
            failure_category=failure_category,
            issues=issues,
            test_cases=test_cases,
            summary=summary,
        )
