from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class QualityCheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class FailureCategory(str, Enum):
    AST_FAILURE = "AST_FAILURE"
    PATCH_APPLICATION_FAILURE = "PATCH_APPLICATION_FAILURE"
    TEST_FAILURE = "TEST_FAILURE"
    LINT_FAILURE = "LINT_FAILURE"
    TYPECHECK_FAILURE = "TYPECHECK_FAILURE"
    SECURITY_FAILURE = "SECURITY_FAILURE"
    TIMEOUT = "TIMEOUT"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class QualityCheck(BaseModel):
    """
    Structured outcome of an individual validation check (AST, pytest, lint, typecheck, security).
    """
    name: str = Field(
        description="Check identifier: ast, pytest, lint, typecheck, or security."
    )
    status: str = Field(
        description="Check status: PASS, FAIL, SKIPPED, or ERROR."
    )
    exit_code: int = Field(
        default=0,
        description="Process exit code, or 0 for internal Python checks."
    )
    duration_ms: int = Field(
        default=0,
        description="Execution duration in milliseconds."
    )
    stdout_summary: str = Field(
        default="",
        description="Concise summary of standard output."
    )
    stderr_summary: str = Field(
        default="",
        description="Concise summary of standard error."
    )
    reason: Optional[str] = Field(
        default=None,
        description="Explanation when status is SKIPPED or FAIL."
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "For a FAIL status, further classifies the cause (e.g. AST_FAILURE, "
            "PATCH_APPLICATION_FAILURE, or MIXED) so callers like the QA judge can "
            "report it accurately without re-deriving it from free-text messages."
        ),
    )


class QAIssue(BaseModel):
    file_path: str = Field(
        description="File associated with the issue."
    )

    issue: str = Field(
        description="Description of the problem."
    )

    severity: str = Field(
        description="Issue severity: LOW, MEDIUM, or HIGH."
    )


class QAResult(BaseModel):
    status: str = Field(
        default="PASS",
        description="Overall validation result: PASS, FAIL, or NEEDS_REVIEW."
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )

    regression_risk: str = Field(
        default="LOW",
        description="Assessed regression risk: LOW, MEDIUM, HIGH, or CRITICAL."
    )

    checks: List[QualityCheck] = Field(
        default_factory=list,
        description="Granular results for each executed or skipped quality check."
    )

    failure_category: Optional[str] = Field(
        default=None,
        description="High-level category if validation failed, e.g. TEST_FAILURE, AST_FAILURE."
    )

    issues: list[QAIssue] = Field(
        default_factory=list,
        description="Problems discovered during validation."
    )

    test_cases: list[str] = Field(
        default_factory=list,
        description="Tests that should be executed or verified."
    )

    summary: str = Field(
        default="",
        description="Short explanation of the QA result."
    )