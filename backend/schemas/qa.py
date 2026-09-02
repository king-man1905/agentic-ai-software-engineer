from pydantic import BaseModel, Field


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
        description="Validation result: PASS or FAIL."
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
        description="Short explanation of the QA result."
    )