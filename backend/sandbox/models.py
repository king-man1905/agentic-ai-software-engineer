from typing import Optional
from pydantic import BaseModel, Field


class TestExecutionResult(BaseModel):
    """
    Represents the structured outcome of running a test suite.
    """
    success: bool = Field(description="Whether the command exited successfully (code 0).")
    exit_code: int = Field(description="Process exit code.")
    passed_count: int = Field(description="Number of passing test cases parsed from stdout.")
    failed_count: int = Field(description="Number of failing test cases parsed from stdout.")
    stdout: str = Field(description="Standard output of the execution.")
    stderr: str = Field(description="Standard error of the execution.")
    duration_seconds: float = Field(description="Total execution time in seconds.")
    error_summary: Optional[str] = Field(
        default=None, description="Concise failure traceback or error message summary."
    )
