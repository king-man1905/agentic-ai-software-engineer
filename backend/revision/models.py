from typing import List, Optional
from pydantic import BaseModel, Field
from backend.developer.models import FilePatch


class RevisionAttempt(BaseModel):
    """
    Represents an individual revision iteration within the self-correction loop.
    """
    attempt_number: int = Field(
        description="1-based attempt counter for this revision iteration."
    )
    failing_tests: List[str] = Field(
        default_factory=list,
        description="List of failing test names/identifiers extracted from execution results."
    )
    error_traceback: str = Field(
        default="",
        description="Captured failure traceback, exception trace, or error summary."
    )
    applied_patch: Optional[FilePatch] = Field(
        default=None,
        description="The FilePatch generated and applied during this revision attempt."
    )
    diagnosis: str = Field(
        default="",
        description="Diagnosis of why the failure occurred and targeted instructions for the fix."
    )


class RevisionHistory(BaseModel):
    """
    Tracks complete history of revision attempts and enforces circuit breaker thresholds.
    """
    attempts: List[RevisionAttempt] = Field(
        default_factory=list,
        description="Chronological record of revision attempts."
    )
    max_retries: int = Field(
        default=3,
        description="Maximum allowed revision retries before circuit breaker trips."
    )
    is_exhausted: bool = Field(
        default=False,
        description="Indicates whether maximum retries have been reached or exceeded."
    )

    def add_attempt(self, attempt: RevisionAttempt) -> None:
        """
        Appends a revision attempt and updates the is_exhausted flag according to max_retries.
        """
        self.attempts.append(attempt)
        if len(self.attempts) >= self.max_retries:
            self.is_exhausted = True

    @property
    def latest_attempt(self) -> Optional[RevisionAttempt]:
        """Returns the most recent RevisionAttempt, if any."""
        return self.attempts[-1] if self.attempts else None

    @property
    def total_attempts(self) -> int:
        """Returns total count of revision attempts made so far."""
        return len(self.attempts)


class ParsedFailure(BaseModel):
    """
    Structured breakdown of a single test failure parsed from pytest output or error traceback.
    """
    test_file: str = Field(
        default="",
        description="Path of the test or source file where failure was triggered."
    )
    test_name: str = Field(
        default="",
        description="Name of the test function, class method, or module."
    )
    full_test_id: str = Field(
        default="",
        description="Full test identifier, e.g. 'tests/test_foo.py::test_bar'."
    )
    exception_type: Optional[str] = Field(
        default=None,
        description="Exception type name, e.g. 'AssertionError', 'ZeroDivisionError'."
    )
    error_message: str = Field(
        default="",
        description="Specific error message, assertion expression, or diff."
    )
    target_lines: List[int] = Field(
        default_factory=list,
        description="Line numbers in test/source files directly associated with the failure."
    )
    traceback_snippet: str = Field(
        default="",
        description="Concise snippet of the traceback corresponding to this failure."
    )


class ErrorTraceAnalysis(BaseModel):
    """
    Aggregated outcome of analyzing test execution error traces.
    """
    failing_tests: List[str] = Field(
        default_factory=list,
        description="List of all unique failing test identifiers."
    )
    failures: List[ParsedFailure] = Field(
        default_factory=list,
        description="List of structured parsed failure details."
    )
    error_traceback: str = Field(
        default="",
        description="Cleaned or raw error traceback string."
    )
    diagnosis: str = Field(
        default="",
        description="Diagnostic summary describing the root causes and target locations to fix."
    )
