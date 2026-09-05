from typing import List, Optional
from pydantic import BaseModel, Field


class GitDiffSummary(BaseModel):
    """
    Structured summary of staged Git changes including unified diff and risk assessment.
    """
    branch_name: str = Field(
        description="Name of the feature branch where changes are staged."
    )
    files_changed: List[str] = Field(
        default_factory=list,
        description="List of file paths affected by the changes."
    )
    lines_added: int = Field(
        default=0,
        description="Total number of lines added across all changed files."
    )
    lines_deleted: int = Field(
        default=0,
        description="Total number of lines deleted across all changed files."
    )
    unified_diff: str = Field(
        default="",
        description="Complete unified diff text representing all changes."
    )
    risk_score: str = Field(
        default="LOW",
        description="Assessed risk level: LOW, MEDIUM, or HIGH."
    )
    risk_reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable explanations for the risk assessment."
    )
    patch_hash: str = Field(
        default="",
        description="SHA-256 cryptographic fingerprint of the unified diff."
    )

    @property
    def is_no_op(self) -> bool:
        """True when no file changes and no diff content were actually produced."""
        return not self.files_changed or not self.unified_diff.strip()


class ApprovalDecision(BaseModel):
    """
    Represents a human reviewer's approval or rejection of proposed changes.
    """
    approved: bool = Field(
        description="Whether the reviewer approved the changes."
    )
    reviewer: Optional[str] = Field(
        default=None,
        description="Identifier or name of the reviewer."
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Reason for rejection, if the changes were not approved."
    )
    patch_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the diff approved by the reviewer."
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of approval submission."
    )
    reviewer_role: Optional[str] = Field(
        default=None,
        description="RBAC role of the reviewer (e.g. OWNER, ADMIN, REVIEWER, SECURITY_REVIEWER)."
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier of the reviewer."
    )
