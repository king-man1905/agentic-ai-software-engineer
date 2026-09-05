from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from backend.vcs.models import GitDiffSummary
from backend.schemas.qa import QAResult
from backend.schemas.policy import PolicyEvaluationResult


class CreateRunRequest(BaseModel):
    """
    Request body for creating a new agent run.
    """
    user_message: str = Field(
        description="The user's task or question for the agent to process."
    )
    project_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional project identifier for workspace-scoped operations "
            "(code patching, knowledge retrieval, VCS)."
        ),
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Optional tenant organization identifier.",
    )
    repository_id: Optional[str] = Field(
        default=None,
        description="Optional repository identifier.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary caller-supplied metadata attached to the run for observability.",
    )


class RunStatusResponse(BaseModel):
    """
    Describes the current status of an agent run.
    """
    run_id: str = Field(
        description="Unique identifier for this run (maps to LangGraph thread_id)."
    )
    status: str = Field(
        description=(
            "Current lifecycle status of the run. "
            "One of: RUNNING, WAITING_APPROVAL, COMPLETED, FAILED."
        )
    )
    current_node: Optional[str] = Field(
        default=None,
        description="The graph node the run is currently paused at, if applicable.",
    )
    git_diff: Optional[GitDiffSummary] = Field(
        default=None,
        description="Structured diff summary available when status is WAITING_APPROVAL.",
    )
    qa_result: Optional[QAResult] = Field(
        default=None,
        description="Structured quality assurance evaluation and check results.",
    )
    policy_result: Optional[PolicyEvaluationResult] = Field(
        default=None,
        description="Structured organization policy evaluation outcome.",
    )
    error_summary: Optional[str] = Field(
        default=None,
        description="Human-readable error summary when status is FAILED.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional human-readable informational message regarding the run dispatch or lifecycle.",
    )


class ResumeRunRequest(BaseModel):
    """
    Request body for resuming a paused run at the HITL approval gate.
    """
    approved: bool = Field(
        description="Whether the human reviewer approved the proposed changes."
    )
    reviewer: Optional[str] = Field(
        default=None,
        description="Identifier or name of the reviewer submitting this decision.",
    )
    reviewer_role: Optional[str] = Field(
        default=None,
        description="Optional role of the reviewer (e.g. REVIEWER, SECURITY_REVIEWER).",
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Reason for rejection. Required when approved=False for auditability.",
    )
    patch_hash: Optional[str] = Field(
        default=None,
        description="Optional SHA-256 hash of the approved diff for cryptographic verification.",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Optional tenant organization identifier for verification.",
    )


class AuditEventView(BaseModel):
    """
    Public schema for audit event query responses.
    """
    event_id: str
    organization_id: str
    user_id: str
    action: str
    timestamp: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any] = Field(default_factory=dict)
    event_hash: str
    previous_hash: str


class CreateApiKeyRequest(BaseModel):
    """
    Request body for creating an API key.
    """
    name: str = Field(default="default", description="Descriptive label for this key.")
    expires_in_days: Optional[int] = Field(default=None, description="Expiration TTL in days.")


class ApiKeyResponse(BaseModel):
    """
    Response schema for API key lifecycle operations.
    raw_key is presented only once upon creation/rotation.
    """
    key_id: str
    key_prefix: str
    raw_key: Optional[str] = Field(
        default=None,
        description="Raw API key token. Displayed only once; never stored in plaintext.",
    )
    user_id: str
    organization_id: str
    created_at: str
    expires_at: Optional[str] = None
    is_revoked: bool = False
    name: str


class PublishPRRequest(BaseModel):
    """
    Request to publish an approved, committed run as a GitHub Pull Request.
    """
    repo_full_name: str = Field(description="Target repository in 'owner/repo' format.")
    title: Optional[str] = Field(default=None, description="Optional custom PR title.")
    base_branch: str = Field(default="main", description="Target base branch to merge into.")
    draft: bool = Field(default=True, description="Whether to open the PR in draft mode.")


class PublishPRResponse(BaseModel):
    """
    Result of publishing a Pull Request to GitHub.
    """
    pr_number: int
    pr_url: str
    head_branch: str
    base_branch: str
    is_draft: bool
    status: str = "PUBLISHED"

