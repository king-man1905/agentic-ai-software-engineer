from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from backend.vcs.models import GitDiffSummary


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
    error_summary: Optional[str] = Field(
        default=None,
        description="Human-readable error summary when status is FAILED.",
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
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Reason for rejection. Required when approved=False for auditability.",
    )
