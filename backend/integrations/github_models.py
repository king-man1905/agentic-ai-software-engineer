from typing import List
from pydantic import BaseModel, Field


class GitHubIssuePayload(BaseModel):
    """
    Represents an ingested GitHub issue with metadata.
    """
    repo_full_name: str = Field(
        ...,
        description="Repository full name in 'owner/repo' format."
    )
    issue_number: int = Field(
        ...,
        description="Issue number within the repository."
    )
    title: str = Field(
        ...,
        description="Issue title."
    )
    body: str = Field(
        default="",
        description="Issue body markdown description."
    )
    labels: List[str] = Field(
        default_factory=list,
        description="List of label names associated with the issue."
    )


class GitHubPRResult(BaseModel):
    """
    Represents the result of creating a GitHub Pull Request.
    """
    pr_number: int = Field(
        ...,
        description="Pull request number."
    )
    pr_url: str = Field(
        ...,
        description="Direct web URL to the created pull request."
    )
    head_branch: str = Field(
        ...,
        description="Source/head branch containing the agent changes."
    )
    base_branch: str = Field(
        default="main",
        description="Target/base branch to merge into."
    )
    is_draft: bool = Field(
        default=True,
        description="Whether the pull request was opened in draft mode."
    )
