"""
Integrations package for external services, VCS platforms, and delivery gates.
"""

from backend.integrations.github_models import (
    GitHubIssuePayload,
    GitHubPRResult,
)
from backend.integrations.github_client import GitHubClient

__all__ = [
    "GitHubIssuePayload",
    "GitHubPRResult",
    "GitHubClient",
]
