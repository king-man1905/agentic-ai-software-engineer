from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class Role(str, Enum):
    """
    Role-Based Access Control (RBAC) roles within an organization.
    """
    OWNER = "OWNER"
    ADMIN = "ADMIN"
    ENGINEER = "ENGINEER"
    REVIEWER = "REVIEWER"
    SECURITY_REVIEWER = "SECURITY_REVIEWER"
    VIEWER = "VIEWER"


class Permission(str, Enum):
    """
    Granular permissions governing operations across tenant resources.
    """
    # Organization management
    ORG_MANAGE = "ORG_MANAGE"
    USER_MANAGE = "USER_MANAGE"

    # Repository management
    REPO_MANAGE = "REPO_MANAGE"
    REPO_READ = "REPO_READ"

    # Policy management
    POLICY_MANAGE = "POLICY_MANAGE"
    POLICY_READ = "POLICY_READ"

    # Agent Run execution & observation
    RUN_CREATE = "RUN_CREATE"
    RUN_READ = "RUN_READ"

    # HITL Approval & Promotion
    RUN_APPROVE = "RUN_APPROVE"
    SECURITY_APPROVE = "SECURITY_APPROVE"

    # Run cancellation - distinct from RUN_APPROVE: stopping work is a
    # lighter-weight action than approving/rejecting its outcome, so it's
    # granted to anyone who can create or approve runs, not just approvers.
    RUN_CANCEL = "RUN_CANCEL"

    # Audit log inspection
    AUDIT_READ = "AUDIT_READ"


class Organization(BaseModel):
    """
    Represents an isolated tenant organization.
    """
    id: str = Field(description="Unique organization identifier (slug or UUID).")
    name: str = Field(description="Human-readable organization display name.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 creation timestamp.",
    )
    status: str = Field(
        default="ACTIVE",
        description="Tenant status: ACTIVE, SUSPENDED, or ARCHIVED.",
    )


class User(BaseModel):
    """
    Represents an authenticated user identity.
    """
    id: str = Field(description="Unique user identifier.")
    email: str = Field(description="User's primary email address.")
    name: str = Field(description="User's full name or handle.")
    key_prefix: Optional[str] = Field(
        default=None,
        description="Optional API key prefix for display and auditing without exposing raw secrets.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Legacy field maintained for constructor compatibility.",
    )


class Membership(BaseModel):
    """
    Represents user membership and role binding in an organization.
    """
    organization_id: str = Field(description="Organization identifier.")
    user_id: str = Field(description="User identifier.")
    role: Role = Field(default=Role.ENGINEER, description="Assigned role.")
    joined_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 membership timestamp.",
    )


class Repository(BaseModel):
    """
    Represents a code repository scoped to an organization.
    """
    id: str = Field(description="Unique repository identifier (e.g. org/repo or slug).")
    organization_id: str = Field(description="Owning organization identifier.")
    name: str = Field(description="Repository name.")
    full_name: Optional[str] = Field(
        default=None,
        description="Repository full name in 'owner/repo' format.",
    )
    default_branch: str = Field(default="main", description="Default branch name.")
    allowed_branches: List[str] = Field(
        default_factory=lambda: ["main", "master", "dev", "agent/*"],
        description="List of branch patterns permitted for automated modifications.",
    )
    is_private: bool = Field(default=True, description="Whether the repository is private.")
    is_authorized: bool = Field(default=True, description="Whether this repository is authorized for agent execution.")
    github_token: Optional[str] = Field(default=None, description="Optional tenant-scoped or installation GitHub token.")


class TenantContext(BaseModel):
    """
    Server-side authenticated context passed to operations.
    """
    user: User
    organization: Organization
    role: Role
    permissions: Set[Permission] = Field(default_factory=set)

    @property
    def organization_id(self) -> str:
        return self.organization.id

    @property
    def user_id(self) -> str:
        return self.user.id
