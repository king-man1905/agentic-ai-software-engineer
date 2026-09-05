from typing import Optional, Set, Tuple
from backend.schemas.tenant import Permission, Role


ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.OWNER: {
        Permission.ORG_MANAGE,
        Permission.USER_MANAGE,
        Permission.REPO_MANAGE,
        Permission.REPO_READ,
        Permission.POLICY_MANAGE,
        Permission.POLICY_READ,
        Permission.RUN_CREATE,
        Permission.RUN_READ,
        Permission.RUN_APPROVE,
        Permission.SECURITY_APPROVE,
        Permission.AUDIT_READ,
    },
    Role.ADMIN: {
        Permission.USER_MANAGE,
        Permission.REPO_MANAGE,
        Permission.REPO_READ,
        Permission.POLICY_MANAGE,
        Permission.POLICY_READ,
        Permission.RUN_CREATE,
        Permission.RUN_READ,
        Permission.RUN_APPROVE,
        Permission.SECURITY_APPROVE,
        Permission.AUDIT_READ,
    },
    Role.SECURITY_REVIEWER: {
        Permission.REPO_READ,
        Permission.POLICY_READ,
        Permission.RUN_READ,
        Permission.RUN_APPROVE,
        Permission.SECURITY_APPROVE,
        Permission.AUDIT_READ,
    },
    Role.REVIEWER: {
        Permission.REPO_READ,
        Permission.POLICY_READ,
        Permission.RUN_READ,
        Permission.RUN_APPROVE,
    },
    Role.ENGINEER: {
        Permission.REPO_READ,
        Permission.REPO_MANAGE,
        Permission.POLICY_READ,
        Permission.RUN_CREATE,
        Permission.RUN_READ,
    },
    Role.VIEWER: {
        Permission.REPO_READ,
        Permission.POLICY_READ,
        Permission.RUN_READ,
        Permission.AUDIT_READ,
    },
}


def get_permissions(role: Role) -> Set[Permission]:
    """
    Returns the set of permissions granted to the specified role.
    """
    return ROLE_PERMISSIONS.get(role, set()).copy()


def has_permission(role: Role, permission: Permission) -> bool:
    """
    Checks if a role has the specified permission.
    """
    return permission in ROLE_PERMISSIONS.get(role, set())


def can_approve_changes(
    role: Role,
    risk_score: Optional[float] = None,
    is_security_sensitive: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether the given role is authorized to approve changes.

    - Low and medium risk changes require RUN_APPROVE.
    - Elevated risk (risk_score >= 70.0) or security-sensitive changes require SECURITY_APPROVE.
    """
    if not has_permission(role, Permission.RUN_APPROVE):
        return False, f"Role '{role.value}' lacks RUN_APPROVE permission."

    elevated_risk = (risk_score is not None and risk_score >= 70.0) or is_security_sensitive
    if elevated_risk and not has_permission(role, Permission.SECURITY_APPROVE):
        return (
            False,
            f"High-risk (risk_score={risk_score}) or security-sensitive approval requires SECURITY_APPROVE permission.",
        )

    return True, None
