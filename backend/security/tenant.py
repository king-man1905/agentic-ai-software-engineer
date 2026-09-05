import fnmatch
import os
from typing import Dict, List, Optional, Tuple
from backend.schemas.tenant import (
    Membership,
    Organization,
    Repository,
    Role,
    TenantContext,
    User,
)
from backend.security.auth import (
    ApiKeyRecord,
    AuthManager,
    AuthMode,
    AuthenticationExpiredError,
    AuthenticationInvalidError,
    AuthenticationRequiredError,
    RepositoryAccessDeniedError,
    TenantAccessDeniedError,
    auth_manager,
)
from backend.security.rbac import get_permissions


class TenantManager:
    """
    In-memory registry managing multi-tenant entities:
    Organizations, Users, Memberships, and Repositories.
    Enforces strict server-side context resolution and credential hashing.
    """

    def __init__(
        self,
        auth_mode: Optional[AuthMode] = None,
        dev_auth_fallback: Optional[bool] = None,
    ) -> None:
        self._organizations: Dict[str, Organization] = {}
        self._users: Dict[str, User] = {}
        self._memberships: Dict[Tuple[str, str], Membership] = {}
        self._repositories: Dict[str, Repository] = {}
        self.auth_manager: AuthManager = auth_manager

        env_mode = os.getenv("AUTH_MODE", "development").strip().lower()
        self.auth_mode: AuthMode = auth_mode or (
            AuthMode.PRODUCTION if env_mode == "production" else AuthMode.DEVELOPMENT
        )

        if dev_auth_fallback is not None:
            self.dev_auth_fallback: bool = dev_auth_fallback
        else:
            env_fallback = os.getenv(
                "DEV_AUTH_FALLBACK",
                "true" if self.auth_mode == AuthMode.DEVELOPMENT else "false",
            ).strip().lower()
            self.dev_auth_fallback = env_fallback == "true"

        # Initialize default sandbox tenant for backward-compatibility
        self._init_defaults()

    def set_mode(self, mode: AuthMode | str, fallback: Optional[bool] = None) -> None:
        """Dynamically configure production vs development authentication mode."""
        self.auth_mode = AuthMode(mode.lower()) if isinstance(mode, str) else mode
        if fallback is not None:
            self.dev_auth_fallback = fallback
        else:
            self.dev_auth_fallback = (self.auth_mode == AuthMode.DEVELOPMENT)

    def _init_defaults(self) -> None:
        default_org = Organization(id="default-org", name="Default Organization")
        default_user = User(
            id="default-user",
            email="admin@default.local",
            name="Default Admin",
            key_prefix="ak_sandbox...",
        )
        self.create_organization(default_org.id, default_org.name)
        self.create_user(
            default_user.id,
            default_user.email,
            default_user.name,
            api_key="default-sandbox-key",
        )
        self.add_membership(default_org.id, default_user.id, Role.OWNER)

    def reset(self) -> None:
        """Clear all registered entities and restore default sandbox tenant."""
        self._organizations.clear()
        self._users.clear()
        self._memberships.clear()
        self._repositories.clear()
        self.auth_manager.clear()
        self._init_defaults()

    # --- Organization Management ---
    def create_organization(self, org_id: str, name: str) -> Organization:
        org = Organization(id=org_id, name=name)
        self._organizations[org_id] = org
        return org

    def get_organization(self, org_id: str) -> Optional[Organization]:
        return self._organizations.get(org_id)

    def list_organizations(self) -> List[Organization]:
        return list(self._organizations.values())

    # --- User Management ---
    def create_user(
        self,
        user_id: str,
        email: str,
        name: str,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> User:
        key_prefix = None
        if api_key:
            target_org = organization_id or "default-org"
            _, record = self.auth_manager.create_api_key(
                user_id=user_id,
                organization_id=target_org,
                name=f"{user_id}_initial_key",
                custom_raw_key=api_key,
            )
            key_prefix = record.key_prefix

        user = User(id=user_id, email=email, name=name, key_prefix=key_prefix)
        self._users[user_id] = user
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        record = self.auth_manager.authenticate_api_key(api_key)
        return self.get_user(record.user_id)

    def create_user_api_key(
        self,
        user_id: str,
        organization_id: str,
        name: str = "default",
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, ApiKeyRecord]:
        if user_id not in self._users:
            raise ValueError(f"User '{user_id}' does not exist.")
        if organization_id not in self._organizations:
            raise ValueError(f"Organization '{organization_id}' does not exist.")
        return self.auth_manager.create_api_key(
            user_id=user_id,
            organization_id=organization_id,
            name=name,
            expires_in_days=expires_in_days,
        )

    # --- Membership Management ---
    def add_membership(self, org_id: str, user_id: str, role: Role) -> Membership:
        if org_id not in self._organizations:
            raise ValueError(f"Organization '{org_id}' does not exist.")
        if user_id not in self._users:
            raise ValueError(f"User '{user_id}' does not exist.")

        membership = Membership(organization_id=org_id, user_id=user_id, role=role)
        self._memberships[(org_id, user_id)] = membership
        return membership

    def get_membership(self, org_id: str, user_id: str) -> Optional[Membership]:
        return self._memberships.get((org_id, user_id))

    def list_user_memberships(self, user_id: str) -> List[Membership]:
        return [m for (oid, uid), m in self._memberships.items() if uid == user_id]

    def list_org_members(self, org_id: str) -> List[Membership]:
        return [m for (oid, uid), m in self._memberships.items() if oid == org_id]

    # --- Repository Management & Authorization ---
    def register_repository(
        self,
        repo_id: str,
        org_id: str,
        name: str,
        full_name: Optional[str] = None,
        default_branch: str = "main",
        allowed_branches: Optional[List[str]] = None,
        is_private: bool = True,
        is_authorized: bool = True,
        github_token: Optional[str] = None,
    ) -> Repository:
        if org_id not in self._organizations:
            raise ValueError(f"Organization '{org_id}' does not exist.")
        repo = Repository(
            id=repo_id,
            organization_id=org_id,
            name=name,
            full_name=full_name or repo_id,
            default_branch=default_branch,
            allowed_branches=allowed_branches or ["main", "master", "dev", "agent/*"],
            is_private=is_private,
            is_authorized=is_authorized,
            github_token=github_token,
        )
        self._repositories[repo_id] = repo
        if full_name:
            self._repositories[full_name] = repo
        return repo

    def get_repository(self, repo_id: str) -> Optional[Repository]:
        return self._repositories.get(repo_id)

    def list_org_repositories(self, org_id: str) -> List[Repository]:
        seen = set()
        repos = []
        for r in self._repositories.values():
            if r.organization_id == org_id and r.id not in seen:
                seen.add(r.id)
                repos.append(r)
        return repos

    def authorize_repository_access(
        self,
        organization_id: str,
        repo_full_name: str,
        branch: Optional[str] = None,
    ) -> Repository:
        """
        Enforces repository authorization before any GitHub operation:
        1. Repository must be registered.
        2. Repository must belong to organization_id.
        3. Repository must be authorized (is_authorized is True).
        4. Target branch (if specified) must match allowed patterns.
        """
        repo = self.get_repository(repo_full_name)
        if not repo:
            raise RepositoryAccessDeniedError(f"Repository '{repo_full_name}' is not registered.")

        if repo.organization_id != organization_id:
            raise TenantAccessDeniedError(
                f"Cross-tenant access violation: Repository '{repo_full_name}' belongs to "
                f"organization '{repo.organization_id}', not '{organization_id}'."
            )

        if not repo.is_authorized:
            raise RepositoryAccessDeniedError(f"Repository '{repo_full_name}' is not authorized for operations.")

        if branch:
            allowed = any(fnmatch.fnmatch(branch, pattern) for pattern in repo.allowed_branches)
            if not allowed:
                raise RepositoryAccessDeniedError(
                    f"Branch '{branch}' is not permitted for repository '{repo_full_name}'. "
                    f"Allowed patterns: {repo.allowed_branches}"
                )

        return repo

    # --- Server-Side Context Resolution ---
    def resolve_context(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> TenantContext:
        """
        Resolves server-side TenantContext.
        Enforces authentication mode (production vs development) and RBAC verification.
        """
        is_production = (self.auth_mode == AuthMode.PRODUCTION) or not self.dev_auth_fallback
        user: Optional[User] = None
        key_record: Optional[ApiKeyRecord] = None

        # 1. Resolve Identity via API Key
        if api_key:
            key_record = self.auth_manager.authenticate_api_key(api_key)
            user = self.get_user(key_record.user_id)
            if not user:
                raise AuthenticationInvalidError(f"User for API key not found.")

        # 2. Production Mode Security Boundary
        if is_production:
            if not user:
                if user_id:
                    raise AuthenticationInvalidError(
                        "Forged identity rejected: Direct X-User-ID headers are not accepted as proof of identity. Provide a valid Authorization Bearer token."
                    )
                raise AuthenticationRequiredError("AUTHENTICATION_REQUIRED: Valid API key credentials are required in production mode.")

        # 3. Development Mode Fallback
        if not user:
            if user_id:
                user = self.get_user(user_id)
                if not user:
                    raise AuthenticationInvalidError(f"User '{user_id}' not found.")
            elif self.dev_auth_fallback and not org_id:
                user = self._users.get("default-user")
                org_id = "default-org"

        if not user:
            raise AuthenticationRequiredError("AUTHENTICATION_REQUIRED: Missing authentication credentials.")

        # 4. Resolve Organization
        if not org_id:
            user_memberships = self.list_user_memberships(user.id)
            if not user_memberships:
                raise TenantAccessDeniedError(f"User '{user.id}' does not belong to any organization.")
            if key_record and any(m.organization_id == key_record.organization_id for m in user_memberships):
                org_id = key_record.organization_id
            else:
                org_id = user_memberships[0].organization_id

        org = self.get_organization(org_id)
        if not org:
            raise TenantAccessDeniedError(f"Organization '{org_id}' not found.")

        # 5. Verify Membership & Granular Role Permissions
        membership = self.get_membership(org.id, user.id)
        if not membership:
            raise TenantAccessDeniedError(
                f"Cross-tenant access violation: User '{user.id}' is not a member of organization '{org.id}'."
            )

        permissions = get_permissions(membership.role)

        return TenantContext(
            user=user,
            organization=org,
            role=membership.role,
            permissions=permissions,
        )


tenant_manager = TenantManager()
