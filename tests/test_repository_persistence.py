"""
Regression tests for the repository registry persistence fix.

Production bug: TenantManager stored repository registrations only in an
in-memory dict. A FastAPI/Uvicorn process restart silently forgot every
registered repository, and the next run against that repository failed
workspace provisioning with:

    WORKSPACE_PROVISIONING_FAILED: repository '...' is not authorized for
    organization 'default-org'.

Fix: backend/security/repository_store.py::RepositoryStore persists only
non-secret repository metadata (never github_token) to a dedicated SQLite
file (workspace/repositories.db by default), and TenantManager writes
through to it on register_repository() and rehydrates its in-memory
registry from it at startup - the actual authorization/branch/tenant
logic in TenantManager.authorize_repository_access is completely
unchanged, it just now operates on a registry that survives a restart.

These tests prove persistence across a NEWLY-CONSTRUCTED TenantManager/
RepositoryStore instance pointing at the same on-disk file - simulating a
real process restart - not merely that the in-memory dict works within a
single instance's lifetime.
"""

import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.graph.runner import AgentRunner
from backend.schemas.tenant import Role
from backend.security.auth import AuthMode, RepositoryAccessDeniedError, TenantAccessDeniedError
from backend.security.repository_store import RepositoryStore
from backend.security.tenant import TenantManager, tenant_manager


def _fresh_manager(db_path: str) -> TenantManager:
    """A brand new TenantManager + brand new RepositoryStore over the
    given file - simulates a fresh process, never reusing any in-memory
    state from a previous instance."""
    return TenantManager(repository_store=RepositoryStore(db_path))


class TestRepositoryRegistrationPersistsAcrossRestart:
    def test_1_register_repository(self, tmp_path):
        """1. Registering a repository succeeds and is immediately visible
        in the same process."""
        db_path = str(tmp_path / "repos.db")
        tm = _fresh_manager(db_path)
        repo = tm.register_repository(
            "king-man1905/agentic-ai-test-repo", "default-org", "agentic-ai-test-repo",
            full_name="king-man1905/agentic-ai-test-repo",
        )
        assert repo.id == "king-man1905/agentic-ai-test-repo"
        assert tm.get_repository("king-man1905/agentic-ai-test-repo", "default-org") is not None

    def test_2_read_repository_back_from_persistent_storage(self, tmp_path):
        """2. The RepositoryStore itself (not TenantManager's in-memory
        dict) can read the record back directly."""
        db_path = str(tmp_path / "repos.db")
        tm = _fresh_manager(db_path)
        tm.register_repository(
            "king-man1905/agentic-ai-test-repo", "default-org", "agentic-ai-test-repo",
            full_name="king-man1905/agentic-ai-test-repo",
        )
        persisted = tm._repository_store.get("king-man1905/agentic-ai-test-repo", "default-org")
        assert persisted is not None
        assert persisted.organization_id == "default-org"
        assert persisted.is_authorized is True

    def test_3_4_5_repository_remains_registered_and_authorizable_after_simulated_restart(self, tmp_path):
        """3+4+5. A brand new TenantManager/RepositoryStore instance over
        the SAME file (simulating a process restart) still has the
        repository registered, and the SAME tenant can still authorize
        it - the exact confirmed production scenario."""
        db_path = str(tmp_path / "repos.db")
        tm1 = _fresh_manager(db_path)
        tm1.register_repository(
            "king-man1905/agentic-ai-test-repo", "default-org", "agentic-ai-test-repo",
            full_name="king-man1905/agentic-ai-test-repo", github_token="secret-abc",
        )

        # Simulated restart: a completely new TenantManager instance, new
        # RepositoryStore instance, same underlying file.
        tm2 = _fresh_manager(db_path)

        repo_after_restart = tm2.get_repository("king-man1905/agentic-ai-test-repo", "default-org")
        assert repo_after_restart is not None
        assert repo_after_restart.organization_id == "default-org"

        authorized = tm2.authorize_repository_access("default-org", "king-man1905/agentic-ai-test-repo")
        assert authorized.id == "king-man1905/agentic-ai-test-repo"

    def test_6_different_tenant_cannot_authorize_after_restart(self, tmp_path):
        """6. Tenant isolation survives the restart too: a DIFFERENT
        organization must still be rejected."""
        db_path = str(tmp_path / "repos.db")
        tm1 = _fresh_manager(db_path)
        tm1.register_repository(
            "king-man1905/agentic-ai-test-repo", "default-org", "agentic-ai-test-repo",
            full_name="king-man1905/agentic-ai-test-repo",
        )

        tm2 = _fresh_manager(db_path)
        tm2.create_organization("other-org", "Other Org")
        with pytest.raises(TenantAccessDeniedError):
            tm2.authorize_repository_access("other-org", "king-man1905/agentic-ai-test-repo")

    def test_7_unauthorized_repository_still_fails_after_restart(self, tmp_path):
        """7. is_authorized=False survives the restart and is still enforced."""
        db_path = str(tmp_path / "repos.db")
        tm1 = _fresh_manager(db_path)
        tm1.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
            is_authorized=False,
        )

        tm2 = _fresh_manager(db_path)
        with pytest.raises(RepositoryAccessDeniedError):
            tm2.authorize_repository_access("default-org", "acme/widgets")

    def test_8_disallowed_branch_still_fails_after_restart(self, tmp_path):
        """8. allowed_branches survives the restart and is still enforced."""
        db_path = str(tmp_path / "repos.db")
        tm1 = _fresh_manager(db_path)
        tm1.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
            allowed_branches=["main"],
        )

        tm2 = _fresh_manager(db_path)
        with pytest.raises(RepositoryAccessDeniedError):
            tm2.authorize_repository_access("default-org", "acme/widgets", branch="not-allowed")
        # The allowed branch still succeeds.
        assert tm2.authorize_repository_access("default-org", "acme/widgets", branch="main") is not None

    def test_9_raw_github_token_never_persisted_in_database(self, tmp_path):
        """9. The raw token must not appear ANYWHERE in the persistent
        database file - not in a dedicated column, not serialized into
        any other column, not anywhere in the raw file bytes."""
        db_path = str(tmp_path / "repos.db")
        secret = "ghp_SuperSecretRawTokenMustNeverPersist12345"
        tm = _fresh_manager(db_path)
        tm.register_repository(
            "king-man1905/agentic-ai-test-repo", "default-org", "agentic-ai-test-repo",
            full_name="king-man1905/agentic-ai-test-repo", github_token=secret,
        )

        conn = sqlite3.connect(db_path)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(repositories);").fetchall()]
        assert "github_token" not in columns

        rows = conn.execute("SELECT * FROM repositories;").fetchall()
        assert secret not in str(rows)
        conn.close()

        # Even the raw file bytes must never contain the secret (guards
        # against it ending up serialized into an unexpected column).
        with open(db_path, "rb") as f:
            raw_bytes = f.read()
        assert secret.encode() not in raw_bytes

    def test_10_runtime_github_token_env_fallback_still_works_after_restart(self, tmp_path, monkeypatch):
        """10. A repository rehydrated after a restart has github_token=None
        (never persisted), so the EXISTING runtime fallback pattern
        (repo.github_token or os.environ.get("GITHUB_TOKEN")) used
        throughout backend/graph/runner.py and backend/api/app.py
        transparently falls back to the environment variable - proving
        credentials and metadata are handled through separate paths."""
        db_path = str(tmp_path / "repos.db")
        tm1 = _fresh_manager(db_path)
        tm1.register_repository(
            "king-man1905/agentic-ai-test-repo", "default-org", "agentic-ai-test-repo",
            full_name="king-man1905/agentic-ai-test-repo", github_token="ephemeral-process1-token",
        )

        tm2 = _fresh_manager(db_path)
        repo = tm2.get_repository("king-man1905/agentic-ai-test-repo", "default-org")
        assert repo.github_token is None

        monkeypatch.setenv("GITHUB_TOKEN", "env-fallback-token")
        effective_token = repo.github_token or os.environ.get("GITHUB_TOKEN")
        assert effective_token == "env-fallback-token"


class TestRepositoryMetadataAndCredentialsHandledSeparately:
    def test_repository_store_get_never_returns_a_token(self, tmp_path):
        """Even for a repo that WAS registered with a token in the same
        process, reading it back through the persistent store (not the
        in-memory dict) never returns a token - proving the store's read
        path is credential-blind by construction, not just by omission."""
        db_path = str(tmp_path / "repos.db")
        tm = _fresh_manager(db_path)
        tm.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
            github_token="should-never-surface-here",
        )
        from_store = tm._repository_store.get("acme/widgets", "default-org")
        assert from_store.github_token is None

        from_memory = tm.get_repository("acme/widgets", "default-org")
        assert from_memory.github_token == "should-never-surface-here"


class TestWorkspaceProvisioningAfterRestart:
    """12+13. Full integration through AgentRunner._ensure_workspace_provisioned -
    the exact production call path - after a simulated restart."""

    def test_provisioning_succeeds_after_restart_for_previously_registered_repository(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        db_path = str(tmp_path / "repos.db")
        monkeypatch.chdir(tmp_path)

        tm1 = _fresh_manager(db_path)
        tm1.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
        )

        # Simulated restart.
        tm2 = _fresh_manager(db_path)
        monkeypatch.setattr("backend.graph.runner.tenant_manager", tm2)

        clone_spy = MagicMock(return_value=True)
        monkeypatch.setattr("backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy)
        monkeypatch.setattr(
            "backend.graph.runner.GitWorkspaceManager.get_remote_url",
            lambda path: "https://github.com/acme/widgets.git",
        )
        monkeypatch.setattr(
            "backend.graph.runner.GitWorkspaceManager.clone_repository",
            lambda clone_url, project_path, timeout=60, auth_header=None: (
                os.makedirs(f"{project_path}/.git", exist_ok=True) or True
            ),
        )

        runner = AgentRunner()
        # Must not raise - the repository is authorized after the
        # simulated restart, exactly like the confirmed production fix.
        runner._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
        )

    def test_fail_closed_protections_remain_intact_after_restart_for_unregistered_repository(self, tmp_path, monkeypatch):
        """13. A repository that was NEVER registered (in this process or
        any prior one) must still fail closed after a restart - proving
        this persistence fix didn't accidentally make provisioning
        permissive by default."""
        db_path = str(tmp_path / "repos.db")
        monkeypatch.chdir(tmp_path)
        tm = _fresh_manager(db_path)
        monkeypatch.setattr("backend.graph.runner.tenant_manager", tm)

        runner = AgentRunner()
        with pytest.raises(RuntimeError, match="WORKSPACE_PROVISIONING_FAILED"):
            runner._ensure_workspace_provisioned(
                project_id="widgets", repository_id="acme/never-registered", organization_id="default-org",
            )


class TestApiRegistrationRbacPreservedWithPersistence:
    """11. POST /api/v1/repositories preserves existing RBAC/response-shape
    behavior with the persistence layer wired in."""

    @pytest.fixture(autouse=True)
    def isolated_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tenant_manager, "_repository_store", RepositoryStore(str(tmp_path / "repos.db")))
        tenant_manager.reset()
        tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
        yield
        tenant_manager.reset()
        tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)

    @pytest.fixture
    def client(self):
        return TestClient(create_app())

    def _headers(self, org_id: str, user_id: str) -> dict:
        return {"X-Organization-ID": org_id, "X-User-ID": user_id}

    def test_requires_repo_manage_permission(self, client):
        org = tenant_manager.create_organization("org-a", "Org A")
        user = tenant_manager.create_user("viewer-a", "v@a.com", "Viewer")
        tenant_manager.add_membership(org.id, user.id, Role.VIEWER)

        resp = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets"},
            headers=self._headers("org-a", "viewer-a"),
        )
        assert resp.status_code == 403

    def test_registration_response_never_exposes_raw_token(self, client):
        org = tenant_manager.create_organization("org-a", "Org A")
        user = tenant_manager.create_user("user-a", "a@a.com", "Alice")
        tenant_manager.add_membership(org.id, user.id, Role.ENGINEER)

        resp = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets", "github_token": "secret-token-abc"},
            headers=self._headers("org-a", "user-a"),
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["has_token"] is True
        assert "github_token" not in body
        assert "secret-token-abc" not in str(body)

    def test_cross_tenant_reregistration_rejected(self, client):
        org_a = tenant_manager.create_organization("org-a", "Org A")
        user_a = tenant_manager.create_user("user-a", "a@a.com", "Alice")
        tenant_manager.add_membership(org_a.id, user_a.id, Role.ENGINEER)
        org_b = tenant_manager.create_organization("org-b", "Org B")
        user_b = tenant_manager.create_user("user-b", "b@b.com", "Bob")
        tenant_manager.add_membership(org_b.id, user_b.id, Role.ENGINEER)

        resp1 = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets"},
            headers=self._headers("org-a", "user-a"),
        )
        assert resp1.status_code == 201

        resp2 = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets"},
            headers=self._headers("org-b", "user-b"),
        )
        assert resp2.status_code == 409

    def test_registration_persists_and_survives_restart_via_api(self, client, tmp_path):
        """Full-stack proof: register via the real API endpoint, then
        simulate a restart and confirm the repository is still
        authorized - not just the lower-level TenantManager unit tests."""
        org = tenant_manager.create_organization("org-a", "Org A")
        user = tenant_manager.create_user("user-a", "a@a.com", "Alice")
        tenant_manager.add_membership(org.id, user.id, Role.ENGINEER)

        resp = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets"},
            headers=self._headers("org-a", "user-a"),
        )
        assert resp.status_code == 201

        # Simulate restart: a brand new TenantManager/RepositoryStore over
        # the SAME db file the API just wrote through to.
        restarted = TenantManager(repository_store=RepositoryStore(tenant_manager._repository_store.db_path))
        restarted.create_organization("org-a", "Org A")
        authorized = restarted.authorize_repository_access("org-a", "acme/widgets")
        assert authorized.id == "acme/widgets"


# ============================================================================
# Multi-tenant registry identity fix: repository identity was keyed
# GLOBALLY by repo_id/full_name alone (self._repositories: Dict[str,
# Repository], and RepositoryStore's `id TEXT PRIMARY KEY`) instead of
# (organization_id, repo_id). A second organization registering the same
# repo_id/full_name (e.g. two tenants both legitimately registering the
# same public repo) silently overwrote the first organization's row,
# in-memory AND on disk - a structural tenant-isolation violation, not
# merely an availability bug, since the persisted table could never even
# recover the first organization's registration afterward.
#
# Fix: self._repositories is keyed by (organization_id, id-or-full_name);
# RepositoryStore's primary key is (organization_id, id); get_repository/
# RepositoryStore.get require organization_id and fail closed (return
# None) without it; the one legitimate cross-tenant need (registration-
# conflict detection in POST /api/v1/repositories) goes through the
# explicitly-named find_registration_any_organization, never through a
# tenant-scoped lookup.
# ============================================================================

class TestMultiTenantRepositoryIdentityIsolation:
    def test_1_register_for_org_a_lookup_succeeds_for_org_a(self, tmp_path):
        """1. Register repository for org A -> lookup succeeds for org A."""
        tm = _fresh_manager(str(tmp_path / "repos.db"))
        tm.create_organization("org-a", "Org A")
        tm.register_repository("shared/widgets", "org-a", "widgets", full_name="shared/widgets")
        assert tm.get_repository("shared/widgets", "org-a") is not None

    def test_2_same_repository_lookup_from_org_b_denied(self, tmp_path):
        """2. Same repository lookup from org B -> not found, per the
        tenant-scoped contract - a strictly scoped lookup must never
        return another tenant's registration."""
        tm = _fresh_manager(str(tmp_path / "repos.db"))
        tm.create_organization("org-a", "Org A")
        tm.create_organization("org-b", "Org B")
        tm.register_repository("shared/widgets", "org-a", "widgets", full_name="shared/widgets")

        assert tm.get_repository("shared/widgets", "org-b") is None
        with pytest.raises(TenantAccessDeniedError):
            tm.authorize_repository_access("org-b", "shared/widgets")

    def test_3_independent_registration_by_org_b_does_not_clobber_org_a(self, tmp_path):
        """3. Register the same repository independently for org B (the
        application's existing model permits this - there is no rule
        against two different tenants registering the same repo name) ->
        both records remain fully isolated, including their own token,
        branch policy, and authorization state."""
        tm = _fresh_manager(str(tmp_path / "repos.db"))
        tm.create_organization("org-a", "Org A")
        tm.create_organization("org-b", "Org B")

        tm.register_repository(
            "shared/widgets", "org-a", "widgets", full_name="shared/widgets",
            github_token="org-a-token", allowed_branches=["main"], is_authorized=True,
        )
        tm.register_repository(
            "shared/widgets", "org-b", "widgets", full_name="shared/widgets",
            github_token="org-b-token", allowed_branches=["dev"], is_authorized=False,
        )

        repo_a = tm.get_repository("shared/widgets", "org-a")
        repo_b = tm.get_repository("shared/widgets", "org-b")
        assert repo_a.github_token == "org-a-token"
        assert repo_b.github_token == "org-b-token"
        assert repo_a.allowed_branches == ["main"]
        assert repo_b.allowed_branches == ["dev"]
        assert repo_a.is_authorized is True
        assert repo_b.is_authorized is False

        # org-a's own registration still authorizes normally...
        assert tm.authorize_repository_access("org-a", "shared/widgets", branch="main") is not None
        # ...while org-b's (unauthorized) registration is still correctly blocked.
        with pytest.raises(RepositoryAccessDeniedError):
            tm.authorize_repository_access("org-b", "shared/widgets")

    def test_4_isolation_survives_restart(self, tmp_path):
        """4 (+3 combined). Both independent registrations, and their
        isolation from each other, survive a simulated process restart."""
        db_path = str(tmp_path / "repos.db")
        tm1 = _fresh_manager(db_path)
        tm1.create_organization("org-a", "Org A")
        tm1.create_organization("org-b", "Org B")
        tm1.register_repository("shared/widgets", "org-a", "widgets", full_name="shared/widgets", allowed_branches=["main"])
        tm1.register_repository("shared/widgets", "org-b", "widgets", full_name="shared/widgets", allowed_branches=["dev"])

        tm2 = _fresh_manager(db_path)
        tm2.create_organization("org-a", "Org A")
        tm2.create_organization("org-b", "Org B")
        assert tm2.get_repository("shared/widgets", "org-a").allowed_branches == ["main"]
        assert tm2.get_repository("shared/widgets", "org-b").allowed_branches == ["dev"]

    def test_5_missing_organization_id_fails_closed(self, tmp_path):
        """5. Missing organization_id -> fail closed (None), never an
        unscoped/global lookup - at both the TenantManager and
        RepositoryStore layers."""
        tm = _fresh_manager(str(tmp_path / "repos.db"))
        tm.create_organization("org-a", "Org A")
        tm.register_repository("shared/widgets", "org-a", "widgets", full_name="shared/widgets")

        assert tm.get_repository("shared/widgets") is None
        assert tm.get_repository("shared/widgets", None) is None
        assert tm.get_repository("shared/widgets", "") is None
        assert tm._repository_store.get("shared/widgets", "") is None

    def test_6_wrong_organization_id_fails_closed(self, tmp_path):
        """6. Wrong organization_id -> fail closed, same as (2)."""
        tm = _fresh_manager(str(tmp_path / "repos.db"))
        tm.create_organization("org-a", "Org A")
        tm.create_organization("org-wrong", "Wrong Org")
        tm.register_repository("shared/widgets", "org-a", "widgets", full_name="shared/widgets")

        assert tm.get_repository("shared/widgets", "org-wrong") is None
        with pytest.raises(TenantAccessDeniedError):
            tm.authorize_repository_access("org-wrong", "shared/widgets")

    def test_7_project_repository_mismatch_cannot_bypass_authorization(self, tmp_path, monkeypatch):
        """7. A project_id that happens to collide with another
        organization's project_id must not grant access to that other
        organization's repository - workspace namespacing
        (workspace/<organization_id>/<project_id>) and repository
        authorization are two independent tenant-scoped checks, and
        neither can be used to bypass the other."""
        import os
        db_path = str(tmp_path / "repos.db")
        monkeypatch.chdir(tmp_path)
        tm = _fresh_manager(db_path)
        tm.create_organization("org-a", "Org A")
        tm.create_organization("org-b", "Org B")
        tm.register_repository("acme/private-repo", "org-a", "private-repo", full_name="acme/private-repo")
        monkeypatch.setattr("backend.graph.runner.tenant_manager", tm)

        runner = AgentRunner()
        # org-b uses the IDENTICAL project_id org-a uses, but requests
        # org-a's repository_id under org-b's organization_id - must fail
        # closed, never silently provision org-a's repo into org-b's
        # workspace.
        with pytest.raises(RuntimeError, match="WORKSPACE_PROVISIONING_FAILED"):
            runner._ensure_workspace_provisioned(
                project_id="private-repo", repository_id="acme/private-repo", organization_id="org-b",
            )
        # And org-a's own workspace path must remain distinct from any
        # path org-b could ever resolve to.
        from backend.vcs.workspace_paths import resolve_workspace_path
        assert resolve_workspace_path("org-a", "private-repo") != resolve_workspace_path("org-b", "private-repo")

    def test_8_update_cannot_cross_tenant_boundaries(self, tmp_path):
        """8. Re-registering (updating) a repository for org B must never
        alter org A's independent record for the "same" repo_id."""
        tm = _fresh_manager(str(tmp_path / "repos.db"))
        tm.create_organization("org-a", "Org A")
        tm.create_organization("org-b", "Org B")
        tm.register_repository("shared/widgets", "org-a", "widgets", full_name="shared/widgets", allowed_branches=["main"])
        tm.register_repository("shared/widgets", "org-b", "widgets", full_name="shared/widgets", allowed_branches=["dev"])

        # org-b updates its OWN registration.
        tm.register_repository(
            "shared/widgets", "org-b", "widgets", full_name="shared/widgets",
            allowed_branches=["dev", "staging"], is_authorized=False,
        )

        # org-a's record must be completely unaffected by org-b's update.
        repo_a = tm.get_repository("shared/widgets", "org-a")
        assert repo_a.allowed_branches == ["main"]
        assert repo_a.is_authorized is True

        repo_b = tm.get_repository("shared/widgets", "org-b")
        assert repo_b.allowed_branches == ["dev", "staging"]
        assert repo_b.is_authorized is False

    def test_9_repeated_registration_does_not_create_inconsistent_state(self, tmp_path):
        """9. Repeated/idempotent registration (simulating concurrent or
        retried calls) for the same (org, repo_id) converges on a single,
        consistent record - never duplicate or conflicting rows."""
        db_path = str(tmp_path / "repos.db")
        tm = _fresh_manager(db_path)
        tm.create_organization("org-a", "Org A")

        for i in range(5):
            tm.register_repository(
                "shared/widgets", "org-a", "widgets", full_name="shared/widgets",
                allowed_branches=["main"], is_authorized=True,
            )

        # Exactly one persisted row for this (organization_id, id).
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT COUNT(*) FROM repositories WHERE organization_id = ? AND id = ?;",
            ("org-a", "shared/widgets"),
        ).fetchone()
        assert rows[0] == 1
        conn.close()

        assert tm.get_repository("shared/widgets", "org-a") is not None

    def test_registration_conflict_detection_still_works_via_explicit_cross_tenant_method(self, tmp_path):
        """The one legitimate cross-tenant lookup (registration-conflict
        detection) still works and is clearly distinct from the
        tenant-scoped get_repository - it must never be used for
        authorization decisions, only to pick an accurate error message
        or reject a hijack attempt at the API layer."""
        tm = _fresh_manager(str(tmp_path / "repos.db"))
        tm.create_organization("org-a", "Org A")
        tm.register_repository("shared/widgets", "org-a", "widgets", full_name="shared/widgets")

        hit = tm.find_registration_any_organization("shared/widgets")
        assert hit is not None
        assert hit.organization_id == "org-a"
        assert tm.find_registration_any_organization("never/registered") is None
