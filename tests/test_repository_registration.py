"""
Focused tests for POST /api/v1/repositories - the API path that lets a
real tenant user/admin register and authorize a repository for their
organization, closing the gap where TenantManager.register_repository()
had no HTTP-reachable caller outside of tests and the internal evaluation
benchmark harness (backend/observability/evaluation.py). Without it, no
real frontend-driven run could ever pass authorize_repository_access(),
so _ensure_workspace_provisioned() silently no-op'd and publish-pr always
403'd.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.graph.runner import AgentRunner
from backend.schemas.tenant import Role
from backend.security.audit import audit_logger
from backend.security.auth import AuthMode
from backend.security.tenant import tenant_manager


@pytest.fixture(autouse=True)
def isolated_tenant_state():
    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
    audit_logger.clear()
    yield
    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
    audit_logger.clear()


@pytest.fixture
def client():
    return TestClient(create_app())


def _headers(org_id: str, user_id: str) -> dict:
    return {"X-Organization-ID": org_id, "X-User-ID": user_id}


class TestRegisterRepositoryEndpoint:
    def test_engineer_can_register_repository_for_own_org(self, client):
        org = tenant_manager.create_organization("org-a", "Org A")
        user = tenant_manager.create_user("user-a", "a@a.com", "Alice")
        tenant_manager.add_membership(org.id, user.id, Role.ENGINEER)

        resp = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets", "github_token": "secret-token-abc"},
            headers=_headers("org-a", "user-a"),
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == "acme/widgets"
        assert body["organization_id"] == "org-a"
        assert body["is_authorized"] is True
        assert body["has_token"] is True
        assert "github_token" not in body
        assert "secret-token-abc" not in resp.text

        # The exact check _ensure_workspace_provisioned and publish-pr use.
        repo = tenant_manager.authorize_repository_access("org-a", "acme/widgets")
        assert repo.github_token == "secret-token-abc"

    def test_token_never_appears_in_audit_trail(self, client):
        org = tenant_manager.create_organization("org-a", "Org A")
        user = tenant_manager.create_user("user-a", "a@a.com", "Alice")
        tenant_manager.add_membership(org.id, user.id, Role.ENGINEER)

        client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets", "github_token": "super-secret-xyz"},
            headers=_headers("org-a", "user-a"),
        )

        events = audit_logger.get_events(organization_id="org-a")
        assert events, "registration must produce an audit trail entry"
        for e in events:
            assert "super-secret-xyz" not in str(e.details)
            assert "super-secret-xyz" not in e.event_hash

    def test_viewer_role_cannot_register_repository(self, client):
        org = tenant_manager.create_organization("org-a", "Org A")
        user = tenant_manager.create_user("viewer-a", "v@a.com", "Viewer")
        tenant_manager.add_membership(org.id, user.id, Role.VIEWER)

        resp = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets"},
            headers=_headers("org-a", "viewer-a"),
        )

        assert resp.status_code == 403
        # find_registration_any_organization (not get_repository, which is
        # tenant-scoped and would return None here regardless) proves the
        # rejected attempt left no registration at all, for any tenant.
        assert tenant_manager.find_registration_any_organization("acme/widgets") is None

    def test_cannot_hijack_another_orgs_registered_repository(self, client):
        org_a = tenant_manager.create_organization("org-a", "Org A")
        user_a = tenant_manager.create_user("user-a", "a@a.com", "Alice")
        tenant_manager.add_membership(org_a.id, user_a.id, Role.ENGINEER)

        org_b = tenant_manager.create_organization("org-b", "Org B")
        user_b = tenant_manager.create_user("user-b", "b@b.com", "Bob")
        tenant_manager.add_membership(org_b.id, user_b.id, Role.ENGINEER)

        first = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets", "github_token": "org-a-token"},
            headers=_headers("org-a", "user-a"),
        )
        assert first.status_code == 201

        hijack = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets", "github_token": "org-b-token"},
            headers=_headers("org-b", "user-b"),
        )
        assert hijack.status_code == 409

        # Org A's registration and token are untouched by the failed hijack.
        repo = tenant_manager.authorize_repository_access("org-a", "acme/widgets")
        assert repo.organization_id == "org-a"
        assert repo.github_token == "org-a-token"

        # Org B still cannot operate against the repository.
        with pytest.raises(Exception):
            tenant_manager.authorize_repository_access("org-b", "acme/widgets")

    def test_reregistering_own_repo_updates_it(self, client):
        """Re-registering under the same org is an update (e.g. rotating
        the stored token), not a conflict."""
        org = tenant_manager.create_organization("org-a", "Org A")
        user = tenant_manager.create_user("user-a", "a@a.com", "Alice")
        tenant_manager.add_membership(org.id, user.id, Role.ENGINEER)

        client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets", "github_token": "old-token"},
            headers=_headers("org-a", "user-a"),
        )
        resp = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets", "github_token": "new-token"},
            headers=_headers("org-a", "user-a"),
        )

        assert resp.status_code == 201
        repo = tenant_manager.authorize_repository_access("org-a", "acme/widgets")
        assert repo.github_token == "new-token"

    def test_registered_repository_enables_workspace_provisioning_and_pr_publish(
        self, tmp_path, monkeypatch, client, fake_clone_creates_real_git_repo
    ):
        """End-to-end wiring proof: a repository registered through this
        endpoint is exactly what _ensure_workspace_provisioned (the run
        creation path, backend/graph/runner.py) and the publish-pr
        endpoint's authorization check both need."""
        org = tenant_manager.create_organization("org-a", "Org A")
        user = tenant_manager.create_user("user-a", "a@a.com", "Alice")
        tenant_manager.add_membership(org.id, user.id, Role.ENGINEER)

        resp = client.post(
            "/api/v1/repositories",
            json={"repo_full_name": "acme/widgets", "github_token": "secret-token-abc"},
            headers=_headers("org-a", "user-a"),
        )
        assert resp.status_code == 201

        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(side_effect=fake_clone_creates_real_git_repo)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
        )

        runner = AgentRunner()
        runner._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="org-a",
        )
        clone_spy.assert_called_once()
        clone_url = clone_spy.call_args[0][0]
        auth_header = clone_spy.call_args.kwargs.get("auth_header")
        # SECURITY: the registered token must reach clone_repository() only
        # via the separate auth_header parameter - never embedded in the
        # URL itself (which is what used to end up persisted verbatim in
        # the clone's .git/config). auth_header carries the token only
        # base64-encoded (standard HTTP Basic auth), never in plaintext.
        assert "secret-token-abc" not in clone_url
        assert "acme/widgets" in clone_url
        assert auth_header is not None
        assert auth_header.startswith("Authorization: Basic ")
        assert "secret-token-abc" not in auth_header
        import base64
        encoded = auth_header.removeprefix("Authorization: Basic ")
        assert base64.b64decode(encoded).decode() == "x-access-token:secret-token-abc"

        # publish-pr's authorization check (backend/api/app.py) also succeeds
        # for the branch pattern the agent commits to.
        repo = tenant_manager.authorize_repository_access(
            "org-a", "acme/widgets", branch="agent/task-x"
        )
        assert repo.is_authorized is True
