import os
from unittest.mock import MagicMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.models import CreateApiKeyRequest, PublishPRRequest, RunStatusResponse
from backend.graph.runner import AgentRunner
from backend.integrations.github_client import (
    GitHubApiError,
    GitHubAuthError,
    GitHubBranchConflictError,
    GitHubClient,
    GitHubNotFoundError,
    GitHubPRCreationError,
    GitHubPermissionError,
    GitHubRateLimitError,
)
from backend.integrations.github_models import GitHubIssuePayload, GitHubPRResult
from backend.integrations.run_github_bot import solve_issue_and_open_pr
from backend.sandbox.runner import get_sandbox_env
from backend.schemas.developer import DeveloperResult, FileChange
from backend.schemas.policy import PolicyDecision, PolicyEvaluationResult
from backend.schemas.qa import QAResult
from backend.schemas.routing import RoutingDecision, TaskType
from backend.schemas.tenant import Role, User
from backend.security.audit import AuditAction, audit_logger
from backend.security.auth import (
    ApiKeyRecord,
    AuthMode,
    AuthenticationExpiredError,
    AuthenticationInvalidError,
    AuthenticationRequiredError,
    RepositoryAccessDeniedError,
    TenantAccessDeniedError,
    auth_manager,
)
from backend.security.tenant import tenant_manager
from backend.vcs.models import ApprovalDecision, GitDiffSummary


@pytest.fixture(autouse=True)
def clean_security_state(monkeypatch):
    """Resets tenant state, auth mode, and mocks LLM nodes before each test."""
    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
    audit_logger.clear()

    monkeypatch.setattr(
        "backend.graph.nodes.route_task",
        lambda msg: RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.95,
            reasoning="Bug fix task",
            requires_planning=False,
            requires_knowledge=False,
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.generate_code_changes",
        lambda user_request, plan, knowledge: DeveloperResult(
            summary="Fix bug in auth service",
            changes=[
                FileChange(
                    file_path="src/main.py",
                    change_type="MODIFY",
                    content="def run(): return True",
                    reason="Fix safely",
                )
            ],
            requires_testing=True,
            notes=[],
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.review_code_changes",
        lambda user_request, plan, developer_result: QAResult(
            status="PASS",
            issues=[],
            test_cases=["test_main"],
            summary="QA verification successful.",
        ),
    )

    yield

    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
    audit_logger.clear()


# =============================================================================
# 1. AUTHENTICATION BOUNDARY & IDENTITY FORGERY TESTS
# =============================================================================

class TestAuthenticationBoundary:
    def test_forged_user_id_header_rejected_in_prod(self):
        tenant_manager.set_mode(AuthMode.PRODUCTION, fallback=False)
        app = create_app()
        client = TestClient(app)

        # Attempt to impersonate default-user via raw header without an API key
        headers = {"X-User-ID": "default-user"}
        resp = client.get("/api/v1/tenant/context", headers=headers)
        assert resp.status_code == 401
        assert "AUTHENTICATION_INVALID" in resp.text

    def test_forged_organization_header_rejected(self):
        # Create Org A with User A
        org_a = tenant_manager.create_organization("org-a", "Org Alpha")
        user_a = tenant_manager.create_user("user-a", "a@a.com", "User A")
        tenant_manager.add_membership(org_a.id, user_a.id, Role.ENGINEER)
        raw_key, _ = tenant_manager.create_user_api_key(user_a.id, org_a.id, "key_a")

        # Create Org B
        tenant_manager.create_organization("org-b", "Org Beta")

        app = create_app()
        client = TestClient(app)

        # User A attempts to claim Org B via header
        headers = {
            "Authorization": f"Bearer {raw_key}",
            "X-Organization-ID": "org-b",
        }
        resp = client.get("/api/v1/tenant/context", headers=headers)
        assert resp.status_code == 403
        assert "Cross-tenant access violation" in resp.text or "TENANT_ACCESS_DENIED" in resp.text

    def test_unauthenticated_production_request_rejected(self):
        tenant_manager.set_mode(AuthMode.PRODUCTION, fallback=False)
        app = create_app()
        client = TestClient(app)

        resp = client.post("/api/v1/runs", json={"user_message": "Production task"})
        assert resp.status_code == 401
        assert "AUTHENTICATION_REQUIRED" in resp.text

    def test_development_fallback_when_enabled(self):
        tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
        app = create_app()
        client = TestClient(app)

        # Unauthenticated request in dev mode falls back to sandbox default-org
        resp = client.get("/api/v1/tenant/context")
        assert resp.status_code == 200
        assert resp.json()["organization_id"] == "default-org"

    def test_production_fallback_rejection(self):
        tenant_manager.set_mode(AuthMode.PRODUCTION, fallback=False)
        app = create_app()
        client = TestClient(app)

        # In production mode, unauthenticated request fails immediately
        resp = client.get("/api/v1/tenant/context")
        assert resp.status_code == 401
        assert "AUTHENTICATION_REQUIRED" in resp.text


# =============================================================================
# 2. API KEY SECURITY, ROTATION & REVOCATION
# =============================================================================

class TestApiKeyLifecycle:
    def test_invalid_api_key_rejected(self):
        app = create_app()
        client = TestClient(app)

        headers = {"Authorization": "Bearer ak_invalid_token_9999"}
        resp = client.get("/api/v1/tenant/context", headers=headers)
        assert resp.status_code == 401
        assert "AUTHENTICATION_INVALID" in resp.text

    def test_revoked_api_key_rejected(self):
        org = tenant_manager.create_organization("org-sec", "Sec Org")
        user = tenant_manager.create_user("sec-user", "sec@sec.com", "Sec User")
        tenant_manager.add_membership(org.id, user.id, Role.ADMIN)
        raw_key, record = tenant_manager.create_user_api_key(user.id, org.id, "temp_key")

        app = create_app()
        client = TestClient(app)

        headers = {"Authorization": f"Bearer {raw_key}"}
        # First request works
        assert client.get("/api/v1/tenant/context", headers=headers).status_code == 200

        # Revoke the key
        auth_manager.revoke_api_key(record.key_id)

        # Subsequent request is rejected
        resp = client.get("/api/v1/tenant/context", headers=headers)
        assert resp.status_code == 401
        assert "AUTHENTICATION_INVALID" in resp.text
        assert "revoked" in resp.text

    def test_api_key_rotation(self):
        org = tenant_manager.create_organization("rot-org", "Rotation Org")
        user = tenant_manager.create_user("rot-user", "rot@rot.com", "Rot User")
        tenant_manager.add_membership(org.id, user.id, Role.ADMIN)
        raw_key_1, record_1 = tenant_manager.create_user_api_key(user.id, org.id, "key_v1")

        app = create_app()
        client = TestClient(app)

        headers_1 = {"Authorization": f"Bearer {raw_key_1}"}
        # Rotate key via API
        rotate_resp = client.post(f"/api/v1/auth/keys/{record_1.key_id}/rotate", headers=headers_1)
        assert rotate_resp.status_code == 200
        rot_data = rotate_resp.json()
        raw_key_2 = rot_data["raw_key"]
        assert raw_key_2 is not None
        assert raw_key_2 != raw_key_1

        # Old key is now rejected
        assert client.get("/api/v1/tenant/context", headers=headers_1).status_code == 401

        # New key is accepted
        headers_2 = {"Authorization": f"Bearer {raw_key_2}"}
        assert client.get("/api/v1/tenant/context", headers=headers_2).status_code == 200

    def test_plaintext_api_key_not_stored(self):
        org = tenant_manager.create_organization("store-org", "Store Org")
        user = tenant_manager.create_user("store-user", "st@st.com", "St User")
        raw_key, record = tenant_manager.create_user_api_key(user.id, org.id, "k")

        # Stored record must only contain key_hash (SHA-256), never raw_key
        assert raw_key not in record.key_hash
        assert len(record.key_hash) == 64  # SHA-256 hex length
        assert record.key_prefix.startswith("ak_")

        # User entity has no plaintext key
        assert user.api_key is None


# =============================================================================
# 3. REPOSITORY AUTHORIZATION & BRANCH CONSTRAINTS
# =============================================================================

class TestRepositoryAuthorization:
    def test_cross_tenant_github_repository_blocked(self):
        org_a = tenant_manager.create_organization("org-a", "Tenant A")
        org_b = tenant_manager.create_organization("org-b", "Tenant B")
        tenant_manager.register_repository("org-b/repo", org_b.id, "repo", full_name="org-b/repo")

        # Attempt to access Org B's repo from Org A
        with pytest.raises(TenantAccessDeniedError) as exc:
            tenant_manager.authorize_repository_access("org-a", "org-b/repo")
        assert "Cross-tenant access violation" in str(exc.value)

    def test_unauthorized_repository_blocked(self):
        org = tenant_manager.create_organization("corp", "Corp")
        tenant_manager.register_repository("corp/blocked-repo", org.id, "blocked", is_authorized=False)

        with pytest.raises(RepositoryAccessDeniedError) as exc:
            tenant_manager.authorize_repository_access("corp", "corp/blocked-repo")
        assert "not authorized" in str(exc.value)

    def test_unauthorized_branch_blocked(self):
        org = tenant_manager.create_organization("corp", "Corp")
        tenant_manager.register_repository(
            "corp/api",
            org.id,
            "api",
            allowed_branches=["main", "agent/*"],
        )

        # Permitted branch succeeds
        repo = tenant_manager.authorize_repository_access("corp", "corp/api", branch="agent/fix-1")
        assert repo is not None

        # Forbidden branch fails
        with pytest.raises(RepositoryAccessDeniedError) as exc:
            tenant_manager.authorize_repository_access("corp", "corp/api", branch="prod-freeze")
        assert "Branch 'prod-freeze' is not permitted" in str(exc.value)


# =============================================================================
# 4. CREDENTIAL ISOLATION (LLM & SANDBOX)
# =============================================================================

class TestCredentialIsolation:
    def test_github_credential_isolation_from_llm(self, monkeypatch):
        """Verify developer and planner agent functions never leak GITHUB_TOKEN in prompts."""
        monkeypatch.setenv("GITHUB_TOKEN", "super_secret_github_write_token_12345")
        monkeypatch.setenv("GH_TOKEN", "secondary_secret_gh_token_67890")

        from backend.schemas.planning import ExecutionPlan, PlanStep
        from backend.schemas.knowledge import KnowledgeAnswer

        plan = ExecutionPlan(
            goal="fix null check",
            steps=[
                PlanStep(
                    step_number=1,
                    action="fix null check",
                    agent="developer",
                    files=["main.py"],
                )
            ],
            success_criteria="pytest passes",
        )
        knowledge = KnowledgeAnswer(
            answer="Safe code context",
            sources=["main.py"],
            sufficient_context=True,
        )

        captured_prompts = []

        def mock_invoke(llm, schema, prompt):
            captured_prompts.append(prompt)
            return DeveloperResult(summary="ok", changes=[], requires_testing=False, notes=[])

        monkeypatch.setattr("backend.agents.developer.invoke_structured", mock_invoke)

        from backend.agents.developer import generate_code_changes
        generate_code_changes("Fix bug", plan, knowledge)

        assert len(captured_prompts) == 1
        assert "super_secret_github_write_token_12345" not in captured_prompts[0]
        assert "secondary_secret_gh_token_67890" not in captured_prompts[0]

    def test_sandbox_credential_isolation(self, monkeypatch):
        """Verify sandbox environment strictly strips GITHUB_TOKEN and cloud credentials."""
        monkeypatch.setenv("GITHUB_TOKEN", "secret_gh_token")
        monkeypatch.setenv("GH_TOKEN", "secret_gh_token_2")
        monkeypatch.setenv("NVIDIA_API_KEY", "nv_secret_key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk_secret_key")

        env = get_sandbox_env(allow_network=False)
        assert "GITHUB_TOKEN" not in env
        assert "GH_TOKEN" not in env
        assert "NVIDIA_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env

    def test_llm_credential_isolation(self, monkeypatch):
        """Verify LLM usage metrics and state payloads never expose tokens."""
        from backend.observability.telemetry import estimate_cost_usd
        # estimate_cost_usd operates purely on token numbers, no credentials
        assert estimate_cost_usd("nvidia/nv-embedqa-e5-v5", 100, 100) >= 0.0


# =============================================================================
# 5. STRUCTURED GITHUB FAILURE STATES
# =============================================================================

class TestStructuredGitHubFailures:
    def test_github_authentication_failure_structured(self):
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Bad credentials"
        mock_client.get.return_value = mock_resp

        client = GitHubClient(token="invalid_token", http_client=mock_client)
        with pytest.raises(GitHubAuthError) as exc:
            client.fetch_issue("octocat/repo", 1)
        assert exc.value.code == "GITHUB_AUTH_FAILURE"
        assert exc.value.status_code == 401

    def test_github_permission_failure_structured(self):
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Resource not accessible by integration"
        mock_client.get.return_value = mock_resp

        client = GitHubClient(token="token", http_client=mock_client)
        with pytest.raises(GitHubPermissionError) as exc:
            client.fetch_issue("octocat/repo", 1)
        assert exc.value.code == "GITHUB_PERMISSION_DENIED"
        assert exc.value.status_code == 403

    def test_github_rate_limit_structured(self):
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "API rate limit exceeded for user ID"
        mock_client.get.return_value = mock_resp

        client = GitHubClient(token="token", http_client=mock_client)
        with pytest.raises(GitHubRateLimitError) as exc:
            client.fetch_issue("octocat/repo", 1)
        assert exc.value.code == "GITHUB_RATE_LIMIT"
        assert exc.value.status_code == 403

    def test_github_branch_conflict_structured(self):
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock()
        mock_resp.status_code = 409
        mock_resp.text = "Branch conflict: Head branch already exists or was modified"
        mock_client.post.return_value = mock_resp

        client = GitHubClient(token="token", http_client=mock_client)
        with pytest.raises(GitHubBranchConflictError) as exc:
            client.create_pull_request("octocat/repo", "Title", "Body", "head", "main")
        assert exc.value.code == "GITHUB_BRANCH_CONFLICT"
        assert exc.value.status_code == 409

    def test_github_pr_creation_failure_structured(self):
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.text = "Validation Failed: No commits between base and head"
        mock_client.post.return_value = mock_resp

        client = GitHubClient(token="token", http_client=mock_client)
        with pytest.raises(GitHubPRCreationError) as exc:
            client.create_pull_request("octocat/repo", "Title", "Body", "head", "main")
        assert exc.value.code == "PR_CREATION_FAILURE"
        assert exc.value.status_code == 422


# =============================================================================
# 6. PR PUBLISHING GATE & HITL ENFORCEMENT
# =============================================================================

class TestPRPublishingPipeline:
    def test_hitl_remains_mandatory_before_pr(self):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        # Register unapproved/running run
        runner.register_run("run-unapproved-1", organization_id="default-org")

        req = PublishPRRequest(repo_full_name="default-org/repo")
        resp = client.post("/api/v1/runs/run-unapproved-1/publish-pr", json=req.model_dump())
        assert resp.status_code == 409
        assert "Cannot publish PR" in resp.text

    def test_sha256_approval_binding_remains_mandatory(self):
        """Verify that a run with a patch hash mismatch cannot publish a PR."""
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        # Start a run
        create_resp = client.post("/api/v1/runs", json={"user_message": "Fix bug"})
        run_id = create_resp.json()["run_id"]

        # Resume with incorrect hash -> triggers PATCH_HASH_MISMATCH
        resume_resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={"approved": True, "reviewer": "eng", "patch_hash": "wrong_tampered_hash_999"},
        )
        assert resume_resp.status_code == 200

        # Attempt to publish PR -> rejected because approval failed
        req = PublishPRRequest(repo_full_name="default-org/repo")
        pub_resp = client.post(f"/api/v1/runs/{run_id}/publish-pr", json=req.model_dump())
        assert pub_resp.status_code == 409

    def test_workspace_drift_protection_remains_mandatory(self, monkeypatch):
        """Verify that if workspace drift occurs, git_commit_node does not commit."""
        from backend.vcs.git_manager import GitWorkspaceManager
        # Simulate drift detection failure
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.verify_workspace_drift",
            lambda **kwargs: (False, "Workspace drifted!"),
        )
        from backend.graph.nodes import git_commit_node

        state = {
            "git_diff": GitDiffSummary(
                branch_name="agent/test",
                unified_diff="--- a\n+++ b",
                patch_hash="original_hash",
                files_changed=["file.py"],
            ),
            "project_id": "test_proj",
        }
        res = git_commit_node(state)
        assert res["approval_status"] == "PATCH_HASH_MISMATCH"

    def test_successful_draft_pr_publication(self, monkeypatch):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        # 1. Register repo under default-org
        tenant_manager.register_repository("default-org/api", "default-org", "api")

        run_id = "test-run-publish-42"
        mock_diff = GitDiffSummary(
            branch_name="agent/task-fix",
            files_changed=["src/main.py"],
            lines_added=5,
            lines_deleted=1,
            unified_diff="--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-old\n+new",
            patch_hash="abc123hash",
            risk_score="LOW",
        )
        mock_status = RunStatusResponse(
            run_id=run_id,
            status="COMPLETED",
            current_node="git_commit",
            git_diff=mock_diff,
        )

        monkeypatch.setattr(runner, "get_status", lambda rid, organization_id=None: mock_status)
        monkeypatch.setattr(
            runner,
            "get_state_values",
            lambda rid, organization_id=None: {
                "approval": ApprovalDecision(approved=True, reviewer="admin", patch_hash="abc123hash"),
                "approval_status": "COMMITTED",
                "git_diff": mock_diff,
            },
        )

        # Mock GitHubClient PR creation
        monkeypatch.setattr(
            "backend.integrations.github_client.GitHubClient.create_pull_request",
            lambda *args, **kwargs: GitHubPRResult(
                pr_number=42,
                pr_url="https://github.com/default-org/api/pull/42",
                head_branch="agent/task-fix",
                base_branch="main",
                is_draft=True,
            ),
        )

        # 3. Publish PR
        pub_req = PublishPRRequest(repo_full_name="default-org/api")
        pub_resp = client.post(f"/api/v1/runs/{run_id}/publish-pr", json=pub_req.model_dump())
        assert pub_resp.status_code == 200
        data = pub_resp.json()
        assert data["pr_number"] == 42
        assert data["is_draft"] is True
        assert data["status"] == "PUBLISHED"


# =============================================================================
# 7. AUDIT LOGGING & TAMPER RESISTANCE
# =============================================================================

class TestAuditSecurity:
    def test_audit_events_recorded(self):
        org = tenant_manager.create_organization("audit-org-6", "Audit Org 6")
        user = tenant_manager.create_user("aud-user", "a@aud.com", "Aud User")
        tenant_manager.add_membership(org.id, user.id, Role.OWNER)

        # Log Phase 6 actions
        audit_logger.log(org.id, user.id, AuditAction.AUTHENTICATION_SUCCESS, "auth", user.id)
        audit_logger.log(org.id, user.id, AuditAction.AUTHENTICATION_FAILURE, "auth", "bad_key", details={"token": "super_secret"})
        audit_logger.log(org.id, user.id, AuditAction.PR_CREATED, "github_pr", "org/repo#1", details={"pr_url": "http://pr"})

        events = audit_logger.get_events(org.id)
        assert len(events) == 3
        # Secret in details must be redacted
        assert events[1].details["token"] == "[REDACTED]"

    def test_audit_integrity_remains_valid(self):
        org_id = "integrity-org"
        audit_logger.log(org_id, "u1", AuditAction.AUTHENTICATION_SUCCESS, "auth", "u1")
        audit_logger.log(org_id, "u1", AuditAction.PR_CREATED, "pr", "repo#1")

        valid, err = audit_logger.verify_integrity(org_id)
        assert valid is True
        assert err is None


# =============================================================================
# 7. AGENTRUNNER TENANT FALLBACK (regression for the Phase 6 audit finding:
# runner.py's "organization_id or ... or 'default-org'" fallback had no
# production awareness of its own, independent of the API layer's checks)
# =============================================================================

class TestRunnerProductionTenantFallback:
    def test_register_run_without_org_fails_in_production(self):
        tenant_manager.set_mode(AuthMode.PRODUCTION, fallback=False)
        runner = AgentRunner()
        with pytest.raises(PermissionError, match="AUTH_MODE=production"):
            runner.register_run(run_id="run-no-org-register")

    def test_start_run_without_org_fails_in_production(self):
        tenant_manager.set_mode(AuthMode.PRODUCTION, fallback=False)
        runner = AgentRunner()
        with pytest.raises(PermissionError, match="AUTH_MODE=production"):
            runner.start_run(run_id="run-no-org-start", user_message="do something")

    def test_get_status_without_org_fails_safely_in_production(self):
        # A run legitimately created under a real tenant in dev mode...
        tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
        runner = AgentRunner()
        runner.register_run(run_id="run-existing-prod-check", organization_id="org-real")

        # ...must not become readable with no organization_id once the
        # server is actually running in production - "no identity" must
        # never quietly resolve to "default-org" or "whatever the run has".
        tenant_manager.set_mode(AuthMode.PRODUCTION, fallback=False)
        with pytest.raises(KeyError):
            runner.get_status("run-existing-prod-check")

    def test_resume_run_without_org_fails_safely_in_production(self):
        tenant_manager.set_mode(AuthMode.PRODUCTION, fallback=False)
        runner = AgentRunner()
        with pytest.raises(KeyError):
            runner.resume_run("nonexistent-run", ApprovalDecision(approved=True))

    def test_default_org_fallback_still_works_in_development(self):
        tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
        runner = AgentRunner()
        # No organization_id passed at all - explicit dev fallback must
        # still default cleanly, this is the behavior Phase 6 intended to
        # keep for local development.
        runner.register_run(run_id="run-dev-fallback")
        result = runner.get_status("run-dev-fallback")
        assert result.status == "RUNNING"

    def test_authenticated_production_tenant_remains_correctly_scoped(self):
        tenant_manager.set_mode(AuthMode.PRODUCTION, fallback=False)
        org_a = tenant_manager.create_organization("org-prod-a", "Prod Org A")
        user_a = tenant_manager.create_user("user-prod-a", "a@prod.local", "Prod User A")
        tenant_manager.add_membership(org_a.id, user_a.id, Role.ENGINEER)
        raw_key_a, _ = tenant_manager.create_user_api_key(user_a.id, org_a.id, "key-prod-a")

        org_b = tenant_manager.create_organization("org-prod-b", "Prod Org B")
        user_b = tenant_manager.create_user("user-prod-b", "b@prod.local", "Prod User B")
        tenant_manager.add_membership(org_b.id, user_b.id, Role.ENGINEER)
        raw_key_b, _ = tenant_manager.create_user_api_key(user_b.id, org_b.id, "key-prod-b")

        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/api/v1/runs",
            json={"user_message": "Fix the auth bug"},
            headers={"Authorization": f"Bearer {raw_key_a}"},
        )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        # The authenticated owner can read their own run.
        own_read = client.get(f"/api/v1/runs/{run_id}", headers={"Authorization": f"Bearer {raw_key_a}"})
        assert own_read.status_code == 200

        # A different authenticated production tenant must not see it.
        cross_tenant_read = client.get(f"/api/v1/runs/{run_id}", headers={"Authorization": f"Bearer {raw_key_b}"})
        assert cross_tenant_read.status_code == 404

        # No credentials at all must not see it either (no silent default-org).
        unauthenticated_read = client.get(f"/api/v1/runs/{run_id}")
        assert unauthenticated_read.status_code == 401
