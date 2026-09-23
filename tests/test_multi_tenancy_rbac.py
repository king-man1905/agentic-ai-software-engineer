import uuid
from typing import Optional
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.models import CreateRunRequest, ResumeRunRequest
from backend.graph.runner import AgentRunner
from backend.rag.indexer import get_vector_store_path
from backend.rag.retriever import load_project_index
from backend.schemas.developer import DeveloperResult, FileChange
from backend.schemas.policy import PolicyDecision, PolicyEvaluationResult
from backend.schemas.qa import QAResult
from backend.schemas.routing import RoutingDecision, TaskType
from backend.schemas.tenant import (
    Membership,
    Organization,
    Permission,
    Repository,
    Role,
    TenantContext,
    User,
)
from backend.security.audit import AuditAction, AuditLogger, audit_logger
from backend.security.rbac import (
    can_approve_changes,
    get_permissions,
    has_permission,
)
from backend.security.tenant import TenantManager, tenant_manager
from backend.vcs.models import ApprovalDecision, GitDiffSummary


@pytest.fixture(autouse=True)
def reset_tenants_and_mocks(monkeypatch):
    """Ensure clean tenant state and mock agents before each test."""
    tenant_manager.reset()
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
            summary="Fix bug",
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
    audit_logger.clear()


# =============================================================================
# 1. Organization and Entity Domain Tests
# =============================================================================

class TestTenantDomainEntities:
    def test_create_organization(self):
        org = tenant_manager.create_organization("acme-corp", "Acme Corporation")
        assert org.id == "acme-corp"
        assert org.name == "Acme Corporation"
        assert org.status == "ACTIVE"

        retrieved = tenant_manager.get_organization("acme-corp")
        assert retrieved is not None
        assert retrieved.id == "acme-corp"

    def test_create_user_and_membership(self):
        org = tenant_manager.create_organization("org-alpha", "Alpha Org")
        user = tenant_manager.create_user("user-alice", "alice@alpha.com", "Alice Dev", api_key="key-alice-123")
        membership = tenant_manager.add_membership(org.id, user.id, Role.ENGINEER)

        assert membership.organization_id == "org-alpha"
        assert membership.user_id == "user-alice"
        assert membership.role == Role.ENGINEER

        user_memberships = tenant_manager.list_user_memberships(user.id)
        assert len(user_memberships) == 1
        assert user_memberships[0].organization_id == "org-alpha"

    def test_register_repository_under_org(self):
        org = tenant_manager.create_organization("org-beta", "Beta Org")
        repo = tenant_manager.register_repository("beta/core-api", org.id, "core-api")

        assert repo.id == "beta/core-api"
        assert repo.organization_id == "org-beta"
        assert repo.default_branch == "main"
        assert repo.is_private is True

        repos = tenant_manager.list_org_repositories("org-beta")
        assert len(repos) == 1
        assert repos[0].id == "beta/core-api"

    def test_cross_org_membership_isolation(self):
        org_a = tenant_manager.create_organization("org-a", "Org A")
        org_b = tenant_manager.create_organization("org-b", "Org B")
        user_a = tenant_manager.create_user("user-a", "a@a.com", "Alice")
        tenant_manager.add_membership(org_a.id, user_a.id, Role.ENGINEER)

        # Alice is in Org A, but not Org B
        assert tenant_manager.get_membership(org_a.id, user_a.id) is not None
        assert tenant_manager.get_membership(org_b.id, user_a.id) is None


# =============================================================================
# 2. RBAC Role Permissions Matrix Tests
# =============================================================================

class TestRBACPermissionsMatrix:
    def test_role_permissions_owner(self):
        perms = get_permissions(Role.OWNER)
        assert Permission.ORG_MANAGE in perms
        assert Permission.USER_MANAGE in perms
        assert Permission.RUN_CREATE in perms
        assert Permission.RUN_APPROVE in perms
        assert Permission.SECURITY_APPROVE in perms
        assert Permission.AUDIT_READ in perms

    def test_role_permissions_admin(self):
        perms = get_permissions(Role.ADMIN)
        assert Permission.USER_MANAGE in perms
        assert Permission.REPO_MANAGE in perms
        assert Permission.RUN_CREATE in perms
        assert Permission.RUN_APPROVE in perms
        assert Permission.SECURITY_APPROVE in perms
        assert Permission.AUDIT_READ in perms

    def test_role_permissions_security_reviewer(self):
        perms = get_permissions(Role.SECURITY_REVIEWER)
        assert Permission.RUN_APPROVE in perms
        assert Permission.SECURITY_APPROVE in perms
        assert Permission.AUDIT_READ in perms
        assert Permission.RUN_CREATE not in perms
        assert Permission.ORG_MANAGE not in perms

    def test_role_permissions_reviewer(self):
        perms = get_permissions(Role.REVIEWER)
        assert Permission.RUN_APPROVE in perms
        assert Permission.RUN_READ in perms
        # REVIEWER does NOT have elevated security approval
        assert Permission.SECURITY_APPROVE not in perms
        assert Permission.RUN_CREATE not in perms

    def test_role_permissions_engineer(self):
        perms = get_permissions(Role.ENGINEER)
        assert Permission.RUN_CREATE in perms
        assert Permission.RUN_READ in perms
        assert Permission.REPO_MANAGE in perms
        # ENGINEER cannot approve
        assert Permission.RUN_APPROVE not in perms
        assert Permission.SECURITY_APPROVE not in perms

    def test_role_permissions_viewer(self):
        perms = get_permissions(Role.VIEWER)
        assert Permission.RUN_READ in perms
        assert Permission.AUDIT_READ in perms
        assert Permission.RUN_CREATE not in perms
        assert Permission.RUN_APPROVE not in perms


# =============================================================================
# 3. Approval Authorization Tests
# =============================================================================

class TestApprovalAuthorization:
    def test_engineer_cannot_approve(self):
        can_approve, reason = can_approve_changes(Role.ENGINEER, risk_score=10.0)
        assert can_approve is False
        assert "lacks RUN_APPROVE" in (reason or "")

    def test_viewer_cannot_approve(self):
        can_approve, reason = can_approve_changes(Role.VIEWER, risk_score=10.0)
        assert can_approve is False
        assert "lacks RUN_APPROVE" in (reason or "")

    def test_reviewer_can_approve_low_risk(self):
        can_approve, reason = can_approve_changes(Role.REVIEWER, risk_score=25.0)
        assert can_approve is True
        assert reason is None

    def test_reviewer_denied_elevated_risk(self):
        # Risk >= 70 requires SECURITY_APPROVE
        can_approve, reason = can_approve_changes(Role.REVIEWER, risk_score=75.0)
        assert can_approve is False
        assert "requires SECURITY_APPROVE" in (reason or "")

    def test_reviewer_denied_security_sensitive(self):
        can_approve, reason = can_approve_changes(Role.REVIEWER, risk_score=30.0, is_security_sensitive=True)
        assert can_approve is False
        assert "requires SECURITY_APPROVE" in (reason or "")

    def test_security_reviewer_can_approve_elevated_risk(self):
        can_approve, reason = can_approve_changes(Role.SECURITY_REVIEWER, risk_score=85.0)
        assert can_approve is True
        assert reason is None

    def test_approval_node_enforces_rbac(self):
        """Verify the graph approval node directly records APPROVAL_UNAUTHORIZED when reviewer lacks role permission."""
        from backend.graph.nodes import approval_node

        state = {
            "git_diff": GitDiffSummary(
                branch_name="feature/test",
                files_changed=["app.py"],
                unified_diff="--- a\n+++ b",
                risk_score="LOW",
                patch_hash="abc123hash",
            ),
            "policy_result": PolicyEvaluationResult(
                decision=PolicyDecision.ALLOW,
                risk_score=15.0,
                evaluated_at="2026-09-05T00:00:00Z",
            ),
        }

        unauth_decision = ApprovalDecision(
            approved=True,
            reviewer="Eve Dev",
            patch_hash="abc123hash",
            reviewer_role=Role.ENGINEER.value,
        )

        with patch("backend.graph.nodes.interrupt", return_value=unauth_decision):
            result = approval_node(state)

        assert result["approval_status"] == "APPROVAL_UNAUTHORIZED"
        assert result["approval"].approved is False
        assert "APPROVAL_UNAUTHORIZED" in result["approval"].rejection_reason


# =============================================================================
# 4. Tenant Isolation & Context Resolution Tests
# =============================================================================

class TestTenantContextAndIsolation:
    def test_context_resolution_valid_api_key(self):
        org = tenant_manager.create_organization("tech-inc", "Tech Inc")
        user = tenant_manager.create_user("tech-user", "tech@inc.com", "Tech User", api_key="secret-tech-key")
        tenant_manager.add_membership(org.id, user.id, Role.ADMIN)

        ctx = tenant_manager.resolve_context(api_key="secret-tech-key")
        assert ctx.organization.id == "tech-inc"
        assert ctx.user.id == "tech-user"
        assert ctx.role == Role.ADMIN
        assert Permission.USER_MANAGE in ctx.permissions

    def test_context_resolution_invalid_credentials(self):
        with pytest.raises(PermissionError, match="Invalid API key"):
            tenant_manager.resolve_context(api_key="nonexistent-key")

        with pytest.raises(PermissionError, match="not found"):
            tenant_manager.resolve_context(user_id="nonexistent-user")

    def test_cross_tenant_header_rejection(self):
        org_1 = tenant_manager.create_organization("org-1", "Org One")
        org_2 = tenant_manager.create_organization("org-2", "Org Two")
        user_1 = tenant_manager.create_user("user-1", "u1@org1.com", "User 1")
        tenant_manager.add_membership(org_1.id, user_1.id, Role.ENGINEER)

        # User 1 claims to belong to Org 2 -> should raise PermissionError
        with pytest.raises(PermissionError, match="Cross-tenant access violation"):
            tenant_manager.resolve_context(org_id=org_2.id, user_id=user_1.id)

    def test_cross_tenant_run_isolation_api(self):
        app = create_app()
        client = TestClient(app)

        # Setup Tenant Alpha and Tenant Beta
        org_a = tenant_manager.create_organization("org-a", "Tenant A")
        user_a = tenant_manager.create_user("user-a", "a@a.com", "Alice")
        tenant_manager.add_membership(org_a.id, user_a.id, Role.ENGINEER)

        org_b = tenant_manager.create_organization("org-b", "Tenant B")
        user_b = tenant_manager.create_user("user-b", "b@b.com", "Bob")
        tenant_manager.add_membership(org_b.id, user_b.id, Role.ENGINEER)

        # User A creates a run in Org A
        headers_a = {"X-Organization-ID": "org-a", "X-User-ID": "user-a"}
        resp = client.post("/api/v1/runs", json={"user_message": "Fix bug in Org A"}, headers=headers_a)
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        # User B in Org B attempts to inspect User A's run -> 404 (prevents tenant enumeration)
        headers_b = {"X-Organization-ID": "org-b", "X-User-ID": "user-b"}
        get_resp = client.get(f"/api/v1/runs/{run_id}", headers=headers_b)
        assert get_resp.status_code == 404

        # User A can inspect their own run
        get_resp_a = client.get(f"/api/v1/runs/{run_id}", headers=headers_a)
        assert get_resp_a.status_code == 200

    def test_cross_tenant_runner_isolation(self):
        runner = AgentRunner()
        runner.register_run("run-test-tenant", organization_id="org-acme")

        # Querying with correct org succeeds
        status_ok = runner.get_status("run-test-tenant", organization_id="org-acme")
        assert status_ok.status == "RUNNING"

        # Querying with wrong org raises KeyError (preventing leaking run existence)
        with pytest.raises(KeyError):
            runner.get_status("run-test-tenant", organization_id="org-rival")


# =============================================================================
# 5. RAG Tenant Isolation Tests
# =============================================================================

class TestRAGTenantIsolation:
    def test_rag_vector_store_tenant_path(self):
        # Two different tenants on the same project_id resolve to disjoint,
        # namespaced paths - never a flat, unnamespaced one. Exact form:
        # vector_store/<organization_id>/<project_id>, not merely a
        # substring match.
        from pathlib import Path

        path_org_a = get_vector_store_path("project-xyz", organization_id="tenant-123")
        assert path_org_a == Path("vector_store") / "tenant-123" / "project-xyz"

        path_org_b = get_vector_store_path("project-xyz", organization_id="tenant-456")
        assert path_org_b == Path("vector_store") / "tenant-456" / "project-xyz"

        # Same project_id, different tenants - must never collide.
        assert path_org_a != path_org_b

    def test_rag_vector_store_missing_organization_id_fails_closed(self):
        # A vector store must always be tenant-bound - no unnamespaced
        # vector_store/<project_id> fallback is allowed.
        with pytest.raises(ValueError):
            get_vector_store_path("project-xyz", organization_id=None)
        with pytest.raises(ValueError):
            get_vector_store_path("project-xyz", organization_id="")

    def test_rag_vector_store_traversal_organization_id_fails_closed(self):
        with pytest.raises(ValueError):
            get_vector_store_path("project-xyz", organization_id="../other-org")

    def test_rag_vector_store_traversal_project_id_fails_closed(self):
        with pytest.raises(ValueError):
            get_vector_store_path("../other-project", organization_id="tenant-123")

    def test_rag_cross_tenant_retrieval_isolation(self):
        # Attempting to load another tenant's vector store index fails with FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_project_index("project-xyz", organization_id="nonexistent-tenant")

    def test_rag_load_project_index_missing_organization_id_fails_closed(self):
        with pytest.raises(ValueError):
            load_project_index("project-xyz", organization_id=None)

    def test_knowledge_node_passes_run_organization_id_to_vector_store(self, monkeypatch):
        """knowledge_node must resolve dense retrieval against the run's own
        tenant, never the vector-store-unaware default."""
        from backend.graph import nodes as nodes_module

        captured = {}

        def fake_load_project_index(project_id, organization_id=None):
            captured["organization_id"] = organization_id
            raise FileNotFoundError("no index for this test")

        monkeypatch.setattr(
            "backend.rag.retriever.load_project_index", fake_load_project_index
        )
        def fake_answer_from_project(project_id, question, k=4, documents=None, organization_id=None):
            captured["answer_from_project_org"] = organization_id
            return nodes_module.KnowledgeAnswer(answer="x", sources=[], sufficient_context=False)

        monkeypatch.setattr(nodes_module, "answer_from_project", fake_answer_from_project)

        state = {
            "project_id": "proj-1",
            "organization_id": "tenant-real",
            "user_message": "what does this do?",
        }
        nodes_module.knowledge_node(state)

        assert captured["organization_id"] == "tenant-real"
        assert captured["answer_from_project_org"] == "tenant-real"

    def test_knowledge_node_fails_closed_when_organization_id_missing(self, monkeypatch):
        """The actual vulnerability this hardening closes: knowledge_node
        used to substitute a real, valid-looking "default-org" whenever a
        run's state had no organization_id at all - silently querying/
        reading THAT tenant's vector store instead of failing closed like
        get_vector_store_path's own contract requires. A run with no
        organization_id must never resolve to any real tenant's data."""
        from backend.graph import nodes as nodes_module

        captured = {}

        def fake_load_project_index(project_id, organization_id=None):
            captured["organization_id"] = organization_id
            raise FileNotFoundError("no index for this test")

        monkeypatch.setattr(
            "backend.rag.retriever.load_project_index", fake_load_project_index
        )

        def fake_answer_from_project(project_id, question, k=4, documents=None, organization_id=None):
            captured["answer_from_project_org"] = organization_id
            return nodes_module.KnowledgeAnswer(answer="x", sources=[], sufficient_context=False)

        monkeypatch.setattr(nodes_module, "answer_from_project", fake_answer_from_project)

        state = {
            "project_id": "proj-1",
            # organization_id deliberately absent from state.
            "user_message": "what does this do?",
        }
        nodes_module.knowledge_node(state)

        assert captured["organization_id"] != "default-org"
        assert captured["organization_id"] is None
        assert captured["answer_from_project_org"] != "default-org"
        assert captured["answer_from_project_org"] is None


# =============================================================================
# 6. Tamper-Evident Audit Logging Tests
# =============================================================================

class TestAuditLoggingAndIntegrity:
    def test_audit_logger_hash_chain_integrity(self):
        logger = AuditLogger()
        org_id = "finance-corp"

        e1 = logger.log(org_id, "user-1", AuditAction.RUN_INITIATED, "run", "run-101")
        e2 = logger.log(org_id, "user-2", AuditAction.APPROVAL_GRANTED, "run", "run-101")
        e3 = logger.log(org_id, "user-1", AuditAction.RUN_COMPLETED, "run", "run-101")

        assert e1.previous_hash == "0" * 64
        assert e2.previous_hash == e1.event_hash
        assert e3.previous_hash == e2.event_hash

        valid, err = logger.verify_integrity(org_id)
        assert valid is True
        assert err is None

    def test_audit_logger_tamper_detection(self):
        logger = AuditLogger()
        org_id = "security-corp"

        logger.log(org_id, "user-1", AuditAction.RUN_INITIATED, "run", "run-201")
        logger.log(org_id, "user-2", AuditAction.APPROVAL_DENIED, "run", "run-201")

        # Simulate malicious tampering of an event's details
        events = logger.get_events(org_id)
        events[0].details["tampered"] = True

        valid, err = logger.verify_integrity(org_id)
        assert valid is False
        assert "Tampered event" in (err or "")

    def test_audit_api_endpoint(self):
        app = create_app()
        client = TestClient(app)

        org = tenant_manager.create_organization("audit-org", "Audit Org")
        user = tenant_manager.create_user("auditor-1", "auditor@audit.com", "Auditor")
        tenant_manager.add_membership(org.id, user.id, Role.VIEWER)

        # Log some events
        audit_logger.log(org.id, user.id, AuditAction.RUN_INITIATED, "run", "run-301")
        audit_logger.log(org.id, user.id, AuditAction.APPROVAL_GRANTED, "run", "run-301")

        headers = {"X-Organization-ID": org.id, "X-User-ID": user.id}
        resp = client.get("/api/v1/audit/events", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["action"] == AuditAction.RUN_INITIATED.value
        assert data[1]["action"] == AuditAction.APPROVAL_GRANTED.value


# =============================================================================
# 7. Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    def test_backward_compatibility_default_tenant(self):
        app = create_app()
        client = TestClient(app)

        # Request with NO tenant headers should resolve to default-org and succeed
        resp = client.get("/api/v1/tenant/context")
        assert resp.status_code == 200
        data = resp.json()
        assert data["organization_id"] == "default-org"
        assert data["role"] == Role.OWNER.value
        assert "RUN_CREATE" in data["permissions"]

    def test_create_run_without_auth_headers_succeeds(self):
        app = create_app()
        client = TestClient(app)

        resp = client.post("/api/v1/runs", json={"user_message": "Legacy unauthenticated run"})
        assert resp.status_code == 202
        assert resp.json()["status"] == "RUNNING"
