import concurrent.futures
import hashlib
import json
import os
import sqlite3
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.models import (
    CreateRunRequest,
    PublishPRRequest,
    PublishPRResponse,
    ResumeRunRequest,
    RunStatusResponse,
)
from backend.graph.runner import AgentRunner
from backend.integrations.github_client import (
    GitHubApiError,
    GitHubAuthError,
    GitHubBranchConflictError,
    GitHubClient,
    GitHubPRCreationError,
)
from backend.integrations.github_models import GitHubPRResult
from backend.observability.collector import telemetry_collector
from backend.observability.store import telemetry_store
from backend.schemas.telemetry import TelemetryEventType
from backend.schemas.tenant import Role
from backend.security.audit import audit_logger
from backend.security.auth import AuthMode
from backend.security.idempotency import (
    IdempotencyConflictError,
    IdempotencyStore,
    compute_fingerprint,
    compute_key_hash,
    idempotency_store,
)
from backend.security.tenant import tenant_manager
from backend.vcs.models import ApprovalDecision, GitDiffSummary


@pytest.fixture(autouse=True)
def clean_idempotency_state(tmp_path, monkeypatch):
    """Resets tenant manager, audit logger, telemetry store, and idempotency store before each test."""
    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
    audit_logger.clear()
    idempotency_store.clear()

    # Several tests share a fixed literal run_id (e.g. "test-pr-run-100")
    # across the file. Without isolating telemetry_store too, state one
    # test persists (PR number, pr_url, ...) leaks into every later test
    # that reuses the same run_id via the real, shared workspace/telemetry.db.
    # Mutate the singletons in place - `from ... import telemetry_store` /
    # `default_store` bindings already taken by other modules won't see a
    # module-attribute reassignment, only an in-place attribute mutation.
    from backend.observability.store import TelemetryStore
    test_telemetry_db = str(tmp_path / "test_idempotency_telemetry.db")
    test_telemetry_store = TelemetryStore(test_telemetry_db)
    monkeypatch.setattr(telemetry_store, "db_path", test_telemetry_db)
    monkeypatch.setattr(telemetry_collector, "store", test_telemetry_store)

    # Isolate test DB if desired
    db_file = str(tmp_path / "test_idempotency.db")
    test_store = IdempotencyStore(db_path=db_file)
    monkeypatch.setattr("backend.api.app.idempotency_store", test_store)
    monkeypatch.setattr("backend.security.idempotency.idempotency_store", test_store)

    # Tests that hit the real POST /api/v1/runs endpoint run the graph's
    # background task synchronously under TestClient - without mocking the
    # LLM-touching nodes, every such test makes a real network call to the
    # configured provider (slow, flaky, and costs real API spend). Match
    # the mocking already used by test_production_identity_github.py and
    # test_observability_analytics.py so idempotency tests only exercise
    # idempotency, not the LLM pipeline.
    from backend.schemas.routing import RoutingDecision, TaskType
    from backend.schemas.developer import DeveloperResult, FileChange
    from backend.schemas.qa import QAResult

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
            summary="Fix applied",
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

    yield test_store

    test_store.clear()


# ============================================================================
# 1. RUN CREATION IDEMPOTENCY (6 tests)
# ============================================================================

class TestRunCreationIdempotency:
    def test_run_creation_first_request_creates_run(self, clean_idempotency_state):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        req_payload = {"user_message": "Fix login crash bug", "project_id": "auth-service"}
        headers = {"Idempotency-Key": "test-key-create-001"}

        resp = client.post("/api/v1/runs", json=req_payload, headers=headers)
        assert resp.status_code == 202
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "RUNNING"
        assert "dispatched successfully" in data["message"]

        # Verify DB reservation
        rec = clean_idempotency_state.get_record("default-org", "test-key-create-001", "create_run")
        assert rec is not None
        assert rec["status"] == "DISPATCHED"
        assert rec["run_id"] == data["run_id"]

    def test_run_creation_replay_returns_same_run(self, clean_idempotency_state):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        req_payload = {"user_message": "Fix login crash bug", "project_id": "auth-service"}
        headers = {"Idempotency-Key": "test-key-create-002"}

        resp1 = client.post("/api/v1/runs", json=req_payload, headers=headers)
        assert resp1.status_code == 202
        run_id_1 = resp1.json()["run_id"]

        # Second request with identical key and payload
        resp2 = client.post("/api/v1/runs", json=req_payload, headers=headers)
        assert resp2.status_code == 202
        run_id_2 = resp2.json()["run_id"]
        assert run_id_1 == run_id_2
        assert "idempotent replay" in resp2.json()["message"]

    def test_run_creation_conflict_with_different_payload(self, clean_idempotency_state):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        headers = {"Idempotency-Key": "test-key-create-003"}
        resp1 = client.post("/api/v1/runs", json={"user_message": "First payload"}, headers=headers)
        assert resp1.status_code == 202

        # Second request with same key but different payload
        resp2 = client.post("/api/v1/runs", json={"user_message": "Completely different payload"}, headers=headers)
        assert resp2.status_code == 409
        assert "IDEMPOTENCY_CONFLICT" in resp2.json()["detail"]

    def test_run_creation_without_key_creates_new_runs(self, clean_idempotency_state):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        req_payload = {"user_message": "Fix login crash bug"}
        resp1 = client.post("/api/v1/runs", json=req_payload)
        resp2 = client.post("/api/v1/runs", json=req_payload)

        assert resp1.status_code == 202
        assert resp2.status_code == 202
        assert resp1.json()["run_id"] != resp2.json()["run_id"]

    def test_run_creation_tenant_scoped_idempotency(self, clean_idempotency_state):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        tenant_manager.create_organization("org-alpha", "Alpha Org")
        tenant_manager.create_user("user-alpha", "alpha@test.com", "Alpha User")
        tenant_manager.add_membership("org-alpha", "user-alpha", Role.ADMIN)

        tenant_manager.create_organization("org-beta", "Beta Org")
        tenant_manager.create_user("user-beta", "beta@test.com", "Beta User")
        tenant_manager.add_membership("org-beta", "user-beta", Role.ADMIN)

        req_payload = {"user_message": "Deploy service"}
        headers_alpha = {
            "Idempotency-Key": "shared-key-scoped",
            "X-Organization-ID": "org-alpha",
            "X-User-ID": "user-alpha",
        }
        headers_beta = {
            "Idempotency-Key": "shared-key-scoped",
            "X-Organization-ID": "org-beta",
            "X-User-ID": "user-beta",
        }

        resp_alpha = client.post("/api/v1/runs", json=req_payload, headers=headers_alpha)
        resp_beta = client.post("/api/v1/runs", json=req_payload, headers=headers_beta)

        assert resp_alpha.status_code == 202
        assert resp_beta.status_code == 202
        # Different tenants must produce different runs even with identical idempotency key
        assert resp_alpha.json()["run_id"] != resp_beta.json()["run_id"]

    def test_run_creation_concurrent_requests_single_dispatch(self, clean_idempotency_state):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        headers = {"Idempotency-Key": "concurrent-run-key-999"}
        payload = {"user_message": "Run concurrent safety check"}

        def do_post():
            return client.post("/api/v1/runs", json=payload, headers=headers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(do_post) for _ in range(8)]
            responses = [f.result() for f in futures]

        status_codes = [r.status_code for r in responses]
        assert all(code == 202 for code in status_codes)

        run_ids = {r.json()["run_id"] for r in responses}
        assert len(run_ids) == 1, f"Expected exactly 1 run_id across concurrent calls, got: {run_ids}"


# ============================================================================
# 2. RESUME IDEMPOTENCY (4 tests)
# ============================================================================

class TestResumeIdempotency:
    def test_resume_first_call_succeeds(self, monkeypatch):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        run_id = "test-resume-run-001"
        mock_diff = GitDiffSummary(
            branch_name="agent/fix",
            files_changed=["main.py"],
            patch_hash="patch_hash_123",
            risk_score="LOW",
        )
        status_paused = RunStatusResponse(
            run_id=run_id,
            status="WAITING_APPROVAL",
            current_node="approval",
            git_diff=mock_diff,
        )
        status_completed = RunStatusResponse(
            run_id=run_id,
            status="COMPLETED",
            current_node="git_commit",
            git_diff=mock_diff,
        )

        monkeypatch.setattr(runner, "get_status", lambda rid, organization_id=None: status_paused)
        monkeypatch.setattr(runner, "resume_run", lambda rid, decision, organization_id=None: status_completed)

        resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={"approved": True, "reviewer": "admin", "patch_hash": "patch_hash_123"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "COMPLETED"

    def test_resume_replay_on_completed_run_succeeds(self, monkeypatch):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        run_id = "test-resume-completed-002"
        mock_diff = GitDiffSummary(
            branch_name="agent/fix",
            files_changed=["main.py"],
            patch_hash="patch_hash_abc",
            risk_score="LOW",
        )
        status_completed = RunStatusResponse(
            run_id=run_id,
            status="COMPLETED",
            current_node="git_commit",
            git_diff=mock_diff,
        )

        monkeypatch.setattr(runner, "get_status", lambda rid, organization_id=None: status_completed)
        monkeypatch.setattr(
            runner,
            "get_state_values",
            lambda rid, organization_id=None: {
                "approval": ApprovalDecision(approved=True, reviewer="admin", patch_hash="patch_hash_abc"),
                "approval_status": "COMMITTED",
                "git_diff": mock_diff,
            },
        )
        resume_mock = MagicMock()
        monkeypatch.setattr(runner, "resume_run", resume_mock)

        # Call resume on already COMPLETED run with matching decision
        resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={"approved": True, "reviewer": "admin", "patch_hash": "patch_hash_abc"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "COMPLETED"
        # resume_run must NOT be re-executed
        resume_mock.assert_not_called()

    def test_resume_conflict_on_completed_run_different_decision(self, monkeypatch):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        run_id = "test-resume-conflict-003"
        mock_diff = GitDiffSummary(
            branch_name="agent/fix",
            files_changed=["main.py"],
            patch_hash="patch_hash_original",
            risk_score="LOW",
        )
        status_completed = RunStatusResponse(
            run_id=run_id,
            status="COMPLETED",
            current_node="git_commit",
            git_diff=mock_diff,
        )

        monkeypatch.setattr(runner, "get_status", lambda rid, organization_id=None: status_completed)
        monkeypatch.setattr(
            runner,
            "get_state_values",
            lambda rid, organization_id=None: {
                "approval": ApprovalDecision(approved=True, reviewer="admin", patch_hash="patch_hash_original"),
                "approval_status": "COMMITTED",
            },
        )

        # Send conflicting decision (approved=False instead of True)
        resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={"approved": False, "reviewer": "admin"},
        )
        assert resp.status_code == 409
        assert "different decision" in resp.json()["detail"].lower()

    def test_resume_concurrency_protection_running_run(self, monkeypatch):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        run_id = "test-resume-running-004"
        status_running = RunStatusResponse(
            run_id=run_id,
            status="RUNNING",
            current_node="developer",
        )

        monkeypatch.setattr(runner, "get_status", lambda rid, organization_id=None: status_running)

        resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={"approved": True, "reviewer": "admin"},
        )
        assert resp.status_code == 409
        assert "currently running" in resp.json()["detail"].lower()


# ============================================================================
# 3. PR PUBLICATION IDEMPOTENCY & RECONCILIATION (6 tests)
# ============================================================================

class TestPRPublicationIdempotencyAndReconciliation:
    @pytest.fixture
    def setup_completed_run(self, monkeypatch):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        tenant_manager.register_repository("default-org/repo", "default-org", "repo")

        run_id = "test-pr-run-100"
        mock_diff = GitDiffSummary(
            branch_name="agent/patch-100",
            files_changed=["app.py"],
            patch_hash="patch_100_hash",
            unified_diff="--- a\n+++ b\n+fix",
            risk_score="LOW",
        )
        status_completed = RunStatusResponse(
            run_id=run_id,
            status="COMPLETED",
            current_node="git_commit",
            git_diff=mock_diff,
        )

        monkeypatch.setattr(runner, "get_status", lambda rid, organization_id=None: status_completed)
        monkeypatch.setattr(
            runner,
            "get_state_values",
            lambda rid, organization_id=None: {
                "approval": ApprovalDecision(approved=True, reviewer="admin", patch_hash="patch_100_hash"),
                "approval_status": "COMMITTED",
                "git_diff": mock_diff,
            },
        )

        return client, run_id, mock_diff

    def test_publish_pr_first_call_creates_pr(self, setup_completed_run, monkeypatch):
        client, run_id, mock_diff = setup_completed_run

        monkeypatch.setattr(
            "backend.integrations.github_client.GitHubClient.find_pull_request",
            lambda *args, **kwargs: None,
        )
        create_mock = MagicMock(
            return_value=GitHubPRResult(
                pr_number=101,
                pr_url="https://github.com/default-org/repo/pull/101",
                head_branch=mock_diff.branch_name,
                base_branch="main",
            )
        )
        monkeypatch.setattr("backend.integrations.github_client.GitHubClient.create_pull_request", create_mock)

        resp = client.post(
            f"/api/v1/runs/{run_id}/publish-pr",
            json={"repo_full_name": "default-org/repo", "base_branch": "main"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pr_number"] == 101
        assert data["status"] == "PUBLISHED"
        create_mock.assert_called_once()

    def test_publish_pr_local_replay_returns_existing_pr(self, setup_completed_run, monkeypatch):
        client, run_id, mock_diff = setup_completed_run

        # Simulate PR already published in local telemetry store
        telemetry_collector.on_pr_published(
            run_id=run_id,
            organization_id="default-org",
            pr_number=101,
            pr_url="https://github.com/default-org/repo/pull/101",
        )

        create_mock = MagicMock()
        monkeypatch.setattr("backend.integrations.github_client.GitHubClient.create_pull_request", create_mock)

        resp = client.post(
            f"/api/v1/runs/{run_id}/publish-pr",
            json={"repo_full_name": "default-org/repo", "base_branch": "main"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pr_number"] == 101
        assert data["status"] == "already_published"
        create_mock.assert_not_called()

    def test_publish_pr_pre_creation_reconciliation(self, setup_completed_run, monkeypatch):
        client, run_id, mock_diff = setup_completed_run

        # PR already exists on GitHub for this branch
        monkeypatch.setattr(
            "backend.integrations.github_client.GitHubClient.find_pull_request",
            lambda *args, **kwargs: GitHubPRResult(
                pr_number=102,
                pr_url="https://github.com/default-org/repo/pull/102",
                head_branch=mock_diff.branch_name,
                base_branch="main",
            ),
        )
        create_mock = MagicMock()
        monkeypatch.setattr("backend.integrations.github_client.GitHubClient.create_pull_request", create_mock)

        resp = client.post(
            f"/api/v1/runs/{run_id}/publish-pr",
            json={"repo_full_name": "default-org/repo", "base_branch": "main"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pr_number"] == 102
        assert data["status"] == "reconciled"
        create_mock.assert_not_called()

    def test_publish_pr_conflict_recovery_reconciliation(self, setup_completed_run, monkeypatch):
        client, run_id, mock_diff = setup_completed_run

        find_calls = 0

        def mock_find(*args, **kwargs):
            nonlocal find_calls
            find_calls += 1
            if find_calls == 1:
                return None  # Pre-creation check misses
            return GitHubPRResult(
                pr_number=103,
                pr_url="https://github.com/default-org/repo/pull/103",
                head_branch=mock_diff.branch_name,
                base_branch="main",
            )

        monkeypatch.setattr("backend.integrations.github_client.GitHubClient.find_pull_request", mock_find)

        def mock_create(*args, **kwargs):
            raise GitHubPRCreationError("A pull request already exists for this branch.")

        monkeypatch.setattr("backend.integrations.github_client.GitHubClient.create_pull_request", mock_create)

        resp = client.post(
            f"/api/v1/runs/{run_id}/publish-pr",
            json={"repo_full_name": "default-org/repo", "base_branch": "main"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pr_number"] == 103
        assert data["status"] == "reconciled"

    def test_publish_pr_unrelated_pr_not_hijacked(self, setup_completed_run, monkeypatch):
        client, run_id, mock_diff = setup_completed_run

        monkeypatch.setattr(
            "backend.integrations.github_client.GitHubClient.find_pull_request",
            lambda *args, **kwargs: None,
        )

        def mock_create(*args, **kwargs):
            raise GitHubPRCreationError("Validation failed: head branch invalid")

        monkeypatch.setattr("backend.integrations.github_client.GitHubClient.create_pull_request", mock_create)

        resp = client.post(
            f"/api/v1/runs/{run_id}/publish-pr",
            json={"repo_full_name": "default-org/repo", "base_branch": "main"},
        )
        assert resp.status_code == 422
        assert "PR_CREATION_FAILURE" in resp.json()["detail"]

    def test_publish_pr_requires_completed_and_committed(self, monkeypatch):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        run_id = "test-pr-uncommitted"
        status_paused = RunStatusResponse(
            run_id=run_id,
            status="WAITING_APPROVAL",
            current_node="approval",
        )
        monkeypatch.setattr(runner, "get_status", lambda rid, organization_id=None: status_paused)

        resp = client.post(
            f"/api/v1/runs/{run_id}/publish-pr",
            json={"repo_full_name": "default-org/repo"},
        )
        assert resp.status_code == 409
        assert "expected 'completed'" in resp.json()["detail"].lower()


# ============================================================================
# 4. SECURITY & PERSISTENCE (4 tests)
# ============================================================================

class TestSecurityAndPersistence:
    def test_raw_idempotency_key_never_logged_or_stored(self, clean_idempotency_state):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        raw_secret_key = "super-secret-api-key-value-999"
        headers = {"Idempotency-Key": raw_secret_key}
        payload = {"user_message": "Audit key storage test"}

        resp = client.post("/api/v1/runs", json=payload, headers=headers)
        assert resp.status_code == 202

        # 1. Check SQLite idempotency table
        with sqlite3.connect(clean_idempotency_state.db_path) as conn:
            cursor = conn.cursor()
            rows = cursor.execute("SELECT key_hash, request_fingerprint, response_json FROM idempotency_records").fetchall()
            for key_hash, fingerprint, resp_json in rows:
                assert raw_secret_key not in key_hash
                assert raw_secret_key not in fingerprint
                if resp_json:
                    assert raw_secret_key not in resp_json

        # 2. Check audit log
        audit_events = audit_logger.get_events("default-org")
        for event in audit_events:
            event_str = json.dumps(event.details)
            assert raw_secret_key not in event_str

    def test_idempotency_records_survive_process_restart(self, tmp_path):
        db_path = str(tmp_path / "restart_test.db")
        store1 = IdempotencyStore(db_path=db_path)

        key = "restart-key-123"
        params = {"action": "sync", "project": "p1"}
        action1, run_id1, _ = store1.check_or_reserve("org-restart", key, "create_run", params, "run-100")
        assert action1 == "NEW"
        store1.complete_reservation("org-restart", key, "create_run", "DISPATCHED", {"run_id": "run-100"})

        # Re-initialize store to simulate process restart
        store2 = IdempotencyStore(db_path=db_path)
        action2, run_id2, cached = store2.check_or_reserve("org-restart", key, "create_run", params, "run-new")
        assert action2 == "REPLAY"
        assert run_id2 == "run-100"
        assert cached == {"run_id": "run-100"}

    def test_cross_tenant_idempotency_key_isolation(self, clean_idempotency_state):
        key = "iso-test-key"
        params = {"task": "isolated"}

        action_a, run_a, _ = clean_idempotency_state.check_or_reserve("tenant-a", key, "op", params, "run-a")
        action_b, run_b, _ = clean_idempotency_state.check_or_reserve("tenant-b", key, "op", params, "run-b")

        assert action_a == "NEW"
        assert action_b == "NEW"
        assert run_a != run_b

        rec_a = clean_idempotency_state.get_record("tenant-a", key, "op")
        rec_b = clean_idempotency_state.get_record("tenant-b", key, "op")
        assert rec_a["organization_id"] == "tenant-a"
        assert rec_b["organization_id"] == "tenant-b"

    def test_drift_protection_preserved_on_resume(self, monkeypatch):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        run_id = "test-drift-run"
        mock_diff = GitDiffSummary(
            branch_name="agent/drift",
            files_changed=["main.py"],
            patch_hash="trusted_original_hash",
            risk_score="LOW",
        )
        status_paused = RunStatusResponse(
            run_id=run_id,
            status="WAITING_APPROVAL",
            current_node="approval",
            git_diff=mock_diff,
        )

        monkeypatch.setattr(runner, "get_status", lambda rid, organization_id=None: status_paused)

        def mock_resume_fail(rid, decision, organization_id=None):
            if decision.patch_hash != "trusted_original_hash":
                raise ValueError("PATCH_HASH_MISMATCH: Workspace drifted!")
            return RunStatusResponse(run_id=rid, status="COMPLETED")

        monkeypatch.setattr(runner, "resume_run", mock_resume_fail)

        # Attempt to resume with altered patch hash
        resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={"approved": True, "reviewer": "admin", "patch_hash": "tampered_hash_999"},
        )
        assert resp.status_code == 409
        assert "PATCH_HASH_MISMATCH" in resp.json()["detail"]


# ============================================================================
# 5. TELEMETRY & OBSERVABILITY (2 tests)
# ============================================================================

class TestTelemetryAndObservability:
    def test_idempotency_replay_event_emitted(self, clean_idempotency_state):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        key = "telemetry-test-key-replay"
        headers = {"Idempotency-Key": key}
        payload = {"user_message": "Telemetry event check"}

        first_run_id = client.post("/api/v1/runs", json=payload, headers=headers).json()["run_id"]
        client.post("/api/v1/runs", json=payload, headers=headers)

        events = telemetry_store.list_events(first_run_id, organization_id="default-org")
        replay_events = [e for e in events if e.event_type == TelemetryEventType.IDEMPOTENCY_REPLAY]
        assert len(replay_events) >= 1
        event = replay_events[0]
        assert event.details.get("key_hash") == compute_key_hash(key)
        assert event.details.get("outcome") == "REPLAY"

    def test_pr_reconciliation_event_emitted(self, monkeypatch):
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        tenant_manager.register_repository("default-org/repo", "default-org", "repo")

        run_id = "test-pr-telemetry-run"
        mock_diff = GitDiffSummary(
            branch_name="agent/patch-telem",
            files_changed=["main.py"],
            patch_hash="hash_telem",
            unified_diff="+telem",
            risk_score="LOW",
        )
        status_completed = RunStatusResponse(
            run_id=run_id,
            status="COMPLETED",
            current_node="git_commit",
            git_diff=mock_diff,
        )

        monkeypatch.setattr(runner, "get_status", lambda rid, organization_id=None: status_completed)
        monkeypatch.setattr(
            runner,
            "get_state_values",
            lambda rid, organization_id=None: {
                "approval": ApprovalDecision(approved=True, reviewer="admin", patch_hash="hash_telem"),
                "approval_status": "COMMITTED",
                "git_diff": mock_diff,
            },
        )
        monkeypatch.setattr(
            "backend.integrations.github_client.GitHubClient.find_pull_request",
            lambda *args, **kwargs: GitHubPRResult(
                pr_number=555,
                pr_url="https://github.com/default-org/repo/pull/555",
                head_branch=mock_diff.branch_name,
                base_branch="main",
            ),
        )

        resp = client.post(
            f"/api/v1/runs/{run_id}/publish-pr",
            json={"repo_full_name": "default-org/repo"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "reconciled"

        events = telemetry_store.list_events(run_id=run_id, organization_id="default-org")
        rec_events = [e for e in events if e.event_type == TelemetryEventType.PR_RECONCILIATION]
        assert len(rec_events) >= 1
        ev = rec_events[0]
        assert ev.details.get("pr_number") == 555
        assert ev.details.get("outcome") == "RECONCILED"
