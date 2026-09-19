import os
import subprocess

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.models import CreateRunRequest, RunStatusResponse, ResumeRunRequest
from backend.graph.runner import AgentRunner
from backend.schemas.routing import RoutingDecision, TaskType
from backend.schemas.planning import ExecutionPlan
from backend.schemas.developer import DeveloperResult, FileChange
from backend.schemas.qa import QAResult
from backend.vcs.models import GitDiffSummary, ApprovalDecision


def _init_git_workspace(tmp_path, monkeypatch, project_id: str):
    """
    Creates a real, git-initialized workspace/default-org/<project_id>
    under tmp_path and points nodes.py's os.getcwd()-based fallback path
    resolution at it, so developer_node's fallback write + git_prepare_node's
    diff computation exercise a genuine git baseline (as every real, cloned
    workspace has - see _ensure_workspace_provisioned/clone_repository)
    instead of the ambient, non-git workspace/test_project used when no
    project_id is given. Without real git history, _read_head_content()
    has no prior baseline to diff a freshly-written file against.

    run_cfba0530500b investigation: this used to create the repo at
    workspace/<project_id>, missing the "default-org" tenant-namespace
    segment that resolve_workspace_path() actually resolves to (every
    caller in this file relies on the dev-mode default organization,
    never an explicit organization_id). That mismatch meant
    git_prepare_node/git_commit_node silently operated against a
    different, freshly auto-created, non-git directory - approval and
    "COMPLETED" still appeared to work only because status derivation
    never checked whether an approved diff was actually committed
    (git_commit_node's real GitWorkspaceManager.stage_and_commit fails
    closed via _is_own_git_repo on a non-git directory). Now that status
    derivation checks this, the workspace must genuinely exist where the
    code actually looks for it.
    """
    from pathlib import Path

    workspace_dir = tmp_path / "workspace" / "default-org" / project_id
    workspace_dir.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(workspace_dir), capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(workspace_dir), capture_output=True, text=True)
    (workspace_dir / "README.md").write_text("# Test Project\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)

    assert not (Path("workspace") / "default-org" / project_id).exists()
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    return workspace_dir


# ============================================================================
# 1. API CONTRACT & MODEL TESTS
# ============================================================================

class TestApiModels:
    def test_create_run_request_minimal(self):
        req = CreateRunRequest(user_message="Fix bug in auth service")
        assert req.user_message == "Fix bug in auth service"
        assert req.project_id is None
        assert req.metadata is None

    def test_create_run_request_full(self):
        req = CreateRunRequest(
            user_message="Add retry loop",
            project_id="backend_svc",
            metadata={"caller": "web_ui", "priority": 1},
        )
        assert req.user_message == "Add retry loop"
        assert req.project_id == "backend_svc"
        assert req.metadata == {"caller": "web_ui", "priority": 1}

    def test_run_status_response_serialization(self):
        diff = GitDiffSummary(
            branch_name="agent/task-test1234",
            files_changed=["main.py"],
            lines_added=10,
            lines_deleted=2,
            unified_diff="+new\n-old",
            risk_score="LOW",
            risk_reasons=["Standard change."],
        )
        res = RunStatusResponse(
            run_id="run_12345",
            status="WAITING_APPROVAL",
            current_node="approval",
            git_diff=diff,
            error_summary=None,
        )
        json_str = res.model_dump_json()
        restored = RunStatusResponse.model_validate_json(json_str)
        assert restored.run_id == "run_12345"
        assert restored.status == "WAITING_APPROVAL"
        assert restored.current_node == "approval"
        assert restored.git_diff is not None
        assert restored.git_diff.branch_name == "agent/task-test1234"

    def test_resume_run_request_approved(self):
        req = ResumeRunRequest(approved=True, reviewer="alice")
        assert req.approved is True
        assert req.reviewer == "alice"
        assert req.rejection_reason is None

    def test_resume_run_request_rejected(self):
        req = ResumeRunRequest(
            approved=False,
            reviewer="bob",
            rejection_reason="Test failures detected.",
        )
        assert req.approved is False
        assert req.reviewer == "bob"
        assert req.rejection_reason == "Test failures detected."


# ============================================================================
# 2. MIDDLEWARES & HEALTH TESTS
# ============================================================================

class TestAppInfrastructure:
    @pytest.fixture
    def client(self):
        runner = AgentRunner()
        app = create_app(runner=runner)
        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_correlation_id_auto_generated(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0

    def test_correlation_id_preserved_when_supplied(self, client):
        custom_id = "trace-client-custom-999"
        response = client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id

    def test_cors_headers_present(self, client):
        response = client.options(
            "/api/v1/runs",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


# ============================================================================
# 3. FASTAPI RUNS API ENDPOINTS & ERROR HANDLING
# ============================================================================

class TestRunEndpointsValidationAndErrors:
    @pytest.fixture
    def client(self):
        runner = AgentRunner()
        app = create_app(runner=runner)
        return TestClient(app)

    def test_get_nonexistent_run_returns_404(self, client):
        response = client.get("/api/v1/runs/run_nonexistent_999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_resume_nonexistent_run_returns_404(self, client):
        response = client.post(
            "/api/v1/runs/run_nonexistent_999/resume",
            json={"approved": True, "reviewer": "admin"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_create_run_missing_user_message_returns_422(self, client):
        response = client.post("/api/v1/runs", json={})
        assert response.status_code == 422

    def test_run_status_while_active_in_memory(self):
        runner = AgentRunner()
        run_id = "run_test_in_progress"
        runner.register_run(run_id)
        status_res = runner.get_status(run_id)
        assert status_res.status == "RUNNING"
        assert status_res.run_id == run_id
        assert status_res.message is not None
        assert "background" in status_res.message.lower()


# ============================================================================
# 4. FULL HITL APPROVAL & REJECTION LIFECYCLE TESTS (MOCKED AGENTS)
# ============================================================================

class TestHITLLifecycleWithAPI:
    HITL_PROJECT_ID = "hitl_lifecycle_project"

    @pytest.fixture
    def client_and_runner(self, tmp_path, monkeypatch):
        # A real, git-initialized workspace (not the ambient, non-git
        # workspace/test_project) so git_prepare_node's diff computation has
        # a genuine baseline to diff the fallback-written file against -
        # every real, cloned workspace has one (see clone_repository).
        _init_git_workspace(tmp_path, monkeypatch, self.HITL_PROJECT_ID)

        # Mock agents in backend.graph.nodes namespace to avoid external LLM calls
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
                summary="Fix null pointer exception",
                changes=[
                    FileChange(
                        file_path="src/service.py",
                        change_type="MODIFY",
                        content="def check(): return True",
                        reason="Handle null safely",
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
                test_cases=["test_check"],
                summary="QA verification successful.",
            ),
        )
        # developer_node's own inline exact-snippet LLM call is a separate,
        # real network call (get_llm + invoke_structured) - independent of
        # the generate_code_changes mock above, and reachable whenever
        # self-scan finds any existing content under the workspace (e.g.
        # the README.md the git init above commits). Stubbed to
        # deterministically return no patches from that branch, so this
        # test exercises its own mocked FileChange (via the fallback write
        # path) and the HITL lifecycle deterministically, without live
        # LLM calls.
        monkeypatch.setattr(
            "backend.graph.nodes.invoke_structured",
            lambda llm, schema_cls, prompt, *a, **k: schema_cls(patches=[]),
        )

        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)
        return client, runner

    def test_full_hitl_approval_lifecycle(self, client_and_runner):
        client, runner = client_and_runner

        # 1. Create run (dispatched asynchronously in background)
        create_resp = client.post(
            "/api/v1/runs",
            json={
                "user_message": "Fix the null check in service",
                "project_id": self.HITL_PROJECT_ID,
                "metadata": {"source": "unit_test"},
            },
        )
        assert create_resp.status_code == 202
        run_data = create_resp.json()
        run_id = run_data["run_id"]
        assert run_id.startswith("run_")
        assert run_data["status"] == "RUNNING"
        assert run_data["message"] == "Run dispatched successfully in background"

        # 2. Inspect run via GET (background execution reached approval gate)
        get_resp = client.get(f"/api/v1/runs/{run_id}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["status"] == "WAITING_APPROVAL"
        assert get_data["current_node"] == "approval"
        patch_hash = get_data["git_diff"]["patch_hash"]
        assert patch_hash

        # 3. Resume run with approval. P0-4: approval is fail-closed on a
        # missing patch_hash, so it must be submitted here to genuinely
        # reach APPROVED - omitting it (as this test previously did) is now
        # correctly rejected instead of silently treated as approved.
        resume_resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={
                "approved": True,
                "reviewer": "lead_engineer",
                "patch_hash": patch_hash,
            },
        )
        assert resume_resp.status_code == 200
        resumed_data = resume_resp.json()
        assert resumed_data["status"] == "COMPLETED"

        # 4. Re-inspect run via GET to verify completed state persistence
        final_get = client.get(f"/api/v1/runs/{run_id}")
        assert final_get.status_code == 200
        assert final_get.json()["status"] == "COMPLETED"

        # 5. Resuming an already-completed run with the SAME decision is now
        # an idempotent replay (Phase 8 Step 4), not a 409 - a client retry
        # after a dropped response must not error just because the first
        # attempt actually succeeded. "Same decision" includes the hash -
        # this replays exactly what step 3 submitted.
        replay_resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={"approved": True, "patch_hash": patch_hash},
        )
        assert replay_resp.status_code == 200
        assert replay_resp.json()["status"] == "COMPLETED"

        # 6. A *different* decision on that same completed run is still a
        # genuine conflict, not silently accepted as a replay.
        conflict_resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={"approved": False, "rejection_reason": "changed my mind"},
        )
        assert conflict_resp.status_code == 409

    def test_full_hitl_rejection_lifecycle(self, client_and_runner):
        client, runner = client_and_runner

        # 1. Create run (dispatched asynchronously in background)
        create_resp = client.post(
            "/api/v1/runs",
            json={"user_message": "Fix the null check in service", "project_id": self.HITL_PROJECT_ID},
        )
        assert create_resp.status_code == 202
        run_id = create_resp.json()["run_id"]
        assert create_resp.json()["status"] == "RUNNING"
        assert create_resp.json()["message"] == "Run dispatched successfully in background"

        # 2. Inspect run via GET (background execution reached approval gate)
        get_resp = client.get(f"/api/v1/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "WAITING_APPROVAL"

        # 3. Resume run with rejection
        resume_resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={
                "approved": False,
                "reviewer": "qa_lead",
                "rejection_reason": "Changes miss edge cases.",
            },
        )
        assert resume_resp.status_code == 200
        assert resume_resp.json()["status"] == "COMPLETED"

        # 4. Verify final state
        get_resp = client.get(f"/api/v1/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "COMPLETED"


# ============================================================================
# 5. CONCURRENT RUN ISOLATION TESTS
# ============================================================================

class TestConcurrentRunIsolation:
    def test_concurrent_runs_remain_isolated(self, tmp_path, monkeypatch):
        # A real, git-initialized workspace (see _init_git_workspace) so
        # git_prepare_node's diff computation has a genuine baseline -
        # Run A and Run B write to distinct file paths within it.
        _init_git_workspace(tmp_path, monkeypatch, "concurrent_isolation_project")

        monkeypatch.setattr(
            "backend.graph.nodes.route_task",
            lambda msg: RoutingDecision(
                task_type=TaskType.BUG_FIX,
                confidence=0.9,
                reasoning="Bug fix",
                requires_planning=False,
                requires_knowledge=False,
            ),
        )
        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary=f"Fix for: {user_request}",
                changes=[
                    FileChange(
                        # Distinct per-request path (not shared between Run
                        # A and Run B) so each run's diff is unambiguously
                        # its own, not a race on the same file.
                        file_path=f"src/isolation_check_{user_request.replace(' ', '_')}.py",
                        change_type="MODIFY",
                        content=f"# {user_request}\n",
                        reason="Isolation check fix",
                    )
                ],
                requires_testing=True,
            ),
        )
        monkeypatch.setattr(
            "backend.graph.nodes.review_code_changes",
            lambda user_request, plan, developer_result: QAResult(
                status="PASS",
                issues=[],
                test_cases=[],
                summary="QA Pass",
            ),
        )
        # See client_and_runner in TestHITLLifecycleWithAPI: developer_node's
        # own inline exact-snippet LLM call is separate from the
        # generate_code_changes mock above and reachable via the workspace's
        # committed README.md - stubbed to deterministically produce no
        # patches from that branch so this test's own mocked FileChange
        # (via the fallback write path) is what reaches approval, not a
        # live/non-deterministic LLM call.
        monkeypatch.setattr(
            "backend.graph.nodes.invoke_structured",
            lambda llm, schema_cls, prompt, *a, **k: schema_cls(patches=[]),
        )

        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        # Start Run A and Run B
        res_a = client.post(
            "/api/v1/runs",
            json={"user_message": "Task A", "project_id": "concurrent_isolation_project"},
        )
        res_b = client.post(
            "/api/v1/runs",
            json={"user_message": "Task B", "project_id": "concurrent_isolation_project"},
        )

        assert res_a.status_code == 202
        assert res_b.status_code == 202

        run_a_id = res_a.json()["run_id"]
        run_b_id = res_b.json()["run_id"]

        assert run_a_id != run_b_id
        assert res_a.json()["status"] == "RUNNING"
        assert res_b.json()["status"] == "RUNNING"

        # Inspect both runs reached WAITING_APPROVAL after background dispatch
        status_a = client.get(f"/api/v1/runs/{run_a_id}")
        assert status_a.status_code == 200
        assert status_a.json()["status"] == "WAITING_APPROVAL"

        status_b = client.get(f"/api/v1/runs/{run_b_id}")
        assert status_b.status_code == 200
        assert status_b.json()["status"] == "WAITING_APPROVAL"

        # Resume Run A only
        resume_a = client.post(
            f"/api/v1/runs/{run_a_id}/resume",
            json={"approved": True, "reviewer": "eng_1"},
        )
        assert resume_a.status_code == 200
        assert resume_a.json()["status"] == "COMPLETED"

        # Run B must still be WAITING_APPROVAL
        status_b_after = client.get(f"/api/v1/runs/{run_b_id}")
        assert status_b_after.status_code == 200
        assert status_b_after.json()["status"] == "WAITING_APPROVAL"

        # Resume Run B with rejection
        resume_b = client.post(
            f"/api/v1/runs/{run_b_id}/resume",
            json={"approved": False, "rejection_reason": "Cancelled"},
        )
        assert resume_b.status_code == 200
        assert resume_b.json()["status"] == "COMPLETED"

    def test_dashboard_static_mount_and_redirect(self):
        client = TestClient(create_app())
        # Test /dashboard/ returns HTML
        res = client.get("/dashboard/")
        assert res.status_code == 200
        assert "Autonomous AI Software Engineer — HITL Control Panel" in res.text
        assert "text/html" in res.headers.get("content-type", "")

        # Test root redirect
        res_root = client.get("/", follow_redirects=True)
        assert res_root.status_code == 200
        assert "Autonomous AI Software Engineer — HITL Control Panel" in res_root.text

