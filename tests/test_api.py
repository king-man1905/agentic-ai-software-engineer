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


# ============================================================================
# 4. FULL HITL APPROVAL & REJECTION LIFECYCLE TESTS (MOCKED AGENTS)
# ============================================================================

class TestHITLLifecycleWithAPI:
    @pytest.fixture
    def client_and_runner(self, monkeypatch):
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

        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)
        return client, runner

    def test_full_hitl_approval_lifecycle(self, client_and_runner):
        client, runner = client_and_runner

        # 1. Create run
        create_resp = client.post(
            "/api/v1/runs",
            json={
                "user_message": "Fix the null check in service",
                "metadata": {"source": "unit_test"},
            },
        )
        assert create_resp.status_code == 201
        run_data = create_resp.json()
        run_id = run_data["run_id"]
        assert run_id.startswith("run_")
        assert run_data["status"] == "WAITING_APPROVAL"
        assert run_data["current_node"] == "approval"

        # 2. Inspect run via GET
        get_resp = client.get(f"/api/v1/runs/{run_id}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["status"] == "WAITING_APPROVAL"
        assert get_data["current_node"] == "approval"

        # 3. Resume run with approval
        resume_resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={
                "approved": True,
                "reviewer": "lead_engineer",
            },
        )
        assert resume_resp.status_code == 200
        resumed_data = resume_resp.json()
        assert resumed_data["status"] == "COMPLETED"

        # 4. Re-inspect run via GET to verify completed state persistence
        final_get = client.get(f"/api/v1/runs/{run_id}")
        assert final_get.status_code == 200
        assert final_get.json()["status"] == "COMPLETED"

        # 5. Attempting to resume already completed run returns 409 Conflict
        conflict_resp = client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={"approved": True},
        )
        assert conflict_resp.status_code == 409
        assert "not awaiting approval" in conflict_resp.json()["detail"]

    def test_full_hitl_rejection_lifecycle(self, client_and_runner):
        client, runner = client_and_runner

        # 1. Create run
        create_resp = client.post(
            "/api/v1/runs",
            json={"user_message": "Fix the null check in service"},
        )
        assert create_resp.status_code == 201
        run_id = create_resp.json()["run_id"]
        assert create_resp.json()["status"] == "WAITING_APPROVAL"

        # 2. Resume run with rejection
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

        # 3. Verify final state
        get_resp = client.get(f"/api/v1/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "COMPLETED"


# ============================================================================
# 5. CONCURRENT RUN ISOLATION TESTS
# ============================================================================

class TestConcurrentRunIsolation:
    def test_concurrent_runs_remain_isolated(self, monkeypatch):
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
                changes=[],
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

        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        # Start Run A and Run B
        res_a = client.post("/api/v1/runs", json={"user_message": "Task A"})
        res_b = client.post("/api/v1/runs", json={"user_message": "Task B"})

        run_a_id = res_a.json()["run_id"]
        run_b_id = res_b.json()["run_id"]

        assert run_a_id != run_b_id
        assert res_a.json()["status"] == "WAITING_APPROVAL"
        assert res_b.json()["status"] == "WAITING_APPROVAL"

        # Resume Run A only
        resume_a = client.post(
            f"/api/v1/runs/{run_a_id}/resume",
            json={"approved": True, "reviewer": "eng_1"},
        )
        assert resume_a.status_code == 200
        assert resume_a.json()["status"] == "COMPLETED"

        # Run B must still be WAITING_APPROVAL
        status_b = client.get(f"/api/v1/runs/{run_b_id}")
        assert status_b.status_code == 200
        assert status_b.json()["status"] == "WAITING_APPROVAL"

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

