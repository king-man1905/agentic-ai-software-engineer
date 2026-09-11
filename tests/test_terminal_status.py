"""
Regression tests for the terminal-status bug: a run whose revision budget is
exhausted while the final QA result is still FAIL must be reported FAILED,
never COMPLETED.

Root cause: AgentRunner._derive_status (backend/graph/runner.py) derived the
overall run status purely from LangGraph's structural "is the graph paused"
signal (state.next empty => COMPLETED), with no reference to the actual
qa_result/approval outcome. qa_router (backend/graph/nodes.py) routes
straight to END once the revision budget (MAX_REVISIONS) is exhausted while
QA is still FAIL, bypassing git_prepare/policy/approval/git_commit entirely
- and that path was indistinguishable, to _derive_status, from a genuine
committed success.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.graph.runner import AgentRunner
from backend.observability.collector import telemetry_collector
from backend.observability.store import telemetry_store
from backend.schemas.developer import DeveloperResult
from backend.schemas.qa import QAResult
from backend.schemas.routing import RoutingDecision, TaskType


@pytest.fixture(autouse=True)
def isolated_telemetry(tmp_path, monkeypatch):
    """Isolates telemetry_store to a tmp_path db so these runs don't leak
    into the real workspace/telemetry.db shared by other test invocations."""
    from backend.observability.store import TelemetryStore

    test_db = str(tmp_path / "test_terminal_status_telemetry.db")
    test_store = TelemetryStore(test_db)
    monkeypatch.setattr(telemetry_store, "db_path", test_db)
    monkeypatch.setattr(telemetry_collector, "store", test_store)


def _snapshot(next_tuple, values):
    """Minimal stand-in for LangGraph's StateSnapshot - _derive_status only
    ever reads .next and .values."""
    snap = MagicMock()
    snap.next = next_tuple
    snap.values = values
    return snap


# ---------------------------------------------------------------------------
# A-C: _derive_status unit tests
# ---------------------------------------------------------------------------


def test_derive_status_terminal_qa_fail_without_committed_approval_is_failed():
    """A. Graph terminal (state.next empty), final QA is FAIL, never
    committed -> FAILED, not COMPLETED."""
    runner = AgentRunner()
    snapshot = _snapshot(
        (),
        {"qa_result": QAResult(status="FAIL", summary="Quality Gate FAILED: ...")},
    )
    assert runner._derive_status(snapshot) == "FAILED"


def test_derive_status_terminal_qa_pass_committed_is_completed():
    """B. Graph terminal, final QA is PASS, approval COMMITTED -> COMPLETED."""
    runner = AgentRunner()
    snapshot = _snapshot(
        (),
        {
            "qa_result": QAResult(status="PASS", summary="All checks passed."),
            "approval_status": "COMMITTED",
        },
    )
    assert runner._derive_status(snapshot) == "COMPLETED"


def test_derive_status_paused_state_is_waiting_approval_regardless_of_qa():
    """C. Graph paused (state.next non-empty) -> WAITING_APPROVAL, even if
    the in-flight qa_result happens to be FAIL (a mid-revision snapshot)."""
    runner = AgentRunner()
    snapshot = _snapshot(("approval",), {"qa_result": QAResult(status="FAIL")})
    assert runner._derive_status(snapshot) == "WAITING_APPROVAL"


def test_derive_status_no_qa_result_still_completes():
    """Simple/no-op runs that never reach qa_node (e.g. a GENERAL task_type
    routed straight to end) must keep the original COMPLETED behavior -
    only qa_result-bearing FAIL terminations are affected by the fix."""
    runner = AgentRunner()
    snapshot = _snapshot((), {})
    assert runner._derive_status(snapshot) == "COMPLETED"


# ---------------------------------------------------------------------------
# D-F: full-graph integration tests
# ---------------------------------------------------------------------------

_NONEXISTENT_PROJECT_ID = "nonexistent_test_project_terminal_status"


@pytest.fixture
def mocked_router_and_developer(monkeypatch):
    """Routes straight to developer (no planning/knowledge LLM calls), and
    gives the developer a trivial no-op result (empty changes -> no
    patches), so git_prepare/check_ast/check_security all take their
    existing, already-safe empty-patches path and check_pytest skips
    cleanly (the project_id used by these tests has no real workspace
    directory, so no real pytest subprocess is spawned)."""
    monkeypatch.setattr(
        "backend.graph.nodes.route_task",
        lambda msg: RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            reasoning="test",
            requires_planning=False,
            requires_knowledge=False,
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.generate_code_changes",
        lambda user_request, plan, knowledge: DeveloperResult(
            summary="test change", changes=[], requires_testing=True, notes=[]
        ),
    )
    monkeypatch.setattr(
        "backend.agents.developer.revise_code_changes",
        lambda user_request, plan, previous_result, qa_result: DeveloperResult(
            summary="revised", changes=[], requires_testing=True, notes=[]
        ),
    )


def test_graph_exhausted_revisions_with_qa_fail_yields_failed_status(
    monkeypatch, mocked_router_and_developer
):
    """
    D. Full graph run: review_code_changes returns FAIL every time (never
    PASS), so the run exhausts MAX_REVISIONS revision attempts, still FAIL.
    The run's final status must be FAILED, and git_prepare/policy/approval/
    git_commit must never have been entered.
    """
    monkeypatch.setattr(
        "backend.graph.nodes.review_code_changes",
        lambda user_request, plan, developer_result: QAResult(
            status="FAIL", summary="", issues=[]
        ),
    )
    on_completed_spy = MagicMock()
    monkeypatch.setattr(telemetry_collector, "on_run_completed", on_completed_spy)

    runner = AgentRunner()
    app = create_app(runner=runner)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/runs",
        json={"user_message": "Fix a bug", "project_id": _NONEXISTENT_PROJECT_ID},
    )
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    status_resp = client.get(f"/api/v1/runs/{run_id}")
    assert status_resp.status_code == 200
    data = status_resp.json()

    # D: final status FAILED, not COMPLETED.
    assert data["status"] == "FAILED"
    assert data["qa_result"]["status"] == "FAIL"

    # git_prepare/policy/approval/git_commit never entered: no git_diff, no
    # policy_result, no PR artifacts (publish_pr is a separate endpoint
    # this test never calls, and could only ever have run after git_commit
    # set approval_status="COMMITTED", which never happened here).
    assert data["git_diff"] is None
    assert data["policy_result"] is None
    assert data["pr_number"] is None
    assert data["pr_url"] is None

    # F: on_run_completed must never fire for this FAILED outcome.
    on_completed_spy.assert_not_called()


def test_graph_qa_pass_after_revision_and_approval_still_completes(
    monkeypatch, mocked_router_and_developer
):
    """
    E. Legitimate success path unaffected: QA fails once, revises, then
    passes; human approves; run reaches git_commit -> END -> COMPLETED.
    (Patches stay empty throughout, so git_prepare/git_commit take their
    existing no-op paths - no real git I/O - this exercises only the
    control-flow/status-derivation fix, not git internals.)
    """
    qa_sequence = [
        QAResult(status="FAIL", summary="needs a fix"),
        QAResult(status="PASS", summary="All checks passed."),
    ]
    monkeypatch.setattr(
        "backend.graph.nodes.review_code_changes",
        lambda user_request, plan, developer_result: qa_sequence.pop(0),
    )

    runner = AgentRunner()
    app = create_app(runner=runner)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/runs",
        json={"user_message": "Fix a bug", "project_id": _NONEXISTENT_PROJECT_ID},
    )
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    status_resp = client.get(f"/api/v1/runs/{run_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "WAITING_APPROVAL"

    resume_resp = client.post(
        f"/api/v1/runs/{run_id}/resume",
        json={"approved": True, "reviewer": "test-reviewer"},
    )
    assert resume_resp.status_code == 200
    assert resume_resp.json()["status"] == "COMPLETED"
