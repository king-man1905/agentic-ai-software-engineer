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

from types import SimpleNamespace
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


def _snapshot(next_tuple, values, tasks=()):
    """Minimal stand-in for LangGraph's StateSnapshot - _derive_status reads
    .next, .values, and (for a non-empty .next) .tasks to tell a genuine
    interrupt() pause apart from a crashed node."""
    snap = MagicMock()
    snap.next = next_tuple
    snap.values = values
    snap.tasks = tasks
    return snap


def _fake_task(name, error=None, interrupts=()):
    """Minimal stand-in for LangGraph's PregelTask - _derive_status only
    ever reads .name, .error, and .interrupts."""
    return SimpleNamespace(name=name, error=error, interrupts=interrupts)


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
# Crashed-node vs genuine-interrupt regression tests
#
# Root cause: state.next is non-empty both when the graph is genuinely
# paused at interrupt() (approval_node) AND when a node raised an exception
# before completing - LangGraph leaves the crashed node's name sitting in
# `next` either way. A task with a populated `interrupts` tuple is a real
# HITL pause; a task with a populated `error` and no `interrupts` is a
# crashed node and must be reported FAILED, never a phantom approval
# request with empty qa_result/policy_result/git_diff.
# ---------------------------------------------------------------------------


def test_derive_status_crashed_developer_node_is_failed():
    """1. developer_node raised (e.g. the real AST pre-flight ValueError)
    and never completed - next=("developer",) with an error and no
    interrupts must be FAILED, not WAITING_APPROVAL."""
    runner = AgentRunner()
    snapshot = _snapshot(
        ("developer",),
        {"routing": object()},
        tasks=(
            _fake_task(
                "developer",
                error="ValueError(\"AST pre-flight validation failed for README.md: "
                "['Target original snippet not found in source file: README.md']\")",
                interrupts=(),
            ),
        ),
    )
    assert runner._derive_status(snapshot) == "FAILED"


def test_derive_status_crashed_arbitrary_node_is_failed():
    """2. Any other node crashing (not just developer_node) must be
    reported FAILED the same way - e.g. cleanup_node raising an OSError
    against a workspace path that doesn't exist."""
    runner = AgentRunner()
    snapshot = _snapshot(
        ("cleanup",),
        {"git_diff": None},
        tasks=(
            _fake_task(
                "cleanup",
                error="NotADirectoryError(20, 'The directory name is invalid')",
                interrupts=(),
            ),
        ),
    )
    assert runner._derive_status(snapshot) == "FAILED"


def test_derive_status_genuine_approval_interrupt_is_waiting_approval():
    """3. A real interrupt() pause at approval_node - task.error is None
    and task.interrupts is populated - must still be WAITING_APPROVAL."""
    runner = AgentRunner()
    snapshot = _snapshot(
        ("approval",),
        {"qa_result": QAResult(status="PASS")},
        tasks=(
            _fake_task(
                "approval",
                error=None,
                interrupts=(MagicMock(value={"task": "approval_required"}),),
            ),
        ),
    )
    assert runner._derive_status(snapshot) == "WAITING_APPROVAL"


def test_derive_status_paused_with_no_matching_task_defaults_to_waiting_approval():
    """Safety net: if the pending task can't be found in .tasks at all
    (e.g. an empty tasks tuple, as every pre-existing test/caller that
    doesn't populate it produces), preserve the original behavior -
    WAITING_APPROVAL - rather than guessing FAILED with no evidence."""
    runner = AgentRunner()
    snapshot = _snapshot(("approval",), {}, tasks=())
    assert runner._derive_status(snapshot) == "WAITING_APPROVAL"


def test_derive_status_policy_blocked_no_op_terminal_status_unaffected():
    """4. A policy-blocked or no-changes-needed run (both terminal -
    state.next is empty, routed through cleanup, never through approval)
    must remain unaffected by this fix: QA passed trivially (nothing to
    check), so it stays COMPLETED exactly as before."""
    runner = AgentRunner()
    snapshot = _snapshot(
        (),
        {
            "qa_result": QAResult(status="PASS", summary="No patches to validate."),
            "approval_status": "NO_CHANGES_NEEDED",
        },
    )
    assert runner._derive_status(snapshot) == "COMPLETED"


def test_derive_status_completed_and_committed_terminal_statuses_unaffected():
    """4 (cont.). Existing committed-success terminal status logic is
    untouched by this fix (state.next is empty for all of these, so the
    new crashed-task check never runs)."""
    runner = AgentRunner()
    completed = _snapshot(
        (),
        {
            "qa_result": QAResult(status="PASS", summary="All checks passed."),
            "approval_status": "COMMITTED",
        },
    )
    assert runner._derive_status(completed) == "COMPLETED"

    rejected = _snapshot(
        (),
        {
            "qa_result": QAResult(status="PASS", summary="All checks passed."),
            "approval_status": "REJECTED_AND_CLEANED",
        },
    )
    assert runner._derive_status(rejected) == "COMPLETED"


def test_derive_status_commit_failed_is_failed():
    """A run whose commit failed must be derived as FAILED, never COMPLETED."""
    runner = AgentRunner()
    commit_failed = _snapshot(
        (),
        {
            "qa_result": QAResult(status="PASS", summary="All checks passed."),
            "approval_status": "COMMIT_FAILED",
        },
    )
    assert runner._derive_status(commit_failed) == "FAILED"


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


def test_graph_qa_pass_after_revision_empty_diff_auto_completes_as_no_changes_needed(
    monkeypatch, mocked_router_and_developer
):
    """
    E. Legitimate success path unaffected: QA fails once, revises, then
    passes. Patches stay empty throughout (a no-op diff), so the no-op
    guard in route_after_policy (backend/graph/nodes.py) routes straight to
    cleanup instead of approval - the run auto-completes as
    COMPLETED/NO_CHANGES_NEEDED without ever pausing for a human, since
    there is nothing to review. This supersedes the old expectation that
    such a run would sit at WAITING_APPROVAL until approved: asking a human
    to approve/reject an empty diff was the bug this guard fixes. Resuming
    it afterward is a 409 conflict, same as resuming any other
    already-terminal run.
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
    assert status_resp.json()["status"] == "COMPLETED"

    state_values = runner.get_state_values(run_id, organization_id=None)
    assert state_values.get("approval_status") == "NO_CHANGES_NEEDED"
    # No ApprovalDecision exists - approval_node was never entered.
    assert state_values.get("approval") is None

    resume_resp = client.post(
        f"/api/v1/runs/{run_id}/resume",
        json={"approved": True, "reviewer": "test-reviewer"},
    )
    assert resume_resp.status_code == 409


def test_graph_real_node_crash_reports_failed_not_waiting_approval(monkeypatch):
    """
    End-to-end reproduction of the investigated bug: a node genuinely
    raising mid-graph (here, developer_node, via a raising route_task so no
    other mocking is needed) must be reported FAILED - with empty
    qa_result/policy_result/git_diff, exactly as a crash should look - by
    the real POST /api/v1/runs -> GET /api/v1/runs/{id} flow, never
    WAITING_APPROVAL with a phantom empty diff.
    """
    def boom(msg):
        raise ValueError("simulated node crash")

    monkeypatch.setattr("backend.graph.nodes.route_task", boom)

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

    assert data["status"] == "FAILED"
    assert data["qa_result"] is None
    assert data["policy_result"] is None
    assert data["git_diff"] is None
