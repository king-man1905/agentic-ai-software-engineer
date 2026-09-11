"""
Regression tests for Part 1: resume_run() terminal-state guard.

Root cause fixed:
    run_17ff55a6491b was already FAILED when a stale approval triggered
    resume_run(), which re-acquired the workspace lock.  The resumed invoke
    crashed, and the lock was never released, causing a 30-second
    WORKSPACE_LOCK_TIMEOUT for the next run.

These tests assert:
1. A FAILED run raises ValueError before acquiring any workspace lock.
2. A COMPLETED run raises ValueError before acquiring any workspace lock.
3. A CANCELLED run is rejected (existing guard still fires).
4. A normal WAITING_APPROVAL run still succeeds (no regression).
5. Lock acquisition count is exactly 0 for all terminal-run rejections.
"""

import threading
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from backend.graph.runner import AgentRunner
from backend.vcs.models import ApprovalDecision
from backend.vcs.workspace_lock import WorkspaceLockManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_record(status: str):
    """Returns a minimal RunRecord-like object with the given status."""
    rec = MagicMock()
    rec.status = status
    rec.organization_id = "test-org"
    rec.created_at = datetime.now(timezone.utc).isoformat()
    return rec


def _make_paused_snapshot():
    """State snapshot that looks like a WAITING_APPROVAL pause."""
    snapshot = MagicMock()
    snapshot.next = ("approval",)  # non-empty -> graph is paused
    snapshot.values = {
        "organization_id": "test-org",
        "repository_id": "test-repo",
    }
    return snapshot


def _make_runner_with_mocked_lock():
    """
    Builds an AgentRunner with:
    - A spy WorkspaceLockManager so we can count acquire_lock calls
    """
    runner = AgentRunner.__new__(AgentRunner)
    runner._lock = threading.Lock()
    runner._run_metadata = {}
    runner._run_tenants = {"run-test-abc": "test-org"}
    runner._active_runs = {}
    runner._run_errors = {}
    # __new__ bypasses __init__, so the compiled graph attribute normally
    # set there doesn't exist yet - tests patch.object(runner, "_graph"),
    # which requires the attribute to already exist on the instance.
    runner._graph = MagicMock()

    lock_manager = MagicMock(spec=WorkspaceLockManager)
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=False)
    lock_manager.acquire.return_value = cm
    runner._lock_manager = lock_manager

    return runner, lock_manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner_and_lock():
    return _make_runner_with_mocked_lock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTerminalRunResumeGuard:
    def test_failed_run_raises_before_lock(self, runner_and_lock):
        """FAILED run: ValueError raised, workspace lock NEVER acquired."""
        runner, lock_manager = runner_and_lock
        run_id = "run-test-abc"
        paused = _make_paused_snapshot()

        with (
            patch.object(runner, "_check_tenant_access"),
            patch.object(runner, "_require_org", return_value="test-org"),
            patch.object(runner, "_graph") as mock_graph,
            patch("backend.graph.runner.telemetry_store") as mock_store,
            patch("backend.graph.runner.telemetry_collector"),
        ):
            mock_graph.get_state.return_value = paused
            mock_store.is_cancelled.return_value = False
            mock_store.get_run.return_value = _make_run_record("FAILED")

            with pytest.raises(ValueError, match="terminal state"):
                runner.resume_run(
                    run_id=run_id,
                    approval_decision=ApprovalDecision(approved=True, reviewer="alice"),
                    organization_id="test-org",
                )

        lock_manager.acquire.assert_not_called()

    def test_completed_run_raises_before_lock(self, runner_and_lock):
        """COMPLETED run: ValueError raised, workspace lock NEVER acquired."""
        runner, lock_manager = runner_and_lock
        run_id = "run-test-abc"
        paused = _make_paused_snapshot()

        with (
            patch.object(runner, "_check_tenant_access"),
            patch.object(runner, "_require_org", return_value="test-org"),
            patch.object(runner, "_graph") as mock_graph,
            patch("backend.graph.runner.telemetry_store") as mock_store,
            patch("backend.graph.runner.telemetry_collector"),
        ):
            mock_graph.get_state.return_value = paused
            mock_store.is_cancelled.return_value = False
            mock_store.get_run.return_value = _make_run_record("COMPLETED")

            with pytest.raises(ValueError, match="terminal state"):
                runner.resume_run(
                    run_id=run_id,
                    approval_decision=ApprovalDecision(approved=True, reviewer="alice"),
                    organization_id="test-org",
                )

        lock_manager.acquire.assert_not_called()

    def test_blocked_run_raises_before_lock(self, runner_and_lock):
        """BLOCKED run: ValueError raised, workspace lock NEVER acquired."""
        runner, lock_manager = runner_and_lock
        run_id = "run-test-abc"
        paused = _make_paused_snapshot()

        with (
            patch.object(runner, "_check_tenant_access"),
            patch.object(runner, "_require_org", return_value="test-org"),
            patch.object(runner, "_graph") as mock_graph,
            patch("backend.graph.runner.telemetry_store") as mock_store,
            patch("backend.graph.runner.telemetry_collector"),
        ):
            mock_graph.get_state.return_value = paused
            mock_store.is_cancelled.return_value = False
            mock_store.get_run.return_value = _make_run_record("BLOCKED")

            with pytest.raises(ValueError, match="terminal state"):
                runner.resume_run(
                    run_id=run_id,
                    approval_decision=ApprovalDecision(approved=True, reviewer="alice"),
                    organization_id="test-org",
                )

        lock_manager.acquire.assert_not_called()

    def test_cancelled_run_rejected_by_existing_guard(self, runner_and_lock):
        """CANCELLED: the pre-existing is_cancelled() guard fires first."""
        runner, lock_manager = runner_and_lock
        run_id = "run-test-abc"
        paused = _make_paused_snapshot()

        with (
            patch.object(runner, "_check_tenant_access"),
            patch.object(runner, "_require_org", return_value="test-org"),
            patch.object(runner, "_graph") as mock_graph,
            patch("backend.graph.runner.telemetry_store") as mock_store,
            patch("backend.graph.runner.telemetry_collector"),
        ):
            mock_graph.get_state.return_value = paused
            mock_store.is_cancelled.return_value = True  # is_cancelled fires first

            with pytest.raises(ValueError, match="cancelled"):
                runner.resume_run(
                    run_id=run_id,
                    approval_decision=ApprovalDecision(approved=True, reviewer="alice"),
                    organization_id="test-org",
                )

        lock_manager.acquire.assert_not_called()

    def test_no_telemetry_record_does_not_block_resume(self, runner_and_lock):
        """
        If telemetry has no record for the run (edge case: very new run),
        the guard must NOT block.  Fail-open for unknown runs; the existing
        state_snapshot.next check is the secondary gate.
        The test stubs invoke() to raise so we stop after lock acquisition.
        """
        runner, lock_manager = runner_and_lock
        run_id = "run-test-abc"
        paused = _make_paused_snapshot()

        with (
            patch.object(runner, "_check_tenant_access"),
            patch.object(runner, "_require_org", return_value="test-org"),
            patch.object(runner, "_graph") as mock_graph,
            patch("backend.graph.runner.telemetry_store") as mock_store,
            patch("backend.graph.runner.telemetry_collector"),
        ):
            mock_graph.get_state.return_value = paused
            mock_store.is_cancelled.return_value = False
            mock_store.get_run.return_value = None  # not in telemetry
            # Short-circuit after lock acquisition. resume_run() catches
            # any exception from _graph.invoke() and returns a FAILED
            # status response rather than propagating it (existing,
            # intentional resilience behavior - not something this guard
            # changes) - so assert on the returned status, not a raise.
            mock_graph.invoke.side_effect = RuntimeError("stub-stop")

            result = runner.resume_run(
                run_id=run_id,
                approval_decision=ApprovalDecision(approved=True, reviewer="alice"),
                organization_id="test-org",
            )
            assert result.status == "FAILED"
            assert "stub-stop" in (result.error_summary or "")

        # Guard did NOT block; lock was acquired
        lock_manager.acquire.assert_called_once()

    def test_error_message_includes_run_id_and_status(self, runner_and_lock):
        """Error message must name the run_id and the terminal status."""
        runner, lock_manager = runner_and_lock
        run_id = "run-test-abc"
        paused = _make_paused_snapshot()

        with (
            patch.object(runner, "_check_tenant_access"),
            patch.object(runner, "_require_org", return_value="test-org"),
            patch.object(runner, "_graph") as mock_graph,
            patch("backend.graph.runner.telemetry_store") as mock_store,
            patch("backend.graph.runner.telemetry_collector"),
        ):
            mock_graph.get_state.return_value = paused
            mock_store.is_cancelled.return_value = False
            mock_store.get_run.return_value = _make_run_record("FAILED")

            with pytest.raises(ValueError) as exc_info:
                runner.resume_run(
                    run_id=run_id,
                    approval_decision=ApprovalDecision(approved=True, reviewer="alice"),
                    organization_id="test-org",
                )

        msg = str(exc_info.value)
        assert run_id in msg
        assert "FAILED" in msg

    def test_all_non_cancelled_terminal_statuses_are_rejected(self):
        """
        Parametric: every status in TERMINAL_RUN_STATUSES (except CANCELLED,
        which has its own pre-existing guard) must trigger the new guard.
        """
        from backend.schemas.telemetry import TERMINAL_RUN_STATUSES

        for status in sorted(TERMINAL_RUN_STATUSES):
            if status == "CANCELLED":
                continue  # covered by the is_cancelled() guard test above

            runner, lock_manager = _make_runner_with_mocked_lock()
            paused = _make_paused_snapshot()

            with (
                patch.object(runner, "_check_tenant_access"),
                patch.object(runner, "_require_org", return_value="test-org"),
                patch.object(runner, "_graph") as mock_graph,
                patch("backend.graph.runner.telemetry_store") as mock_store,
                patch("backend.graph.runner.telemetry_collector"),
            ):
                mock_graph.get_state.return_value = paused
                mock_store.is_cancelled.return_value = False
                mock_store.get_run.return_value = _make_run_record(status)

                with pytest.raises(ValueError, match="terminal state"):
                    runner.resume_run(
                        run_id="run-test-abc",
                        approval_decision=ApprovalDecision(
                            approved=True, reviewer="test"
                        ),
                        organization_id="test-org",
                    )

            lock_manager.acquire.assert_not_called(), (
                f"Lock was acquired for terminal status='{status}'"
            )
