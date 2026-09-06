"""
Cooperative cancellation for graph execution (Phase 8 Step 5).

Cancellation is never enforced by killing threads or processes. A run is
cancelled by durably recording the request (backend.observability.store),
and every graph node checks that record at its own entry point and stops
itself. This module is the small, shared check every node calls.
"""

from backend.graph.state import AgentState


class RunCancelledException(Exception):
    """
    Raised by `check_cancelled()` to unwind out of `graph.invoke()` when a
    run has been cancelled. Caught specifically by AgentRunner.start_run -
    never expected to reach an API caller directly.
    """

    def __init__(self, run_id: str, reason: str = "Cancelled by user request"):
        self.run_id = run_id
        self.reason = reason
        super().__init__(f"Run '{run_id}' cancelled: {reason}")


def check_cancelled(state: AgentState) -> None:
    """
    Raises RunCancelledException if cancellation has been requested for this
    run. Call at the start of every graph node, and again immediately before
    any irreversible action (sandbox execution, git commit, PR publication).
    A no-op for runs without a run_id (e.g. ad-hoc/legacy callers).
    """
    run_id = state.get("run_id")
    if not run_id:
        return
    from backend.observability.store import telemetry_store

    if telemetry_store.is_cancel_requested(run_id, state.get("organization_id")):
        raise RunCancelledException(run_id)


def mark_activity(state: AgentState, phase: str) -> None:
    """
    Records a heartbeat for the watchdog at a node boundary. Best-effort:
    telemetry must never fail the run it's observing.
    """
    run_id = state.get("run_id")
    if not run_id:
        return
    try:
        from backend.observability.store import telemetry_store

        telemetry_store.update_activity(run_id, state.get("organization_id"), phase=phase)
    except Exception:
        pass
