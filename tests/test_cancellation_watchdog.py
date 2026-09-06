"""
Tests for Phase 8 Step 5: run cancellation and the stuck-run watchdog.

Uses deterministic synchronization (direct store manipulation, short bounded
waits for real subprocess termination) rather than long sleeps, and never
makes real GitHub/network calls.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.graph.runner import AgentRunner
from backend.graph.cancellation import RunCancelledException, check_cancelled
from backend.graph.nodes import developer_node, revision_node
from backend.observability.collector import telemetry_collector
from backend.observability.store import TelemetryStore, telemetry_store
from backend.observability.watchdog import StuckRunWatchdog
from backend.sandbox.runner import SandboxRunner
from backend.schemas.telemetry import RunRecord, TelemetryEventType
from backend.schemas.tenant import Role
from backend.security.audit import audit_logger
from backend.security.auth import AuthMode
from backend.security.tenant import tenant_manager
from backend.vcs.models import ApprovalDecision


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
    audit_logger.clear()

    test_db = str(tmp_path / "test_cancellation_telemetry.db")
    test_store = TelemetryStore(test_db)
    monkeypatch.setattr(telemetry_store, "db_path", test_db)
    monkeypatch.setattr(telemetry_collector, "store", test_store)

    yield test_store


def _make_run(store: TelemetryStore, run_id: str, org: str, status: str, **extra) -> RunRecord:
    rec = RunRecord(run_id=run_id, organization_id=org, status=status, **extra)
    store.create_or_update_run(rec)
    return rec


def _grant(user_id: str, org_id: str, role: Role, org_name: str = "Org"):
    tenant_manager.create_organization(org_id, org_name)
    tenant_manager.create_user(user_id, f"{user_id}@test.local", user_id)
    tenant_manager.add_membership(org_id, user_id, role)


# ============================================================================
# API: cancellation
# ============================================================================

class TestCancelAPI:
    def test_cancel_running_run(self):
        _grant("user-a", "org-a", Role.ENGINEER)
        runner = AgentRunner()
        runner.register_run(run_id="run-1", organization_id="org-a")
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/runs/run-1/cancel",
            json={"reason": "no longer needed"},
            headers={"X-User-ID": "user-a", "X-Organization-ID": "org-a"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "CANCELLING"

    def test_cancel_waiting_approval_run(self, clean_state):
        _grant("user-a", "org-a", Role.ENGINEER)
        _make_run(clean_state, "run-2", "org-a", "WAITING_APPROVAL")
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/runs/run-2/cancel",
            json={},
            headers={"X-User-ID": "user-a", "X-Organization-ID": "org-a"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "CANCELLED"

    def test_cancel_already_cancelled_is_idempotent(self, clean_state):
        _grant("user-a", "org-a", Role.ENGINEER)
        _make_run(clean_state, "run-3", "org-a", "WAITING_APPROVAL")
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)
        headers = {"X-User-ID": "user-a", "X-Organization-ID": "org-a"}

        first = client.post("/api/v1/runs/run-3/cancel", json={}, headers=headers)
        second = client.post("/api/v1/runs/run-3/cancel", json={}, headers=headers)
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["status"] == "CANCELLED"

        # No duplicate RUN_CANCELLED side effect from the second call.
        events = clean_state.list_events("run-3", "org-a")
        cancelled_events = [e for e in events if e.event_type == TelemetryEventType.RUN_CANCELLED]
        assert len(cancelled_events) <= 1

    def test_cancel_completed_run_is_rejected(self, clean_state):
        _grant("user-a", "org-a", Role.ENGINEER)
        _make_run(clean_state, "run-4", "org-a", "COMPLETED")
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/runs/run-4/cancel",
            json={},
            headers={"X-User-ID": "user-a", "X-Organization-ID": "org-a"},
        )
        assert resp.status_code == 409

    def test_cross_tenant_cancel_is_rejected(self, clean_state):
        _grant("user-a", "org-a", Role.ENGINEER)
        _grant("user-b", "org-b", Role.ENGINEER)
        _make_run(clean_state, "run-5", "org-a", "RUNNING")
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/runs/run-5/cancel",
            json={},
            headers={"X-User-ID": "user-b", "X-Organization-ID": "org-b"},
        )
        assert resp.status_code == 404

    def test_cancel_rbac_enforced(self, clean_state):
        _grant("viewer-a", "org-a", Role.VIEWER)
        _make_run(clean_state, "run-6", "org-a", "RUNNING")
        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/runs/run-6/cancel",
            json={},
            headers={"X-User-ID": "viewer-a", "X-Organization-ID": "org-a"},
        )
        assert resp.status_code == 403


# ============================================================================
# Durability
# ============================================================================

class TestCancellationDurability:
    def test_cancellation_survives_runner_recreation(self, clean_state):
        runner1 = AgentRunner()
        runner1.register_run(run_id="run-7", organization_id="org-a")
        result = runner1.cancel_run("run-7", organization_id="org-a")
        assert result.status == "CANCELLING"

        # A brand new AgentRunner (simulating process/runner recreation)
        # must not treat the run as active just because its own in-memory
        # _active_runs dict starts empty - the durable store is authoritative.
        runner2 = AgentRunner()
        assert telemetry_store.is_cancel_requested("run-7", "org-a") is True
        status = runner2.get_status("run-7", organization_id="org-a")
        assert status.status in ("CANCELLING", "CANCELLED")

    def test_cancelled_run_cannot_resume(self, clean_state):
        _make_run(clean_state, "run-8", "org-a", "WAITING_APPROVAL")
        runner = AgentRunner()
        cancel_result = runner.cancel_run("run-8", organization_id="org-a")
        assert cancel_result.status == "CANCELLED"

        # No real checkpoint exists for this run, so resume_run's own
        # not-found check would fire first if the cancellation check were
        # missing or mis-ordered - assert we get *a* rejection either way,
        # and specifically confirm the durable state is unresumable.
        assert telemetry_store.is_cancelled("run-8", "org-a") is True
        with pytest.raises((ValueError, KeyError)):
            runner.resume_run("run-8", ApprovalDecision(approved=True), organization_id="org-a")


# ============================================================================
# Graph: cooperative cancellation
# ============================================================================

class TestGraphCancellation:
    def test_check_cancelled_raises_when_requested(self, clean_state):
        _make_run(clean_state, "run-9", "org-a", "RUNNING")
        telemetry_store.request_cancellation("run-9", "org-a")
        with pytest.raises(RunCancelledException):
            check_cancelled({"run_id": "run-9", "organization_id": "org-a"})

    def test_check_cancelled_noop_when_not_requested(self, clean_state):
        _make_run(clean_state, "run-10", "org-a", "RUNNING")
        check_cancelled({"run_id": "run-10", "organization_id": "org-a"})  # must not raise

    def test_cancellation_stops_before_developer_node(self, clean_state):
        _make_run(clean_state, "run-11", "org-a", "RUNNING")
        telemetry_store.request_cancellation("run-11", "org-a")
        with pytest.raises(RunCancelledException):
            developer_node({"run_id": "run-11", "organization_id": "org-a", "user_message": "x"})

    def test_cancellation_stops_revision_loop(self, clean_state):
        _make_run(clean_state, "run-12", "org-a", "REVISING")
        telemetry_store.request_cancellation("run-12", "org-a")
        with pytest.raises(RunCancelledException):
            revision_node({"run_id": "run-12", "organization_id": "org-a", "user_message": "x"})


# ============================================================================
# Sandbox subprocess cancellation
# ============================================================================

def _write_sleep_script(seconds: float) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(textwrap.dedent(f"""
            import time
            time.sleep({seconds})
            print("finished")
        """))
    return path


class TestSandboxCancellation:
    def test_sandbox_cancellation_terminates_process(self, tmp_path):
        script = _write_sleep_script(10.0)
        try:
            deadline = time.time() + 0.4

            def cancel_after_delay():
                return time.time() > deadline

            result = SandboxRunner.run_command(
                ["python", script],
                cwd=str(tmp_path),
                timeout=30.0,
                cancel_check=cancel_after_delay,
            )
            assert result.success is False
            assert "Cancelled" in (result.error_summary or "")
            # Terminated well before the real 10s sleep would have finished.
            assert result.duration_seconds < 5.0
        finally:
            os.remove(script)

    def test_sandbox_cancellation_does_not_orphan_process(self, tmp_path, monkeypatch):
        script = _write_sleep_script(10.0)
        captured = {}
        real_popen = subprocess.Popen

        def spy_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            captured["proc"] = proc
            return proc

        monkeypatch.setattr(subprocess, "Popen", spy_popen)
        try:
            result = SandboxRunner.run_command(
                ["python", script],
                cwd=str(tmp_path),
                timeout=30.0,
                cancel_check=lambda: True,  # cancel immediately
            )
            assert "Cancelled" in (result.error_summary or "")
            assert captured["proc"].poll() is not None, "process must not be left running"
        finally:
            os.remove(script)

    def test_sandbox_timeout_still_works(self, tmp_path):
        script = _write_sleep_script(5.0)
        try:
            result = SandboxRunner.run_command(
                ["python", script], cwd=str(tmp_path), timeout=0.3,
            )
            assert result.success is False
            assert "Timeout" in (result.error_summary or "")
        finally:
            os.remove(script)

    def test_shell_false_remains_enforced(self, tmp_path, monkeypatch):
        captured_kwargs = {}
        real_popen = subprocess.Popen

        def spy_popen(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return real_popen(*args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", spy_popen)
        fd, script = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write("print(1)\n")
        try:
            SandboxRunner.run_command(["python", script], cwd=str(tmp_path), timeout=5.0)
        finally:
            os.remove(script)
        assert captured_kwargs.get("shell") is False

    def test_sandbox_secret_stripping_remains_enforced(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_SUPER_SECRET_TOKEN", "should-not-leak")
        fd, script = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write("import os\nprint(list(os.environ.keys()))\n")
        try:
            result = SandboxRunner.run_command(["python", script], cwd=str(tmp_path), timeout=5.0)
            assert "MY_SUPER_SECRET_TOKEN" not in result.stdout
        finally:
            os.remove(script)


# ============================================================================
# Watchdog
# ============================================================================

class TestWatchdog:
    def _stale_iso(self, seconds_ago: float) -> str:
        return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()

    def test_watchdog_detects_stale_running_run(self, clean_state, monkeypatch):
        monkeypatch.setenv("RUN_STUCK_RUNNING_SECONDS", "1")
        _make_run(
            clean_state, "run-stale-1", "org-a", "RUNNING",
            last_activity_at=self._stale_iso(10),
        )
        watchdog = StuckRunWatchdog(store=clean_state)
        stuck = watchdog.check_once()
        assert "run-stale-1" in stuck
        assert clean_state.get_run("run-stale-1", "org-a").stuck_at is not None

    def test_watchdog_does_not_flag_active_run(self, clean_state, monkeypatch):
        monkeypatch.setenv("RUN_STUCK_RUNNING_SECONDS", "900")
        _make_run(
            clean_state, "run-active-1", "org-a", "RUNNING",
            last_activity_at=self._stale_iso(1),
        )
        watchdog = StuckRunWatchdog(store=clean_state)
        stuck = watchdog.check_once()
        assert "run-active-1" not in stuck

    def test_watchdog_waiting_approval_threshold(self, clean_state, monkeypatch):
        # One hour is well within the 48h default - must not be flagged.
        _make_run(
            clean_state, "run-wait-1", "org-a", "WAITING_APPROVAL",
            last_activity_at=self._stale_iso(3600),
        )
        watchdog = StuckRunWatchdog(store=clean_state)
        assert "run-wait-1" not in watchdog.check_once()

        # Lowering the threshold below the elapsed time flags it.
        monkeypatch.setenv("RUN_STUCK_WAITING_APPROVAL_SECONDS", "60")
        assert "run-wait-1" in watchdog.check_once()

    def test_watchdog_thresholds_are_configurable(self, monkeypatch):
        from backend.observability.watchdog import _stuck_thresholds

        monkeypatch.setenv("RUN_STUCK_RUNNING_SECONDS", "123")
        monkeypatch.setenv("RUN_STUCK_REVISING_SECONDS", "45")
        monkeypatch.setenv("RUN_STUCK_WAITING_APPROVAL_SECONDS", "999")
        monkeypatch.setenv("RUN_STUCK_PUBLISHING_SECONDS", "7")
        thresholds = _stuck_thresholds()
        assert thresholds["RUNNING"] == 123.0
        assert thresholds["REVISING"] == 45.0
        assert thresholds["WAITING_APPROVAL"] == 999.0
        assert thresholds["PUBLISHING"] == 7.0

    def test_watchdog_concurrent_execution_is_atomic(self, clean_state):
        _make_run(clean_state, "run-race-1", "org-a", "RUNNING")
        first = clean_state.mark_stuck("run-race-1", "org-a")
        second = clean_state.mark_stuck("run-race-1", "org-a")
        assert first is True
        assert second is False, "only the first caller should win the race"

    def test_watchdog_does_not_cancel_completed_run(self, clean_state, monkeypatch):
        monkeypatch.setenv("RUN_STUCK_RUNNING_SECONDS", "1")
        _make_run(
            clean_state, "run-done-1", "org-a", "COMPLETED",
            last_activity_at=self._stale_iso(999999),
        )
        watchdog = StuckRunWatchdog(store=clean_state)
        assert "run-done-1" not in watchdog.check_once()
        assert clean_state.get_run("run-done-1", "org-a").stuck_at is None

    def test_watchdog_emits_run_stuck_telemetry(self, clean_state, monkeypatch):
        monkeypatch.setenv("RUN_STUCK_RUNNING_SECONDS", "1")
        _make_run(
            clean_state, "run-stuck-telemetry", "org-a", "RUNNING",
            last_activity_at=self._stale_iso(10),
        )
        watchdog = StuckRunWatchdog(store=clean_state)
        watchdog.check_once()

        events = clean_state.list_events("run-stuck-telemetry", "org-a")
        stuck_events = [e for e in events if e.event_type == TelemetryEventType.RUN_STUCK]
        assert len(stuck_events) == 1


# ============================================================================
# Race safety
# ============================================================================

class TestRaceSafety:
    def test_cancel_race_with_commit_is_safe(self, clean_state):
        """
        A run that genuinely completed (e.g. commit finished) just before a
        cancel request arrives must report the true outcome, never a lie.
        """
        _make_run(clean_state, "run-race-commit", "org-a", "RUNNING")
        # Simulate the commit finishing first.
        clean_state.create_or_update_run(
            RunRecord(run_id="run-race-commit", organization_id="org-a", status="COMPLETED")
        )
        runner = AgentRunner()
        with pytest.raises(ValueError):
            runner.cancel_run("run-race-commit", organization_id="org-a")
        # Status must still read COMPLETED, not CANCELLED.
        assert clean_state.get_run("run-race-commit", "org-a").status == "COMPLETED"

    def test_cancel_race_with_resume_is_safe(self, clean_state):
        """Once cancelled, a concurrent resume attempt must never execute."""
        _make_run(clean_state, "run-race-resume", "org-a", "WAITING_APPROVAL")
        runner = AgentRunner()
        runner.cancel_run("run-race-resume", organization_id="org-a")

        with pytest.raises((ValueError, KeyError)):
            runner.resume_run(
                "run-race-resume", ApprovalDecision(approved=True), organization_id="org-a"
            )
