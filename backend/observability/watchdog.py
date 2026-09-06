"""
Stuck-run watchdog (Phase 8 Step 5).

Deliberately conservative: this module only *observes* and, at most,
*requests* cancellation of runs that have crossed their configured
staleness threshold. It never kills a process directly and never destroys
a workspace. Wiring `StuckRunWatchdog.check_once()` into a periodic
scheduler (cron, a background task loop, etc.) is a separate integration
decision left for later - this module is the detection logic itself,
built so it can be tested deterministically without one.
"""

import os
from datetime import datetime, timezone
from typing import List, Optional

from backend.observability.collector import telemetry_collector
from backend.observability.store import TelemetryStore, telemetry_store
from backend.schemas.telemetry import RunRecord, TERMINAL_RUN_STATUSES


def _threshold_seconds(env_var: str, default: float) -> float:
    try:
        return float(os.getenv(env_var, str(default)))
    except (TypeError, ValueError):
        return default


def _stuck_thresholds() -> dict:
    """Re-read on every call so tests can monkeypatch env vars per-test."""
    return {
        "RUNNING": _threshold_seconds("RUN_STUCK_RUNNING_SECONDS", 900.0),
        "REVISING": _threshold_seconds("RUN_STUCK_REVISING_SECONDS", 300.0),
        "WAITING_APPROVAL": _threshold_seconds("RUN_STUCK_WAITING_APPROVAL_SECONDS", 172800.0),
        "PUBLISHING": _threshold_seconds("RUN_STUCK_PUBLISHING_SECONDS", 120.0),
    }


def _auto_cancel_enabled() -> bool:
    return os.getenv("WATCHDOG_AUTO_CANCEL_STUCK", "false").strip().lower() in ("true", "1")


def _last_activity(rec: RunRecord) -> Optional[str]:
    return rec.last_activity_at or rec.started_at or rec.created_at


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


class StuckRunWatchdog:
    """
    Detects genuinely stuck runs by comparing each active run's last
    recorded activity against a per-status threshold. Never flags
    WAITING_APPROVAL as stuck just because a human hasn't responded yet -
    that threshold defaults to 48 hours specifically so a normal review
    delay is never mistaken for a hang.
    """

    def __init__(self, store: Optional[TelemetryStore] = None):
        self.store = store or telemetry_store

    def check_once(self) -> List[str]:
        """
        Runs one detection pass over all active runs. Returns the run_ids
        this call newly marked stuck (not ones already flagged, and not
        ones another concurrent caller won the race on).
        """
        thresholds = _stuck_thresholds()
        now = datetime.now(timezone.utc)
        newly_stuck: List[str] = []

        for rec in self.store.list_active_runs():
            if rec.status in TERMINAL_RUN_STATUSES or rec.stuck_at:
                continue

            threshold = thresholds.get(rec.status)
            if threshold is None:
                continue

            last_activity_dt = _parse_iso(_last_activity(rec))
            if last_activity_dt is None:
                continue
            if last_activity_dt.tzinfo is None:
                last_activity_dt = last_activity_dt.replace(tzinfo=timezone.utc)

            elapsed = (now - last_activity_dt).total_seconds()
            if elapsed <= threshold:
                continue

            # Re-verify against the durable record immediately before
            # mutating - the atomic UPDATE inside mark_stuck() is the real
            # guard against two watchdogs racing, but re-fetching first
            # avoids flagging a run that completed between list and here.
            current = self.store.get_run(rec.run_id, rec.organization_id)
            if not current or current.status in TERMINAL_RUN_STATUSES or current.stuck_at:
                continue

            won = self.store.mark_stuck(rec.run_id, rec.organization_id)
            if not won:
                # Another watchdog instance/pass already claimed this run.
                continue

            telemetry_collector.on_run_stuck(
                run_id=rec.run_id,
                organization_id=rec.organization_id,
                status=rec.status,
                current_phase=rec.current_phase,
                last_activity_at=_last_activity(rec),
                threshold_seconds=threshold,
            )
            newly_stuck.append(rec.run_id)

            if _auto_cancel_enabled():
                self.store.request_cancellation(
                    run_id=rec.run_id,
                    organization_id=rec.organization_id,
                    reason=f"Automatically cancelled: stuck in {rec.status} for over {threshold:.0f}s.",
                    actor="watchdog",
                )
                telemetry_collector.on_cancel_requested(
                    run_id=rec.run_id,
                    organization_id=rec.organization_id,
                    actor="watchdog",
                    reason="stuck_run_auto_cancel",
                )

        return newly_stuck


# Platform singleton, mirroring the other observability singletons.
stuck_run_watchdog = StuckRunWatchdog()
