"""
Thread-safe persistent storage and querying engine for runs, lifecycle events, and evaluation analytics.
Uses SQLite with optimized indexes and tenant-isolated SQL queries.
"""

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.observability.sanitizer import sanitize_telemetry_payload
from backend.schemas.telemetry import (
    AnalyticsOverview,
    EvaluationResult,
    EvaluationSummary,
    FailureAnalytics,
    ModelAnalytics,
    ModelUsage,
    ProviderUsage,
    QualityAnalytics,
    RAGAnalytics,
    RunRecord,
    TelemetryEvent,
    TelemetryEventType,
    TERMINAL_RUN_STATUSES,
    WATCHDOG_TRACKED_STATUSES,
)

# Columns added after the original `runs` table shipped. CREATE TABLE IF NOT
# EXISTS is a no-op against an existing table, so a real migration step is
# needed for any store already on disk (e.g. workspace/telemetry.db from a
# prior deployment) - not just for brand new ones.
_CANCELLATION_COLUMNS = {
    "cancel_requested": "INTEGER DEFAULT 0",
    "cancellation_requested_at": "TEXT",
    "cancellation_requested_by": "TEXT",
    "cancellation_reason": "TEXT",
    "cancelled_at": "TEXT",
    "last_activity_at": "TEXT",
    "current_phase": "TEXT",
    "stuck_at": "TEXT",
}


class TelemetryStore:
    """
    Persistent SQLite repository for telemetry, events, and evaluations.
    Thread-safe with connections managed per thread or via connection locks.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            workspace_dir = Path("workspace")
            workspace_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(workspace_dir / "telemetry.db")
        else:
            self.db_path = db_path

        # RLock, not Lock: get_quality_analytics() calls get_rag_analytics()
        # while still holding the lock, which would self-deadlock a plain Lock.
        self._lock = threading.RLock()
        self._closed = False
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        with self._lock:
            if self._closed:
                raise RuntimeError("TelemetryStore is closed.")
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def close(self) -> None:
        """Closes the telemetry store gracefully and idempotently."""
        with self._lock:
            self._closed = True

    def reopen(self) -> None:
        """Re-arms the telemetry store if closed."""
        with self._lock:
            self._closed = False

    def is_ready(self) -> bool:
        """Verifies database connectivity and readiness."""
        with self._lock:
            if self._closed:
                return False
        try:
            with self._lock, self._get_conn() as conn:
                cursor = conn.execute("SELECT 1;")
                return cursor.fetchone() is not None
        except Exception:
            return False

    def _init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    user_id TEXT,
                    repository TEXT,
                    branch TEXT,
                    user_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    duration_ms REAL,
                    status TEXT NOT NULL,
                    failure_category TEXT,
                    safe_failure_message TEXT,
                    provider TEXT,
                    model TEXT,
                    revision_count INTEGER DEFAULT 0,
                    qa_status TEXT,
                    qa_summary TEXT,
                    rag_status TEXT,
                    rag_quality_summary TEXT,
                    policy_decision TEXT,
                    risk_score REAL,
                    approval_required INTEGER DEFAULT 0,
                    approval_status TEXT,
                    approval_latency_ms REAL,
                    approval_reviewer TEXT,
                    approval_decision TEXT,
                    patch_hash TEXT,
                    commit_status TEXT,
                    github_status TEXT,
                    pr_status TEXT,
                    pr_url TEXT,
                    pr_number INTEGER,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    estimated_input_cost REAL,
                    estimated_output_cost REAL,
                    estimated_total_cost REAL,
                    currency TEXT DEFAULT 'USD',
                    cancel_requested INTEGER DEFAULT 0,
                    cancellation_requested_at TEXT,
                    cancellation_requested_by TEXT,
                    cancellation_reason TEXT,
                    cancelled_at TEXT,
                    last_activity_at TEXT,
                    current_phase TEXT,
                    stuck_at TEXT
                );
                """
            )
            self._migrate_runs_table(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    duration_ms REAL,
                    safe_metadata TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_stuck ON runs (status, stuck_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_org_created ON runs (organization_id, created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_org_status ON runs (organization_id, status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events (run_id, timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_org ON events (organization_id, timestamp);")
            conn.commit()

    def _migrate_runs_table(self, conn: sqlite3.Connection) -> None:
        """Adds any cancellation/watchdog columns missing from an existing
        `runs` table (e.g. a telemetry.db created before Phase 8 Step 5).
        CREATE TABLE IF NOT EXISTS alone would silently skip these on a
        pre-existing table."""
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(runs);").fetchall()}
        for column, decl in _CANCELLATION_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {decl};")
        conn.commit()

    def reset(self) -> None:
        """Clears all records for clean test isolation."""
        with self._lock, self._get_conn() as conn:
            conn.execute("DELETE FROM runs;")
            conn.execute("DELETE FROM events;")
            conn.execute("DELETE FROM evaluations;")
            conn.commit()

    # =========================================================================
    # Run Records CRUD
    # =========================================================================

    def create_or_update_run(self, record: RunRecord) -> None:
        """Persists or updates a RunRecord after applying redaction."""
        sanitized_msg = sanitize_telemetry_payload(record.user_message)
        sanitized_fail = sanitize_telemetry_payload(record.safe_failure_message)

        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, organization_id, user_id, repository, branch, user_message,
                    created_at, started_at, completed_at, duration_ms, status,
                    failure_category, safe_failure_message, provider, model,
                    revision_count, qa_status, qa_summary, rag_status, rag_quality_summary,
                    policy_decision, risk_score, approval_required, approval_status,
                    approval_latency_ms, approval_reviewer, approval_decision,
                    patch_hash, commit_status, github_status,
                    pr_status, pr_url, pr_number, input_tokens, output_tokens, total_tokens,
                    estimated_input_cost, estimated_output_cost, estimated_total_cost, currency,
                    cancel_requested, cancellation_requested_at, cancellation_requested_by,
                    cancellation_reason, cancelled_at, last_activity_at, current_phase, stuck_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(run_id) DO UPDATE SET
                    organization_id=excluded.organization_id,
                    user_id=coalesce(excluded.user_id, runs.user_id),
                    repository=coalesce(excluded.repository, runs.repository),
                    branch=coalesce(excluded.branch, runs.branch),
                    user_message=coalesce(excluded.user_message, runs.user_message),
                    started_at=coalesce(excluded.started_at, runs.started_at),
                    completed_at=coalesce(excluded.completed_at, runs.completed_at),
                    duration_ms=coalesce(excluded.duration_ms, runs.duration_ms),
                    status=excluded.status,
                    failure_category=coalesce(excluded.failure_category, runs.failure_category),
                    safe_failure_message=coalesce(excluded.safe_failure_message, runs.safe_failure_message),
                    provider=coalesce(excluded.provider, runs.provider),
                    model=coalesce(excluded.model, runs.model),
                    revision_count=max(excluded.revision_count, runs.revision_count),
                    qa_status=coalesce(excluded.qa_status, runs.qa_status),
                    qa_summary=coalesce(excluded.qa_summary, runs.qa_summary),
                    rag_status=coalesce(excluded.rag_status, runs.rag_status),
                    rag_quality_summary=coalesce(excluded.rag_quality_summary, runs.rag_quality_summary),
                    policy_decision=coalesce(excluded.policy_decision, runs.policy_decision),
                    risk_score=coalesce(excluded.risk_score, runs.risk_score),
                    approval_required=max(excluded.approval_required, runs.approval_required),
                    approval_status=coalesce(excluded.approval_status, runs.approval_status),
                    approval_latency_ms=coalesce(excluded.approval_latency_ms, runs.approval_latency_ms),
                    approval_reviewer=coalesce(excluded.approval_reviewer, runs.approval_reviewer),
                    approval_decision=coalesce(excluded.approval_decision, runs.approval_decision),
                    patch_hash=coalesce(excluded.patch_hash, runs.patch_hash),
                    commit_status=coalesce(excluded.commit_status, runs.commit_status),
                    github_status=coalesce(excluded.github_status, runs.github_status),
                    pr_status=coalesce(excluded.pr_status, runs.pr_status),
                    pr_url=coalesce(excluded.pr_url, runs.pr_url),
                    pr_number=coalesce(excluded.pr_number, runs.pr_number),
                    input_tokens=max(excluded.input_tokens, runs.input_tokens),
                    output_tokens=max(excluded.output_tokens, runs.output_tokens),
                    total_tokens=max(excluded.total_tokens, runs.total_tokens),
                    estimated_input_cost=coalesce(excluded.estimated_input_cost, runs.estimated_input_cost),
                    estimated_output_cost=coalesce(excluded.estimated_output_cost, runs.estimated_output_cost),
                    estimated_total_cost=coalesce(excluded.estimated_total_cost, runs.estimated_total_cost),
                    currency=excluded.currency,
                    cancel_requested=max(excluded.cancel_requested, runs.cancel_requested),
                    cancellation_requested_at=coalesce(excluded.cancellation_requested_at, runs.cancellation_requested_at),
                    cancellation_requested_by=coalesce(excluded.cancellation_requested_by, runs.cancellation_requested_by),
                    cancellation_reason=coalesce(excluded.cancellation_reason, runs.cancellation_reason),
                    cancelled_at=coalesce(excluded.cancelled_at, runs.cancelled_at),
                    last_activity_at=coalesce(excluded.last_activity_at, runs.last_activity_at),
                    current_phase=coalesce(excluded.current_phase, runs.current_phase),
                    stuck_at=coalesce(excluded.stuck_at, runs.stuck_at);
                """,
                (
                    record.run_id, record.organization_id, record.user_id, record.repository, record.branch, sanitized_msg,
                    record.created_at, record.started_at, record.completed_at, record.duration_ms, str(record.status),
                    record.failure_category.value if record.failure_category else None, sanitized_fail, record.provider, record.model,
                    record.revision_count, record.qa_status, record.qa_summary, record.rag_status, record.rag_quality_summary,
                    record.policy_decision, record.risk_score, 1 if record.approval_required else 0, record.approval_status,
                    record.approval_latency_ms, record.approval_reviewer, record.approval_decision,
                    record.patch_hash, record.commit_status, record.github_status,
                    record.pr_status, record.pr_url, record.pr_number, record.input_tokens, record.output_tokens, record.total_tokens,
                    record.estimated_input_cost, record.estimated_output_cost, record.estimated_total_cost, record.currency,
                    1 if record.cancel_requested else 0, record.cancellation_requested_at, record.cancellation_requested_by,
                    record.cancellation_reason, record.cancelled_at, record.last_activity_at, record.current_phase, record.stuck_at,
                ),
            )
            conn.commit()

    def get_run(self, run_id: str, organization_id: Optional[str] = None) -> Optional[RunRecord]:
        """Fetches a RunRecord with optional or strict tenant scoping."""
        with self._lock, self._get_conn() as conn:
            if organization_id:
                cursor = conn.execute(
                    "SELECT * FROM runs WHERE run_id = ? AND organization_id = ?;",
                    (run_id, organization_id),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM runs WHERE run_id = ?;",
                    (run_id,),
                )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_run_record(row)

    def list_runs(
        self,
        organization_id: str,
        status: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[RunRecord]:
        """Lists tenant runs ordered by creation timestamp descending."""
        query = "SELECT * FROM runs WHERE organization_id = ?"
        params: List[Any] = [organization_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        if project_id:
            query += " AND repository = ?"
            params.append(project_id)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_run_record(row) for row in cursor.fetchall()]

    def _row_to_run_record(self, row: sqlite3.Row) -> RunRecord:
        d = dict(row)
        d["approval_required"] = bool(d.get("approval_required", 0))
        d["cancel_requested"] = bool(d.get("cancel_requested", 0))
        return RunRecord(**d)

    # =========================================================================
    # Cancellation (Phase 8 Step 5)
    # =========================================================================
    #
    # All transitions here are single, atomic SQLite UPDATE statements with a
    # WHERE clause that only matches rows in the expected state - the
    # equivalent of a compare-and-swap. `cursor.rowcount` tells the caller
    # whether *this* call actually made the change, which is what makes it
    # safe for concurrent callers (two API requests, or two watchdog
    # instances) to race on the same run without a separate lock: only one
    # UPDATE can ever match and mutate the row.

    def request_cancellation(
        self,
        run_id: str,
        organization_id: Optional[str],
        reason: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> str:
        """
        Atomically requests cancellation of a run. Returns one of:
          "REQUESTED"         - this call recorded the request
          "ALREADY_CANCELLED" - the run was already CANCELLED (idempotent)
          "ALREADY_TERMINAL"  - the run already finished in some other way
          "NOT_FOUND"         - no such run for this organization
        Never overwrites a run that's already in a terminal state, and never
        requests cancellation twice for the same already-cancelled run.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            existing = self._get_run_row(conn, run_id, organization_id)
            if not existing:
                return "NOT_FOUND"
            if existing["status"] == "CANCELLED":
                return "ALREADY_CANCELLED"
            if existing["status"] in TERMINAL_RUN_STATUSES:
                return "ALREADY_TERMINAL"

            # WAITING_APPROVAL has no in-flight execution to wait for -
            # cancellation is immediate. Actively executing states move to
            # CANCELLING and rely on cooperative checks to finalize it.
            new_status = "CANCELLED" if existing["status"] == "WAITING_APPROVAL" else "CANCELLING"
            cursor = conn.execute(
                """
                UPDATE runs SET
                    cancel_requested = 1,
                    cancellation_requested_at = ?,
                    cancellation_requested_by = ?,
                    cancellation_reason = ?,
                    status = ?,
                    cancelled_at = CASE WHEN ? = 'CANCELLED' THEN ? ELSE cancelled_at END
                WHERE run_id = ? AND organization_id = ? AND cancel_requested = 0
                  AND status NOT IN ('COMPLETED', 'FAILED', 'CANCELLED', 'BLOCKED');
                """,
                (
                    now_iso, actor, reason, new_status,
                    new_status, now_iso,
                    run_id, existing["organization_id"],
                ),
            )
            conn.commit()
            return "REQUESTED" if cursor.rowcount > 0 else "ALREADY_TERMINAL"

    def mark_cancelled(self, run_id: str, organization_id: Optional[str]) -> bool:
        """
        Finalizes a run as CANCELLED once execution has actually stopped.
        Returns True only if this call performed the transition (guards
        against a run that raced to COMPLETED/FAILED in the meantime -
        the final status must reflect what truly happened, never a lie).
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            existing = self._get_run_row(conn, run_id, organization_id)
            if not existing:
                return False
            cursor = conn.execute(
                """
                UPDATE runs SET status = 'CANCELLED', cancelled_at = ?
                WHERE run_id = ? AND organization_id = ?
                  AND status NOT IN ('COMPLETED', 'FAILED', 'CANCELLED', 'BLOCKED');
                """,
                (now_iso, run_id, existing["organization_id"]),
            )
            conn.commit()
            return cursor.rowcount > 0

    def is_cancel_requested(self, run_id: str, organization_id: Optional[str] = None) -> bool:
        """Cheap read for cooperative cancellation checks inside graph nodes."""
        with self._lock, self._get_conn() as conn:
            row = self._get_run_row(conn, run_id, organization_id)
            return bool(row and (row["cancel_requested"] or row["status"] == "CANCELLED"))

    def is_cancelled(self, run_id: str, organization_id: Optional[str] = None) -> bool:
        with self._lock, self._get_conn() as conn:
            row = self._get_run_row(conn, run_id, organization_id)
            return bool(row and row["status"] == "CANCELLED")

    def update_activity(
        self,
        run_id: str,
        organization_id: Optional[str],
        phase: Optional[str] = None,
    ) -> None:
        """
        Records a heartbeat at a meaningful lifecycle boundary (node
        started/completed, revision, approval, git op, sandbox run) so the
        watchdog can distinguish a genuinely stuck run from a slow-but-active
        one. Intentionally not called on every tiny internal operation.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            if organization_id:
                conn.execute(
                    "UPDATE runs SET last_activity_at = ?, current_phase = coalesce(?, current_phase) "
                    "WHERE run_id = ? AND organization_id = ?;",
                    (now_iso, phase, run_id, organization_id),
                )
            else:
                conn.execute(
                    "UPDATE runs SET last_activity_at = ?, current_phase = coalesce(?, current_phase) "
                    "WHERE run_id = ?;",
                    (now_iso, phase, run_id),
                )
            conn.commit()

    def mark_stuck(self, run_id: str, organization_id: Optional[str]) -> bool:
        """
        Atomically flags a run as stuck. Returns True only for the caller
        that actually wins the race - if two watchdog instances (or two
        passes) both observe the same stale run, only one UPDATE can match
        `stuck_at IS NULL`, so only one proceeds to emit telemetry / request
        cancellation.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE runs SET stuck_at = ? WHERE run_id = ? AND organization_id = ? AND stuck_at IS NULL;",
                (now_iso, run_id, organization_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_active_runs(self) -> List[RunRecord]:
        """
        Cross-tenant listing of runs in a non-terminal, watchdog-relevant
        state. Internal/system use only (the watchdog) - never expose this
        through the tenant-scoped API, which always filters by
        organization_id.
        """
        placeholders = ",".join("?" for _ in WATCHDOG_TRACKED_STATUSES)
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                f"SELECT * FROM runs WHERE status IN ({placeholders});",
                tuple(WATCHDOG_TRACKED_STATUSES),
            )
            return [self._row_to_run_record(row) for row in cursor.fetchall()]

    @staticmethod
    def _get_run_row(conn: sqlite3.Connection, run_id: str, organization_id: Optional[str]) -> Optional[sqlite3.Row]:
        if organization_id:
            cursor = conn.execute(
                "SELECT * FROM runs WHERE run_id = ? AND organization_id = ?;",
                (run_id, organization_id),
            )
        else:
            cursor = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run_id,))
        return cursor.fetchone()

    # =========================================================================
    # Event Log
    # =========================================================================

    def record_event(self, event: TelemetryEvent) -> None:
        """Stores a lifecycle event with sanitized safe metadata."""
        sanitized_meta = sanitize_telemetry_payload(event.safe_metadata)
        meta_json = json.dumps(sanitized_meta)

        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO events (event_id, run_id, organization_id, timestamp, event_type, duration_ms, safe_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.organization_id,
                    event.timestamp,
                    event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
                    event.duration_ms,
                    meta_json,
                ),
            )
            conn.commit()

    def list_events(self, run_id: str, organization_id: Optional[str] = None) -> List[TelemetryEvent]:
        """Returns deterministic, chronologically ordered events for a run."""
        with self._lock, self._get_conn() as conn:
            if organization_id:
                cursor = conn.execute(
                    """
                    SELECT event_id, run_id, organization_id, timestamp, event_type, duration_ms, safe_metadata
                    FROM events
                    WHERE run_id = ? AND organization_id = ?
                    ORDER BY timestamp ASC, rowid ASC;
                    """,
                    (run_id, organization_id),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT event_id, run_id, organization_id, timestamp, event_type, duration_ms, safe_metadata
                    FROM events
                    WHERE run_id = ?
                    ORDER BY timestamp ASC, rowid ASC;
                    """,
                    (run_id,),
                )
            events = []
            for row in cursor.fetchall():
                meta = json.loads(row["safe_metadata"]) if row["safe_metadata"] else {}
                events.append(
                    TelemetryEvent(
                        event_id=row["event_id"],
                        run_id=row["run_id"],
                        organization_id=row["organization_id"],
                        timestamp=row["timestamp"],
                        event_type=TelemetryEventType(row["event_type"]) if row["event_type"] in TelemetryEventType.__members__ else TelemetryEventType.RUN_CREATED,
                        duration_ms=row["duration_ms"],
                        safe_metadata=meta,
                    )
                )
            return events

    get_run_events = list_events

    # =========================================================================
    # Aggregated Tenant Analytics
    # =========================================================================

    def get_overview(self, organization_id: str) -> AnalyticsOverview:
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT
                    count(*) as total_runs,
                    sum(case when status = 'COMPLETED' then 1 else 0 end) as successful_runs,
                    sum(case when status in ('FAILED', 'BLOCKED') then 1 else 0 end) as failed_runs,
                    sum(case when status = 'WAITING_APPROVAL' then 1 else 0 end) as waiting_approval_runs,
                    avg(duration_ms) as avg_duration_ms,
                    sum(coalesce(estimated_total_cost, 0.0)) as total_cost,
                    sum(total_tokens) as total_tokens
                FROM runs
                WHERE organization_id = ?;
                """,
                (organization_id,),
            )
            row = cursor.fetchone()
            if not row or row["total_runs"] == 0:
                return AnalyticsOverview()

            return AnalyticsOverview(
                total_runs=row["total_runs"] or 0,
                successful_runs=row["successful_runs"] or 0,
                failed_runs=row["failed_runs"] or 0,
                waiting_approval_runs=row["waiting_approval_runs"] or 0,
                avg_duration_ms=round(row["avg_duration_ms"] or 0.0, 2),
                total_estimated_cost_usd=round(row["total_cost"] or 0.0, 4),
                total_tokens=row["total_tokens"] or 0,
            )

    get_analytics_overview = get_overview

    def get_quality_analytics(self, organization_id: str) -> QualityAnalytics:
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT
                    count(*) as total_evaluated,
                    sum(case when qa_status = 'PASS' then 1 else 0 end) as qa_passes,
                    sum(case when failure_category = 'TEST_FAILURE' then 1 else 0 end) as test_failures,
                    sum(case when failure_category = 'SECURITY_FAILURE' then 1 else 0 end) as security_failures,
                    sum(coalesce(revision_count, 0)) as total_revisions,
                    avg(coalesce(revision_count, 0)) as avg_revisions
                FROM runs
                WHERE organization_id = ? AND qa_status IS NOT NULL;
                """,
                (organization_id,),
            )
            row = cursor.fetchone()
            if not row or row["total_evaluated"] == 0:
                return QualityAnalytics(rag_analytics=self.get_rag_analytics(organization_id))

            total = row["total_evaluated"]
            passes = row["qa_passes"] or 0
            rag_an = self.get_rag_analytics(organization_id)
            return QualityAnalytics(
                total_evaluated=total,
                qa_pass_rate=round(passes / total, 2) if total > 0 else 0.0,
                test_failure_count=row["test_failures"] or 0,
                security_failure_count=row["security_failures"] or 0,
                avg_revisions_per_run=round(row["avg_revisions"] or 0.0, 2),
                total_revisions=row["total_revisions"] or 0,
                rag_analytics=rag_an,
            )

    def get_rag_analytics(self, organization_id: str) -> RAGAnalytics:
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT
                    count(*) as total_retrievals,
                    sum(case when rag_status = 'SUFFICIENT' then 1 else 0 end) as sufficient_count,
                    sum(case when rag_status = 'INSUFFICIENT' then 1 else 0 end) as insufficient_count
                FROM runs
                WHERE organization_id = ? AND rag_status IS NOT NULL;
                """,
                (organization_id,),
            )
            row = cursor.fetchone()
            if not row or row["total_retrievals"] == 0:
                return RAGAnalytics()

            total = row["total_retrievals"]
            suff = row["sufficient_count"] or 0
            return RAGAnalytics(
                total_retrievals=total,
                retrieval_success_rate=round(suff / total, 2) if total > 0 else 0.0,
                insufficient_context_count=row["insufficient_count"] or 0,
                sufficient_context_count=suff,
            )

    def get_model_analytics(self, organization_id: str) -> ModelAnalytics:
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT
                    coalesce(provider, 'unknown') as provider,
                    coalesce(model, 'unknown') as model,
                    count(*) as request_count,
                    sum(input_tokens) as in_tokens,
                    sum(output_tokens) as out_tokens,
                    sum(total_tokens) as all_tokens,
                    sum(coalesce(estimated_total_cost, 0.0)) as total_cost
                FROM runs
                WHERE organization_id = ?
                GROUP BY provider, model;
                """,
                (organization_id,),
            )
            by_provider: Dict[str, ProviderUsage] = {}
            by_model: Dict[str, ModelUsage] = {}
            overall_cost = 0.0

            for row in cursor.fetchall():
                p = row["provider"]
                m = row["model"]
                cost = row["total_cost"] or 0.0
                tokens = row["all_tokens"] or 0
                count = row["request_count"] or 0

                overall_cost += cost

                # Aggregate by provider
                if p not in by_provider:
                    by_provider[p] = ProviderUsage(provider=p)
                by_provider[p].request_count += count
                by_provider[p].total_tokens += tokens
                by_provider[p].estimated_cost_usd += cost

                # Aggregate by model
                by_model[m] = ModelUsage(
                    model=m,
                    request_count=count,
                    input_tokens=row["in_tokens"] or 0,
                    output_tokens=row["out_tokens"] or 0,
                    total_tokens=tokens,
                    estimated_cost_usd=round(cost, 6),
                )

            for p_usage in by_provider.values():
                p_usage.estimated_cost_usd = round(p_usage.estimated_cost_usd, 6)

            return ModelAnalytics(
                by_provider=by_provider,
                by_model=by_model,
                total_cost_usd=round(overall_cost, 6),
            )

    def get_failure_analytics(self, organization_id: str) -> FailureAnalytics:
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT failure_category, count(*) as count
                FROM runs
                WHERE organization_id = ? AND failure_category IS NOT NULL
                GROUP BY failure_category;
                """,
                (organization_id,),
            )
            by_cat = {}
            total = 0
            for row in cursor.fetchall():
                cat = row["failure_category"]
                cnt = row["count"]
                by_cat[cat] = cnt
                total += cnt

            return FailureAnalytics(
                by_category=by_cat,
                total_failures=total,
            )

    # =========================================================================
    # Evaluation Benchmarks
    # =========================================================================

    def record_evaluation(self, summary: EvaluationSummary) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO evaluations (evaluation_id, organization_id, timestamp, summary_json)
                VALUES (?, ?, ?, ?);
                """,
                (summary.evaluation_id, summary.organization_id, summary.timestamp, summary.model_dump_json()),
            )
            conn.commit()

    def save_evaluation(self, item: Any) -> None:
        if isinstance(item, EvaluationSummary):
            self.record_evaluation(item)
            return

        with self._lock, self._get_conn() as conn:
            eval_id = getattr(item, "evaluation_id", None) or f"eval_{uuid.uuid4().hex[:12]}"
            org_id = getattr(item, "tenant_id", None) or getattr(item, "organization_id", "default-org")
            ts = getattr(item, "timestamp", None) or datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO evaluations (evaluation_id, organization_id, timestamp, summary_json)
                VALUES (?, ?, ?, ?);
                """,
                (eval_id, org_id, ts, item.model_dump_json()),
            )
            conn.commit()

    def list_evaluations(self, organization_id: str, benchmark_name: Optional[str] = None) -> List[Any]:
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT summary_json FROM evaluations
                WHERE organization_id = ?
                ORDER BY timestamp DESC;
                """,
                (organization_id,),
            )
            out = []
            for row in cursor.fetchall():
                sj = row["summary_json"]
                try:
                    res = EvaluationResult.model_validate_json(sj)
                    if benchmark_name and res.benchmark_name and res.benchmark_name != benchmark_name:
                        continue
                    out.append(res)
                except Exception:
                    try:
                        summ = EvaluationSummary.model_validate_json(sj)
                        out.append(summ)
                    except Exception:
                        pass
            return out


# Platform singleton instance pointing to workspace/telemetry.db
telemetry_store = TelemetryStore()
