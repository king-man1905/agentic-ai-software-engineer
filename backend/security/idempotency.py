import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


class IdempotencyConflictError(ValueError):
    """
    Raised when an Idempotency-Key is reused with a different request payload
    or when concurrent/subsequent operations violate idempotency invariants.
    """
    pass


def compute_key_hash(key: str) -> str:
    """
    Computes a cryptographic SHA-256 hash of an idempotency key.
    Raw keys and auth credentials must never be stored or exposed in plaintext.
    """
    if not key:
        raise ValueError("Idempotency key cannot be empty.")
    return hashlib.sha256(key.strip().encode("utf-8")).hexdigest()


def compute_fingerprint(params: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 fingerprint of canonicalized request parameters.
    Volatile/non-deterministic fields (run_id, timestamps, auth tokens, random UUIDs)
    are excluded from the fingerprint.
    """
    excluded_fields = {
        "run_id",
        "created_at",
        "timestamp",
        "updated_at",
        "auth_token",
        "authorization",
        "token",
        "api_key",
        "idempotency_key",
    }

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: _normalize(v)
                for k, v in sorted(obj.items())
                if k.lower() not in excluded_fields and v is not None
            }
        elif isinstance(obj, list):
            return [_normalize(item) for item in obj]
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        else:
            return str(obj)

    normalized = _normalize(params)
    canonical_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class IdempotencyStore:
    """
    Durable, thread-safe, SQLite-backed idempotency store supporting
    atomic reservation and replay across process restarts and workers.
    """

    def __init__(self, db_path: str = "workspace/idempotency.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_records (
                    organization_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id, key_hash, operation)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_idempotency_run_id
                ON idempotency_records (organization_id, run_id);
                """
            )
            conn.commit()

    def check_or_reserve(
        self,
        organization_id: str,
        idempotency_key: str,
        operation: str,
        params: Dict[str, Any],
        pending_run_id: str,
    ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Atomically checks if an operation with (organization_id, key_hash, operation)
        already exists.
        
        Returns:
            ("NEW", pending_run_id, None) if this is the first request.
            ("REPLAY", existing_run_id, cached_response) if identical request is repeated.
            
        Raises:
            IdempotencyConflictError if key is reused with different request payload/fingerprint.
        """
        key_hash = compute_key_hash(idempotency_key)
        fingerprint = compute_fingerprint(params)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            try:
                # Attempt atomic insertion
                conn.execute(
                    """
                    INSERT INTO idempotency_records (
                        organization_id, key_hash, operation, request_fingerprint,
                        run_id, status, response_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'PENDING', NULL, ?, ?)
                    """,
                    (
                        organization_id,
                        key_hash,
                        operation,
                        fingerprint,
                        pending_run_id,
                        now_iso,
                        now_iso,
                    ),
                )
                conn.commit()
                return ("NEW", pending_run_id, None)

            except sqlite3.IntegrityError:
                # Record already exists for this (org, key_hash, operation)
                cursor = conn.execute(
                    """
                    SELECT request_fingerprint, run_id, status, response_json
                    FROM idempotency_records
                    WHERE organization_id = ? AND key_hash = ? AND operation = ?
                    """,
                    (organization_id, key_hash, operation),
                )
                row = cursor.fetchone()
                if not row:
                    # Race condition where record was deleted; retry insert
                    conn.execute(
                        """
                        INSERT INTO idempotency_records (
                            organization_id, key_hash, operation, request_fingerprint,
                            run_id, status, response_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 'PENDING', NULL, ?, ?)
                        """,
                        (
                            organization_id,
                            key_hash,
                            operation,
                            fingerprint,
                            pending_run_id,
                            now_iso,
                            now_iso,
                        ),
                    )
                    conn.commit()
                    return ("NEW", pending_run_id, None)

                existing_fingerprint = row["request_fingerprint"]
                existing_run_id = row["run_id"]
                existing_response_json = row["response_json"]

                if existing_fingerprint != fingerprint:
                    raise IdempotencyConflictError(
                        f"Idempotency key reuse detected with mismatched request payload for operation '{operation}'."
                    )

                cached_response = None
                if existing_response_json:
                    try:
                        cached_response = json.loads(existing_response_json)
                    except json.JSONDecodeError:
                        cached_response = None

                return ("REPLAY", existing_run_id, cached_response)

    def complete_reservation(
        self,
        organization_id: str,
        idempotency_key: str,
        operation: str,
        status: str,
        response_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Updates the status and cached response JSON of an existing idempotency record.
        """
        key_hash = compute_key_hash(idempotency_key)
        now_iso = datetime.now(timezone.utc).isoformat()
        response_json = json.dumps(response_data) if response_data is not None else None

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE idempotency_records
                SET status = ?, response_json = ?, updated_at = ?
                WHERE organization_id = ? AND key_hash = ? AND operation = ?
                """,
                (status, response_json, now_iso, organization_id, key_hash, operation),
            )
            conn.commit()

    def get_record(
        self, organization_id: str, idempotency_key: str, operation: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves a record by organization, key, and operation.
        """
        key_hash = compute_key_hash(idempotency_key)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT organization_id, key_hash, operation, request_fingerprint,
                       run_id, status, response_json, created_at, updated_at
                FROM idempotency_records
                WHERE organization_id = ? AND key_hash = ? AND operation = ?
                """,
                (organization_id, key_hash, operation),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

    def delete_record(
        self, organization_id: str, idempotency_key: str, operation: str
    ) -> bool:
        """
        Deletes a specific idempotency record.
        """
        key_hash = compute_key_hash(idempotency_key)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM idempotency_records
                WHERE organization_id = ? AND key_hash = ? AND operation = ?
                """,
                (organization_id, key_hash, operation),
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> None:
        """
        Clears all idempotency records (intended for test teardown and cleanup).
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM idempotency_records;")
            conn.commit()


# Default singleton instance
idempotency_store = IdempotencyStore()
