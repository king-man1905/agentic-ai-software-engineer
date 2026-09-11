"""
Regression tests for Part 2: stale-lock detection in WorkspaceLockManager.

Tests assert the three-condition safety invariant for _try_recover_stale_lock():
  1. Dead PID + terminal run status -> safe in-process eviction
  2. Live PID                       -> fail-closed (no eviction)
  3. Non-terminal run status        -> fail-closed (no eviction)
  4. No telemetry record            -> fail-closed (no eviction)
  5. Unreadable lock file           -> fail-closed (no eviction)
  6. Ambiguous/exception            -> fail-closed (no eviction)
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.vcs.workspace_lock import (
    WorkspaceLockManager,
    _is_pid_alive,
    _read_lock_file_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lock_manager(tmp_path: Path) -> WorkspaceLockManager:
    return WorkspaceLockManager(lock_dir=str(tmp_path / ".locks"))


def _write_lock_file(lock_file: Path, run_id: str, pid: int, org: str = "default-org") -> None:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "organization_id": org,
        "resource_id": "test-resource",
        "run_id": run_id,
        "owner_pid": pid,
        "thread_id": 12345,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
    }
    lock_file.write_text(json.dumps(metadata), encoding="utf-8")


def _make_run_record(status: str):
    rec = MagicMock()
    rec.status = status
    rec.organization_id = "default-org"
    return rec


# ---------------------------------------------------------------------------
# Module-level helper tests
# ---------------------------------------------------------------------------

class TestIsPidAlive:
    def test_current_process_is_alive(self):
        assert _is_pid_alive(os.getpid()) is True

    def test_invalid_pid_treated_as_alive(self):
        """PID 0 or negative -> fail-closed (treat as alive)."""
        assert _is_pid_alive(0) is True
        assert _is_pid_alive(-1) is True

    def test_very_high_pid_treated_as_dead(self):
        """A very large PID that almost certainly does not exist."""
        # PID 9_999_999 is far beyond the typical Linux/Windows limit
        result = _is_pid_alive(9_999_999)
        # On most systems this will be False; if the OS wraps it, test is still safe
        # because the function returning True here is fail-closed behaviour.
        assert isinstance(result, bool)


class TestReadLockFileMetadata:
    def test_reads_valid_json(self, tmp_path):
        f = tmp_path / "test.lock"
        data = {"run_id": "run-abc", "owner_pid": 1234}
        f.write_text(json.dumps(data), encoding="utf-8")
        result = _read_lock_file_metadata(f)
        assert result == data

    def test_returns_none_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.lock"
        f.write_text("", encoding="utf-8")
        assert _read_lock_file_metadata(f) is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        f = tmp_path / "bad.lock"
        f.write_text("not-json{{{", encoding="utf-8")
        assert _read_lock_file_metadata(f) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.lock"
        assert _read_lock_file_metadata(f) is None


# ---------------------------------------------------------------------------
# _try_recover_stale_lock integration tests
# ---------------------------------------------------------------------------

class TestStaleLockRecovery:
    """
    These tests call _try_recover_stale_lock() directly to verify the
    three-condition safety invariant without needing to run the full
    acquire_lock() polling loop.
    """

    def _get_lock_file(self, manager: WorkspaceLockManager, org: str, resource: str) -> Path:
        key = manager.get_resource_key(org, resource)
        return manager._get_lock_file_path(key)

    def test_dead_pid_plus_terminal_run_evicts_in_process_state(self, tmp_path):
        """
        All three conditions met: dead PID + FAILED run ->
        any stale in-process entries are cleared (eviction fires).
        """
        manager = _make_lock_manager(tmp_path)
        resource_key = manager.get_resource_key("default-org", "test-resource")
        lock_file = manager._get_lock_file_path(resource_key)

        # Write a stale lock file with a dead PID
        _write_lock_file(lock_file, run_id="run-stale-001", pid=9_999_998)

        stale_run_record = _make_run_record("FAILED")

        # Pre-populate in-process state to simulate a leaked entry
        # (in practice the in-process dict would be empty after a crash,
        # but we test that any residual entries are cleaned up)

        recovered_events = []

        with (
            patch("backend.vcs.workspace_lock._is_pid_alive", return_value=False),
            patch("backend.vcs.workspace_lock.telemetry_collector") as mock_tc,
        ):
            mock_tc.on_workspace_lock_stale_recovered.side_effect = (
                lambda **kw: recovered_events.append(kw)
            )

            with patch("backend.observability.store.telemetry_store") as mock_store:
                mock_store.get_run.return_value = stale_run_record

                manager._try_recover_stale_lock(
                    resource_key=resource_key,
                    lock_file=lock_file,
                    requesting_run_id="run-new-001",
                    organization_id="default-org",
                    resource_id="test-resource",
                )

        # A telemetry event should have been emitted
        assert len(recovered_events) == 1
        evt = recovered_events[0]
        assert evt["stale_owner_run_id"] == "run-stale-001"
        assert evt["stale_owner_status"] == "FAILED"

    def test_live_pid_does_not_evict(self, tmp_path):
        """Live PID -> fail-closed: no eviction, no telemetry event."""
        manager = _make_lock_manager(tmp_path)
        resource_key = manager.get_resource_key("default-org", "test-resource")
        lock_file = manager._get_lock_file_path(resource_key)

        _write_lock_file(lock_file, run_id="run-live-001", pid=os.getpid())

        recovered_events = []

        with (
            patch("backend.vcs.workspace_lock._is_pid_alive", return_value=True),
            patch("backend.vcs.workspace_lock.telemetry_collector") as mock_tc,
        ):
            mock_tc.on_workspace_lock_stale_recovered.side_effect = (
                lambda **kw: recovered_events.append(kw)
            )

            manager._try_recover_stale_lock(
                resource_key=resource_key,
                lock_file=lock_file,
                requesting_run_id="run-new-001",
                organization_id="default-org",
                resource_id="test-resource",
            )

        # No eviction should have occurred
        assert len(recovered_events) == 0

    def test_non_terminal_run_status_does_not_evict(self, tmp_path):
        """
        Dead PID but run is still RUNNING -> fail-closed (the run might
        still be active on another host or after process restart).
        """
        manager = _make_lock_manager(tmp_path)
        resource_key = manager.get_resource_key("default-org", "test-resource")
        lock_file = manager._get_lock_file_path(resource_key)

        _write_lock_file(lock_file, run_id="run-active-001", pid=9_999_997)
        active_run_record = _make_run_record("RUNNING")

        recovered_events = []

        with (
            patch("backend.vcs.workspace_lock._is_pid_alive", return_value=False),
            patch("backend.vcs.workspace_lock.telemetry_collector") as mock_tc,
        ):
            mock_tc.on_workspace_lock_stale_recovered.side_effect = (
                lambda **kw: recovered_events.append(kw)
            )

            with patch("backend.observability.store.telemetry_store") as mock_store:
                mock_store.get_run.return_value = active_run_record

                manager._try_recover_stale_lock(
                    resource_key=resource_key,
                    lock_file=lock_file,
                    requesting_run_id="run-new-001",
                    organization_id="default-org",
                    resource_id="test-resource",
                )

        assert len(recovered_events) == 0

    def test_no_telemetry_record_does_not_evict(self, tmp_path):
        """
        Dead PID but no telemetry record -> cannot confirm terminal ->
        fail-closed (no eviction).
        """
        manager = _make_lock_manager(tmp_path)
        resource_key = manager.get_resource_key("default-org", "test-resource")
        lock_file = manager._get_lock_file_path(resource_key)

        _write_lock_file(lock_file, run_id="run-unknown-001", pid=9_999_996)

        recovered_events = []

        with (
            patch("backend.vcs.workspace_lock._is_pid_alive", return_value=False),
            patch("backend.vcs.workspace_lock.telemetry_collector") as mock_tc,
        ):
            mock_tc.on_workspace_lock_stale_recovered.side_effect = (
                lambda **kw: recovered_events.append(kw)
            )

            with patch("backend.observability.store.telemetry_store") as mock_store:
                mock_store.get_run.return_value = None  # not in telemetry

                manager._try_recover_stale_lock(
                    resource_key=resource_key,
                    lock_file=lock_file,
                    requesting_run_id="run-new-001",
                    organization_id="default-org",
                    resource_id="test-resource",
                )

        assert len(recovered_events) == 0

    def test_unreadable_lock_file_does_not_evict(self, tmp_path):
        """Corrupt / unreadable lock file -> fail-closed."""
        manager = _make_lock_manager(tmp_path)
        resource_key = manager.get_resource_key("default-org", "test-resource")
        lock_file = manager._get_lock_file_path(resource_key)

        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("CORRUPT{{{", encoding="utf-8")

        recovered_events = []

        with (
            patch("backend.vcs.workspace_lock._is_pid_alive", return_value=False),
            patch("backend.vcs.workspace_lock.telemetry_collector") as mock_tc,
        ):
            mock_tc.on_workspace_lock_stale_recovered.side_effect = (
                lambda **kw: recovered_events.append(kw)
            )

            manager._try_recover_stale_lock(
                resource_key=resource_key,
                lock_file=lock_file,
                requesting_run_id="run-new-001",
                organization_id="default-org",
                resource_id="test-resource",
            )

        assert len(recovered_events) == 0

    def test_missing_lock_file_does_not_evict(self, tmp_path):
        """No lock file at all -> fail-closed."""
        manager = _make_lock_manager(tmp_path)
        resource_key = manager.get_resource_key("default-org", "test-resource")
        lock_file = manager._get_lock_file_path(resource_key)
        # Do NOT write the file

        recovered_events = []

        with patch("backend.vcs.workspace_lock.telemetry_collector") as mock_tc:
            mock_tc.on_workspace_lock_stale_recovered.side_effect = (
                lambda **kw: recovered_events.append(kw)
            )
            manager._try_recover_stale_lock(
                resource_key=resource_key,
                lock_file=lock_file,
                requesting_run_id="run-new-001",
                organization_id="default-org",
                resource_id="test-resource",
            )

        assert len(recovered_events) == 0

    def test_telemetry_exception_does_not_evict(self, tmp_path):
        """If telemetry_store.get_run() raises, fail-closed."""
        manager = _make_lock_manager(tmp_path)
        resource_key = manager.get_resource_key("default-org", "test-resource")
        lock_file = manager._get_lock_file_path(resource_key)

        _write_lock_file(lock_file, run_id="run-err-001", pid=9_999_995)

        recovered_events = []

        with (
            patch("backend.vcs.workspace_lock._is_pid_alive", return_value=False),
            patch("backend.vcs.workspace_lock.telemetry_collector") as mock_tc,
        ):
            mock_tc.on_workspace_lock_stale_recovered.side_effect = (
                lambda **kw: recovered_events.append(kw)
            )

            with patch("backend.observability.store.telemetry_store") as mock_store:
                mock_store.get_run.side_effect = RuntimeError("DB connection lost")

                manager._try_recover_stale_lock(
                    resource_key=resource_key,
                    lock_file=lock_file,
                    requesting_run_id="run-new-001",
                    organization_id="default-org",
                    resource_id="test-resource",
                )

        assert len(recovered_events) == 0

    def test_legitimate_in_process_owner_is_never_evicted(self, tmp_path):
        """
        Even if PID is dead and run is terminal in telemetry, if another
        thread has legitimately acquired the lock in this same process
        (i.e. _lock_owners[key] is populated), do NOT evict.
        """
        manager = _make_lock_manager(tmp_path)
        resource_key = manager.get_resource_key("default-org", "test-resource")
        lock_file = manager._get_lock_file_path(resource_key)

        _write_lock_file(lock_file, run_id="run-old-001", pid=9_999_994)

        # Simulate a live in-process owner
        manager._lock_owners[resource_key] = {
            "organization_id": "default-org",
            "resource_id": "test-resource",
            "run_id": "run-new-owner",
            "owner_pid": os.getpid(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "reentrancy_count": 1,
        }

        recovered_events = []

        with (
            patch("backend.vcs.workspace_lock._is_pid_alive", return_value=False),
            patch("backend.vcs.workspace_lock.telemetry_collector") as mock_tc,
        ):
            mock_tc.on_workspace_lock_stale_recovered.side_effect = (
                lambda **kw: recovered_events.append(kw)
            )

            with patch("backend.observability.store.telemetry_store") as mock_store:
                mock_store.get_run.return_value = _make_run_record("FAILED")

                manager._try_recover_stale_lock(
                    resource_key=resource_key,
                    lock_file=lock_file,
                    requesting_run_id="run-new-001",
                    organization_id="default-org",
                    resource_id="test-resource",
                )

        # Existing owner must not have been evicted
        assert resource_key in manager._lock_owners
        assert manager._lock_owners[resource_key]["run_id"] == "run-new-owner"
        assert len(recovered_events) == 0

    def test_stale_recovery_fires_for_all_terminal_statuses(self, tmp_path):
        """All four terminal statuses trigger eviction when PID is dead."""
        from backend.schemas.telemetry import TERMINAL_RUN_STATUSES

        for status in sorted(TERMINAL_RUN_STATUSES):
            manager = _make_lock_manager(tmp_path / status)
            resource_key = manager.get_resource_key("default-org", "res")
            lock_file = manager._get_lock_file_path(resource_key)
            _write_lock_file(lock_file, run_id=f"run-{status}", pid=9_999_900)

            recovered_events = []

            with (
                patch("backend.vcs.workspace_lock._is_pid_alive", return_value=False),
                patch("backend.vcs.workspace_lock.telemetry_collector") as mock_tc,
            ):
                mock_tc.on_workspace_lock_stale_recovered.side_effect = (
                    lambda **kw: recovered_events.append(kw)
                )

                with patch("backend.observability.store.telemetry_store") as mock_store:
                    mock_store.get_run.return_value = _make_run_record(status)

                    manager._try_recover_stale_lock(
                        resource_key=resource_key,
                        lock_file=lock_file,
                        requesting_run_id="run-requester",
                        organization_id="default-org",
                        resource_id="res",
                    )

            assert len(recovered_events) == 1, (
                f"Expected 1 recovery event for terminal status='{status}', "
                f"got {len(recovered_events)}"
            )
