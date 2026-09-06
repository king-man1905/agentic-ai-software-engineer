"""
Production Workspace Locking and Concurrency Safety.

Provides a robust two-tier locking mechanism (in-process thread coordination +
cross-process OS file locking) ensuring safe serialization of runs touching
the same mutable workspace while allowing independent resources and distinct tenants
to execute concurrently with strict isolation.
"""

import contextlib
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.core.config import WORKSPACE_LOCK_TIMEOUT_SECONDS
from backend.observability.collector import telemetry_collector

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


class WorkspaceLockError(Exception):
    """Base exception for workspace locking errors."""
    pass


class WorkspaceLockTimeoutError(RuntimeError):
    """Raised when lock acquisition exceeds the configured timeout."""
    pass


def _try_os_lock(fd: int) -> bool:
    """Attempts non-blocking OS lock on open file descriptor."""
    if HAS_MSVCRT:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except (OSError, IOError):
            return False
    elif HAS_FCNTL:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, IOError):
            return False
    return True


def _release_os_lock(fd: int) -> None:
    """Releases OS lock on open file descriptor."""
    if HAS_MSVCRT:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except (OSError, IOError):
            pass
    elif HAS_FCNTL:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except (OSError, IOError):
            pass


class WorkspaceLockManager:
    """
    Production workspace locking coordinator.

    Guarantees:
    1. Same mutable resource -> same lock (serialized access).
    2. Distinct projects / resources -> independent locks (concurrent access).
    3. Distinct tenants -> strictly isolated locks (zero cross-tenant collision).
    4. Bounded timeout with clear error and telemetry.
    5. Re-entrant for the same run_id (eliminates self-deadlocks).
    6. Safe ownership verification (a run cannot release another run's lock).
    7. Process-exit auto-release via kernel OS file locking.
    """

    def __init__(self, lock_dir: Optional[str] = None):
        if lock_dir is None:
            lock_dir = os.getenv("WORKSPACE_LOCK_DIR", "workspace/.locks")
        self.lock_dir = Path(lock_dir)
        self._global_lock = threading.Lock()
        self._thread_locks: Dict[str, threading.RLock] = {}
        self._open_fds: Dict[str, int] = {}
        self._lock_owners: Dict[str, Dict[str, Any]] = {}
        self._held_start_times: Dict[str, float] = {}

    def _ensure_lock_dir(self) -> Path:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        return self.lock_dir

    @staticmethod
    def normalize_resource_id(resource_id: Optional[str]) -> str:
        """Normalizes resource identifier and prevents path traversal."""
        raw = (resource_id or "default").strip()
        # Keep alphanumeric, hyphens, underscores, dots
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", raw)
        return sanitized or "default"

    def get_resource_key(self, organization_id: str, resource_id: Optional[str]) -> str:
        """Computes canonical resource key: organization_id + ':' + normalized_resource_id."""
        clean_org = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", (organization_id or "default-org").strip())
        clean_res = self.normalize_resource_id(resource_id)
        return f"{clean_org}:{clean_res}"

    def _get_lock_file_path(self, resource_key: str) -> Path:
        """Produces a deterministic, safe, traversal-proof lock file path."""
        key_hash = hashlib.sha256(resource_key.encode("utf-8")).hexdigest()[:32]
        return self._ensure_lock_dir() / f"{key_hash}.lock"

    def _get_thread_lock(self, resource_key: str) -> threading.RLock:
        with self._global_lock:
            if resource_key not in self._thread_locks:
                self._thread_locks[resource_key] = threading.RLock()
            return self._thread_locks[resource_key]

    def acquire_lock(
        self,
        organization_id: str,
        resource_id: Optional[str],
        run_id: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquires workspace lock for the specified resource and run.

        Blocks up to timeout seconds (defaults to WORKSPACE_LOCK_TIMEOUT_SECONDS).
        Raises WorkspaceLockTimeoutError if lock cannot be acquired within timeout.
        """
        if timeout is None:
            timeout = WORKSPACE_LOCK_TIMEOUT_SECONDS

        resource_key = self.get_resource_key(organization_id, resource_id)
        norm_resource = self.normalize_resource_id(resource_id)
        thread_lock = self._get_thread_lock(resource_key)

        start_time = time.time()
        deadline = start_time + timeout

        # 1. Re-entrancy check: if this run already holds the lock, increment and return
        with self._global_lock:
            owner_info = self._lock_owners.get(resource_key)
            if owner_info and owner_info.get("run_id") == run_id and owner_info.get("organization_id") == organization_id:
                owner_info["reentrancy_count"] = owner_info.get("reentrancy_count", 1) + 1
                return True

        # 2. Acquire thread lock and OS lock with bounded polling
        poll_interval = 0.05
        while True:
            now = time.time()
            remaining = max(0.0, deadline - now)

            acquired_thread = thread_lock.acquire(blocking=True, timeout=min(poll_interval, remaining))
            if acquired_thread:
                # Thread lock held; now check re-entrancy under global lock
                with self._global_lock:
                    owner_info = self._lock_owners.get(resource_key)
                    if owner_info and owner_info.get("run_id") == run_id and owner_info.get("organization_id") == organization_id:
                        owner_info["reentrancy_count"] = owner_info.get("reentrancy_count", 1) + 1
                        return True

                # Try acquiring OS file lock
                lock_file = self._get_lock_file_path(resource_key)
                fd = None
                try:
                    fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT)
                    if _try_os_lock(fd):
                        # Successfully acquired OS file lock
                        wait_duration_ms = (time.time() - start_time) * 1000.0

                        # Write sanitized metadata to lock file
                        metadata = {
                            "organization_id": organization_id,
                            "resource_id": norm_resource,
                            "run_id": run_id,
                            "owner_pid": os.getpid(),
                            "thread_id": threading.get_ident(),
                            "acquired_at": datetime.now(timezone.utc).isoformat(),
                        }
                        try:
                            os.lseek(fd, 0, os.SEEK_SET)
                            os.write(fd, json.dumps(metadata).encode("utf-8"))
                        except Exception:
                            pass

                        with self._global_lock:
                            self._open_fds[resource_key] = fd
                            self._held_start_times[resource_key] = time.time()
                            self._lock_owners[resource_key] = {
                                "organization_id": organization_id,
                                "resource_id": norm_resource,
                                "run_id": run_id,
                                "owner_pid": os.getpid(),
                                "acquired_at": metadata["acquired_at"],
                                "reentrancy_count": 1,
                            }

                        telemetry_collector.on_workspace_lock_acquired(
                            run_id=run_id,
                            organization_id=organization_id,
                            resource_id=norm_resource,
                            wait_duration_ms=wait_duration_ms,
                        )
                        return True
                    else:
                        # Another process holds the OS lock
                        os.close(fd)
                except Exception:
                    if fd is not None:
                        try:
                            os.close(fd)
                        except Exception:
                            pass

                # Failed OS lock; release thread lock and wait
                thread_lock.release()

            if time.time() >= deadline:
                wait_duration_ms = (time.time() - start_time) * 1000.0
                telemetry_collector.on_workspace_lock_timeout(
                    run_id=run_id,
                    organization_id=organization_id,
                    resource_id=norm_resource,
                    wait_duration_ms=wait_duration_ms,
                )
                raise WorkspaceLockTimeoutError(
                    f"WORKSPACE_LOCK_TIMEOUT: Failed to acquire workspace lock for resource '{norm_resource}' "
                    f"(org: '{organization_id}') after {wait_duration_ms:.1f}ms (timeout: {timeout:.1f}s)."
                )

            time.sleep(poll_interval)

    def release_lock(
        self,
        organization_id: str,
        resource_id: Optional[str],
        run_id: str,
    ) -> None:
        """
        Releases the workspace lock for the specified resource and run.

        Enforces ownership: a run cannot release another run's lock.
        Decrements re-entrancy count; only releases OS lock and thread lock
        when count reaches zero.
        """
        resource_key = self.get_resource_key(organization_id, resource_id)
        norm_resource = self.normalize_resource_id(resource_id)

        with self._global_lock:
            owner_info = self._lock_owners.get(resource_key)
            if not owner_info:
                # Lock not held
                return

            if owner_info.get("organization_id") != organization_id or owner_info.get("run_id") != run_id:
                raise PermissionError(
                    f"Lock release rejected: Run '{run_id}' (org: '{organization_id}') does not own "
                    f"the lock for resource '{norm_resource}' (owned by run '{owner_info.get('run_id')}')."
                )

            reentrancy = owner_info.get("reentrancy_count", 1) - 1
            if reentrancy > 0:
                owner_info["reentrancy_count"] = reentrancy
                return

            # Completely releasing lock
            start_held = self._held_start_times.pop(resource_key, time.time())
            held_duration_ms = (time.time() - start_held) * 1000.0
            self._lock_owners.pop(resource_key, None)
            fd = self._open_fds.pop(resource_key, None)

        if fd is not None:
            try:
                _release_os_lock(fd)
                os.close(fd)
            except Exception:
                pass

        thread_lock = self._get_thread_lock(resource_key)
        try:
            thread_lock.release()
        except RuntimeError:
            pass

        telemetry_collector.on_workspace_lock_released(
            run_id=run_id,
            organization_id=organization_id,
            resource_id=norm_resource,
            held_duration_ms=held_duration_ms,
        )

    @contextlib.contextmanager
    def acquire(
        self,
        organization_id: str,
        resource_id: Optional[str],
        run_id: str,
        timeout: Optional[float] = None,
    ):
        """Context manager guaranteeing lock acquisition and safe release."""
        self.acquire_lock(organization_id, resource_id, run_id, timeout)
        try:
            yield
        finally:
            self.release_lock(organization_id, resource_id, run_id)

    def is_locked(self, organization_id: str, resource_id: Optional[str]) -> bool:
        """Returns True if the resource is currently locked."""
        resource_key = self.get_resource_key(organization_id, resource_id)
        with self._global_lock:
            return resource_key in self._lock_owners

    def get_lock_owner(self, organization_id: str, resource_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Returns sanitized metadata of the current lock owner, if any."""
        resource_key = self.get_resource_key(organization_id, resource_id)
        with self._global_lock:
            owner = self._lock_owners.get(resource_key)
            if not owner:
                return None
            return {
                "organization_id": owner["organization_id"],
                "resource_id": owner["resource_id"],
                "run_id": owner["run_id"],
                "owner_pid": owner["owner_pid"],
                "acquired_at": owner["acquired_at"],
            }


# Platform singleton instance
workspace_lock_manager = WorkspaceLockManager()
