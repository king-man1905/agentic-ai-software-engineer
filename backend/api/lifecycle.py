"""
Application Lifecycle Management (Phase 8 Step 6).

Coordinates graceful startup, readiness evaluation, active-run draining,
and clean resource teardown for the FastAPI engineering control plane.
"""

from enum import Enum
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from backend.core.config import SHUTDOWN_DRAIN_TIMEOUT_SECONDS


class AppLifecycleState(str, Enum):
    """
    Lifecycle phases of the FastAPI application.
    """
    STARTING = "STARTING"
    READY = "READY"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"


class LifecycleManager:
    """
    Coordinates application lifecycle state transitions, readiness checks,
    and graceful draining during shutdown.
    """

    def __init__(
        self,
        drain_timeout_seconds: Optional[float] = None,
        initial_state: AppLifecycleState = AppLifecycleState.READY,
    ):
        self.drain_timeout_seconds = (
            drain_timeout_seconds
            if drain_timeout_seconds is not None
            else SHUTDOWN_DRAIN_TIMEOUT_SECONDS
        )
        self._state: AppLifecycleState = initial_state
        self.startup_time: Optional[float] = time.time()
        self.shutdown_time: Optional[float] = None

    @property
    def state(self) -> AppLifecycleState:
        return self._state

    def set_state(self, new_state: AppLifecycleState) -> None:
        self._state = new_state
        if new_state in (AppLifecycleState.DRAINING, AppLifecycleState.STOPPED) and self.shutdown_time is None:
            self.shutdown_time = time.time()

    def is_ready(self) -> bool:
        """Returns True only when the application is in the READY state."""
        return self._state == AppLifecycleState.READY

    def is_accepting_work(self) -> bool:
        """Returns True if new or resumed runs can be accepted."""
        return self._state == AppLifecycleState.READY

    def is_draining_or_stopped(self) -> bool:
        """Returns True if the application is shutting down or has stopped."""
        return self._state in (AppLifecycleState.DRAINING, AppLifecycleState.STOPPED)

    def check_readiness(
        self,
        runner: Optional[Any] = None,
        store: Optional[Any] = None,
        lock_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes granular readiness checks for production load balancers and orchestrators.
        Never exposes secrets, tokens, or tenant-sensitive data.
        """
        checks: Dict[str, Dict[str, Any]] = {}
        all_ok = True

        # 1. Lifecycle state check
        lifecycle_ready = (self._state == AppLifecycleState.READY)
        checks["lifecycle"] = {
            "ready": lifecycle_ready,
            "state": self._state.value,
        }
        if not lifecycle_ready:
            all_ok = False

        # 2. Checkpointer database readiness
        if runner is not None:
            runner_ready = False
            try:
                runner_ready = bool(runner.is_ready())
            except Exception:
                runner_ready = False

            checks["checkpointer"] = {
                "ready": runner_ready,
            }
            if not runner_ready:
                all_ok = False
        else:
            checks["checkpointer"] = {
                "ready": True,
                "note": "runner_not_attached",
            }

        # 3. Telemetry store database readiness
        if store is not None:
            store_ready = False
            try:
                store_ready = bool(store.is_ready())
            except Exception:
                store_ready = False

            checks["telemetry_store"] = {
                "ready": store_ready,
            }
            if not store_ready:
                all_ok = False
        else:
            checks["telemetry_store"] = {
                "ready": True,
                "note": "store_not_attached",
            }

        # 4. Workspace lock directory accessibility
        target_lock_dir = Path(lock_dir) if lock_dir else Path("workspace/locks")
        lock_ready = False
        try:
            target_lock_dir.mkdir(parents=True, exist_ok=True)
            lock_ready = target_lock_dir.is_dir() and os_access_writable(target_lock_dir)
        except Exception:
            lock_ready = False

        checks["workspace_lock"] = {
            "ready": lock_ready,
        }
        if not lock_ready:
            all_ok = False

        # 5. Git CLI executable availability
        git_path = shutil.which("git")
        git_ready = git_path is not None
        checks["git"] = {
            "ready": git_ready,
        }
        if not git_ready:
            all_ok = False

        uptime_seconds = (
            round(time.time() - self.startup_time, 2)
            if self.startup_time is not None
            else None
        )

        return {
            "status": "ready" if all_ok else "not_ready",
            "is_ready": all_ok,
            "lifecycle_state": self._state.value,
            "uptime_seconds": uptime_seconds,
            "checks": checks,
        }


def os_access_writable(path: Path) -> bool:
    """Tests write access to a directory using a temporary file probe."""
    try:
        test_file = path / f".readiness_probe_{time.time_ns()}.tmp"
        test_file.write_text("probe", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False
