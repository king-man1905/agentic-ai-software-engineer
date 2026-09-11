"""
AgentRunner: Durable graph runner with per-instance MemorySaver checkpointer.

Key design notes for LangGraph 1.2.10:
- graph.invoke() does NOT raise GraphInterrupt; instead it returns with an
  '__interrupt__' key in the result dict when the graph hits interrupt().
- graph.get_state(config).next is a non-empty tuple when the graph is paused.
- Resumption is performed via graph.invoke(Command(resume=value), config=config).
"""

import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

import time
from datetime import datetime, timezone

from backend.api.models import RunStatusResponse
from backend.graph.state import AgentState
from backend.vcs.models import ApprovalDecision, GitDiffSummary
from backend.schemas.qa import QAResult, QAIssue
from backend.schemas.policy import (
    PolicyConfig,
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyTelemetry,
    PolicyViolation,
)
from backend.schemas.routing import RoutingDecision, TaskType
from backend.schemas.planning import ExecutionPlan, PlanStep
from backend.schemas.knowledge import KnowledgeAnswer
from backend.schemas.developer import DeveloperResult, FileChange
from backend.indexer.models import CodeChunk
from backend.developer.models import FilePatch
from backend.sandbox.models import TestExecutionResult
from backend.revision.models import RevisionHistory
from backend.schemas.rag import RetrievalEvaluation, RAGTelemetry
from backend.observability.collector import telemetry_collector
from backend.observability.store import telemetry_store
from backend.observability.telemetry import run_context
from backend.schemas.telemetry import TelemetryEventType, TERMINAL_RUN_STATUSES
from backend.security.auth import AuthMode, RepositoryAccessDeniedError, TenantAccessDeniedError
from backend.security.tenant import tenant_manager
from backend.vcs.git_manager import GitWorkspaceManager
from backend.vcs.workspace_lock import (
    WorkspaceLockManager,
    workspace_lock_manager,
)
from backend.graph.cancellation import RunCancelledException

_SERIALIZATION_CLASSES = [
    GitDiffSummary,
    ApprovalDecision,
    PolicyEvaluationResult,
    PolicyDecision,
    PolicyViolation,
    PolicyConfig,
    PolicyTelemetry,
    QAResult,
    QAIssue,
    RoutingDecision,
    TaskType,
    ExecutionPlan,
    PlanStep,
    KnowledgeAnswer,
    DeveloperResult,
    FileChange,
    CodeChunk,
    FilePatch,
    TestExecutionResult,
    RevisionHistory,
    RetrievalEvaluation,
    RAGTelemetry,
]
_ALLOWLIST = {(cls.__module__, cls.__name__) for cls in _SERIALIZATION_CLASSES}


def _build_graph(checkpointer: BaseCheckpointSaver):
    """
    Build and compile the StateGraph with the given checkpointer.
    Isolated from the module-level graph to allow per-runner checkpointers.
    """
    from langgraph.graph import StateGraph, START, END
    from backend.graph.nodes import (
        router_node,
        planner_node,
        knowledge_node,
        developer_node,
        qa_node,
        qa_router,
        route_after_router,
        route_after_planner,
        route_after_knowledge,
        revision_node,
        git_prepare_node,
        policy_node,
        route_after_policy,
        approval_node,
        route_after_approval,
        git_commit_node,
        cleanup_node,
    )

    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("planner", planner_node)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("developer", developer_node)
    builder.add_node("qa", qa_node)
    builder.add_node("revision", revision_node)
    builder.add_node("git_prepare", git_prepare_node)
    builder.add_node("policy", policy_node)
    builder.add_node("approval", approval_node)
    builder.add_node("git_commit", git_commit_node)
    builder.add_node("cleanup", cleanup_node)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"planner": "planner", "knowledge": "knowledge", "developer": "developer", "end": END},
    )
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"knowledge": "knowledge", "developer": "developer"},
    )
    builder.add_conditional_edges(
        "knowledge",
        route_after_knowledge,
        {"developer": "developer", "end": END},
    )

    builder.add_edge("developer", "qa")
    builder.add_conditional_edges(
        "qa",
        qa_router,
        {"pass": "git_prepare", "fail": "revision", "max_retries": END},
    )
    builder.add_edge("git_prepare", "policy")
    builder.add_conditional_edges(
        "policy",
        route_after_policy,
        {"approval": "approval", "cleanup": "cleanup"},
    )
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"git_commit": "git_commit", "cleanup": "cleanup"},
    )
    builder.add_edge("git_commit", END)
    builder.add_edge("cleanup", END)
    builder.add_edge("revision", "developer")

    return builder.compile(checkpointer=checkpointer)


class AgentRunner:
    """
    Thread-safe agent runner with durable SQLite-backed checkpointer.

    Each AgentRunner instance maintains its own SQLite connection to checkpoints.db
    with WAL mode and busy timeout enabled, ensuring graph state survives server
    restarts and concurrent runners are safely supported.
    """

    def __init__(
        self,
        checkpoint_db_path: Optional[str] = None,
        lock_manager: Optional[WorkspaceLockManager] = None,
    ):
        if checkpoint_db_path is None:
            checkpoint_db_path = os.getenv("CHECKPOINT_DB_PATH", "workspace/checkpoints.db")
        self._checkpoint_db_path = checkpoint_db_path
        self._lock_manager: WorkspaceLockManager = lock_manager or workspace_lock_manager
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._checkpointer: BaseCheckpointSaver = self._init_checkpointer(checkpoint_db_path)
        self._graph = _build_graph(self._checkpointer)
        # Track metadata supplied at run-creation time (not stored in graph state)
        self._run_metadata: Dict[str, Dict[str, Any]] = {}
        self._run_tenants: Dict[str, str] = {}
        self._active_runs: Dict[str, str] = {}
        self._run_errors: Dict[str, str] = {}

    def _init_checkpointer(self, db_path: str) -> BaseCheckpointSaver:
        """
        Initializes the durable SQLite checkpointer with WAL mode and concurrency settings.
        Fails safely in production without silently falling back to volatile MemorySaver.
        """
        last_error = None
        for attempt in range(5):
            try:
                if db_path != ":memory:":
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(
                    db_path,
                    timeout=30.0,
                    check_same_thread=False,
                )
                conn.execute("PRAGMA busy_timeout=30000;")
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                self._conn = conn
                serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWLIST)
                checkpointer = SqliteSaver(conn, serde=serde)
                checkpointer.setup()
                return checkpointer
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as oe:
                last_error = oe
                time.sleep(0.05 * (attempt + 1))
            except Exception as e:
                last_error = e
                break

        e = last_error
        if self._is_production_mode():
            telemetry_collector.record_event(
                run_id="system",
                organization_id="system",
                event_type=TelemetryEventType.RUN_FAILED,
                metadata={"error": f"DURABLE_CHECKPOINT_INIT_FAILED: {str(e)}"},
            )
            raise RuntimeError(f"DURABLE_CHECKPOINT_INIT_FAILED: {str(e)}") from e

        allow_volatile = os.getenv("ALLOW_VOLATILE_CHECKPOINTER", "false").lower() in ("true", "1")
        if allow_volatile:
            return MemorySaver()
        raise RuntimeError(f"CHECKPOINT_INIT_FAILED: {str(e)}") from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _config(self, run_id: str) -> Dict[str, Any]:
        """Returns the LangGraph thread config for a given run_id."""
        return {"configurable": {"thread_id": run_id}}

    def _derive_status(
        self,
        state_snapshot,
        invoke_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Derives the RunStatusResponse.status from graph state.

        LangGraph 1.2.10 behaviour:
          - Interrupted: invoke() returns with '__interrupt__' key, state.next is non-empty.
          - Completed: state.next is empty tuple.
        """
        # Primary signal: state.next populated means graph is paused
        if state_snapshot and state_snapshot.next:
            return "WAITING_APPROVAL"

        # Secondary signal: invoke result carried an interrupt marker
        if invoke_result and "__interrupt__" in invoke_result:
            return "WAITING_APPROVAL"

        # Reaching LangGraph's END with nothing pending is a structural
        # signal, not a business outcome - qa_router (backend/graph/nodes.py)
        # routes straight to END once the revision budget is exhausted while
        # QA is still failing, bypassing git_prepare/policy/approval/
        # git_commit entirely. That path must not be reported the same as a
        # genuine committed success: if the final QA result is not PASS and
        # the run never reached a committed approval, the run failed.
        qa_result = self._extract_qa_result(state_snapshot)
        values = state_snapshot.values if state_snapshot else {}
        approval_status = (values or {}).get("approval_status")
        if qa_result is not None and qa_result.status != "PASS" and approval_status != "COMMITTED":
            return "FAILED"

        return "COMPLETED"

    def _extract_current_node(self, state_snapshot) -> Optional[str]:
        """Returns the node the graph is currently paused at."""
        if state_snapshot and state_snapshot.next:
            return state_snapshot.next[0]
        return None

    def _extract_git_diff(self, state_snapshot) -> Optional[GitDiffSummary]:
        """Extracts GitDiffSummary from the checkpointed state values."""
        if not state_snapshot:
            return None
        values: Dict[str, Any] = state_snapshot.values or {}
        raw = values.get("git_diff")
        if raw is None:
            return None
        if isinstance(raw, GitDiffSummary):
            return raw
        try:
            return GitDiffSummary(**raw)
        except Exception:
            return None

    def _extract_qa_result(self, state_snapshot) -> Optional[QAResult]:
        """Extracts QAResult from the checkpointed state values."""
        if not state_snapshot:
            return None
        values: Dict[str, Any] = state_snapshot.values or {}
        raw = values.get("qa_result")
        if raw is None:
            return None
        if isinstance(raw, QAResult):
            return raw
        try:
            return QAResult(**raw)
        except Exception:
            return None

    def _extract_policy_result(self, state_snapshot) -> Optional[PolicyEvaluationResult]:
        """Extracts PolicyEvaluationResult from the checkpointed state values."""
        if not state_snapshot:
            return None
        values: Dict[str, Any] = state_snapshot.values or {}
        raw = values.get("policy_result")
        if raw is None:
            return None
        if isinstance(raw, PolicyEvaluationResult):
            return raw
        try:
            return PolicyEvaluationResult(**raw)
        except Exception:
            return None

    def _build_status_response(
        self,
        run_id: str,
        status: str,
        state_snapshot=None,
        error_summary: Optional[str] = None,
        pr_number: Optional[int] = None,
        pr_url: Optional[str] = None,
        pr_status: Optional[str] = None,
    ) -> RunStatusResponse:
        return RunStatusResponse(
            run_id=run_id,
            status=status,
            current_node=self._extract_current_node(state_snapshot),
            git_diff=self._extract_git_diff(state_snapshot),
            qa_result=self._extract_qa_result(state_snapshot),
            policy_result=self._extract_policy_result(state_snapshot),
            error_summary=error_summary,
            pr_number=pr_number,
            pr_url=pr_url,
            pr_status=pr_status,
        )


    # ------------------------------------------------------------------
    # Tenant Enforcement Helper
    # ------------------------------------------------------------------

    def _is_production_mode(self) -> bool:
        """True when AUTH_MODE=production (or dev fallback is explicitly
        disabled) - the mode in which a missing tenant identity must fail
        closed rather than default to the sandbox tenant."""
        return tenant_manager.auth_mode == AuthMode.PRODUCTION or not tenant_manager.dev_auth_fallback

    def _require_org(
        self,
        organization_id: Optional[str],
        run_id: Optional[str] = None,
        state_snapshot=None,
    ) -> str:
        """
        Resolves the organization to use for a call, refusing to invent
        "default-org" when running in production. A caller reaching here
        with no organization_id in production means no authenticated
        identity was ever resolved upstream - treating that as
        "default-org" would let a missing/misconfigured auth layer read or
        mutate the sandbox tenant's data. Development fallback (the
        pre-existing behavior) remains available when AUTH_MODE isn't
        production and dev_auth_fallback hasn't been explicitly disabled.
        """
        if organization_id:
            return organization_id
        if run_id:
            tracked = self._run_tenants.get(run_id)
            if not tracked and state_snapshot and state_snapshot.values:
                tracked = state_snapshot.values.get("organization_id")
                if tracked:
                    with self._lock:
                        self._run_tenants[run_id] = tracked
            if not tracked:
                rec = telemetry_store.get_run(run_id)
                if rec and rec.organization_id:
                    tracked = rec.organization_id
                    with self._lock:
                        self._run_tenants[run_id] = tracked
            if tracked:
                return tracked
        if self._is_production_mode():
            raise PermissionError(
                "AUTHENTICATION_REQUIRED: No organization identity was provided; "
                "AUTH_MODE=production forbids an implicit 'default-org' fallback."
            )
        return "default-org"

    def _check_tenant_access(
        self,
        run_id: str,
        organization_id: Optional[str],
        state_snapshot=None,
    ) -> None:
        if not organization_id:
            if self._is_production_mode():
                # No authenticated identity in production: fail the same
                # way as "run not found" rather than silently allowing
                # access - don't reveal whether the run exists either.
                raise KeyError(f"Run not found: {run_id}")
            return
        assigned_org = self._run_tenants.get(run_id)
        if not assigned_org and state_snapshot and state_snapshot.values:
            assigned_org = state_snapshot.values.get("organization_id")
            if assigned_org:
                with self._lock:
                    self._run_tenants[run_id] = assigned_org
        if not assigned_org:
            rec = telemetry_store.get_run(run_id)
            if rec and rec.organization_id:
                assigned_org = rec.organization_id
                with self._lock:
                    self._run_tenants[run_id] = assigned_org
        if assigned_org and assigned_org != organization_id:
            raise KeyError(f"Run not found: {run_id}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_run(
        self,
        run_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        organization_id: Optional[str] = None,
    ) -> None:
        """
        Registers a run in memory before starting execution in the background,
        marking its initial status as RUNNING.
        """
        effective_org = self._require_org(organization_id)
        with self._lock:
            if metadata:
                self._run_metadata[run_id] = metadata
            if organization_id:
                self._run_tenants[run_id] = organization_id
            self._active_runs[run_id] = "RUNNING"

        telemetry_collector.on_run_created(
            run_id=run_id,
            organization_id=effective_org,
            metadata=metadata,
        )

    def _ensure_workspace_provisioned(
        self,
        project_id: Optional[str],
        repository_id: Optional[str],
        organization_id: str,
    ) -> None:
        """
        Clones the registered, authorized repository behind `repository_id`
        into workspace/<project_id> when that directory doesn't already
        exist - the only place in the API-driven run path that turns a
        registered GitHub repository into a local checkout (previously
        only backend/integrations/run_github_bot.py's standalone CLI had
        this, unreachable from POST /api/v1/runs).

        A no-op (returns immediately, nothing cloned) when:
        - project_id or repository_id is missing - runs with no associated
          repository are unaffected.
        - the workspace already exists - idempotent, never re-clones.
        - the repository isn't registered/authorized for this tenant -
          never clones a repo this caller isn't authorized to access;
          existing RAG_INSUFFICIENT_CONTEXT/"don't guess" behavior takes
          over unchanged, exactly as if this method didn't exist.

        Raises RuntimeError (caught by start_run's existing exception
        handler, which records an explicit FAILED run) if the repository
        IS authorized but cloning it fails - never falls through silently
        to a misleading RAG_INSUFFICIENT_CONTEXT/QA-failure trail.

        SECURITY: the resolved GitHub token is embedded only in the local
        `clone_url` variable and the argv GitWorkspaceManager.clone_repository
        passes to the git subprocess - never logged, printed, or included
        in the RuntimeError message (which names only the project/repo,
        never the URL) raised on failure.
        """
        if not project_id or not repository_id:
            return

        project_path = Path("workspace") / project_id
        if not project_path.exists():
            project_path = Path(os.getcwd()) / "workspace" / project_id
        if project_path.exists():
            return

        try:
            repo = tenant_manager.authorize_repository_access(
                organization_id=organization_id,
                repo_full_name=repository_id,
            )
        except (TenantAccessDeniedError, RepositoryAccessDeniedError):
            return

        full_name = repo.full_name or repository_id
        token = repo.github_token or os.environ.get("GITHUB_TOKEN")
        if token:
            clone_url = f"https://x-access-token:{token}@github.com/{full_name}.git"
        else:
            clone_url = f"https://github.com/{full_name}.git"

        cloned = GitWorkspaceManager.clone_repository(clone_url, str(project_path))
        if not cloned:
            raise RuntimeError(
                f"WORKSPACE_PROVISIONING_FAILED: could not clone repository "
                f"for project '{project_id}'."
            )

    def start_run(
        self,
        run_id: str,
        user_message: str,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        repository_id: Optional[str] = None,
    ) -> RunStatusResponse:
        """
        Starts a new agent run by invoking the graph with the given input.
        Returns immediately; if the graph hits interrupt() the status will
        be WAITING_APPROVAL.

        Args:
            run_id: Unique identifier for this run (used as thread_id).
            user_message: The user's task description.
            project_id: Optional project workspace identifier.
            metadata: Caller-supplied metadata stored for observability.
            organization_id: Tenant organization identifier.
            user_id: Initiating user identifier.
            repository_id: Associated repository identifier.

        Returns:
            RunStatusResponse with the initial status.
        """
        t0 = time.time()
        effective_org = self._require_org(organization_id)

        with self._lock:
            if metadata:
                self._run_metadata[run_id] = metadata
            if organization_id:
                self._run_tenants[run_id] = organization_id
            self._active_runs[run_id] = "RUNNING"

        telemetry_collector.on_run_created(
            run_id=run_id,
            organization_id=effective_org,
            user_id=user_id,
            repository=repository_id or project_id,
            user_message=user_message,
            metadata=metadata,
        )
        telemetry_collector.on_run_started(run_id, effective_org)

        initial_state: AgentState = {"user_message": user_message, "run_id": run_id}

        if project_id:
            initial_state["project_id"] = project_id
        if organization_id:
            initial_state["organization_id"] = organization_id
        if user_id:
            initial_state["user_id"] = user_id
        if repository_id:
            initial_state["repository_id"] = repository_id

        resource_id = project_id or repository_id or "default"
        config = self._config(run_id)

        try:
            with self._lock_manager.acquire(effective_org, resource_id, run_id):
                with run_context(run_id, effective_org):
                    self._ensure_workspace_provisioned(
                        project_id=project_id,
                        repository_id=repository_id,
                        organization_id=effective_org,
                    )
                    invoke_result = self._graph.invoke(initial_state, config=config)
            state_snapshot = self._graph.get_state(config)
            status = self._derive_status(state_snapshot, invoke_result)
            duration_ms = (time.time() - t0) * 1000.0

            with self._lock:
                self._active_runs.pop(run_id, None)

            if status == "WAITING_APPROVAL":
                git_diff = self._extract_git_diff(state_snapshot)
                pol_res = self._extract_policy_result(state_snapshot)
                telemetry_collector.on_approval_requested(
                    run_id=run_id,
                    organization_id=effective_org,
                    patch_hash=git_diff.patch_hash if git_diff else None,
                    risk_score=getattr(pol_res, "risk_score", None),
                )
            elif status == "COMPLETED":
                vals = state_snapshot.values if state_snapshot else {}
                telemetry_collector.on_run_completed(
                    run_id=run_id,
                    organization_id=effective_org,
                    state_values=vals,
                    duration_ms=duration_ms,
                )

            return self._build_status_response(run_id, status, state_snapshot)
        except RunCancelledException as ce:
            # Cooperative cancellation unwound out of graph.invoke() - this
            # is a clean stop, not a failure. The workspace lock is already
            # released by the `with self._lock_manager.acquire(...)` block
            # above unwinding through its `finally`.
            duration_ms = (time.time() - t0) * 1000.0
            state_snapshot = self._try_get_state(config)
            with self._lock:
                self._active_runs.pop(run_id, None)

            telemetry_store.mark_cancelled(run_id, effective_org)
            telemetry_collector.on_run_cancelled(
                run_id=run_id,
                organization_id=effective_org,
                current_phase=self._extract_current_node(state_snapshot),
                reason=ce.reason,
            )
            return self._build_status_response(run_id, "CANCELLED", state_snapshot)
        except Exception as e:
            duration_ms = (time.time() - t0) * 1000.0
            state_snapshot = self._try_get_state(config)
            with self._lock:
                self._active_runs.pop(run_id, None)
                self._run_errors[run_id] = str(e)

            telemetry_collector.on_run_failed(
                run_id=run_id,
                organization_id=effective_org,
                error_message=str(e),
                duration_ms=duration_ms,
            )
            return self._build_status_response(
                run_id, "FAILED", state_snapshot, error_summary=str(e)
            )

    def get_status(
        self,
        run_id: str,
        organization_id: Optional[str] = None,
    ) -> RunStatusResponse:
        """
        Inspects the checkpointed state of a run.

        Args:
            run_id: The run identifier to query.
            organization_id: Optional tenant organization identifier for isolation.

        Returns:
            RunStatusResponse with current status.

        Raises:
            KeyError: If run_id does not correspond to any known thread or tenant mismatch.
        """
        self._check_tenant_access(run_id, organization_id)
        config = self._config(run_id)
        state_snapshot = self._graph.get_state(config)
        self._check_tenant_access(run_id, organization_id, state_snapshot)

        with self._lock:
            is_active = run_id in self._active_runs
            error_msg = self._run_errors.get(run_id)

        # CANCELLED/CANCELLING is authoritative from the durable store, not
        # derivable from the checkpoint: LangGraph's own state.next still
        # reflects whatever node would have run next, since cancellation
        # unwinds out of graph.invoke() rather than advancing it. Without
        # this, a later poll after cancellation would resurrect a stale
        # WAITING_APPROVAL/etc. guess instead of the true outcome.
        effective_org_for_cancel = organization_id or self._run_tenants.get(run_id)
        try:
            durable_status = telemetry_store.get_run(run_id, effective_org_for_cancel)
        except Exception:
            durable_status = None

        pr_number = durable_status.pr_number if durable_status else None
        pr_url = durable_status.pr_url if durable_status else None
        pr_status = durable_status.pr_status if durable_status else None

        if durable_status and durable_status.status in ("CANCELLED", "CANCELLING"):
            return self._build_status_response(
                run_id,
                durable_status.status,
                state_snapshot,
                pr_number=pr_number,
                pr_url=pr_url,
                pr_status=pr_status,
            )

        if error_msg and (state_snapshot is None or not state_snapshot.values):
            return self._build_status_response(
                run_id,
                "FAILED",
                state_snapshot,
                error_summary=error_msg,
                pr_number=pr_number,
                pr_url=pr_url,
                pr_status=pr_status,
            )

        if is_active and (state_snapshot is None or not state_snapshot.values):
            return RunStatusResponse(
                run_id=run_id,
                status="RUNNING",
                message="Run is currently executing in background",
                pr_number=pr_number,
                pr_url=pr_url,
                pr_status=pr_status,
            )

        if state_snapshot is None or not state_snapshot.values:
            raise KeyError(f"Run not found: {run_id}")

        if is_active:
            return RunStatusResponse(
                run_id=run_id,
                status="RUNNING",
                current_node=self._extract_current_node(state_snapshot),
                message="Run is currently executing in background",
                pr_number=pr_number,
                pr_url=pr_url,
                pr_status=pr_status,
            )

        status = self._derive_status(state_snapshot)
        return self._build_status_response(
            run_id,
            status,
            state_snapshot,
            pr_number=pr_number,
            pr_url=pr_url,
            pr_status=pr_status,
        )

    def get_state_values(
        self,
        run_id: str,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves the raw state dictionary of a run from its checkpoint.

        Args:
            run_id: The run identifier to query.
            organization_id: Optional tenant organization identifier for isolation.

        Returns:
            Dict[str, Any] containing all AgentState fields stored at the checkpoint.

        Raises:
            KeyError: If run_id does not correspond to any known thread or tenant mismatch.
        """
        self._check_tenant_access(run_id, organization_id)
        config = self._config(run_id)
        state_snapshot = self._graph.get_state(config)
        self._check_tenant_access(run_id, organization_id, state_snapshot)

        if state_snapshot is None or not state_snapshot.values:
            raise KeyError(f"Run not found: {run_id}")

        return state_snapshot.values

    def resume_run(
        self,
        run_id: str,
        approval_decision: ApprovalDecision,
        organization_id: Optional[str] = None,
    ) -> RunStatusResponse:
        """
        Resumes a paused run by injecting an ApprovalDecision via Command(resume=...).

        Args:
            run_id: The run to resume.
            approval_decision: The reviewer's structured decision.
            organization_id: Optional tenant organization identifier for isolation.

        Returns:
            RunStatusResponse reflecting the post-resume state.

        Raises:
            KeyError: If run_id is not found or tenant mismatch.
            ValueError: If the run is not in WAITING_APPROVAL state.
        """
        self._check_tenant_access(run_id, organization_id)
        config = self._config(run_id)
        state_snapshot = self._graph.get_state(config)
        self._check_tenant_access(run_id, organization_id, state_snapshot)

        if state_snapshot is None or not state_snapshot.values:
            raise KeyError(f"Run not found: {run_id}")

        effective_org = self._require_org(organization_id, run_id=run_id, state_snapshot=state_snapshot)

        # A cancelled run must never resume, even though the checkpoint
        # itself still looks "paused" (state.next is untouched by
        # cancelling a WAITING_APPROVAL run - only the durable status
        # changes). The durable store is authoritative here, not the
        # checkpoint's own interrupt marker.
        if telemetry_store.is_cancelled(run_id, effective_org):
            raise ValueError(f"Run '{run_id}' has been cancelled and cannot be resumed.")

        # Guard: reject resume if the run has already reached a terminal state.
        # The telemetry store is the authoritative source of run status — the
        # LangGraph checkpoint's state.next may still appear "paused" even after
        # a run has FAILED (the checkpoint records graph position, not run
        # outcome).  Allowing a resume on a terminal run caused the
        # WORKSPACE_LOCK_TIMEOUT incident: resume_run() re-acquired the lock,
        # the resumed graph invocation crashed, and the lock was never released.
        run_record = telemetry_store.get_run(run_id, effective_org)
        if run_record is not None and str(run_record.status) in TERMINAL_RUN_STATUSES:
            raise ValueError(
                f"Run '{run_id}' is already in a terminal state "
                f"(status='{run_record.status}') and cannot be resumed."
            )

        if not state_snapshot.next:
            raise ValueError(
                f"Run '{run_id}' is not awaiting approval "
                f"(current status: COMPLETED or FAILED)."
            )

        resume_value = approval_decision.model_dump()

        rec = telemetry_store.get_run(run_id, effective_org)
        latency_ms = None
        if rec and rec.created_at:
            try:
                req_time = datetime.fromisoformat(rec.created_at)
                latency_ms = (datetime.now(timezone.utc) - req_time).total_seconds() * 1000.0
            except Exception:
                pass

        telemetry_collector.on_approval_decision(
            run_id=run_id,
            organization_id=effective_org,
            approved=approval_decision.approved,
            reviewer=approval_decision.reviewer,
            latency_ms=latency_ms,
        )

        with self._lock:
            self._active_runs[run_id] = "RUNNING"

        values = state_snapshot.values or {}
        resource_id = values.get("project_id") or values.get("repository_id") or "default"

        t_resume = time.time()
        try:
            with self._lock_manager.acquire(effective_org, resource_id, run_id):
                with run_context(run_id, effective_org):
                    invoke_result = self._graph.invoke(
                        Command(resume=resume_value),
                        config=config,
                    )
            state_snapshot = self._graph.get_state(config)
            status = self._derive_status(state_snapshot, invoke_result)
            duration_ms = (time.time() - t_resume) * 1000.0

            with self._lock:
                self._active_runs.pop(run_id, None)

            if status == "COMPLETED":
                vals = state_snapshot.values if state_snapshot else {}
                telemetry_collector.on_run_completed(
                    run_id=run_id,
                    organization_id=effective_org,
                    state_values=vals,
                    duration_ms=duration_ms,
                )

            return self._build_status_response(run_id, status, state_snapshot)
        except RunCancelledException as ce:
            duration_ms = (time.time() - t_resume) * 1000.0
            state_snapshot = self._try_get_state(config)
            with self._lock:
                self._active_runs.pop(run_id, None)

            telemetry_store.mark_cancelled(run_id, effective_org)
            telemetry_collector.on_run_cancelled(
                run_id=run_id,
                organization_id=effective_org,
                current_phase=self._extract_current_node(state_snapshot),
                reason=ce.reason,
            )
            return self._build_status_response(run_id, "CANCELLED", state_snapshot)
        except Exception as e:
            duration_ms = (time.time() - t_resume) * 1000.0
            state_snapshot = self._try_get_state(config)
            with self._lock:
                self._active_runs.pop(run_id, None)
                self._run_errors[run_id] = str(e)

            telemetry_collector.on_run_failed(
                run_id=run_id,
                organization_id=effective_org,
                error_message=str(e),
                duration_ms=duration_ms,
            )
            return self._build_status_response(
                run_id, "FAILED", state_snapshot, error_summary=str(e)
            )

    def cancel_run(
        self,
        run_id: str,
        organization_id: Optional[str] = None,
        reason: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> RunStatusResponse:
        """
        Requests cancellation of a run.

        For a run paused at WAITING_APPROVAL, cancellation is immediate -
        there is no in-flight execution to wait for, so the durable store
        transitions straight to CANCELLED and `resume_run` is permanently
        blocked for this run_id from then on.

        For an actively executing run (RUNNING/REVISING), this only
        *requests* cancellation (durable `cancel_requested` flag, status
        CANCELLING); the executing background thread notices at its next
        node boundary via `check_cancelled()` and finalizes to CANCELLED
        itself. This method does not wait for that to happen.

        Raises:
            KeyError: If run_id is not found or belongs to another tenant.
            ValueError: If the run has already reached a terminal state
                other than CANCELLED (COMPLETED/FAILED/BLOCKED) - a
                completed run cannot retroactively be cancelled.
        """
        self._check_tenant_access(run_id, organization_id)
        config = self._config(run_id)
        state_snapshot = self._try_get_state(config)
        self._check_tenant_access(run_id, organization_id, state_snapshot)

        effective_org = self._require_org(organization_id, run_id=run_id, state_snapshot=state_snapshot)

        outcome = telemetry_store.request_cancellation(
            run_id=run_id, organization_id=effective_org, reason=reason, actor=actor,
        )

        if outcome == "NOT_FOUND":
            raise KeyError(f"Run not found: {run_id}")
        if outcome == "ALREADY_TERMINAL":
            raise ValueError(
                f"Run '{run_id}' has already reached a terminal state and cannot be cancelled."
            )

        # ALREADY_CANCELLED and REQUESTED both return the current
        # (idempotent) state - repeated cancellation is a no-op, not an error.
        if outcome == "REQUESTED":
            telemetry_collector.on_cancel_requested(
                run_id=run_id, organization_id=effective_org, actor=actor, reason=reason,
            )
            rec = telemetry_store.get_run(run_id, effective_org)
            if rec and rec.status == "CANCELLED":
                # Was WAITING_APPROVAL - already finalized, no execution to wait for.
                telemetry_collector.on_run_cancelled(
                    run_id=run_id,
                    organization_id=effective_org,
                    current_phase=self._extract_current_node(state_snapshot),
                    reason=reason,
                )

        rec = telemetry_store.get_run(run_id, effective_org)
        status_val = rec.status if rec else "CANCELLING"
        return self._build_status_response(run_id, status_val, state_snapshot)

    def _try_get_state(self, config: Dict[str, Any]):
        """Best-effort state retrieval; returns None on failure."""
        try:
            return self._graph.get_state(config)
        except Exception:
            return None

    def get_active_runs(self) -> Dict[str, str]:
        """Returns a snapshot of currently executing runs."""
        with self._lock:
            return dict(self._active_runs)

    def get_active_run_count(self) -> int:
        """Returns the number of currently executing runs."""
        with self._lock:
            return len(self._active_runs)

    def is_ready(self) -> bool:
        """Verifies runner readiness and checkpointer database connectivity."""
        with self._lock:
            if self._conn is not None:
                try:
                    cursor = self._conn.execute("SELECT 1;")
                    return cursor.fetchone() is not None
                except Exception:
                    return False
            if isinstance(self._checkpointer, MemorySaver):
                return True
            return False

    def drain(
        self,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Drains in-flight active runs during application shutdown.
        Waits up to timeout_seconds for active runs to complete normally.
        If timeout expires, triggers cooperative cancellation for remaining
        active runs and allows a brief grace period for cleanup.
        """
        start_time = time.time()
        cancelled_runs = []
        deadline = start_time + max(0.0, timeout_seconds)

        while time.time() < deadline:
            active = self.get_active_runs()
            if not active:
                return {
                    "drained": True,
                    "active_runs_remaining": [],
                    "cancelled_runs": cancelled_runs,
                    "duration_seconds": time.time() - start_time,
                }
            time.sleep(min(poll_interval_seconds, max(0.01, deadline - time.time())))

        # Drain timed out - cooperatively cancel remaining active runs
        remaining = self.get_active_runs()
        for run_id in list(remaining.keys()):
            try:
                org_id = self._run_tenants.get(run_id)
                self.cancel_run(
                    run_id=run_id,
                    organization_id=org_id,
                    reason="Server shutdown drain deadline exceeded",
                    actor="system_shutdown",
                )
                cancelled_runs.append(run_id)
            except Exception:
                pass

        # Bounded grace period for cancellations to unwind
        grace_deadline = time.time() + min(3.0, max(0.5, timeout_seconds * 0.1))
        while time.time() < grace_deadline:
            active = self.get_active_runs()
            if not active:
                break
            time.sleep(0.1)

        final_active = list(self.get_active_runs().keys())
        return {
            "drained": len(final_active) == 0,
            "active_runs_remaining": final_active,
            "cancelled_runs": cancelled_runs,
            "duration_seconds": time.time() - start_time,
        }

    def close(self) -> None:
        """Closes the underlying SQLite connection if open."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
