"""
AgentRunner: Durable graph runner with per-instance MemorySaver checkpointer.

Key design notes for LangGraph 1.2.10:
- graph.invoke() does NOT raise GraphInterrupt; instead it returns with an
  '__interrupt__' key in the result dict when the graph hits interrupt().
- graph.get_state(config).next is a non-empty tuple when the graph is paused.
- Resumption is performed via graph.invoke(Command(resume=value), config=config).
"""

import threading
import uuid
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import time
from datetime import datetime, timezone

from backend.api.models import RunStatusResponse
from backend.graph.state import AgentState
from backend.vcs.models import ApprovalDecision, GitDiffSummary
from backend.schemas.qa import QAResult
from backend.schemas.policy import PolicyEvaluationResult
from backend.observability.collector import telemetry_collector
from backend.observability.store import telemetry_store
from backend.schemas.telemetry import TelemetryEventType
from backend.security.auth import AuthMode
from backend.security.tenant import tenant_manager


def _build_graph(checkpointer: MemorySaver):
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
    Thread-safe agent runner with durable MemorySaver checkpointer.

    Each AgentRunner instance owns its own checkpointer so concurrent
    runners and test instances are fully isolated from one another.
    The checkpointer backend can be swapped to SqliteSaver or
    PostgresSaver by changing the factory in __init__.
    """

    def __init__(self):
        self._checkpointer = MemorySaver()
        self._graph = _build_graph(self._checkpointer)
        # Track metadata supplied at run-creation time (not stored in graph state)
        self._run_metadata: Dict[str, Dict[str, Any]] = {}
        self._run_tenants: Dict[str, str] = {}
        self._active_runs: Dict[str, str] = {}
        self._run_errors: Dict[str, str] = {}
        self._lock = threading.Lock()

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
    ) -> RunStatusResponse:
        return RunStatusResponse(
            run_id=run_id,
            status=status,
            current_node=self._extract_current_node(state_snapshot),
            git_diff=self._extract_git_diff(state_snapshot),
            qa_result=self._extract_qa_result(state_snapshot),
            policy_result=self._extract_policy_result(state_snapshot),
            error_summary=error_summary,
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

        config = self._config(run_id)

        try:
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

        if error_msg and (state_snapshot is None or not state_snapshot.values):
            return self._build_status_response(
                run_id, "FAILED", state_snapshot, error_summary=error_msg
            )

        if is_active and (state_snapshot is None or not state_snapshot.values):
            return RunStatusResponse(
                run_id=run_id,
                status="RUNNING",
                message="Run is currently executing in background",
            )

        if state_snapshot is None or not state_snapshot.values:
            raise KeyError(f"Run not found: {run_id}")

        if is_active:
            return RunStatusResponse(
                run_id=run_id,
                status="RUNNING",
                current_node=self._extract_current_node(state_snapshot),
                message="Run is currently executing in background",
            )

        status = self._derive_status(state_snapshot)
        return self._build_status_response(run_id, status, state_snapshot)

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

        if not state_snapshot.next:
            raise ValueError(
                f"Run '{run_id}' is not awaiting approval "
                f"(current status: COMPLETED or FAILED)."
            )

        resume_value = approval_decision.model_dump()
        effective_org = self._require_org(organization_id, run_id=run_id)

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

        t_resume = time.time()
        try:
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

    def _try_get_state(self, config: Dict[str, Any]):
        """Best-effort state retrieval; returns None on failure."""
        try:
            return self._graph.get_state(config)
        except Exception:
            return None
