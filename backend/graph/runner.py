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

from backend.api.models import RunStatusResponse
from backend.graph.state import AgentState
from backend.vcs.models import ApprovalDecision, GitDiffSummary


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
    builder.add_edge("git_prepare", "approval")
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
            error_summary=error_summary,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_id: str,
        user_message: str,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
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

        Returns:
            RunStatusResponse with the initial status.
        """
        with self._lock:
            if metadata:
                self._run_metadata[run_id] = metadata

        initial_state: AgentState = {"user_message": user_message, "run_id": run_id}

        if project_id:
            initial_state["project_id"] = project_id

        config = self._config(run_id)

        try:
            invoke_result = self._graph.invoke(initial_state, config=config)
            state_snapshot = self._graph.get_state(config)
            status = self._derive_status(state_snapshot, invoke_result)
            return self._build_status_response(run_id, status, state_snapshot)
        except Exception as e:
            state_snapshot = self._try_get_state(config)
            return self._build_status_response(
                run_id, "FAILED", state_snapshot, error_summary=str(e)
            )

    def get_status(self, run_id: str) -> RunStatusResponse:
        """
        Inspects the checkpointed state of a run.

        Args:
            run_id: The run identifier to query.

        Returns:
            RunStatusResponse with current status.

        Raises:
            KeyError: If run_id does not correspond to any known thread.
        """
        config = self._config(run_id)
        state_snapshot = self._graph.get_state(config)

        if state_snapshot is None or not state_snapshot.values:
            raise KeyError(f"Run not found: {run_id}")

        status = self._derive_status(state_snapshot)
        return self._build_status_response(run_id, status, state_snapshot)

    def get_state_values(self, run_id: str) -> Dict[str, Any]:
        """
        Retrieves the raw state dictionary of a run from its checkpoint.

        Args:
            run_id: The run identifier to query.

        Returns:
            Dict[str, Any] containing all AgentState fields stored at the checkpoint.

        Raises:
            KeyError: If run_id does not correspond to any known thread.
        """
        config = self._config(run_id)
        state_snapshot = self._graph.get_state(config)

        if state_snapshot is None or not state_snapshot.values:
            raise KeyError(f"Run not found: {run_id}")

        return state_snapshot.values


    def resume_run(
        self,
        run_id: str,
        approval_decision: ApprovalDecision,
    ) -> RunStatusResponse:
        """
        Resumes a paused run by injecting an ApprovalDecision via Command(resume=...).

        Args:
            run_id: The run to resume.
            approval_decision: The reviewer's structured decision.

        Returns:
            RunStatusResponse reflecting the post-resume state.

        Raises:
            KeyError: If run_id is not found.
            ValueError: If the run is not in WAITING_APPROVAL state.
        """
        config = self._config(run_id)
        state_snapshot = self._graph.get_state(config)

        if state_snapshot is None or not state_snapshot.values:
            raise KeyError(f"Run not found: {run_id}")

        if not state_snapshot.next:
            raise ValueError(
                f"Run '{run_id}' is not awaiting approval "
                f"(current status: COMPLETED or FAILED)."
            )

        resume_value = approval_decision.model_dump()

        try:
            invoke_result = self._graph.invoke(
                Command(resume=resume_value),
                config=config,
            )
            state_snapshot = self._graph.get_state(config)
            status = self._derive_status(state_snapshot, invoke_result)
            return self._build_status_response(run_id, status, state_snapshot)
        except Exception as e:
            state_snapshot = self._try_get_state(config)
            return self._build_status_response(
                run_id, "FAILED", state_snapshot, error_summary=str(e)
            )

    def _try_get_state(self, config: Dict[str, Any]):
        """Best-effort state retrieval; returns None on failure."""
        try:
            return self._graph.get_state(config)
        except Exception:
            return None
