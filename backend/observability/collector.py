"""
TelemetryCollector: High-level operational manager for telemetry ingestion and lifecycle transitions.
Binds graph lifecycle checkpoints, node events, token accounting, and pricing to the persistent TelemetryStore.
"""

import contextlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.observability.pricing import ModelPricingManager, default_pricing, pricing_manager
from backend.observability.store import TelemetryStore, telemetry_store
from backend.schemas.policy import PolicyDecision
from backend.schemas.telemetry import (
    FailureCategory,
    RunRecord,
    TelemetryEvent,
    TelemetryEventType,
)


@contextlib.contextmanager
def _telemetry_guard(warning: str):
    """
    Ensures a telemetry operation never raises out to its caller - every
    on_* method below has always had this fail-safe contract (a telemetry
    capture error must never crash the engineering run); this just removes
    the repeated try/except/print boilerplate that enforced it individually
    in each method. Prints the exact same "[Telemetry] Warning: ...: {e}"
    message any given call site printed before this refactor.
    """
    try:
        yield
    except Exception as e:
        print(f"[Telemetry] Warning: {warning}: {e}", flush=True)


class TelemetryCollector:
    """
    Central operational collector for recording structured events and updating RunRecords.
    Thread-safe and failsafe: telemetry capture errors never crash the engineering run.
    """

    def __init__(
        self,
        store: Optional[TelemetryStore] = None,
        pricing_manager: Optional[ModelPricingManager] = None,
    ):
        self.store = store or telemetry_store
        self.pricing_manager = pricing_manager or default_pricing
        self._node_start_times: Dict[str, float] = {}

    def _resolve_org(
        self,
        run_id: str,
        organization_id: str = "default-org",
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Resolves the effective organization for a telemetry call. An explicit
        tenant_id/organization_id always wins; otherwise, since most call
        sites only pass the tenant once (at on_run_started/on_run_created)
        and every later call for the same run_id omits it, falls back to
        looking up the run's already-recorded organization rather than
        defaulting to "default-org" and silently mutating the run onto the
        wrong tenant.
        """
        if tenant_id:
            return tenant_id
        if organization_id and organization_id != "default-org":
            return organization_id
        existing = self.store.get_run(run_id, organization_id=None)
        if existing:
            return existing.organization_id
        return organization_id or "default-org"

    def on_run_created(
        self,
        run_id: str,
        organization_id: str = "default-org",
        user_id: Optional[str] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
        user_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> RunRecord:
        """Called when a run is registered/queued."""
        effective_org = self._resolve_org(run_id, organization_id, tenant_id)
        effective_repo = project_id or repository
        record = RunRecord(
            run_id=run_id,
            organization_id=effective_org,
            user_id=user_id,
            repository=effective_repo,
            branch=branch,
            user_message=user_message,
            status="QUEUED",
        )
        with _telemetry_guard("failed to record on_run_created"):
            self.store.create_or_update_run(record)
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.RUN_CREATED,
                metadata={"user_message": (user_message or "")[:100], "repository": effective_repo},
            )
        return record

    def on_run_started(
        self,
        run_id: str,
        organization_id: str = "default-org",
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called when graph execution commences in the background worker."""
        effective_org = self._resolve_org(run_id, organization_id, tenant_id)
        now_iso = datetime.now(timezone.utc).isoformat()
        with _telemetry_guard("failed to record on_run_started"):
            rec = self.store.get_run(run_id, effective_org)
            if not rec:
                rec = self.on_run_created(
                    run_id=run_id,
                    organization_id=effective_org,
                    repository=project_id,
                    metadata=metadata,
                )
            rec.status = "RUNNING"
            rec.started_at = now_iso
            self.store.create_or_update_run(rec)
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.RUN_STARTED,
            )

    def on_node_started(
        self,
        run_id: str,
        node: str,
        organization_id: str = "default-org",
        tenant_id: Optional[str] = None,
    ) -> None:
        """Tracks start time for node-level duration computation."""
        key = f"{run_id}:{node}"
        self._node_start_times[key] = time.time()

    def on_node_completed(
        self,
        run_id: str,
        node: str,
        organization_id: str = "default-org",
        tenant_id: Optional[str] = None,
        qa_result: Optional[Any] = None,
        policy_result: Optional[Any] = None,
        failure_category: Optional[Any] = None,
        revision_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        event_type: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Emits canonical typed lifecycle event (or NODE_COMPLETE) and stores node-specific telemetry on the RunRecord."""
        effective_org = self._resolve_org(run_id, organization_id, tenant_id)
        key = f"{run_id}:{node}"
        t0 = self._node_start_times.pop(key, None)
        duration_ms = (time.time() - t0) * 1000.0 if t0 else kwargs.get("duration_ms", 25.0)

        meta = dict(details or {})
        rec = self.store.get_run(run_id, effective_org)
        if rec:
            if qa_result:
                status_val = getattr(qa_result, "status", None) or (qa_result.get("status") if isinstance(qa_result, dict) else None)
                rec.qa_status = status_val
                meta["qa_status"] = status_val
                meta["qa_passed"] = (status_val or "").upper() == "PASS"
            if policy_result:
                rec.policy_decision = getattr(policy_result, "decision", None) or (policy_result.get("decision") if isinstance(policy_result, dict) else None)
                if hasattr(rec.policy_decision, "value"):
                    rec.policy_decision = rec.policy_decision.value
                meta["policy_decision"] = rec.policy_decision
            if failure_category:
                rec.failure_category = failure_category
                meta["failure_category"] = getattr(failure_category, "value", str(failure_category))
            if revision_count is not None:
                rec.revision_count = revision_count
                meta["revision_count"] = revision_count
            self.store.create_or_update_run(rec)

        # Resolve typed lifecycle event type
        if event_type is not None:
            resolved_event_type = event_type
        else:
            node_key = str(node).strip().lower() if node else ""
            node_map = {
                "policy": TelemetryEventType.POLICY_EVALUATED,
                "qa": TelemetryEventType.QA_COMPLETED,
                "revision": TelemetryEventType.REVISION_COMPLETED,
                "router": TelemetryEventType.ROUTING_COMPLETED,
                "planner": TelemetryEventType.PLANNER_COMPLETE,
                "knowledge": TelemetryEventType.RAG_COMPLETED,
                "rag": TelemetryEventType.RAG_COMPLETED,
                "git_commit": TelemetryEventType.COMMIT_COMPLETED,
                "developer": TelemetryEventType.NODE_COMPLETE,
            }
            if node_key in node_map:
                resolved_event_type = node_map[node_key]
            elif policy_result is not None:
                resolved_event_type = TelemetryEventType.POLICY_EVALUATED
            elif qa_result is not None:
                resolved_event_type = TelemetryEventType.QA_COMPLETED
            elif revision_count is not None:
                resolved_event_type = TelemetryEventType.REVISION_COMPLETED
            else:
                resolved_event_type = TelemetryEventType.NODE_COMPLETE

        # Policy event normalization
        if resolved_event_type == TelemetryEventType.POLICY_EVALUATED:
            raw_decision = None
            if policy_result is not None:
                raw_decision = getattr(policy_result, "decision", None) or (policy_result.get("decision") if isinstance(policy_result, dict) else None)
            if raw_decision is None:
                raw_decision = meta.get("decision") or (rec.policy_decision if rec else None)

            raw_str = raw_decision.value if hasattr(raw_decision, "value") else (str(raw_decision) if raw_decision is not None else None)

            if raw_decision == PolicyDecision.REVIEW or raw_str in ("REVIEW", "HUMAN_REVIEW"):
                meta["decision"] = "HUMAN_REVIEW"
            elif raw_str:
                meta["decision"] = raw_str

            req_human = False
            if (
                raw_decision == PolicyDecision.REVIEW
                or raw_str in ("REVIEW", "HUMAN_REVIEW")
                or getattr(policy_result, "requires_human_approval", False) is True
                or (isinstance(policy_result, dict) and policy_result.get("requires_human_approval") is True)
                or meta.get("requires_human_approval") is True
            ):
                req_human = True
            meta["requires_human_approval"] = req_human

            if "risk_score" not in meta and rec and rec.risk_score is not None:
                meta["risk_score"] = rec.risk_score

        self.emit_event(
            run_id=run_id,
            event_type=resolved_event_type,
            node=node,
            duration_ms=duration_ms,
            details=meta,
            organization_id=effective_org,
            **kwargs,
        )

    def emit_event(
        self,
        run_id: str,
        event_type: Any,
        node: Optional[str] = None,
        duration_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        organization_id: str = "default-org",
        tenant_id: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        **kwargs,
    ) -> None:
        """Universal event emitter that also accumulates model, tokens, and cost onto RunRecord."""
        effective_org = self._resolve_org(run_id, organization_id, tenant_id)
        rec = self.store.get_run(run_id, effective_org)
        if rec:
            if model_name:
                rec.model = model_name
            if provider:
                rec.provider = provider
            if prompt_tokens:
                rec.input_tokens += prompt_tokens
            if completion_tokens:
                rec.output_tokens += completion_tokens
            if prompt_tokens or completion_tokens:
                rec.total_tokens = rec.input_tokens + rec.output_tokens
                if rec.model:
                    in_c, out_c, tot_c, curr = self.pricing_manager.calculate_cost(
                        rec.model, rec.input_tokens, rec.output_tokens
                    )
                    rec.estimated_input_cost = in_c
                    rec.estimated_output_cost = out_c
                    rec.estimated_total_cost = tot_c
            if str(event_type).endswith("PR_PUBLISHED") or event_type == TelemetryEventType.PR_CREATED:
                rec.pr_status = "PUBLISHED"
                if details:
                    rec.pr_url = details.get("pr_url")
                    rec.pr_number = details.get("pr_number")
            self.store.create_or_update_run(rec)

        meta = dict(details or {})
        if node:
            meta["node"] = node
        if model_name:
            meta["model"] = model_name
        if provider:
            meta["provider"] = provider
        if prompt_tokens:
            meta["prompt_tokens"] = prompt_tokens
        if completion_tokens:
            meta["completion_tokens"] = completion_tokens

        ev_enum = event_type if isinstance(event_type, TelemetryEventType) else (
            TelemetryEventType(str(event_type)) if str(event_type) in TelemetryEventType.__members__ else TelemetryEventType.RUN_CREATED
        )
        self.record_event(
            run_id=run_id,
            organization_id=effective_org,
            event_type=ev_enum,
            duration_ms=duration_ms,
            metadata=meta,
        )

    def record_event(
        self,
        run_id: str,
        organization_id: str = "default-org",
        event_type: TelemetryEventType = TelemetryEventType.RUN_CREATED,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records an arbitrary ordered lifecycle event."""
        with _telemetry_guard(f"failed to record event {event_type}"):
            event = TelemetryEvent(
                run_id=run_id,
                organization_id=organization_id,
                event_type=event_type,
                duration_ms=duration_ms,
                safe_metadata=metadata or {},
            )
            self.store.record_event(event)

    def on_approval_requested(
        self,
        run_id: str,
        organization_id: str = "default-org",
        patch_hash: Optional[str] = None,
        risk_score: Optional[float] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Called when HITL interrupt is triggered prior to git_commit."""
        effective_org = self._resolve_org(run_id, organization_id, tenant_id)
        with _telemetry_guard("failed on_approval_requested"):
            rec = self.store.get_run(run_id, effective_org)
            if rec:
                rec.status = "WAITING_APPROVAL"
                rec.approval_required = True
                rec.approval_status = "PENDING"
                rec.patch_hash = patch_hash
                rec.risk_score = risk_score
                self.store.create_or_update_run(rec)

            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.APPROVAL_REQUESTED,
                metadata={"patch_hash": patch_hash, "risk_score": risk_score},
            )

    def on_waiting_approval(
        self,
        run_id: str,
        organization_id: str = "default-org",
        patch_hash: Optional[str] = None,
        risk_score: Optional[float] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        self.on_approval_requested(run_id, organization_id, patch_hash, risk_score, tenant_id=tenant_id)

    def on_approval_decision(
        self,
        run_id: str,
        organization_id: str = "default-org",
        approved: bool = True,
        reviewer: Optional[str] = None,
        latency_ms: Optional[float] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Called when a reviewer resumes a run."""
        effective_org = self._resolve_org(run_id, organization_id, tenant_id)
        with _telemetry_guard("failed on_approval_decision"):
            rec = self.store.get_run(run_id, effective_org)
            if rec:
                rec.approval_status = "APPROVED" if approved else "REJECTED"
                if latency_ms is None and rec.started_at:
                    try:
                        st = datetime.fromisoformat(rec.started_at)
                        latency_ms = (datetime.now(timezone.utc) - st).total_seconds() * 1000.0
                    except Exception:
                        latency_ms = 50.0
                rec.approval_latency_ms = latency_ms
                rec.approval_required = True
                rec.approval_reviewer = reviewer
                rec.approval_decision = "APPROVED" if approved else "REJECTED"
                self.store.create_or_update_run(rec)

            evt = TelemetryEventType.APPROVAL_GRANTED if approved else TelemetryEventType.APPROVAL_DENIED
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=evt,
                duration_ms=latency_ms,
                metadata={"reviewer": reviewer, "approved": approved},
            )

    def on_run_completed(
        self,
        run_id: str,
        organization_id: str = "default-org",
        state_values: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        status: str = "COMPLETED",
        tenant_id: Optional[str] = None,
    ) -> None:
        """Called when graph finishes execution."""
        effective_org = self._resolve_org(run_id, organization_id, tenant_id)
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        state = state_values or {}
        with _telemetry_guard("failed on_run_completed"):
            rec = self.store.get_run(run_id, effective_org)
            if not rec:
                rec = RunRecord(run_id=run_id, organization_id=effective_org)

            rec.completed_at = now_iso
            if duration_ms is None and rec.started_at:
                try:
                    st = datetime.fromisoformat(rec.started_at)
                    duration_ms = (now - st).total_seconds() * 1000.0
                except Exception:
                    duration_ms = 100.0
            rec.duration_ms = duration_ms
            rec.status = status

            # Parse QA results
            qa_res = state.get("qa_result")
            if qa_res:
                rec.qa_status = getattr(qa_res, "status", None) or (qa_res.get("status") if isinstance(qa_res, dict) else None)
                rec.qa_summary = getattr(qa_res, "summary", None) or (qa_res.get("summary") if isinstance(qa_res, dict) else None)

            # Parse Policy results
            pol_res = state.get("policy_result")
            if pol_res:
                rec.policy_decision = getattr(pol_res, "decision", None) or (pol_res.get("decision") if isinstance(pol_res, dict) else None)
                rec.risk_score = getattr(pol_res, "risk_score", None) or (pol_res.get("risk_score") if isinstance(pol_res, dict) else None)

            # Parse RAG
            rec.rag_status = state.get("rag_status")
            rag_eval = state.get("rag_evaluation")
            if rag_eval:
                rec.rag_quality_summary = f"Confidence: {getattr(rag_eval, 'confidence_score', 0.0)}"

            # Revision count
            rec.revision_count = state.get("revision_count", rec.revision_count)

            # Patch & Commit
            git_diff = state.get("git_diff")
            if git_diff:
                rec.patch_hash = getattr(git_diff, "patch_hash", None)
                rec.branch = getattr(git_diff, "branch_name", None)
            rec.commit_status = state.get("approval_status")

            # Metrics & Token accounting
            metrics = state.get("metrics") or {}
            in_tokens = metrics.get("prompt_tokens", 0)
            out_tokens = metrics.get("completion_tokens", 0)
            if in_tokens or out_tokens:
                rec.input_tokens = in_tokens
                rec.output_tokens = out_tokens
                rec.total_tokens = in_tokens + out_tokens

            # Cost calculation via centralized PricingManager
            model_used = state.get("model") or rec.model or "gpt-4o"
            provider_used = state.get("provider") or rec.provider or "openai"
            rec.model = model_used
            rec.provider = provider_used

            if rec.total_tokens > 0:
                in_cost, out_cost, tot_cost, curr = self.pricing_manager.calculate_cost(
                    model_used, rec.input_tokens, rec.output_tokens
                )
                rec.estimated_input_cost = in_cost
                rec.estimated_output_cost = out_cost
                rec.estimated_total_cost = tot_cost
                rec.currency = curr

            self.store.create_or_update_run(rec)
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.RUN_COMPLETED,
                duration_ms=duration_ms,
                metadata={
                    "qa_status": rec.qa_status,
                    "revisions": rec.revision_count,
                    "total_tokens": rec.total_tokens,
                    "total_cost": rec.estimated_total_cost,
                },
            )

    def on_run_failed(
        self,
        run_id: str,
        organization_id: str = "default-org",
        error_message: str = "",
        failure_category: Optional[FailureCategory] = None,
        duration_ms: Optional[float] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Called when a run fails or is blocked."""
        effective_org = self._resolve_org(run_id, organization_id, tenant_id)
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        with _telemetry_guard("failed on_run_failed"):
            rec = self.store.get_run(run_id, effective_org)
            if not rec:
                rec = RunRecord(run_id=run_id, organization_id=effective_org)

            rec.completed_at = now_iso
            if duration_ms is None and rec.started_at:
                try:
                    st = datetime.fromisoformat(rec.started_at)
                    duration_ms = (now - st).total_seconds() * 1000.0
                except Exception:
                    duration_ms = 100.0
            rec.duration_ms = duration_ms
            rec.status = "FAILED"
            rec.failure_category = failure_category or self._infer_failure_category(error_message)
            rec.safe_failure_message = error_message[:500]

            self.store.create_or_update_run(rec)
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.RUN_FAILED,
                duration_ms=duration_ms,
                metadata={
                    "error": rec.safe_failure_message,
                    "failure_category": rec.failure_category.value if rec.failure_category else None,
                },
            )

    def on_pr_published(
        self,
        run_id: str,
        organization_id: str = "default-org",
        pr_number: int = 1,
        pr_url: str = "",
        is_draft: bool = False,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Called when a Pull Request is successfully published to GitHub."""
        effective_org = self._resolve_org(run_id, organization_id, tenant_id)
        with _telemetry_guard("failed on_pr_published"):
            rec = self.store.get_run(run_id, effective_org)
            if not rec:
                # No RunRecord yet for this run_id (e.g. a run whose
                # creation telemetry was never recorded, or a caller that
                # publishes a PR for a run created outside this collector).
                # Create one rather than silently dropping the PR status -
                # a later idempotency check reading this run must see it.
                rec = RunRecord(run_id=run_id, organization_id=effective_org)
            rec.pr_status = "PUBLISHED"
            rec.pr_number = pr_number
            rec.pr_url = pr_url
            rec.github_status = "SUCCESS"
            self.store.create_or_update_run(rec)

            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.PR_CREATED,
                metadata={"pr_number": pr_number, "pr_url": pr_url, "is_draft": is_draft},
            )

    def on_provider_fallback(
        self,
        run_id: str,
        organization_id: str = "default-org",
        primary_provider: str = "",
        fallback_provider: str = "",
        failure_category: str = "LLM_TIMEOUT",
        failure_type: str = "",
        attempt_number: int = 1,
        elapsed_ms: float = 0.0,
        model: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Emits structured PROVIDER_FALLBACK resilience event with sanitized metadata."""
        effective_org = self._resolve_org(run_id, organization_id, tenant_id)
        with _telemetry_guard("failed on_provider_fallback"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.PROVIDER_FALLBACK,
                duration_ms=elapsed_ms,
                metadata={
                    "run_id": run_id,
                    "organization_id": effective_org,
                    "primary_provider": primary_provider,
                    "fallback_provider": fallback_provider,
                    "failure_category": failure_category,
                    "failure_type": failure_type,
                    "attempt_number": attempt_number,
                    "elapsed_ms": elapsed_ms,
                    "model": model,
                },
            )

    def on_workspace_lock_acquired(
        self,
        run_id: str,
        organization_id: str = "default-org",
        resource_id: str = "default",
        wait_duration_ms: float = 0.0,
    ) -> None:
        """Records WORKSPACE_LOCK_ACQUIRED event with sanitized timing metadata."""
        effective_org = self._resolve_org(run_id, organization_id)
        with _telemetry_guard("failed on_workspace_lock_acquired"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.WORKSPACE_LOCK_ACQUIRED,
                duration_ms=round(wait_duration_ms, 2),
                metadata={
                    "run_id": run_id,
                    "organization_id": effective_org,
                    "resource_id": resource_id,
                    "wait_duration_ms": round(wait_duration_ms, 2),
                    "outcome": "ACQUIRED",
                },
            )

    def on_workspace_lock_released(
        self,
        run_id: str,
        organization_id: str = "default-org",
        resource_id: str = "default",
        held_duration_ms: float = 0.0,
    ) -> None:
        """Records WORKSPACE_LOCK_RELEASED event with sanitized timing metadata."""
        effective_org = self._resolve_org(run_id, organization_id)
        with _telemetry_guard("failed on_workspace_lock_released"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.WORKSPACE_LOCK_RELEASED,
                duration_ms=round(held_duration_ms, 2),
                metadata={
                    "run_id": run_id,
                    "organization_id": effective_org,
                    "resource_id": resource_id,
                    "held_duration_ms": round(held_duration_ms, 2),
                    "outcome": "RELEASED",
                },
            )

    def on_workspace_lock_timeout(
        self,
        run_id: str,
        organization_id: str = "default-org",
        resource_id: str = "default",
        wait_duration_ms: float = 0.0,
    ) -> None:
        """Records WORKSPACE_LOCK_TIMEOUT event with sanitized timing metadata."""
        effective_org = self._resolve_org(run_id, organization_id)
        with _telemetry_guard("failed on_workspace_lock_timeout"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.WORKSPACE_LOCK_TIMEOUT,
                duration_ms=round(wait_duration_ms, 2),
                metadata={
                    "run_id": run_id,
                    "organization_id": effective_org,
                    "resource_id": resource_id,
                    "wait_duration_ms": round(wait_duration_ms, 2),
                    "outcome": "TIMEOUT",
                },
            )

    def on_workspace_lock_stale_recovered(
        self,
        run_id: str,
        organization_id: str = "default-org",
        resource_id: str = "default",
        stale_owner_run_id: str = "unknown",
        stale_owner_pid: int = 0,
        stale_owner_status: str = "unknown",
    ) -> None:
        """Records WORKSPACE_LOCK_STALE_RECOVERED when an orphaned lock entry is
        safely evicted from in-process state because the owning PID is confirmed
        dead and the owning run is in a terminal state in telemetry."""
        effective_org = self._resolve_org(run_id, organization_id)
        with _telemetry_guard("failed on_workspace_lock_stale_recovered"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.WORKSPACE_LOCK_STALE_RECOVERED,
                metadata={
                    "run_id": run_id,
                    "organization_id": effective_org,
                    "resource_id": resource_id,
                    "stale_owner_run_id": stale_owner_run_id,
                    "stale_owner_pid": stale_owner_pid,
                    "stale_owner_status": stale_owner_status,
                    "outcome": "STALE_RECOVERED",
                },
            )

    def on_idempotency_replay(
        self,
        run_id: str,
        organization_id: str,
        operation: str,
        key_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records IDEMPOTENCY_REPLAY event with sanitized metadata."""
        effective_org = self._resolve_org(run_id, organization_id)
        meta = {
            "run_id": run_id,
            "organization_id": effective_org,
            "operation": operation,
            "key_hash": key_hash,
            "outcome": "REPLAY",
        }
        if metadata:
            meta.update(metadata)
        with _telemetry_guard("failed on_idempotency_replay"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.IDEMPOTENCY_REPLAY,
                metadata=meta,
            )

    def on_idempotency_conflict(
        self,
        run_id: str,
        organization_id: str,
        operation: str,
        key_hash: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records IDEMPOTENCY_CONFLICT event with sanitized metadata."""
        effective_org = self._resolve_org(run_id, organization_id)
        meta = {
            "run_id": run_id,
            "organization_id": effective_org,
            "operation": operation,
            "key_hash": key_hash,
            "reason": reason,
            "outcome": "CONFLICT",
        }
        if metadata:
            meta.update(metadata)
        with _telemetry_guard("failed on_idempotency_conflict"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.IDEMPOTENCY_CONFLICT,
                metadata=meta,
            )

    def on_pr_reconciled(
        self,
        run_id: str,
        organization_id: str,
        pr_number: int,
        pr_url: str,
        reconciliation_source: str = "github_reconciliation",
    ) -> None:
        """Records PR_RECONCILIATION event when an existing PR is matched."""
        effective_org = self._resolve_org(run_id, organization_id)
        with _telemetry_guard("failed on_pr_reconciled"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.PR_RECONCILIATION,
                metadata={
                    "run_id": run_id,
                    "organization_id": effective_org,
                    "pr_number": pr_number,
                    "pr_url": pr_url,
                    "reconciliation_source": reconciliation_source,
                    "outcome": "RECONCILED",
                },
            )

    def classify_error(self, msg: str) -> FailureCategory:
        """Maps freeform or exception strings to standard FailureCategory."""
        return self._infer_failure_category(msg)

    def _infer_failure_category(self, msg: str) -> FailureCategory:
        """Maps freeform or exception strings to standard FailureCategory."""
        m = (msg or "").lower()
        if "workspace_lock_timeout" in m or "workspace lock timeout" in m:
            return FailureCategory.WORKSPACE_LOCK_TIMEOUT
        if "llmtimeouterror" in m or ("llm" in m and ("timeout" in m or "timed out" in m)):
            return FailureCategory.LLM_TIMEOUT
        if "llmtransienterror" in m or ("llm" in m and "transient" in m):
            return FailureCategory.LLM_TRANSIENT_FAILURE
        if "llmpermanenterror" in m or "llmerror" in m:
            return FailureCategory.LLM_PERMANENT_FAILURE
        if "auth" in m or "unauthorized" in m:
            return FailureCategory.AUTHENTICATION_FAILURE
        if "tenant" in m or "cross-tenant" in m:
            return FailureCategory.TENANT_ACCESS_FAILURE
        if "policy" in m or "blocked by policy" in m:
            return FailureCategory.POLICY_BLOCK
        if "insufficient context" in m:
            return FailureCategory.RAG_INSUFFICIENT_CONTEXT
        if "ast" in m:
            return FailureCategory.AST_FAILURE
        if "test" in m or "pytest" in m:
            return FailureCategory.TEST_FAILURE
        if "security" in m:
            return FailureCategory.SECURITY_FAILURE
        if "revision" in m:
            return FailureCategory.REVISION_EXHAUSTED
        if "sandbox" in m or "timeout" in m or "timed out" in m:
            return FailureCategory.SANDBOX_FAILURE
        if "commit" in m or "patch hash" in m or "drift" in m:
            return FailureCategory.COMMIT_FAILURE
        if "rate limit" in m:
            return FailureCategory.GITHUB_RATE_LIMIT
        if "branch" in m or "conflict" in m:
            return FailureCategory.GITHUB_BRANCH_CONFLICT
        if "pull request" in m:
            return FailureCategory.PR_CREATION_FAILURE
        return FailureCategory.INTERNAL_ERROR

    # ------------------------------------------------------------------
    # Cancellation & Watchdog (Phase 8 Step 5)
    # ------------------------------------------------------------------

    def on_cancel_requested(
        self,
        run_id: str,
        organization_id: str = "default-org",
        actor: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Records RUN_CANCEL_REQUESTED - the request, not necessarily the outcome."""
        effective_org = self._resolve_org(run_id, organization_id)
        with _telemetry_guard("failed on_cancel_requested"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.RUN_CANCEL_REQUESTED,
                metadata={
                    "run_id": run_id,
                    "organization_id": effective_org,
                    "actor": actor,
                    "reason": reason,
                },
            )

    def on_run_cancelled(
        self,
        run_id: str,
        organization_id: str = "default-org",
        current_phase: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Records RUN_CANCELLED once execution has actually stopped."""
        effective_org = self._resolve_org(run_id, organization_id)
        with _telemetry_guard("failed on_run_cancelled"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.RUN_CANCELLED,
                metadata={
                    "run_id": run_id,
                    "organization_id": effective_org,
                    "current_phase": current_phase,
                    "reason": reason,
                },
            )

    def on_run_stuck(
        self,
        run_id: str,
        organization_id: str = "default-org",
        status: Optional[str] = None,
        current_phase: Optional[str] = None,
        last_activity_at: Optional[str] = None,
        threshold_seconds: Optional[float] = None,
    ) -> None:
        """Records RUN_STUCK - an observation, not a mutation of the run."""
        effective_org = self._resolve_org(run_id, organization_id)
        with _telemetry_guard("failed on_run_stuck"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.RUN_STUCK,
                metadata={
                    "run_id": run_id,
                    "organization_id": effective_org,
                    "status": status,
                    "current_phase": current_phase,
                    "last_activity_at": last_activity_at,
                    "threshold_seconds": threshold_seconds,
                },
            )

    def on_sandbox_cancelled(
        self,
        run_id: str,
        organization_id: str = "default-org",
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Records SANDBOX_CANCELLED when a sandbox subprocess was terminated for cancellation."""
        effective_org = self._resolve_org(run_id, organization_id)
        with _telemetry_guard("failed on_sandbox_cancelled"):
            self.record_event(
                run_id=run_id,
                organization_id=effective_org,
                event_type=TelemetryEventType.SANDBOX_CANCELLED,
                duration_ms=round(duration_seconds * 1000.0, 2) if duration_seconds is not None else None,
                metadata={"run_id": run_id, "organization_id": effective_org},
            )

    def on_shutdown_started(
        self,
        active_runs: int = 0,
        drain_timeout_seconds: Optional[float] = None,
        organization_id: str = "system",
    ) -> None:
        """Records SHUTDOWN_STARTED when graceful shutdown sequence commences."""
        with _telemetry_guard("failed on_shutdown_started"):
            self.record_event(
                run_id="system_shutdown",
                organization_id=organization_id,
                event_type=TelemetryEventType.SHUTDOWN_STARTED,
                metadata={
                    "active_runs": active_runs,
                    "drain_timeout_seconds": drain_timeout_seconds,
                },
            )

    def on_shutdown_completed(
        self,
        duration_seconds: Optional[float] = None,
        cancelled_runs: int = 0,
        organization_id: str = "system",
    ) -> None:
        """Records SHUTDOWN_COMPLETED when graceful shutdown finishes successfully."""
        with _telemetry_guard("failed on_shutdown_completed"):
            self.record_event(
                run_id="system_shutdown",
                organization_id=organization_id,
                event_type=TelemetryEventType.SHUTDOWN_COMPLETED,
                duration_ms=round(duration_seconds * 1000.0, 2) if duration_seconds is not None else None,
                metadata={
                    "cancelled_runs": cancelled_runs,
                    "duration_seconds": duration_seconds,
                },
            )

    def on_shutdown_interrupted(
        self,
        reason: str = "",
        active_runs: int = 0,
        organization_id: str = "system",
    ) -> None:
        """Records SHUTDOWN_INTERRUPTED if shutdown encounters an unhandled error or timeout."""
        with _telemetry_guard("failed on_shutdown_interrupted"):
            self.record_event(
                run_id="system_shutdown",
                organization_id=organization_id,
                event_type=TelemetryEventType.SHUTDOWN_INTERRUPTED,
                metadata={
                    "reason": reason,
                    "active_runs": active_runs,
                },
            )


# Platform singleton collector
telemetry_collector = TelemetryCollector()
