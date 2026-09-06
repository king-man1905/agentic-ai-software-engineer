import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Response, status, Depends, BackgroundTasks, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.lifecycle import AppLifecycleState, LifecycleManager
from backend.core.config import SHUTDOWN_DRAIN_TIMEOUT_SECONDS

from backend.api.models import (
    ApiKeyResponse,
    AuditEventView,
    CancelRunRequest,
    CreateApiKeyRequest,
    CreateRunRequest,
    PublishPRRequest,
    PublishPRResponse,
    RunStatusResponse,
    ResumeRunRequest,
    RunListResponse,
    RunEventsResponse,
    RunEvaluationRequest,
    AnalyticsOverview,
    QualityAnalytics,
    ModelAnalytics,
    FailureAnalytics,
    EvaluationSummary,
)
from backend.graph.runner import AgentRunner
from backend.schemas.qa import QAResult
from backend.schemas.policy import PolicyEvaluationResult
from backend.schemas.tenant import Permission, Role, TenantContext
from backend.observability.store import telemetry_store
from backend.observability.collector import telemetry_collector
from backend.observability.evaluation import EvaluationEngine
from backend.security.audit import AuditAction, audit_logger
from backend.security.auth import (
    AuthenticationError,
    AuthenticationExpiredError,
    AuthenticationInvalidError,
    AuthenticationRequiredError,
    RepositoryAccessDeniedError,
    TenantAccessDeniedError,
)
from backend.security.idempotency import (
    IdempotencyConflictError,
    compute_key_hash,
    idempotency_store,
)
from backend.security.rbac import can_approve_changes
from backend.security.tenant import tenant_manager
from backend.vcs.models import ApprovalDecision


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Ensures every request has an X-Request-ID correlation header for observability.
    If incoming request carries one, it is preserved; otherwise a new UUID is generated.
    """
    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        return response


def get_tenant_context(request: Request) -> TenantContext:
    """
    FastAPI dependency extracting and resolving server-side TenantContext.
    Enforces authentication mode (production vs development) and structured error handling.
    """
    org_id = request.headers.get("X-Organization-ID") or request.headers.get("X-Tenant-ID")
    user_id = request.headers.get("X-User-ID")
    auth_header = request.headers.get("Authorization")
    api_key = None
    if auth_header:
        if auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()
        else:
            api_key = auth_header.strip()

    try:
        ctx = tenant_manager.resolve_context(
            org_id=org_id,
            user_id=user_id,
            api_key=api_key,
        )
        return ctx
    except (AuthenticationRequiredError, AuthenticationInvalidError, AuthenticationExpiredError) as e:
        audit_logger.log(
            organization_id=org_id or "unauthenticated",
            user_id=user_id or "anonymous",
            action=AuditAction.AUTHENTICATION_FAILURE,
            resource_type="auth",
            resource_id="credentials",
            details={"error": e.code, "message": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{e.code}: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (TenantAccessDeniedError, RepositoryAccessDeniedError) as e:
        audit_logger.log(
            organization_id=org_id or "unauthorized",
            user_id=user_id or "unknown",
            action=AuditAction.AUTHENTICATION_FAILURE,
            resource_type="auth",
            resource_id="tenant",
            details={"error": e.code, "message": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{e.code}: {str(e)}",
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Production lifespan context manager.
    Coordinates startup validation and graceful draining during shutdown.
    """
    lifecycle: LifecycleManager = getattr(app.state, "lifecycle", None)
    if lifecycle is None:
        lifecycle = LifecycleManager(initial_state=AppLifecycleState.READY)
        app.state.lifecycle = lifecycle

    if hasattr(telemetry_store, "reopen"):
        telemetry_store.reopen()

    runner: Optional[AgentRunner] = getattr(app.state, "runner", None)

    # 1. Startup validation
    lifecycle.set_state(AppLifecycleState.STARTING)
    lifecycle.check_readiness(runner=runner, store=telemetry_store)
    lifecycle.set_state(AppLifecycleState.READY)

    try:
        yield
    finally:
        # 2. Graceful Shutdown & Drain
        lifecycle.set_state(AppLifecycleState.DRAINING)
        active_count = runner.get_active_run_count() if runner else 0
        telemetry_collector.on_shutdown_started(
            active_runs=active_count,
            drain_timeout_seconds=lifecycle.drain_timeout_seconds,
        )

        t0 = time.time()
        cancelled_count = 0
        try:
            if runner:
                drain_res = await asyncio.to_thread(
                    runner.drain,
                    timeout_seconds=lifecycle.drain_timeout_seconds,
                )
                cancelled_count = len(drain_res.get("cancelled_runs", []))
        except Exception as e:
            telemetry_collector.on_shutdown_interrupted(
                reason=f"Drain error: {str(e)}",
                active_runs=runner.get_active_run_count() if runner else 0,
            )

        duration = time.time() - t0
        lifecycle.set_state(AppLifecycleState.STOPPED)

        try:
            telemetry_collector.on_shutdown_completed(
                duration_seconds=duration,
                cancelled_runs=cancelled_count,
            )
        except Exception:
            pass

        try:
            if runner:
                runner.close()
        except Exception:
            pass

        try:
            telemetry_store.close()
        except Exception:
            pass


def create_app(
    runner: Optional[AgentRunner] = None,
    drain_timeout_seconds: Optional[float] = None,
) -> FastAPI:
    """
    Factory creating a configured FastAPI application gateway with lifespan management.
    """
    lifecycle_mgr = LifecycleManager(
        drain_timeout_seconds=drain_timeout_seconds,
        initial_state=AppLifecycleState.READY,
    )
    runner_instance = runner or AgentRunner()

    app = FastAPI(
        title="Agentic AI Software Engineer API Gateway",
        version="1.0.0",
        description="Production API Gateway for autonomous agent workflows with durable HITL persistence.",
        lifespan=lifespan,
    )

    # Attach shared instances
    app.state.lifecycle = lifecycle_mgr
    app.state.runner = runner_instance

    # Middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(CorrelationIdMiddleware)

    def get_agent_runner(request: Request) -> AgentRunner:
        return request.app.state.runner

    def get_lifecycle_manager(request: Request) -> LifecycleManager:
        lifecycle = getattr(request.app.state, "lifecycle", None)
        if lifecycle is None:
            lifecycle = LifecycleManager(initial_state=AppLifecycleState.READY)
            request.app.state.lifecycle = lifecycle
        return lifecycle

    def check_accepting_work(request: Request) -> LifecycleManager:
        lifecycle = get_lifecycle_manager(request)
        if not lifecycle.is_accepting_work():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Server is shutting down (state: {lifecycle.state.value}). New runs are rejected.",
                headers={"Retry-After": "10"},
            )
        return lifecycle

    # -------------------------------------------------------------------------
    # Health and Diagnostics
    # -------------------------------------------------------------------------
    @app.get("/health", tags=["Health"])
    def health_check():
        return {
            "status": "healthy",
            "service": "agentic-ai-software-engineer-api",
            "version": "1.0.0",
        }

    @app.get("/health/ready", tags=["Health"])
    def readiness_check(
        response: Response,
        runner_inst: AgentRunner = Depends(get_agent_runner),
        lifecycle_inst: LifecycleManager = Depends(get_lifecycle_manager),
    ):
        report = lifecycle_inst.check_readiness(
            runner=runner_inst,
            store=telemetry_store,
        )
        if not report["is_ready"]:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return report

    # -------------------------------------------------------------------------
    # Tenancy & Context Inspection
    # -------------------------------------------------------------------------
    @app.get(
        "/api/v1/tenant/context",
        tags=["Tenancy"],
        summary="Get authenticated tenant and role context",
    )
    def get_context(
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ):
        return {
            "organization_id": tenant_ctx.organization_id,
            "organization_name": tenant_ctx.organization.name,
            "user_id": tenant_ctx.user_id,
            "user_name": tenant_ctx.user.name,
            "role": tenant_ctx.role.value,
            "permissions": [p.value for p in tenant_ctx.permissions],
        }

    # -------------------------------------------------------------------------
    # Runs API (v1)
    # -------------------------------------------------------------------------
    @app.post(
        "/api/v1/runs",
        response_model=RunStatusResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Runs"],
        summary="Create and dispatch a new agent workflow run in the background",
    )
    def create_run(
        request: CreateRunRequest,
        background_tasks: BackgroundTasks,
        idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
        lifecycle_mgr: LifecycleManager = Depends(check_accepting_work),
        runner_instance: AgentRunner = Depends(get_agent_runner),
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> RunStatusResponse:

        # RBAC Permission Check
        if Permission.RUN_CREATE not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks RUN_CREATE permission.",
            )

        # Tenant isolation check
        effective_org = request.organization_id or tenant_ctx.organization_id
        if request.organization_id and request.organization_id != tenant_ctx.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Cross-tenant access violation: Caller belongs to organization '{tenant_ctx.organization_id}', "
                    f"cannot create runs in '{request.organization_id}'."
                ),
            )

        if idempotency_key:
            key_hash = compute_key_hash(idempotency_key)
            pending_run_id = f"run_{uuid.uuid4().hex[:12]}"
            payload_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
            try:
                action, reserved_run_id, cached_resp = idempotency_store.check_or_reserve(
                    organization_id=effective_org,
                    idempotency_key=idempotency_key,
                    operation="create_run",
                    params=payload_dict,
                    pending_run_id=pending_run_id,
                )
            except IdempotencyConflictError as e:
                telemetry_collector.on_idempotency_conflict(
                    run_id="unknown",
                    organization_id=effective_org,
                    operation="create_run",
                    key_hash=key_hash,
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"IDEMPOTENCY_CONFLICT: {str(e)}",
                )

            if action == "REPLAY":
                telemetry_collector.on_idempotency_replay(
                    run_id=reserved_run_id,
                    organization_id=effective_org,
                    operation="create_run",
                    key_hash=key_hash,
                    metadata={"replay_source": "idempotency_store"},
                )
                return RunStatusResponse(
                    run_id=reserved_run_id,
                    status="RUNNING",
                    message="Run previously dispatched (idempotent replay)",
                )

            run_id = reserved_run_id
        else:
            run_id = f"run_{uuid.uuid4().hex[:12]}"

        try:
            runner_instance.register_run(
                run_id=run_id,
                metadata=request.metadata,
                organization_id=effective_org,
            )
        except PermissionError as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
        background_tasks.add_task(
            runner_instance.start_run,
            run_id=run_id,
            user_message=request.user_message,
            project_id=request.project_id,
            metadata=request.metadata,
            organization_id=effective_org,
            user_id=tenant_ctx.user_id,
            repository_id=request.repository_id,
        )

        if idempotency_key:
            idempotency_store.complete_reservation(
                organization_id=effective_org,
                idempotency_key=idempotency_key,
                operation="create_run",
                status="DISPATCHED",
                response_data={"run_id": run_id, "status": "RUNNING"},
            )

        # Tamper-evident Audit Logging
        audit_logger.log(
            organization_id=effective_org,
            user_id=tenant_ctx.user_id,
            action=AuditAction.RUN_INITIATED,
            resource_type="run",
            resource_id=run_id,
            details={
                "project_id": request.project_id,
                "user_message": request.user_message[:60],
            },
        )

        return RunStatusResponse(
            run_id=run_id,
            status="RUNNING",
            message="Run dispatched successfully in background",
        )

    @app.get(
        "/api/v1/runs",
        response_model=RunListResponse,
        tags=["Runs"],
        summary="List engineering runs for caller's organization",
    )
    def list_runs(
        status_filter: Optional[str] = Query(None, alias="status"),
        project_id: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> RunListResponse:
        if Permission.RUN_READ not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks RUN_READ permission.",
            )
        runs = telemetry_store.list_runs(
            organization_id=tenant_ctx.organization_id,
            status=status_filter,
            project_id=project_id,
            limit=limit,
            offset=offset,
        )
        return RunListResponse(
            runs=runs,
            total=len(runs),
            limit=limit,
            offset=offset,
        )

    @app.get(
        "/api/v1/runs/{run_id}/events",
        response_model=RunEventsResponse,
        tags=["Runs"],
        summary="List chronological lifecycle events for an engineering run",
    )
    def get_run_events(
        run_id: str,
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> RunEventsResponse:
        if Permission.RUN_READ not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks RUN_READ permission.",
            )
        events = telemetry_store.list_events(run_id=run_id, organization_id=tenant_ctx.organization_id)
        if not events:
            rec = telemetry_store.get_run(run_id, tenant_ctx.organization_id)
            if not rec:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Run '{run_id}' not found.")
        return RunEventsResponse(run_id=run_id, events=events)

    @app.get(
        "/api/v1/runs/{run_id}",
        response_model=RunStatusResponse,
        tags=["Runs"],
        summary="Get current status and state of an agent run",
    )
    def get_run_status(
        run_id: str,
        runner_instance: AgentRunner = Depends(get_agent_runner),
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> RunStatusResponse:
        # RBAC Permission Check
        if Permission.RUN_READ not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks RUN_READ permission.",
            )

        try:
            res = runner_instance.get_status(run_id, organization_id=tenant_ctx.organization_id)
            if hasattr(res, "qa_result") and not isinstance(res.qa_result, (QAResult, dict, type(None))):
                res.qa_result = None
            if hasattr(res, "policy_result") and not isinstance(res.policy_result, (PolicyEvaluationResult, dict, type(None))):
                res.policy_result = None
            return res
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found.",
            )

    @app.post(
        "/api/v1/runs/{run_id}/resume",
        response_model=RunStatusResponse,
        tags=["Runs"],
        summary="Resume a paused agent run with HITL approval decision",
    )
    def resume_run(
        run_id: str,
        request: ResumeRunRequest,
        lifecycle_mgr: LifecycleManager = Depends(check_accepting_work),
        runner_instance: AgentRunner = Depends(get_agent_runner),
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> RunStatusResponse:

        # 1. Tenant Verification
        if request.organization_id and request.organization_id != tenant_ctx.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Cross-tenant access violation: Caller belongs to organization '{tenant_ctx.organization_id}', "
                    f"cannot approve runs in '{request.organization_id}'."
                ),
            )

        # 2. RBAC Approval Permission Check
        if Permission.RUN_APPROVE not in tenant_ctx.permissions:
            audit_logger.log(
                organization_id=tenant_ctx.organization_id,
                user_id=tenant_ctx.user_id,
                action=AuditAction.APPROVAL_DENIED,
                resource_type="run",
                resource_id=run_id,
                details={"reason": f"Role '{tenant_ctx.role.value}' lacks RUN_APPROVE permission"},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"APPROVAL_UNAUTHORIZED: Role '{tenant_ctx.role.value}' lacks RUN_APPROVE permission.",
            )

        # 3. Retrieve run state to evaluate risk level
        try:
            current_status = runner_instance.get_status(run_id, organization_id=tenant_ctx.organization_id)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found.",
            )

        # Resume Idempotency Protection:
        if current_status.status == "COMPLETED":
            state_values = runner_instance.get_state_values(run_id, organization_id=tenant_ctx.organization_id)
            existing_approval = state_values.get("approval")
            if existing_approval and getattr(existing_approval, "approved", None) == request.approved:
                if request.patch_hash is None or getattr(existing_approval, "patch_hash", None) == request.patch_hash:
                    telemetry_collector.on_idempotency_replay(
                        run_id=run_id,
                        organization_id=tenant_ctx.organization_id,
                        operation="resume_run",
                        metadata={
                            "replay_source": "runner_state",
                            "patch_hash": request.patch_hash,
                            "approved": request.approved,
                        },
                    )
                    if hasattr(current_status, "qa_result") and not isinstance(current_status.qa_result, (QAResult, dict, type(None))):
                        current_status.qa_result = None
                    if hasattr(current_status, "policy_result") and not isinstance(current_status.policy_result, (PolicyEvaluationResult, dict, type(None))):
                        current_status.policy_result = None
                    return current_status
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Run '{run_id}' has already completed with a different decision or state.",
            )

        if current_status.status == "RUNNING":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot resume run '{run_id}': run is currently running.",
            )

        # 4. Elevated / Security Risk Evaluation
        risk_score = None
        if current_status.policy_result and hasattr(current_status.policy_result, "risk_score"):
            risk_score = float(current_status.policy_result.risk_score)
        elif current_status.git_diff and current_status.git_diff.risk_score:
            if current_status.git_diff.risk_score.upper() == "HIGH":
                risk_score = 80.0
            elif current_status.git_diff.risk_score.upper() == "MEDIUM":
                risk_score = 50.0
            else:
                risk_score = 20.0

        is_security_sensitive = bool(
            current_status.policy_result
            and getattr(current_status.policy_result, "decision", None) == "REQUIRE_APPROVAL"
        )

        can_approve, denial_reason = can_approve_changes(
            role=tenant_ctx.role,
            risk_score=risk_score,
            is_security_sensitive=is_security_sensitive,
        )

        if not can_approve:
            audit_logger.log(
                organization_id=tenant_ctx.organization_id,
                user_id=tenant_ctx.user_id,
                action=AuditAction.APPROVAL_DENIED,
                resource_type="run",
                resource_id=run_id,
                details={"reason": denial_reason},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"APPROVAL_UNAUTHORIZED: {denial_reason}",
            )

        try:
            decision = ApprovalDecision(
                approved=request.approved,
                reviewer=request.reviewer or tenant_ctx.user.name,
                rejection_reason=request.rejection_reason,
                patch_hash=request.patch_hash,
                reviewer_role=tenant_ctx.role.value,
                user_id=tenant_ctx.user.id,
            )
            res = runner_instance.resume_run(
                run_id,
                decision,
                organization_id=tenant_ctx.organization_id,
            )
            if hasattr(res, "qa_result") and not isinstance(res.qa_result, (QAResult, dict, type(None))):
                res.qa_result = None
            if hasattr(res, "policy_result") and not isinstance(res.policy_result, (PolicyEvaluationResult, dict, type(None))):
                res.policy_result = None

            audit_logger.log(
                organization_id=tenant_ctx.organization_id,
                user_id=tenant_ctx.user_id,
                action=AuditAction.APPROVAL_GRANTED if request.approved else AuditAction.APPROVAL_DENIED,
                resource_type="run",
                resource_id=run_id,
                details={
                    "approved": request.approved,
                    "reviewer": request.reviewer or tenant_ctx.user.name,
                    "patch_hash": request.patch_hash,
                },
            )

            return res
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found.",
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        except PermissionError as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    @app.post(
        "/api/v1/runs/{run_id}/cancel",
        response_model=RunStatusResponse,
        tags=["Runs"],
        summary="Request cancellation of an agent run",
    )
    def cancel_run(
        run_id: str,
        request: CancelRunRequest,
        runner_instance: AgentRunner = Depends(get_agent_runner),
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> RunStatusResponse:
        # RBAC Permission Check
        if Permission.RUN_CANCEL not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks RUN_CANCEL permission.",
            )

        try:
            res = runner_instance.cancel_run(
                run_id,
                organization_id=tenant_ctx.organization_id,
                reason=request.reason,
                actor=tenant_ctx.user_id,
            )
        except KeyError:
            # Also covers cross-tenant access - deliberately indistinguishable
            # from "not found" so a cancel attempt can't be used to probe
            # for another tenant's run_ids.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found.",
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

        audit_logger.log(
            organization_id=tenant_ctx.organization_id,
            user_id=tenant_ctx.user_id,
            action=AuditAction.RUN_CANCELLED if res.status == "CANCELLED" else AuditAction.RUN_CANCEL_REQUESTED,
            resource_type="run",
            resource_id=run_id,
            details={"reason": request.reason, "status": res.status},
        )

        return res

    # -------------------------------------------------------------------------
    # Audit Logs API (v1)
    # -------------------------------------------------------------------------
    @app.get(
        "/api/v1/audit/events",
        response_model=List[AuditEventView],
        tags=["Audit"],
        summary="Get append-only audit events for tenant organization",
    )
    def get_audit_events(
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> List[AuditEventView]:
        if Permission.AUDIT_READ not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks AUDIT_READ permission.",
            )

        events = audit_logger.get_events(organization_id=tenant_ctx.organization_id)
        return [
            AuditEventView(
                event_id=e.event_id,
                organization_id=e.organization_id,
                user_id=e.user_id,
                action=e.action,
                timestamp=e.timestamp,
                resource_type=e.resource_type,
                resource_id=e.resource_id,
                details=e.details,
                event_hash=e.event_hash,
                previous_hash=e.previous_hash,
            )
            for e in events
        ]

    # -------------------------------------------------------------------------
    # API Keys API (v1)
    # -------------------------------------------------------------------------
    @app.post(
        "/api/v1/auth/keys",
        response_model=ApiKeyResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Auth"],
        summary="Create a new API key for the authenticated tenant user",
    )
    def create_api_key(
        request: CreateApiKeyRequest,
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> ApiKeyResponse:
        raw_key, record = tenant_manager.create_user_api_key(
            user_id=tenant_ctx.user_id,
            organization_id=tenant_ctx.organization_id,
            name=request.name,
            expires_in_days=request.expires_in_days,
        )
        return ApiKeyResponse(
            key_id=record.key_id,
            key_prefix=record.key_prefix,
            raw_key=raw_key,
            user_id=record.user_id,
            organization_id=record.organization_id,
            created_at=record.created_at,
            expires_at=record.expires_at,
            is_revoked=record.is_revoked,
            name=record.name,
        )

    @app.post(
        "/api/v1/auth/keys/{key_id}/rotate",
        response_model=ApiKeyResponse,
        tags=["Auth"],
        summary="Rotate an API key",
    )
    def rotate_api_key(
        key_id: str,
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> ApiKeyResponse:
        record = tenant_manager.auth_manager.get_key_record(key_id)
        if not record or record.organization_id != tenant_ctx.organization_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Key '{key_id}' not found.")
        raw_key, new_record = tenant_manager.auth_manager.rotate_api_key(key_id)
        return ApiKeyResponse(
            key_id=new_record.key_id,
            key_prefix=new_record.key_prefix,
            raw_key=raw_key,
            user_id=new_record.user_id,
            organization_id=new_record.organization_id,
            created_at=new_record.created_at,
            expires_at=new_record.expires_at,
            is_revoked=new_record.is_revoked,
            name=new_record.name,
        )

    @app.delete(
        "/api/v1/auth/keys/{key_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["Auth"],
        summary="Revoke an API key",
    )
    def revoke_api_key(
        key_id: str,
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ):
        record = tenant_manager.auth_manager.get_key_record(key_id)
        if not record or record.organization_id != tenant_ctx.organization_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Key '{key_id}' not found.")
        tenant_manager.auth_manager.revoke_api_key(key_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # -------------------------------------------------------------------------
    # GitHub Integration & PR Publishing (v1)
    # -------------------------------------------------------------------------
    @app.post(
        "/api/v1/runs/{run_id}/publish-pr",
        response_model=PublishPRResponse,
        tags=["GitHub"],
        summary="Publish an approved, committed agent run as a GitHub Pull Request",
    )
    def publish_pr(
        run_id: str,
        request: PublishPRRequest,
        runner_instance: AgentRunner = Depends(get_agent_runner),
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> PublishPRResponse:
        import os
        from backend.integrations.github_client import (
            GitHubClient,
            GitHubAuthError,
            GitHubPermissionError,
            GitHubNotFoundError,
            GitHubBranchConflictError,
            GitHubRateLimitError,
            GitHubPRCreationError,
            GitHubApiError,
        )

        # 1. Permission check
        if Permission.REPO_MANAGE not in tenant_ctx.permissions and Permission.RUN_APPROVE not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks REPO_MANAGE or RUN_APPROVE permission.",
            )

        # 2. Verify run status and tenant isolation
        try:
            run_status = runner_instance.get_status(run_id, organization_id=tenant_ctx.organization_id)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found.",
            )

        # 3. Enforce HITL approval & Git commit sequence
        if run_status.status != "COMPLETED":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot publish PR: Run '{run_id}' status is '{run_status.status}', expected 'COMPLETED' after HITL approval.",
            )

        # Explicit re-check immediately before publication, per the same
        # "check again before the irreversible action" rule as commit -
        # redundant with the COMPLETED check above (a cancelled run can
        # never read as COMPLETED), but this is the actual gate that must
        # never be silently removed by an unrelated refactor of step 3.
        if telemetry_store.is_cancelled(run_id, tenant_ctx.organization_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot publish PR: Run '{run_id}' was cancelled.",
            )

        state_values = runner_instance.get_state_values(run_id, organization_id=tenant_ctx.organization_id)
        approval = state_values.get("approval")
        if not approval or not getattr(approval, "approved", False):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot publish PR: Run '{run_id}' has not received human approval.",
            )

        approval_status = state_values.get("approval_status")
        if approval_status != "COMMITTED":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot publish PR: Changes have not been committed (approval_status: {approval_status}).",
            )

        git_diff = run_status.git_diff or state_values.get("git_diff")
        if not git_diff or git_diff.is_no_op:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot publish PR: No code changes were produced.",
            )

        # 4. PR Publication Idempotency (Local Telemetry Store Check)
        rec = telemetry_store.get_run(run_id, tenant_ctx.organization_id)
        if rec and rec.pr_number and rec.pr_url:
            telemetry_collector.on_idempotency_replay(
                run_id=run_id,
                organization_id=tenant_ctx.organization_id,
                operation="publish_pr",
                metadata={"pr_number": rec.pr_number, "pr_url": rec.pr_url},
            )
            return PublishPRResponse(
                pr_number=rec.pr_number,
                pr_url=rec.pr_url,
                head_branch=git_diff.branch_name,
                base_branch=request.base_branch,
                is_draft=request.draft,
                status="already_published",
            )

        # 5. Repository authorization check
        try:
            repo = tenant_manager.authorize_repository_access(
                organization_id=tenant_ctx.organization_id,
                repo_full_name=request.repo_full_name,
                branch=git_diff.branch_name,
            )
        except (TenantAccessDeniedError, RepositoryAccessDeniedError) as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{e.code}: {str(e)}",
            )

        # 6. Create Pull Request using scoped token or client
        token = repo.github_token or os.environ.get("GITHUB_TOKEN")
        client = GitHubClient(token=token)

        # Pre-creation reconciliation check:
        # Check if an existing PR already exists on GitHub for this branch
        try:
            existing_pr = client.find_pull_request(
                repo_full_name=request.repo_full_name,
                head_branch=git_diff.branch_name,
                base_branch=request.base_branch,
            )
        except Exception:
            existing_pr = None

        if existing_pr:
            telemetry_collector.on_pr_reconciled(
                run_id=run_id,
                organization_id=tenant_ctx.organization_id,
                pr_number=existing_pr.pr_number,
                pr_url=existing_pr.pr_url,
                reconciliation_source="pre_creation_check",
            )
            telemetry_collector.on_pr_published(
                run_id=run_id,
                organization_id=tenant_ctx.organization_id,
                pr_number=existing_pr.pr_number,
                pr_url=existing_pr.pr_url,
                is_draft=existing_pr.is_draft,
            )
            return PublishPRResponse(
                pr_number=existing_pr.pr_number,
                pr_url=existing_pr.pr_url,
                head_branch=existing_pr.head_branch,
                base_branch=existing_pr.base_branch,
                is_draft=existing_pr.is_draft,
                status="reconciled",
            )

        title = request.title or f"fix: automated agent changes for {git_diff.branch_name}"
        body = f"## Automated Agent PR\n\n- **Run ID:** `{run_id}`\n- **Approved Patch Hash:** `{git_diff.patch_hash}`\n\n```diff\n{git_diff.unified_diff}\n```"

        try:
            pr_res = client.create_pull_request(
                repo_full_name=request.repo_full_name,
                title=title,
                body=body,
                head_branch=git_diff.branch_name,
                base_branch=request.base_branch,
                draft=request.draft,
            )

            audit_logger.log(
                organization_id=tenant_ctx.organization_id,
                user_id=tenant_ctx.user_id,
                action=AuditAction.PR_CREATED,
                resource_type="github_pr",
                resource_id=f"{request.repo_full_name}#{pr_res.pr_number}",
                details={"pr_url": pr_res.pr_url, "run_id": run_id, "patch_hash": git_diff.patch_hash},
            )

            telemetry_collector.on_pr_published(
                run_id=run_id,
                organization_id=tenant_ctx.organization_id,
                pr_number=pr_res.pr_number,
                pr_url=pr_res.pr_url,
                is_draft=request.draft,
            )

            return PublishPRResponse(
                pr_number=pr_res.pr_number,
                pr_url=pr_res.pr_url,
                head_branch=pr_res.head_branch,
                base_branch=pr_res.base_branch,
                is_draft=pr_res.is_draft,
                status="PUBLISHED",
            )
        except (GitHubPRCreationError, GitHubBranchConflictError, GitHubApiError) as e:
            # Reconcile in case PR was created or already exists on GitHub
            try:
                reconciled_pr = client.find_pull_request(
                    repo_full_name=request.repo_full_name,
                    head_branch=git_diff.branch_name,
                    base_branch=request.base_branch,
                )
                if reconciled_pr:
                    telemetry_collector.on_pr_reconciled(
                        run_id=run_id,
                        organization_id=tenant_ctx.organization_id,
                        pr_number=reconciled_pr.pr_number,
                        pr_url=reconciled_pr.pr_url,
                        reconciliation_source="error_recovery_reconciliation",
                    )
                    telemetry_collector.on_pr_published(
                        run_id=run_id,
                        organization_id=tenant_ctx.organization_id,
                        pr_number=reconciled_pr.pr_number,
                        pr_url=reconciled_pr.pr_url,
                        is_draft=reconciled_pr.is_draft,
                    )
                    return PublishPRResponse(
                        pr_number=reconciled_pr.pr_number,
                        pr_url=reconciled_pr.pr_url,
                        head_branch=reconciled_pr.head_branch,
                        base_branch=reconciled_pr.base_branch,
                        is_draft=reconciled_pr.is_draft,
                        status="reconciled",
                    )
            except Exception:
                pass

            audit_logger.log(tenant_ctx.organization_id, tenant_ctx.user_id, AuditAction.GITHUB_OPERATION_FAILED, "github", request.repo_full_name, {"error": getattr(e, "code", "UNKNOWN")})
            if isinstance(e, GitHubPRCreationError):
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"PR_CREATION_FAILURE: {str(e)}")
            elif isinstance(e, GitHubBranchConflictError):
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"GITHUB_BRANCH_CONFLICT: {str(e)}")
            else:
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"GITHUB_API_FAILURE: {str(e)}")
        except GitHubAuthError as e:
            audit_logger.log(tenant_ctx.organization_id, tenant_ctx.user_id, AuditAction.GITHUB_OPERATION_FAILED, "github", request.repo_full_name, {"error": e.code})
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"GITHUB_AUTH_FAILURE: {str(e)}")
        except GitHubPermissionError as e:
            audit_logger.log(tenant_ctx.organization_id, tenant_ctx.user_id, AuditAction.GITHUB_OPERATION_FAILED, "github", request.repo_full_name, {"error": e.code})
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"GITHUB_PERMISSION_DENIED: {str(e)}")
        except GitHubNotFoundError as e:
            audit_logger.log(tenant_ctx.organization_id, tenant_ctx.user_id, AuditAction.GITHUB_OPERATION_FAILED, "github", request.repo_full_name, {"error": e.code})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"GITHUB_REPO_NOT_FOUND: {str(e)}")
        except GitHubRateLimitError as e:
            audit_logger.log(tenant_ctx.organization_id, tenant_ctx.user_id, AuditAction.GITHUB_OPERATION_FAILED, "github", request.repo_full_name, {"error": e.code})
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"GITHUB_RATE_LIMIT: {str(e)}")

    # -------------------------------------------------------------------------
    # Analytics API (v1)
    # -------------------------------------------------------------------------
    @app.get(
        "/api/v1/analytics/overview",
        response_model=AnalyticsOverview,
        tags=["Analytics"],
        summary="Retrieve high-level run and cost analytics for tenant",
    )
    def get_analytics_overview(
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> AnalyticsOverview:
        if Permission.RUN_READ not in tenant_ctx.permissions and Permission.AUDIT_READ not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks RUN_READ or AUDIT_READ permission.",
            )
        return telemetry_store.get_overview(organization_id=tenant_ctx.organization_id)

    @app.get(
        "/api/v1/analytics/quality",
        response_model=QualityAnalytics,
        tags=["Analytics"],
        summary="Retrieve QA, security, and revision quality analytics for tenant",
    )
    def get_quality_analytics(
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> QualityAnalytics:
        if Permission.RUN_READ not in tenant_ctx.permissions and Permission.AUDIT_READ not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks RUN_READ or AUDIT_READ permission.",
            )
        return telemetry_store.get_quality_analytics(organization_id=tenant_ctx.organization_id)

    @app.get(
        "/api/v1/analytics/models",
        response_model=ModelAnalytics,
        tags=["Analytics"],
        summary="Retrieve provider and model token usage and cost analytics",
    )
    def get_model_analytics(
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> ModelAnalytics:
        if Permission.RUN_READ not in tenant_ctx.permissions and Permission.AUDIT_READ not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks RUN_READ or AUDIT_READ permission.",
            )
        return telemetry_store.get_model_analytics(organization_id=tenant_ctx.organization_id)

    @app.get(
        "/api/v1/analytics/failures",
        response_model=FailureAnalytics,
        tags=["Analytics"],
        summary="Retrieve failure category distribution for tenant",
    )
    def get_failure_analytics(
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> FailureAnalytics:
        if Permission.RUN_READ not in tenant_ctx.permissions and Permission.AUDIT_READ not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks RUN_READ or AUDIT_READ permission.",
            )
        return telemetry_store.get_failure_analytics(organization_id=tenant_ctx.organization_id)

    # -------------------------------------------------------------------------
    # Evaluation Benchmark API (v1)
    # -------------------------------------------------------------------------
    @app.post(
        "/api/v1/evaluation/run",
        response_model=EvaluationSummary,
        tags=["Evaluation"],
        summary="Execute deterministic evaluation benchmark tasks under tenant boundaries",
    )
    def run_evaluation(
        request: RunEvaluationRequest,
        runner_instance: AgentRunner = Depends(get_agent_runner),
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> EvaluationSummary:
        engine = EvaluationEngine(runner=runner_instance)
        try:
            return engine.run_benchmark(
                tenant_ctx=tenant_ctx,
                tasks=request.tasks,
                model=request.model,
                provider=request.provider,
            )
        except PermissionError as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    @app.get(
        "/api/v1/evaluation/results",
        response_model=List[EvaluationSummary],
        tags=["Evaluation"],
        summary="List past evaluation benchmark results for tenant",
    )
    def list_evaluations(
        tenant_ctx: TenantContext = Depends(get_tenant_context),
    ) -> List[EvaluationSummary]:
        if Permission.RUN_READ not in tenant_ctx.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Role '{tenant_ctx.role.value}' lacks RUN_READ permission.",
            )
        return telemetry_store.list_evaluations(organization_id=tenant_ctx.organization_id)

    # -------------------------------------------------------------------------
    # Static Dashboard Mounting
    # -------------------------------------------------------------------------
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import RedirectResponse

    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="static")

    @app.get("/", include_in_schema=False)
    def root_redirect():
        return RedirectResponse(url="/dashboard/")

    return app



# Default application instance for uvicorn/ASGI entrypoints
app = create_app()


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=port, reload=False)
