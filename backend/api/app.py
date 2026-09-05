import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Response, status, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.models import CreateRunRequest, RunStatusResponse, ResumeRunRequest
from backend.graph.runner import AgentRunner
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


def create_app(runner: Optional[AgentRunner] = None) -> FastAPI:
    """
    Factory creating a configured FastAPI application gateway.
    """
    app = FastAPI(
        title="Agentic AI Software Engineer API Gateway",
        version="1.0.0",
        description="Production API Gateway for autonomous agent workflows with durable HITL persistence.",
    )

    # Attach shared AgentRunner instance
    app.state.runner = runner or AgentRunner()

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
        runner_instance: AgentRunner = Depends(get_agent_runner),
    ) -> RunStatusResponse:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        runner_instance.register_run(
            run_id=run_id,
            metadata=request.metadata,
        )
        background_tasks.add_task(
            runner_instance.start_run,
            run_id=run_id,
            user_message=request.user_message,
            project_id=request.project_id,
            metadata=request.metadata,
        )
        return RunStatusResponse(
            run_id=run_id,
            status="RUNNING",
            message="Run dispatched successfully in background",
        )

    @app.get(
        "/api/v1/runs/{run_id}",
        response_model=RunStatusResponse,
        tags=["Runs"],
        summary="Get current status and state of an agent run",
    )
    def get_run_status(
        run_id: str,
        runner_instance: AgentRunner = Depends(get_agent_runner),
    ) -> RunStatusResponse:
        try:
            return runner_instance.get_status(run_id)
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
        runner_instance: AgentRunner = Depends(get_agent_runner),
    ) -> RunStatusResponse:
        try:
            decision = ApprovalDecision(
                approved=request.approved,
                reviewer=request.reviewer,
                rejection_reason=request.rejection_reason,
            )
            return runner_instance.resume_run(run_id, decision)
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
