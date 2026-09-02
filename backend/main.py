from fastapi import FastAPI
from pydantic import BaseModel
from backend.services.llm import get_llm

from backend.agents.router import route_task
from backend.agents.planner import create_plan
from backend.agents.knowledge import answer_from_project

from backend.orchestrator.workflow import run_workflow



class ChatRequest(BaseModel):
    message: str

class KnowledgeRequest(BaseModel):
    project_id: str
    question: str

class WorkflowRequest(BaseModel):
    message: str
    project_id: str | None = None    

app = FastAPI(
    title="Agentic AI Software Engineer",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Agentic AI Software Engineer API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    llm = get_llm()

    response = llm.invoke(
        request.message
    )

    return {
        "response": response.content
    }


@app.post("/route")
def route(request: ChatRequest):
    decision = route_task(request.message)

    return {
        "routing": decision.model_dump()
    }


@app.post("/plan")
def plan(request: ChatRequest):

    routing = route_task(request.message)

    execution_plan = create_plan(
        request.message,
        routing
    )

    return {
        "routing": routing.model_dump(),
        "plan": execution_plan.model_dump()
    }


@app.post("/knowledge")
def knowledge(request: KnowledgeRequest):

    result = answer_from_project(
        project_id=request.project_id,
        question=request.question,
    )

    return {
        "knowledge": result.model_dump()
    }


@app.post("/run")
def run_agentic_workflow(request: WorkflowRequest):

    result = run_workflow(
        user_message=request.message,
        project_id=request.project_id,
    )

    return result