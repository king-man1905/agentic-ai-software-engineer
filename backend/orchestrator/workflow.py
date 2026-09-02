from backend.agents.router import route_task
from backend.agents.planner import create_plan
from backend.agents.knowledge import answer_from_project


def run_workflow(
    user_message: str,
    project_id: str | None = None,
):
    # 1. Route the request
    routing = route_task(user_message)

    result = {
        "routing": routing.model_dump(),
        "plan": None,
        "knowledge": None,
    }

    # 2. Create plan only when required
    if routing.requires_planning:
        plan = create_plan(
            user_message,
            routing,
        )

        result["plan"] = plan.model_dump()

    # 3. Retrieve project knowledge only when required
    if routing.requires_knowledge:

        if not project_id:
            result["knowledge"] = {
                "answer": (
                    "Project context is required, "
                    "but no project_id was provided."
                ),
                "sources": [],
                "sufficient_context": False,
            }

            return result

        knowledge = answer_from_project(
            project_id=project_id,
            question=user_message,
        )

        result["knowledge"] = knowledge.model_dump()

    return result 