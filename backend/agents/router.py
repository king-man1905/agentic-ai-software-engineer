from backend.services.llm import get_llm
from backend.observability.telemetry import invoke_structured
from backend.schemas.routing import RoutingDecision


def route_task(user_message: str) -> RoutingDecision:
    llm = get_llm()

    # Kept short deliberately: this is a single classification call, and a
    # long, example-heavy prompt makes reasoning models (e.g. NVIDIA's
    # openai/gpt-oss-20b) spend far more hidden "thinking" time before
    # answering, risking request timeouts for no gain on this simple task.
    prompt = f"""Classify this software engineering request. This is a simple classification task - do not reason at length.

task_type - exactly one of:
CODE_GENERATION, BUG_FIX, CODE_REVIEW, CODE_EXPLANATION, DOCUMENTATION, DATA_ANALYSIS, KNOWLEDGE_SEARCH, GENERAL

Rules:
- BUG_FIX: fixing an error or broken behavior.
- CODE_REVIEW: reviewing or improving existing code.
- CODE_EXPLANATION: explaining code the user provided or pointed to.
- DOCUMENTATION: README, comments, or technical docs.
- DATA_ANALYSIS: analyzing CSV/SQL/database/structured data.
- KNOWLEDGE_SEARCH: finding/locating/searching where something exists in the project (not CODE_EXPLANATION).
- CODE_GENERATION: new code, feature, function, API, or component.
- GENERAL: none of the above.

requires_planning - true only if the request needs multiple engineering steps.
requires_knowledge - true only if it needs existing project context, files, docs, data, or external info.
reasoning - one short sentence. No long explanation.

Return only the required structured fields, nothing else.

User request:
{user_message}
"""

    return invoke_structured(llm, RoutingDecision, prompt)