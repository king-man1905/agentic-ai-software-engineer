from typing import Any, Dict, Optional
from backend.services.llm import get_llm
from backend.observability.telemetry import invoke_structured
from backend.schemas.planning import ExecutionPlan
from backend.schemas.routing import RoutingDecision


def create_plan(
    user_message: str,
    routing: RoutingDecision,
    repo_evidence: Optional[Dict[str, Any]] = None,
) -> ExecutionPlan:

    llm = get_llm()

    evidence_block = ""
    if repo_evidence:
        files = repo_evidence.get("files") or []
        symbols = repo_evidence.get("symbols") or []
        tests = repo_evidence.get("tests") or []
        evidence_block = f"""
REPOSITORY EVIDENCE:
Identified Target Files: {files}
Identified Symbols: {symbols}
Related Test Files: {tests}

When available, ground each step by referencing verified files, symbols, and tests in its 'files', 'symbols', and 'tests' fields. Do not invent or force unverified references.
"""

    prompt = f"""
You are the Planner Agent of an Agentic AI Software Engineer.

Your job is to convert a software engineering request into a clear,
ordered and executable plan.

USER REQUEST:
{user_message}

ROUTING INFORMATION:
Task type: {routing.task_type.value}
Requires knowledge: {routing.requires_knowledge}
{evidence_block}
AVAILABLE EXECUTION AGENTS:

Knowledge:
- Search and retrieve relevant project files
- Retrieve documentation using RAG
- Search external information when required
- Analyze SQL/CSV data when required

Developer:
- Generate code
- Modify existing code
- Debug code
- Refactor code
- Generate documentation

QA:
- Generate tests
- Execute tests
- Analyze failures
- Verify the final implementation

PLANNING RULES:

1. Create only steps necessary to complete the request.
2. Steps must be in logical execution order.
3. Use only these agent names:
   Knowledge
   Developer
   QA
4. Do not write implementation code.
5. If project context is required, retrieve it before development.
6. Software changes should normally be verified by QA.
7. Avoid duplicate or vague steps.
8. Every step must describe one concrete action.
9. Define a measurable success criterion.
10. Keep the plan concise.

Return the structured execution plan.
"""

    return invoke_structured(llm, ExecutionPlan, prompt)