from backend.services.llm import get_llm
from backend.observability.telemetry import invoke_structured
from backend.schemas.developer import DeveloperResult
from backend.schemas.planning import ExecutionPlan
from backend.schemas.knowledge import KnowledgeAnswer


def generate_code_changes(
    user_request: str,
    plan: ExecutionPlan,
    knowledge: KnowledgeAnswer,
) -> DeveloperResult:

    llm = get_llm()

    prompt = f"""
You are the Developer Agent of an Agentic AI Software Engineer.

Your responsibility is to propose safe and precise code changes
based on the user's request, execution plan, and retrieved
project context.

USER REQUEST:
{user_request}

EXECUTION PLAN:
{plan.model_dump_json(indent=2)}

PROJECT KNOWLEDGE:
{knowledge.model_dump_json(indent=2)}

RULES:

1. Only modify files supported by the retrieved context.
2. Never invent a filename.
3. Do not modify unrelated files.
4. For CREATE or MODIFY, provide the complete proposed file content.
5. For DELETE, content must be empty.
6. Explain the reason for every change.
7. Do not execute the code.
8. Do not generate tests.
9. QA will handle testing.
10. If the available context is insufficient, do not guess.
11. Keep the implementation focused on the user's request.

Return a structured DeveloperResult.
"""

    return invoke_structured(llm, DeveloperResult, prompt)

def revise_code_changes(
    user_request: str,
    plan: ExecutionPlan,
    previous_result: DeveloperResult,
    qa_result,
) -> DeveloperResult:

    llm = get_llm()

    prompt = f"""
You are the Developer Agent revising a previous implementation.

USER REQUEST:
{user_request}

EXECUTION PLAN:
{plan.model_dump_json(indent=2)}

PREVIOUS DEVELOPER RESULT:
{previous_result.model_dump_json(indent=2)}

QA FEEDBACK:
{qa_result.model_dump_json(indent=2)}

The previous implementation failed QA.

RULES:

1. Fix the important issues identified by QA.
2. Only modify relevant files.
3. Never invent filenames.
4. For CREATE or MODIFY, provide complete file content.
5. Do not generate tests.
6. QA will test the revised implementation.
7. Do not ignore HIGH or MEDIUM severity issues.
8. Explain what was corrected.

Return a revised DeveloperResult.
"""

    return invoke_structured(llm, DeveloperResult, prompt)