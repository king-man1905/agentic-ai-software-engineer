from backend.services.llm import get_llm
from backend.observability.telemetry import invoke_structured
from backend.schemas.qa import QAResult
from backend.schemas.planning import ExecutionPlan
from backend.schemas.developer import DeveloperResult


def review_code_changes(
    user_request: str,
    plan: ExecutionPlan,
    developer_result: DeveloperResult,
) -> QAResult:

    llm = get_llm()


    prompt = f"""
You are the QA Agent of an Agentic AI Software Engineer.

Your job is to review the Developer Agent's proposed changes
before they are accepted.

USER REQUEST:
{user_request}

EXECUTION PLAN:
{plan.model_dump_json(indent=2)}

DEVELOPER CHANGES:
{developer_result.model_dump_json(indent=2)}

QA RESPONSIBILITIES:

1. Check whether the proposed changes address the user request.
2. Check whether the changes follow the execution plan.
3. Check for obvious logic errors.
4. Check for missing validation or edge cases.
5. Check whether unrelated files are being modified.
6. Identify potential regressions.
7. Suggest concrete test cases.
8. Do not modify the code.
9. Return PASS only when the proposed implementation appears
   correct and sufficiently complete.
10. Return FAIL when important problems are found.

SEVERITY:

LOW:
Minor concern that does not block the implementation.

MEDIUM:
Important issue that should be fixed.

HIGH:
Critical issue that makes the implementation incorrect,
unsafe, or unusable.

Return a structured QAResult.
"""

    return invoke_structured(llm, QAResult, prompt)