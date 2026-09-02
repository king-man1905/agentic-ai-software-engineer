from typing import List, Optional
from backend.services.llm import get_llm
from backend.observability.telemetry import invoke_structured
from backend.schemas.developer import DeveloperResult
from backend.schemas.planning import ExecutionPlan
from backend.schemas.qa import QAResult
from backend.sandbox.models import TestExecutionResult
from backend.indexer.models import CodeChunk
from backend.developer.models import FilePatch
from backend.developer.patcher import SafePatcher
from backend.revision.models import (
    RevisionHistory,
    ErrorTraceAnalysis,
)
from backend.revision.analyzer import ErrorTraceAnalyzer
from pydantic import BaseModel, Field


class PatchResponse(BaseModel):
    patches: List[FilePatch] = Field(description="List of proposed file patches.")


def revise_code_with_traceback(
    user_request: str,
    plan: ExecutionPlan,
    previous_result: DeveloperResult,
    qa_result: Optional[QAResult] = None,
    test_result: Optional[TestExecutionResult] = None,
    error_analysis: Optional[ErrorTraceAnalysis] = None,
    revision_history: Optional[RevisionHistory] = None,
    repo_context: Optional[List[CodeChunk]] = None,
) -> DeveloperResult:
    """
    Revision Agent responsible for refining code changes using structured error traces,
    failing tests, and previous revision attempt history.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(DeveloperResult)

    if error_analysis is None:
        if test_result is not None:
            error_analysis = ErrorTraceAnalyzer.analyze_test_result(test_result)
        elif qa_result is not None:
            error_analysis = ErrorTraceAnalyzer.analyze(qa_result.summary)
        else:
            error_analysis = ErrorTraceAnalysis()

    # Format previous revision history
    history_str = "None"
    if revision_history and revision_history.attempts:
        history_lines = []
        for att in revision_history.attempts:
            history_lines.append(
                f"- Attempt #{att.attempt_number}:\n"
                f"  Failing Tests: {', '.join(att.failing_tests) if att.failing_tests else 'None'}\n"
                f"  Diagnosis: {att.diagnosis}\n"
            )
        history_str = "\n".join(history_lines)

    # Format QA Feedback
    qa_str = "No specific QA issues noted."
    if qa_result:
        qa_str = qa_result.model_dump_json(indent=2)

    # Format Repository Context
    context_str = "No repository context provided."
    if repo_context:
        c_lines = []
        for chunk in repo_context:
            c_lines.append(f"\nFILE: {chunk.file_path}\nCONTENT:\n{chunk.content}\n---")
        context_str = "\n".join(c_lines)

    prompt = f"""
You are the Revision Agent of an Agentic AI Software Engineer.
Your responsibility is to correct and fix the code implementation that previously failed verification.

USER REQUEST:
{user_request}

EXECUTION PLAN:
{plan.model_dump_json(indent=2)}

PREVIOUS DEVELOPER IMPLEMENTATION:
{previous_result.model_dump_json(indent=2)}

QA REVIEW FEEDBACK:
{qa_str}

ERROR TRACE & FAILING TESTS ANALYSIS:
Failing Test Cases: {error_analysis.failing_tests}
Diagnosis:
{error_analysis.diagnosis}

RAW ERROR TRACEBACK:
{error_analysis.error_traceback}

PREVIOUS REVISION ATTEMPTS:
{history_str}

PROJECT CODE CONTEXT:
{context_str}

REVISION RULES:
1. Directly fix the root cause of the failing test(s) and assertion mismatches.
2. Pay close attention to target line numbers and exception types identified in the diagnosis.
3. Do not repeat failed modifications recorded in previous revision attempts.
4. Only modify relevant files required to make tests pass and satisfy the user request.
5. Provide the complete updated file content for each modified or created file.
6. Clearly explain what root cause was identified and how it was resolved in the summary.

Return a structured DeveloperResult.
"""

    return structured_llm.invoke(prompt)


def generate_revision_patches(
    user_request: str,
    plan: ExecutionPlan,
    error_analysis: ErrorTraceAnalysis,
    repo_context: Optional[List[CodeChunk]] = None,
    revision_history: Optional[RevisionHistory] = None,
    project_id: str = "test_project",
) -> List[FilePatch]:
    """
    Generates updated FilePatches based on failing error traces, and validates syntax with AST SafePatcher.
    """
    if not repo_context:
        return []

    context_str = ""
    for chunk in repo_context:
        context_str += f"\nFILE: {chunk.file_path}\n"
        if chunk.symbol_name:
            context_str += f"SYMBOL: {chunk.symbol_name}\n"
        context_str += f"CONTENT:\n{chunk.content}\n---\n"

    history_str = "None"
    if revision_history and revision_history.attempts:
        h_lines = []
        for att in revision_history.attempts:
            h_lines.append(
                f"- Attempt #{att.attempt_number}: {att.diagnosis}"
            )
        history_str = "\n".join(h_lines)

    prompt = f"""
You are the Revision Agent Patch Generator.
Your task is to generate precise, surgical code patches to resolve test failures and execution errors.

USER REQUEST:
{user_request}

EXECUTION PLAN:
{plan.model_dump_json(indent=2)}

FAILING TESTS & ERROR DIAGNOSIS:
Failing Tests: {error_analysis.failing_tests}
Diagnosis: {error_analysis.diagnosis}
Traceback:
{error_analysis.error_traceback}

PREVIOUS ATTEMPTS:
{history_str}

REPOSITORY CONTEXT:
{context_str}

Generate precise FilePatch objects replacing the broken snippet with the corrected snippet.
"""
    llm = get_llm()
    patch_result = invoke_structured(llm, PatchResponse, prompt)
    patches = patch_result.patches

    import os
    from pathlib import Path

    validated_patches: List[FilePatch] = []
    for patch in patches:
        project_path = Path("workspace") / project_id
        if not project_path.exists():
            project_path = Path(os.getcwd()) / "workspace" / project_id

        abs_file_path = project_path / patch.file_path
        source_content = ""
        if abs_file_path.exists():
            try:
                with open(abs_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    source_content = f.read()
            except Exception:
                pass

        val_result = SafePatcher.apply_patch(source_content, patch)
        if not val_result.is_valid:
            raise ValueError(
                f"AST pre-flight validation failed during revision for {patch.file_path}: {val_result.syntax_errors}"
            )
        validated_patches.append(patch)

    return validated_patches
