from backend.graph.state import AgentState
from backend.schemas.routing import TaskType
from backend.schemas.knowledge import KnowledgeAnswer
from backend.schemas.planning import ExecutionPlan
from backend.observability.telemetry import collect_usage, invoke_structured, merge_usage

from backend.agents.router import route_task
from backend.agents.planner import create_plan
from backend.agents.knowledge import answer_from_project
from backend.agents.developer import generate_code_changes
from backend.agents.qa import review_code_changes

from langgraph.types import interrupt


def router_node(state: AgentState) -> dict:
    routing = route_task(state["user_message"])

    return {
        "routing": routing
    }


def planner_node(state: AgentState) -> dict:
    with collect_usage() as usage:
        plan = create_plan(
            state["user_message"],
            state["routing"],
        )

    return {
        "plan": plan,
        "metrics": merge_usage(state.get("metrics"), usage),
    }


def knowledge_node(state: AgentState) -> dict:
    project_id = state.get("project_id")

    if not project_id:
        return {
            "knowledge": KnowledgeAnswer(
                answer="No project_id was provided to retrieve project context.",
                sources=[],
                sufficient_context=False,
            ),
            "repo_context": [],
        }

    try:
        knowledge = answer_from_project(
            project_id=project_id,
            question=state["user_message"],
        )
    except Exception as e:
        knowledge = KnowledgeAnswer(
            answer=f"Vector store not indexed yet: {e}",
            sources=[],
            sufficient_context=False,
        )


    import os
    from pathlib import Path
    from backend.indexer.scanner import scan_repository
    from backend.indexer.ast_chunker import chunk_file
    from backend.indexer.retriever import SimpleBM25Index, HybridRetriever
    from backend.indexer.models import CodeChunk

    sparse_candidates = []
    project_path = Path("workspace") / project_id
    if not project_path.exists():
        project_path = Path(os.getcwd()) / "workspace" / project_id

    if project_path.exists():
        try:
            scanned_files = scan_repository(str(project_path))
            all_chunks = []
            for sf in scanned_files:
                chunks = chunk_file(sf.absolute_path, sf.relative_path)
                all_chunks.extend(chunks)

            if all_chunks:
                bm25 = SimpleBM25Index(all_chunks)
                sparse_pairs = bm25.search(state["user_message"], top_n=20)
                sparse_candidates = [pair[0] for pair in sparse_pairs]
        except Exception as e:
            print(f"BM25 retrieval error: {e}")

    dense_candidates = []
    try:
        from backend.rag.retriever import load_project_index
        vector_store = load_project_index(project_id)
        docs = vector_store.similarity_search(state["user_message"], k=20)
        for doc in docs:
            dense_candidates.append(
                CodeChunk(
                    file_path=doc.metadata.get("source", "unknown"),
                    chunk_type="dense",
                    symbol_name=None,
                    content=doc.page_content,
                    start_line=doc.metadata.get("start_line", 1),
                    end_line=doc.metadata.get("end_line", 1),
                    docstring=None,
                    decorators=[],
                )
            )
    except Exception as e:
        print(f"Vector search retrieval error: {e}")

    retriever = HybridRetriever()
    search_results = retriever.retrieve(
        sparse_candidates=sparse_candidates,
        dense_candidates=dense_candidates,
        top_k=4,
    )
    repo_context = [res.chunk for res in search_results]

    return {
        "knowledge": knowledge,
        "repo_context": repo_context,
    }


def developer_node(state: AgentState) -> dict:
    plan = state.get("plan")
    if plan is None:
        plan = ExecutionPlan(
            goal=state["user_message"],
            steps=[],
            success_criteria="Complete requested task directly.",
        )

    knowledge = state.get("knowledge")
    if knowledge is None:
        knowledge = KnowledgeAnswer(
            answer="No external project context required.",
            sources=[],
            sufficient_context=True,
        )

    with collect_usage() as usage:
        developer_result = generate_code_changes(
            user_request=state["user_message"],
            plan=plan,
            knowledge=knowledge,
        )

        repo_context = state.get("repo_context")
        if not repo_context and state.get("project_id"):
            import os
            from pathlib import Path
            project_path = Path("workspace") / state["project_id"]
            if not project_path.exists():
                project_path = Path(os.getcwd()) / "workspace" / state["project_id"]
            if project_path.exists():
                try:
                    from backend.indexer.scanner import scan_repository
                    from backend.indexer.ast_chunker import chunk_file
                    scanned_files = scan_repository(str(project_path))
                    all_chunks = []
                    for sf in scanned_files:
                        all_chunks.extend(chunk_file(sf.absolute_path, sf.relative_path))
                    repo_context = all_chunks
                except Exception as e:
                    print(f"Developer node context scan notice: {e}")

        generated_patches = []

        if repo_context:
            context_str = ""
            for chunk in repo_context:
                context_str += f"\nFILE: {chunk.file_path}\n"
                if chunk.symbol_name:
                    context_str += f"SYMBOL: {chunk.symbol_name}\n"
                context_str += f"CONTENT:\n{chunk.content}\n---\n"

            prompt = f"""
You are the Developer Agent Patch Generator.
Your task is to generate precise and safe code patches for the user request and execution plan.

USER REQUEST:
{state["user_message"]}

EXECUTION PLAN:
{plan.model_dump_json(indent=2)}

REPOSITORY CONTEXT:
{context_str}

Return a list of precise FilePatches. For each patch, provide the file path, the exact original code snippet to be replaced, and the updated code snippet.
"""
            from backend.services.llm import get_llm
            from backend.developer.models import FilePatch
            from pydantic import BaseModel, Field

            class PatchResponse(BaseModel):
                patches: list[FilePatch] = Field(description="List of proposed file patches.")

            llm = get_llm()
            patch_result = invoke_structured(llm, PatchResponse, prompt)
            patches = patch_result.patches

            # Perform AST pre-flight validation
            from backend.developer.patcher import SafePatcher
            import os
            from pathlib import Path

            for patch in patches:
                project_id = state.get("project_id", "test_project")
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
                        f"AST pre-flight validation failed for {patch.file_path}: {val_result.syntax_errors}"
                    )
                if val_result.applied_content is not None and abs_file_path.parent.exists():
                    try:
                        with open(abs_file_path, "w", encoding="utf-8") as f:
                            f.write(val_result.applied_content)
                    except Exception as e:
                        print(f"Notice: could not write patched file: {e}")
                generated_patches.append(patch)

        if not generated_patches and developer_result and developer_result.changes:
            from backend.developer.models import FilePatch
            import os
            from pathlib import Path
            project_id = state.get("project_id", "test_project")
            project_path = Path("workspace") / project_id
            if not project_path.exists():
                project_path = Path(os.getcwd()) / "workspace" / project_id
            for ch in developer_result.changes:
                abs_f = project_path / ch.file_path
                if abs_f.parent.exists() and ch.content:
                    try:
                        with open(abs_f, "w", encoding="utf-8") as f:
                            f.write(ch.content)
                        generated_patches.append(
                            FilePatch(
                                file_path=ch.file_path,
                                original_code_snippet="",
                                updated_code_snippet=ch.content,
                                explanation=ch.reason,
                            )
                        )
                    except Exception as e:
                        print(f"Notice: could not write full file change: {e}")


    return {
        "developer_result": developer_result,
        "plan": plan,
        "generated_patches": generated_patches,
        "metrics": merge_usage(state.get("metrics"), usage),
    }


def qa_node(state: AgentState) -> dict:
    plan = state.get("plan")
    if plan is None:
        plan = ExecutionPlan(
            goal=state["user_message"],
            steps=[],
            success_criteria="Complete requested task directly.",
        )

    qa_result = review_code_changes(
        user_request=state["user_message"],
        plan=plan,
        developer_result=state["developer_result"],
    )

    project_id = state.get("project_id", "test_project")
    import os
    from pathlib import Path
    from backend.sandbox.runner import SandboxRunner
    from backend.schemas.qa import QAIssue

    project_path = Path("workspace") / project_id
    if not project_path.exists():
        project_path = Path(os.getcwd()) / "workspace" / project_id

    test_result = None
    if project_path.exists():
        try:
            cmd = ["python", "-m", "pytest"]
            test_result = SandboxRunner.run_command(cmd, cwd=str(project_path))
        except Exception as e:
            from backend.sandbox.models import TestExecutionResult
            test_result = TestExecutionResult(
                success=False,
                exit_code=-1,
                passed_count=0,
                failed_count=0,
                stdout="",
                stderr=str(e),
                duration_seconds=0.0,
                error_summary=str(e),
            )

    if test_result:
        # Fail the validation if test execution fails
        # Exit code 5 is 'no tests collected' in pytest, which is not treated as a failure.
        tests_passed = test_result.success or test_result.exit_code == 5
        if not tests_passed:
            qa_result.status = "FAIL"
            qa_result.issues.append(
                QAIssue(
                    file_path="tests",
                    issue=test_result.error_summary or "Pytest run failed.",
                    severity="HIGH",
                )
            )
            qa_result.summary = (
                f"Test suite execution failed.\nError Summary: {test_result.error_summary}\n\n"
                + qa_result.summary
            )
        elif test_result.success and test_result.failed_count == 0:
            qa_result.status = "PASS"



    return {
        "qa_result": qa_result,
        "test_result": test_result,
    }


def route_after_router(state: AgentState) -> str:
    routing = state["routing"]

    if routing.task_type == TaskType.GENERAL:
        return "end"

    if routing.requires_planning:
        return "planner"

    if routing.task_type == TaskType.KNOWLEDGE_SEARCH or routing.requires_knowledge:
        return "knowledge"

    return "developer"


def route_after_planner(state: AgentState) -> str:
    routing = state.get("routing")
    if routing and routing.requires_knowledge and state.get("project_id"):
        return "knowledge"

    return "developer"


def route_after_knowledge(state: AgentState) -> str:
    routing = state.get("routing")
    if routing and routing.task_type == TaskType.KNOWLEDGE_SEARCH and not routing.requires_planning:
        return "end"

    return "developer"


MAX_REVISIONS = 3


def qa_router(state: AgentState) -> str:
    qa_result = state.get("qa_result")
    revision_count = state.get("revision_count", 0)

    if qa_result is None:
        return "max_retries" if revision_count >= MAX_REVISIONS else "fail"

    status = (qa_result.status or "").strip().upper()

    if status == "PASS":
        return "pass"

    if revision_count >= MAX_REVISIONS:
        return "max_retries"

    return "fail"


def revision_node(state: AgentState) -> dict:
    from backend.agents.developer import revise_code_changes
    from backend.agents.revision import generate_revision_patches
    from backend.revision.models import RevisionAttempt, RevisionHistory
    from backend.revision.analyzer import ErrorTraceAnalyzer
    from backend.schemas.developer import DeveloperResult

    revision_count = state.get("revision_count", 0)

    # 1. Fallback plan persistence
    plan = state.get("plan")
    if plan is None:
        plan = ExecutionPlan(
            goal=state["user_message"],
            steps=[],
            success_criteria="Complete requested task directly.",
        )

    # 2. Extract and analyze error traces from sandbox execution or QA summary
    test_result = state.get("test_result")
    qa_result = state.get("qa_result")

    if test_result is not None:
        analysis = ErrorTraceAnalyzer.analyze_test_result(test_result)
    elif qa_result is not None and getattr(qa_result, "summary", None):
        analysis = ErrorTraceAnalyzer.analyze(qa_result.summary)
    else:
        analysis = ErrorTraceAnalyzer.analyze("")

    # 3. Retrieve or initialize revision history
    existing_history = state.get("revision_history")
    if existing_history is not None:
        revision_history = existing_history
    else:
        revision_history = RevisionHistory(max_retries=MAX_REVISIONS)

    # 4. Fallback previous developer result
    prev_result = state.get("developer_result")
    if prev_result is None:
        prev_result = DeveloperResult(
            summary="Initial implementation attempt.",
            changes=[],
            requires_testing=True,
            notes=[],
        )

    # 5. Revise code changes
    with collect_usage() as usage:
        revised_result = revise_code_changes(
            user_request=state["user_message"],
            plan=plan,
            previous_result=prev_result,
            qa_result=qa_result,
        )

        # 6. Generate revised patches and perform AST pre-flight validation if repo context is available
        repo_context = state.get("repo_context")
        generated_patches = state.get("generated_patches") or []

        if repo_context:
            try:
                revised_patches = generate_revision_patches(
                    user_request=state["user_message"],
                    plan=plan,
                    error_analysis=analysis,
                    repo_context=repo_context,
                    revision_history=revision_history,
                    project_id=state.get("project_id", "test_project"),
                )
                if revised_patches:
                    generated_patches = revised_patches
            except Exception as e:
                print(f"Revision patch generation notice: {e}")

    # 7. Record this revision attempt
    new_revision_count = revision_count + 1
    applied_patch = generated_patches[0] if generated_patches else None
    diagnosis_summary = analysis.diagnosis or (
        qa_result.summary if qa_result else "Revision performed based on test feedback."
    )

    attempt = RevisionAttempt(
        attempt_number=new_revision_count,
        failing_tests=analysis.failing_tests,
        error_traceback=analysis.error_traceback,
        applied_patch=applied_patch,
        diagnosis=diagnosis_summary,
    )
    revision_history.add_attempt(attempt)

    return {
        "developer_result": revised_result,
        "plan": plan,
        "revision_count": new_revision_count,
        "revision_history": revision_history,
        "generated_patches": generated_patches,
        "metrics": merge_usage(state.get("metrics"), usage),
    }


def git_prepare_node(state: AgentState) -> dict:
    """
    Prepares a Git diff summary by applying validated patches to the workspace,
    computing unified diffs, and evaluating risk.
    """
    from backend.vcs.git_manager import GitWorkspaceManager
    from backend.vcs.models import GitDiffSummary
    import os
    from pathlib import Path

    patches = state.get("generated_patches") or []
    project_id = state.get("project_id", "test_project")
    task_id = state.get("run_id") or project_id


    project_path = Path("workspace") / project_id
    if not project_path.exists():
        project_path = Path(os.getcwd()) / "workspace" / project_id

    if not patches or not project_path.exists():
        # No patches to stage or no workspace; produce an empty diff summary
        branch_name = GitWorkspaceManager.generate_branch_name(task_id)
        return {
            "git_diff": GitDiffSummary(
                branch_name=branch_name,
                files_changed=[],
                lines_added=0,
                lines_deleted=0,
                unified_diff="",
                risk_score="LOW",
                risk_reasons=["No file patches to apply."],
            ),
        }

    try:
        diff_summary = GitWorkspaceManager.prepare_diff_summary(
            repo_path=str(project_path),
            patches=patches,
            task_id=task_id,
        )
    except Exception as e:
        branch_name = GitWorkspaceManager.generate_branch_name(task_id)
        diff_summary = GitDiffSummary(
            branch_name=branch_name,
            files_changed=[p.file_path for p in patches],
            lines_added=0,
            lines_deleted=0,
            unified_diff="",
            risk_score="MEDIUM",
            risk_reasons=[f"Diff preparation encountered an error: {e}"],
        )

    return {
        "git_diff": diff_summary,
    }


def approval_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop approval gate. Pauses execution via LangGraph interrupt()
    presenting the Git diff summary and risk assessment, then resumes with
    an ApprovalDecision.
    """
    from backend.vcs.models import ApprovalDecision

    git_diff = state.get("git_diff")
    developer_result = state.get("developer_result")

    interrupt_payload = {
        "task": "approval_required",
        "diff": git_diff.model_dump() if git_diff else None,
        "developer_result": developer_result.model_dump() if developer_result else None,
        "message": "Review the proposed code changes, diff, and risk assessment.",
    }

    decision = interrupt(interrupt_payload)

    # Handle structured ApprovalDecision or simple boolean
    if isinstance(decision, dict):
        approval = ApprovalDecision(**decision)
    elif isinstance(decision, ApprovalDecision):
        approval = decision
    elif decision is True:
        approval = ApprovalDecision(approved=True)
    else:
        rejection_reason = decision if isinstance(decision, str) else "Rejected by reviewer."
        approval = ApprovalDecision(approved=False, rejection_reason=rejection_reason)

    return {
        "approval": approval,
        "approval_status": "APPROVED" if approval.approved else "REJECTED",
    }


def route_after_approval(state: AgentState) -> str:
    """
    Routes after human approval decision:
    - approved -> git_commit
    - rejected -> cleanup
    """
    approval = state.get("approval")
    if approval is not None and approval.approved:
        return "git_commit"
    return "cleanup"


def git_commit_node(state: AgentState) -> dict:
    """
    Stages and commits approved changes in the feature branch.
    Skips committing entirely when the diff is a no-op (no files actually
    changed) - e.g. the developer agent found nothing to fix - so an empty
    or junk commit is never pushed or turned into a PR.
    """
    from backend.vcs.git_manager import GitWorkspaceManager
    import os
    from pathlib import Path

    git_diff = state.get("git_diff")
    if git_diff is None or git_diff.is_no_op:
        return {
            "approval_status": "NO_CHANGES_NEEDED",
        }

    project_id = state.get("project_id", "test_project")
    project_path = Path("workspace") / project_id
    if not project_path.exists():
        project_path = Path(os.getcwd()) / "workspace" / project_id

    commit_message = f"agent: apply approved changes for {project_id}"
    developer_result = state.get("developer_result")
    if developer_result:
        commit_message = f"agent: {developer_result.summary[:80]}"

    git_diff = state.get("git_diff")
    if git_diff and git_diff.branch_name:
        GitWorkspaceManager.create_feature_branch(str(project_path), git_diff.branch_name)

    success = GitWorkspaceManager.stage_and_commit(
        repo_path=str(project_path),
        message=commit_message,
    )


    return {
        "approval_status": "COMMITTED" if success else "COMMIT_FAILED",
    }


def cleanup_node(state: AgentState) -> dict:
    """
    Cleans up the feature branch after rejection by checking out the
    previous branch and deleting the feature branch.
    """
    from backend.vcs.git_manager import GitWorkspaceManager
    import os
    from pathlib import Path

    git_diff = state.get("git_diff")
    project_id = state.get("project_id", "test_project")
    project_path = Path("workspace") / project_id
    if not project_path.exists():
        project_path = Path(os.getcwd()) / "workspace" / project_id

    branch_name = git_diff.branch_name if git_diff else "agent/task-unknown"

    GitWorkspaceManager.cleanup_branch(
        repo_path=str(project_path),
        branch_name=branch_name,
    )

    rejection_reason = ""
    approval = state.get("approval")
    if approval and approval.rejection_reason:
        rejection_reason = f" Reason: {approval.rejection_reason}"

    return {
        "approval_status": f"REJECTED_AND_CLEANED{rejection_reason}",
    }