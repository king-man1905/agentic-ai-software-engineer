from backend.graph.state import AgentState
from backend.schemas.routing import TaskType
from backend.schemas.knowledge import KnowledgeAnswer
from backend.schemas.planning import ExecutionPlan
from backend.observability.telemetry import collect_usage, invoke_structured, merge_usage
from backend.observability.collector import telemetry_collector
from backend.schemas.telemetry import TelemetryEventType
from backend.graph.cancellation import check_cancelled, mark_activity

from backend.agents.router import route_task
from backend.agents.planner import create_plan
from backend.agents.knowledge import answer_from_project
from backend.agents.developer import generate_code_changes
from backend.agents.qa import review_code_changes

from langgraph.types import interrupt


def router_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "router")
    routing = route_task(state["user_message"])

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.ROUTING_COMPLETED,
            metadata={
                "task_type": getattr(routing.task_type, "value", str(routing.task_type)),
                "requires_planning": getattr(routing, "requires_planning", False),
                "requires_knowledge": getattr(routing, "requires_knowledge", False),
            },
        )

    return {
        "routing": routing
    }


def format_categorized_context(chunks: list) -> str:
    """
    Groups retrieved CodeChunks into 5 structured sections:
    PRIMARY IMPLEMENTATION, RELATED CODE, TESTS, DEPENDENCIES, CONFIGURATION.
    """
    from backend.indexer.ast_chunker import is_test_path, is_config_path

    primary = []
    related = []
    tests = []
    dependencies = []
    configs = []

    for chunk in chunks:
        fp = chunk.file_path.replace("\\", "/").lower()
        st = getattr(chunk, "symbol_type", "") or ""

        # Collect dependencies from imports
        if getattr(chunk, "imports", None):
            for imp in chunk.imports:
                if imp not in dependencies:
                    dependencies.append(imp)

        if st == "test" or is_test_path(fp):
            tests.append(chunk)
        elif st == "config" or is_config_path(fp):
            configs.append(chunk)
        elif chunk.symbol_name and st in {"class", "function", "method"}:
            primary.append(chunk)
        else:
            related.append(chunk)

    sections = []

    if primary:
        lines = ["[PRIMARY IMPLEMENTATION]"]
        for c in primary:
            sym = f" | SYMBOL: {c.symbol_name}" if c.symbol_name else ""
            lines.append(f"FILE: {c.file_path} (Lines {c.start_line}-{c.end_line}){sym}")
            lines.append(c.content)
            lines.append("---")
        sections.append("\n".join(lines))

    if related:
        lines = ["[RELATED CODE]"]
        for c in related:
            sym = f" | SYMBOL: {c.symbol_name}" if c.symbol_name else ""
            lines.append(f"FILE: {c.file_path} (Lines {c.start_line}-{c.end_line}){sym}")
            lines.append(c.content)
            lines.append("---")
        sections.append("\n".join(lines))

    if tests:
        lines = ["[TESTS]"]
        for c in tests:
            sym = f" | SYMBOL: {c.symbol_name}" if c.symbol_name else ""
            lines.append(f"FILE: {c.file_path} (Lines {c.start_line}-{c.end_line}){sym}")
            lines.append(c.content)
            lines.append("---")
        sections.append("\n".join(lines))

    if dependencies:
        lines = ["[DEPENDENCIES]"]
        lines.extend(dependencies[:20])
        lines.append("---")
        sections.append("\n".join(lines))

    if configs:
        lines = ["[CONFIGURATION]"]
        for c in configs:
            lines.append(f"FILE: {c.file_path} (Lines {c.start_line}-{c.end_line})")
            lines.append(c.content)
            lines.append("---")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def planner_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "planner")
    repo_context = state.get("repo_context")
    repo_evidence = None
    if repo_context:
        repo_evidence = {
            "files": list(dict.fromkeys(c.file_path for c in repo_context)),
            "symbols": list(dict.fromkeys(c.symbol_name for c in repo_context if c.symbol_name)),
            "tests": list(dict.fromkeys(c.file_path for c in repo_context if getattr(c, "symbol_type", "") == "test" or "test" in c.file_path.lower())),
        }
    elif state.get("project_id"):
        import os
        from pathlib import Path
        project_path = Path("workspace") / state["project_id"]
        if not project_path.exists():
            project_path = Path(os.getcwd()) / "workspace" / state["project_id"]
        if project_path.exists():
            try:
                from backend.indexer.scanner import scan_repository
                scanned = scan_repository(str(project_path))
                files = [sf.relative_path for sf in scanned]
                tests = [f for f in files if "test" in f.lower()]
                repo_evidence = {"files": files[:10], "symbols": [], "tests": tests[:5]}
            except Exception:
                pass

    with collect_usage() as usage:
        plan = create_plan(
            state["user_message"],
            state["routing"],
            repo_evidence=repo_evidence,
        )

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.PLAN_CREATED,
            metadata={"goal": plan.goal, "steps_count": len(plan.steps)},
        )

    return {
        "plan": plan,
        "metrics": merge_usage(state.get("metrics"), usage),
    }


def knowledge_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "knowledge")
    import time
    start_time = time.time()

    project_id = state.get("project_id")

    if not project_id:
        return {
            "knowledge": KnowledgeAnswer(
                answer="No project_id was provided to retrieve project context.",
                sources=[],
                sufficient_context=False,
            ),
            "repo_context": [],
            "rag_status": "RAG_INSUFFICIENT_CONTEXT",
        }

    import os
    from pathlib import Path
    from backend.indexer.scanner import scan_repository
    from backend.indexer.ast_chunker import chunk_file
    from backend.indexer.retriever import SimpleBM25Index, HybridRetriever
    from backend.indexer.models import CodeChunk
    from backend.rag.evaluator import RetrievalEvaluator, QueryRewriter
    from backend.schemas.rag import RAGTelemetry, RAGStatus

    sparse_candidates = []
    project_path = Path("workspace") / project_id
    if not project_path.exists():
        project_path = Path(os.getcwd()) / "workspace" / project_id

    all_chunks = []
    if project_path.exists():
        try:
            scanned_files = scan_repository(str(project_path))
            for sf in scanned_files:
                chunks = chunk_file(sf.absolute_path, sf.relative_path)
                all_chunks.extend(chunks)

            if all_chunks:
                bm25 = SimpleBM25Index(all_chunks)
                sparse_pairs = bm25.search(state["user_message"], top_n=20)
                sparse_candidates = [pair[0] for pair in sparse_pairs]
        except Exception as e:
            print(f"BM25 retrieval error: {e}")

    # Dense retrieval is run once here (k=20, for hybrid BM25+dense fusion
    # below) and its top results are handed to answer_from_project instead
    # of it loading the same FAISS index and re-running the same query a
    # second time. `docs` stays None (not just empty) if this attempt
    # raises before assignment, so answer_from_project can tell "retrieval
    # genuinely failed" apart from "retrieval succeeded with zero matches"
    # and fall back to its own retrieval/error-handling accordingly.
    docs = None
    dense_candidates = []
    try:
        from backend.rag.retriever import load_project_index
        vector_store = load_project_index(project_id)
        docs = vector_store.similarity_search(state["user_message"], k=20)
        for doc in docs:
            dense_candidates.append(
                CodeChunk(
                    file_path=doc.metadata.get("file") or doc.metadata.get("source", "unknown"),
                    chunk_type=doc.metadata.get("symbol_type") or "dense",
                    symbol_name=doc.metadata.get("symbol"),
                    content=doc.page_content,
                    start_line=doc.metadata.get("line_start") or doc.metadata.get("start_line", 1),
                    end_line=doc.metadata.get("line_end") or doc.metadata.get("end_line", 1),
                    docstring=None,
                    decorators=[],
                    module=doc.metadata.get("module"),
                    symbol_type=doc.metadata.get("symbol_type"),
                    source_hash=doc.metadata.get("source_hash"),
                )
            )
    except Exception as e:
        print(f"Vector search retrieval error: {e}")

    try:
        knowledge = answer_from_project(
            project_id=project_id,
            question=state["user_message"],
            documents=docs,
        )
    except Exception as e:
        knowledge = KnowledgeAnswer(
            answer=f"Vector store not indexed yet: {e}",
            sources=[],
            sufficient_context=False,
        )

    retriever = HybridRetriever()
    search_results = retriever.retrieve(
        sparse_candidates=sparse_candidates,
        dense_candidates=dense_candidates,
        top_k=4,
    )
    repo_context = [res.chunk for res in search_results]
    scores = [res.score for res in search_results]

    current_query = state["user_message"]
    eval_result = RetrievalEvaluator.evaluate(
        issue_text=state["user_message"],
        query=current_query,
        chunks=repo_context,
        scores=scores,
    )

    query_rewrite = None
    attempt = 1

    # Bounded query rewriting if retrieval evaluation is INSUFFICIENT
    if eval_result.status == "INSUFFICIENT":
        rewritten = QueryRewriter.rewrite(
            issue_text=state["user_message"],
            current_query=current_query,
            missing_context=eval_result.missing_context,
            attempt=attempt,
        )
        if rewritten and rewritten != current_query:
            query_rewrite = rewritten
            attempt += 1
            if all_chunks:
                try:
                    bm25 = SimpleBM25Index(all_chunks)
                    sparse_pairs = bm25.search(rewritten, top_n=20)
                    sparse_candidates = [pair[0] for pair in sparse_pairs]
                except Exception:
                    pass

            try:
                if 'vector_store' in locals():
                    docs = vector_store.similarity_search(rewritten, k=20)
                    dense_candidates = [
                        CodeChunk(
                            file_path=doc.metadata.get("file") or doc.metadata.get("source", "unknown"),
                            chunk_type=doc.metadata.get("symbol_type") or "dense",
                            symbol_name=doc.metadata.get("symbol"),
                            content=doc.page_content,
                            start_line=doc.metadata.get("line_start") or doc.metadata.get("start_line", 1),
                            end_line=doc.metadata.get("line_end") or doc.metadata.get("end_line", 1),
                            docstring=None,
                            decorators=[],
                        ) for doc in docs
                    ]
            except Exception:
                pass

            new_search_results = retriever.retrieve(
                sparse_candidates=sparse_candidates,
                dense_candidates=dense_candidates,
                top_k=4,
            )
            if new_search_results:
                repo_context = [res.chunk for res in new_search_results]
                scores = [res.score for res in new_search_results]
                eval_result = RetrievalEvaluator.evaluate(
                    issue_text=state["user_message"],
                    query=rewritten,
                    chunks=repo_context,
                    scores=scores,
                )

    rag_status = (
        RAGStatus.RAG_INSUFFICIENT_CONTEXT.value
        if eval_result.status == "INSUFFICIENT"
        else RAGStatus.RAG_RETRIEVAL_SUCCESS.value
    )

    duration = round(time.time() - start_time, 3)
    telemetry = RAGTelemetry(
        retrieval_query=state["user_message"],
        retrieval_attempt=attempt,
        documents_retrieved=len(sparse_candidates) + len(dense_candidates),
        scores=[round(s, 4) for s in scores],
        selected_documents=[c.file_path for c in repo_context],
        evaluation_status=eval_result.status,
        evaluation_confidence=eval_result.confidence,
        query_rewrite=query_rewrite,
        missing_context=eval_result.missing_context,
        retrieval_duration_seconds=duration,
    )

    is_sufficient = eval_result.status != "INSUFFICIENT"
    knowledge = KnowledgeAnswer(
        answer=knowledge.answer,
        sources=list(dict.fromkeys(knowledge.sources + [c.file_path for c in repo_context])),
        sufficient_context=is_sufficient,
    )

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.RAG_COMPLETED,
            metadata={"rag_status": rag_status, "docs_count": len(repo_context)},
        )

    return {
        "knowledge": knowledge,
        "repo_context": repo_context,
        "rag_evaluation": eval_result,
        "rag_telemetry": telemetry,
        "rag_status": rag_status,
    }


def developer_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "developer")
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

        # Resolved only when a real project_id is present, matching the
        # self-scan gate below - never inferred/defaulted, so consolidation
        # can't accidentally read a different project's workspace when
        # repo_context was supplied without a project_id.
        project_path = None
        if state.get("project_id"):
            import os
            from pathlib import Path
            project_path = Path("workspace") / state["project_id"]
            if not project_path.exists():
                project_path = Path(os.getcwd()) / "workspace" / state["project_id"]

        if not repo_context and project_path is not None and project_path.exists():
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
            # Give the exact-snippet patch-generation prompt below one
            # authoritative, contiguous view of small non-Python files
            # (README.md, config, etc.) instead of fallback_chunk()'s
            # overlapping 100-line fragments - the LLM was generating
            # original_code_snippet anchors from fragmented/duplicated
            # context that didn't byte-match the real file. RAG/indexing
            # (chunk_file/fallback_chunk themselves) are untouched.
            if project_path is not None and project_path.exists():
                try:
                    from backend.indexer.ast_chunker import build_patch_context_chunks
                    repo_context = build_patch_context_chunks(repo_context, str(project_path))
                except Exception as e:
                    print(f"Patch context consolidation notice: {e}")
            context_str = format_categorized_context(repo_context)
            rag_eval = state.get("rag_evaluation")
            context_warning = ""
            if rag_eval and getattr(rag_eval, "status", None) == "INSUFFICIENT":
                context_warning = "\nWARNING: Repository context is evaluated as INSUFFICIENT. Do not invent non-existent files or functions. Only modify verified code.\n"

            prompt = f"""
You are the Developer Agent Patch Generator.
Your task is to generate precise and safe code patches for the user request and execution plan.
{context_warning}
USER REQUEST:
{state["user_message"]}

EXECUTION PLAN:
{plan.model_dump_json(indent=2)}

STRUCTURED REPOSITORY CONTEXT:
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


    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.DEVELOPMENT_COMPLETED,
            metadata={"patches_count": len(generated_patches)},
        )

    return {
        "developer_result": developer_result,
        "plan": plan,
        "generated_patches": generated_patches,
        "metrics": merge_usage(state.get("metrics"), usage),
    }


def qa_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "qa")
    plan = state.get("plan")
    if plan is None:
        plan = ExecutionPlan(
            goal=state["user_message"],
            steps=[],
            success_criteria="Complete requested task directly.",
        )

    # 1. LLM Semantic Review
    llm_qa_result = review_code_changes(
        user_request=state["user_message"],
        plan=plan,
        developer_result=state["developer_result"],
    )

    project_id = state.get("project_id", "test_project")
    import os
    from pathlib import Path
    from backend.qa.pipeline import QualityPipeline
    from backend.qa.judge import StructuredQAJudge

    project_path = Path("workspace") / project_id
    if not project_path.exists():
        project_path = Path(os.getcwd()) / "workspace" / project_id

    patches = state.get("generated_patches") or []

    # 2. Execute Multi-Check Quality Pipeline (AST, pytest, security, lint, typecheck)
    run_id = state.get("run_id")
    org_id = state.get("organization_id")
    cancel_check = None
    if run_id:
        from backend.observability.store import telemetry_store as _telemetry_store
        cancel_check = lambda: _telemetry_store.is_cancel_requested(run_id, org_id)
    checks, test_result = QualityPipeline.run_all(
        repo_path=str(project_path),
        patches=patches,
        timeout=30.0,
        cancel_check=cancel_check,
    )

    # 3. Structured QA Judge Evaluation with Strict Objective Priority
    qa_result = StructuredQAJudge.evaluate(
        checks=checks,
        test_result=test_result,
        patches=patches,
        llm_qa_result=llm_qa_result,
    )

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.QA_COMPLETED,
            metadata={
                "qa_status": getattr(qa_result, "status", None),
                "summary": (getattr(qa_result, "summary", "") or "")[:100],
            },
        )

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
    check_cancelled(state)
    mark_activity(state, "revision")
    import time
    start_rev = time.time()

    from backend.agents.developer import revise_code_changes
    from backend.agents.revision import generate_revision_patches
    from backend.revision.models import RevisionAttempt, RevisionHistory
    from backend.revision.analyzer import ErrorTraceAnalyzer
    from backend.schemas.developer import DeveloperResult
    from backend.vcs.git_manager import GitWorkspaceManager
    from backend.core.config import LLM_MODEL_NAME, LLM_PROVIDER

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

    # 7. Record this revision attempt with structured telemetry
    new_revision_count = revision_count + 1
    applied_patch = generated_patches[0] if generated_patches else None
    diagnosis_summary = analysis.diagnosis or (
        qa_result.summary if qa_result else "Revision performed based on test feedback."
    )

    # Identify failed check & category
    failed_check = None
    failure_category = getattr(qa_result, "failure_category", None) or "TEST_FAILURE"
    if qa_result and getattr(qa_result, "checks", None):
        failed_obj = next((c for c in qa_result.checks if c.status == "FAIL"), None)
        if failed_obj:
            failed_check = failed_obj.name

    existing_patch = state.get("generated_patches", [None])[0] if state.get("generated_patches") else None
    previous_patch_hash = state.get("patch_hash") or (
        state.get("git_diff").patch_hash if state.get("git_diff") else None
    ) or (
        GitWorkspaceManager.compute_patch_hash(existing_patch.updated_code_snippet)
        if existing_patch and existing_patch.updated_code_snippet else None
    )

    new_patch_hash = None
    if applied_patch and applied_patch.updated_code_snippet:
        new_patch_hash = GitWorkspaceManager.compute_patch_hash(applied_patch.updated_code_snippet)

    attempt = RevisionAttempt(
        attempt_number=new_revision_count,
        failing_tests=analysis.failing_tests,
        error_traceback=analysis.error_traceback,
        applied_patch=applied_patch,
        diagnosis=diagnosis_summary,
        failed_check=failed_check,
        failure_category=failure_category,
        previous_patch_hash=previous_patch_hash,
        new_patch_hash=new_patch_hash,
        model=LLM_MODEL_NAME or LLM_PROVIDER,
        provider=LLM_PROVIDER,
        duration_seconds=round(time.time() - start_rev, 2),
    )
    revision_history.add_attempt(attempt)

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.REVISION_COMPLETED,
            metadata={"revision_count": new_revision_count, "failure_category": failure_category},
        )

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
    check_cancelled(state)
    mark_activity(state, "git_prepare")
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
        empty_hash = GitWorkspaceManager.compute_patch_hash("")
        return {
            "git_diff": GitDiffSummary(
                branch_name=branch_name,
                files_changed=[],
                lines_added=0,
                lines_deleted=0,
                unified_diff="",
                patch_hash=empty_hash,
                risk_score="LOW",
                risk_reasons=["No file patches to apply."],
            ),
            "patch_hash": empty_hash,
        }

    try:
        diff_summary = GitWorkspaceManager.prepare_diff_summary(
            repo_path=str(project_path),
            patches=patches,
            task_id=task_id,
        )
    except Exception as e:
        branch_name = GitWorkspaceManager.generate_branch_name(task_id)
        empty_hash = GitWorkspaceManager.compute_patch_hash("")
        diff_summary = GitDiffSummary(
            branch_name=branch_name,
            files_changed=[p.file_path for p in patches],
            lines_added=0,
            lines_deleted=0,
            unified_diff="",
            patch_hash=empty_hash,
            risk_score="MEDIUM",
            risk_reasons=[f"Diff preparation encountered an error: {e}"],
        )

    return {
        "git_diff": diff_summary,
        "patch_hash": diff_summary.patch_hash,
    }


def policy_node(state: AgentState) -> dict:
    """
    Evaluates organization policy on staged changes and test results
    prior to Human-in-the-Loop approval gate and Git mutation.
    """
    check_cancelled(state)
    mark_activity(state, "policy")
    from backend.policy.evaluator import PolicyEvaluator
    from backend.schemas.policy import PolicyConfig, PolicyDecision

    policy_config = state.get("policy_config")
    if policy_config is None:
        policy_config = PolicyConfig()

    git_diff = state.get("git_diff")
    qa_result = state.get("qa_result")
    project_id = state.get("project_id")

    changed_files = git_diff.files_changed if git_diff else []
    lines_added = git_diff.lines_added if git_diff else 0
    lines_deleted = git_diff.lines_deleted if git_diff else 0
    risk_score = git_diff.risk_score if git_diff else "LOW"
    risk_reasons = git_diff.risk_reasons if git_diff else []
    branch_name = git_diff.branch_name if git_diff else None

    result = PolicyEvaluator.evaluate(
        policy=policy_config,
        repository=project_id,
        branch=branch_name,
        changed_files=changed_files,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        risk_score=risk_score,
        risk_reasons=risk_reasons,
        quality_results=qa_result,
        revision_count=state.get("revision_count", 0),
    )

    failed_checks = []
    if qa_result and getattr(qa_result, "checks", None):
        failed_checks = [c.name for c in qa_result.checks if c.status == "FAIL"]

    telemetry = PolicyEvaluator.create_telemetry(
        result=result,
        risk_score=risk_score,
        repository=project_id,
        branch=branch_name,
        changed_files=changed_files,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        required_checks=policy_config.required_quality_checks,
        failed_checks=failed_checks,
    )

    approval_status = state.get("approval_status")
    if result.decision == PolicyDecision.BLOCK:
        approval_status = "POLICY_BLOCKED"

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.POLICY_EVALUATED,
            metadata={
                "decision": getattr(result.decision, "value", str(result.decision)),
                "risk_score": str(risk_score),
            },
        )

    return {
        "policy_result": result,
        "policy_telemetry": telemetry,
        "approval_status": approval_status,
    }


def route_after_policy(state: AgentState) -> str:
    """
    Routes after policy evaluation:
    - If policy decision is BLOCK, route immediately to cleanup (prevents Git mutation).
    - If policy decision is ALLOW or REVIEW, route to approval gate.
    """
    from backend.schemas.policy import PolicyDecision

    policy_result = state.get("policy_result")
    if policy_result and policy_result.decision == PolicyDecision.BLOCK:
        return "cleanup"

    return "approval"


def approval_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop approval gate. Pauses execution via LangGraph interrupt()
    presenting the Git diff summary, cryptographic patch_hash, policy evaluation,
    and risk assessment, then resumes with an ApprovalDecision bound to the patch hash.
    """
    check_cancelled(state)
    mark_activity(state, "approval")
    from backend.vcs.models import ApprovalDecision

    git_diff = state.get("git_diff")
    developer_result = state.get("developer_result")
    policy_result = state.get("policy_result")
    patch_hash = git_diff.patch_hash if git_diff else ""

    interrupt_payload = {
        "task": "approval_required",
        "diff": git_diff.model_dump() if git_diff else None,
        "patch_hash": patch_hash,
        "developer_result": developer_result.model_dump() if developer_result else None,
        "policy": policy_result.model_dump() if policy_result else None,
        "policy_decision": policy_result.decision.value if policy_result else None,
        "message": "Review the proposed code changes, diff, policy compliance, and risk assessment.",
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

    # Cryptographic Approval Integrity Check:
    # If the approval decision explicitly specifies a patch_hash and it does not match
    # the staged diff's hash, mark approval invalid and reject publication.
    if approval.approved and approval.patch_hash and patch_hash:
        if approval.patch_hash != patch_hash:
            approval = ApprovalDecision(
                approved=False,
                reviewer=approval.reviewer,
                rejection_reason=(
                    f"PATCH_HASH_MISMATCH: Approved diff hash '{approval.patch_hash}' "
                    f"does not match staged diff hash '{patch_hash}'."
                ),
                patch_hash=approval.patch_hash,
                timestamp=approval.timestamp,
                reviewer_role=approval.reviewer_role,
                user_id=approval.user_id,
            )
            return {
                "approval": approval,
                "approval_status": "PATCH_HASH_MISMATCH",
            }

    # RBAC Authorization Check:
    if approval.approved and approval.reviewer_role:
        from backend.schemas.tenant import Role
        from backend.security.rbac import can_approve_changes

        try:
            role = Role(approval.reviewer_role)
        except Exception:
            role = None

        if role is not None:
            risk_val = None
            if policy_result and hasattr(policy_result, "risk_score"):
                risk_val = float(policy_result.risk_score)
            elif git_diff and git_diff.risk_score:
                if git_diff.risk_score.upper() == "HIGH":
                    risk_val = 80.0
                elif git_diff.risk_score.upper() == "MEDIUM":
                    risk_val = 50.0
                else:
                    risk_val = 20.0

            authorized, auth_reason = can_approve_changes(
                role=role,
                risk_score=risk_val,
                is_security_sensitive=bool(
                    policy_result and getattr(policy_result, "decision", None) == "REQUIRE_APPROVAL"
                ),
            )
            if not authorized:
                approval = ApprovalDecision(
                    approved=False,
                    reviewer=approval.reviewer,
                    rejection_reason=f"APPROVAL_UNAUTHORIZED: {auth_reason}",
                    patch_hash=approval.patch_hash,
                    timestamp=approval.timestamp,
                    reviewer_role=approval.reviewer_role,
                    user_id=approval.user_id,
                )
                return {
                    "approval": approval,
                    "approval_status": "APPROVAL_UNAUTHORIZED",
                }

    status_label = "APPROVED" if approval.approved else "REJECTED"
    return {
        "approval": approval,
        "approval_status": status_label,
    }


def route_after_approval(state: AgentState) -> str:
    """
    Routes after human approval decision:
    - approved -> git_commit (provided policy decision is not BLOCK)
    - rejected or policy blocked -> cleanup
    """
    policy_result = state.get("policy_result")
    if policy_result and getattr(policy_result, "decision", None) == "BLOCK":
        return "cleanup"

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

    Cryptographically validates workspace drift prior to commit.
    """
    check_cancelled(state)
    mark_activity(state, "git_commit")
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

    # Pre-commit drift and tamper verification:
    # Ensure working tree diff still matches the approved patch_hash
    if git_diff.patch_hash:
        is_valid, drift_err = GitWorkspaceManager.verify_workspace_drift(
            repo_path=str(project_path),
            expected_diff=git_diff.unified_diff,
            expected_hash=git_diff.patch_hash,
            files_changed=git_diff.files_changed,
        )
        if not is_valid:
            return {
                "approval_status": "PATCH_HASH_MISMATCH",
            }

    commit_message = f"agent: apply approved changes for {project_id}"
    developer_result = state.get("developer_result")
    if developer_result:
        commit_message = f"agent: {developer_result.summary[:80]}"

    # Final check immediately before the irreversible mutation: a
    # cancellation that arrived while drift verification ran above must
    # still stop the commit, not just be noticed at the top of the node.
    check_cancelled(state)

    if git_diff.branch_name:
        GitWorkspaceManager.create_feature_branch(str(project_path), git_diff.branch_name)

    success = GitWorkspaceManager.stage_and_commit(
        repo_path=str(project_path),
        message=commit_message,
    )

    status_val = "COMMITTED" if success else "COMMIT_FAILED"
    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.COMMIT_COMPLETED,
            metadata={"status": status_val, "branch": git_diff.branch_name},
        )

    return {
        "approval_status": status_val,
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