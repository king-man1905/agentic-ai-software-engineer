"""
Regression tests for the confirmed production E2E developer-node failure:

Two real EC2 runs against king-man1905/agentic-ai-test-repo (whose real
README.md is exactly "# agentic-ai-test-repo" with no trailing newline)
both failed identically at developer_node with:
    AST pre-flight validation failed for README.md:
    ['Target original snippet not found in source file: README.md']

Root cause: developer_node's exact-snippet patch loop treated ANY
SafePatcher.apply_patch() failure - including a plain anchor/snippet
mismatch on a non-Python file, which is neither a security violation nor
a syntax/AST problem - as an unconditional hard-fail ValueError. This
happened during developer_node's own pre-flight validation, before
qa_node/revision_node ever ran, so the existing bounded self-correction
loop never got a chance to regenerate the patch.

Fix: developer_node now classifies this one specific case (SafePatcher
returns is_valid=False, and the failing patch's file is NOT a .py file -
mirroring QualityPipeline.check_ast's own existing distinction in
backend/qa/pipeline.py) as FailureCategory.PATCH_APPLICATION_FAILURE,
builds a real QAResult for it via the existing StructuredQAJudge, and a
new route_after_developer conditional edge sends that one case to the
existing bounded revision loop instead of qa_node. Every other developer
pre-flight failure (unsafe workspace path, unsafe repo-relative path, a
genuine .py syntax/AST error) is untouched and still raises immediately.
"""

import os
import subprocess

from backend.developer.models import FilePatch
from backend.graph.nodes import developer_node, route_after_developer, MAX_REVISIONS
from backend.schemas.developer import DeveloperResult
from backend.schemas.qa import FailureCategory, QAResult
from backend.indexer.models import CodeChunk


def _real_git_workspace_with_file(tmp_path, monkeypatch, project_id, file_name, content):
    workspace_dir = tmp_path / "workspace" / "default-org" / project_id
    workspace_dir.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(workspace_dir), capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(workspace_dir), capture_output=True, text=True)
    (workspace_dir / file_name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    return workspace_dir


def _run_developer_node_with_patch(tmp_path, monkeypatch, project_id, file_name, real_content, patch):
    class _FakePatchResult:
        patches = [patch]

    _real_git_workspace_with_file(tmp_path, monkeypatch, project_id, file_name, real_content)

    monkeypatch.setattr(
        "backend.graph.nodes.generate_code_changes",
        lambda user_request, plan, knowledge: DeveloperResult(
            summary="stub", changes=[], requires_testing=True, notes=[]
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.invoke_structured",
        lambda llm, schema_cls, prompt, *a, **k: _FakePatchResult(),
    )

    state = {
        "user_message": f"Update {file_name}",
        "project_id": project_id,
        "repo_context": [
            CodeChunk(
                file_path=file_name,
                content=real_content,
                start_line=1,
                end_line=max(len(real_content.splitlines()), 1),
                chunk_type="module",
            )
        ],
    }
    return developer_node(state)


# The exact confirmed production repository content: a single line, no
# trailing newline.
_REAL_README_CONTENT = "# agentic-ai-test-repo"

# An anchor that does NOT byte-match the real file (e.g. the LLM guessed a
# trailing newline that isn't actually there) - reproduces the exact
# confirmed failure.
_MISMATCHED_README_PATCH = FilePatch(
    file_path="README.md",
    original_code_snippet="# agentic-ai-test-repo\n",
    updated_code_snippet="# agentic-ai-test-repo\n\n## E2E Test\n\nVerifies the pipeline.\n",
    explanation="Add E2E Test section",
)


class TestDeveloperNodeRecoverablePatchApplicationFailure:
    def test_readme_anchor_mismatch_is_recoverable_not_raised(self, tmp_path, monkeypatch):
        """The exact confirmed production scenario must no longer raise -
        developer_node returns normally with no patches applied and a
        classified, revision-loop-consumable failure."""
        output = _run_developer_node_with_patch(
            tmp_path, monkeypatch, "e2e-repro", "README.md", _REAL_README_CONTENT, _MISMATCHED_README_PATCH,
        )

        assert output["generated_patches"] == []
        qa_result = output.get("qa_result")
        assert qa_result is not None
        assert qa_result.status == "FAIL"
        assert qa_result.failure_category == FailureCategory.PATCH_APPLICATION_FAILURE.value
        assert "README.md" in qa_result.summary
        assert "not found" in qa_result.summary.lower()

    def test_readme_was_never_written_to_disk(self, tmp_path, monkeypatch):
        """A failed patch must never be applied partially (or at all)."""
        _run_developer_node_with_patch(
            tmp_path, monkeypatch, "e2e-repro-2", "README.md", _REAL_README_CONTENT, _MISMATCHED_README_PATCH,
        )
        readme_path = tmp_path / "workspace" / "default-org" / "e2e-repro-2" / "README.md"
        assert readme_path.read_text(encoding="utf-8") == _REAL_README_CONTENT

    def test_recoverable_failure_routes_to_revision(self, tmp_path, monkeypatch):
        output = _run_developer_node_with_patch(
            tmp_path, monkeypatch, "e2e-repro-3", "README.md", _REAL_README_CONTENT, _MISMATCHED_README_PATCH,
        )
        state = {**output, "revision_count": 0}
        assert route_after_developer(state) == "revision"

    def test_successful_developer_run_still_routes_to_qa(self, tmp_path, monkeypatch):
        """An ordinary successful patch must be completely unaffected."""
        matching_patch = FilePatch(
            file_path="README.md",
            original_code_snippet="# agentic-ai-test-repo",
            updated_code_snippet="# agentic-ai-test-repo\n\n## E2E Test\n",
            explanation="Add E2E Test section",
        )
        output = _run_developer_node_with_patch(
            tmp_path, monkeypatch, "e2e-repro-4", "README.md", _REAL_README_CONTENT, matching_patch,
        )
        assert len(output["generated_patches"]) == 1
        assert "qa_result" not in output
        assert route_after_developer({**output, "revision_count": 0}) == "qa"


class TestGenuineHardFailuresUnaffected:
    """Requirement: security/unsafe-path/AST violations must still hard-fail
    immediately, unchanged."""

    def test_python_file_anchor_mismatch_still_hard_fails(self, tmp_path, monkeypatch):
        patch = FilePatch(
            file_path="main.py",
            original_code_snippet="def missing():\n    pass\n",
            updated_code_snippet="def missing():\n    return 1\n",
            explanation="edit",
        )
        try:
            _run_developer_node_with_patch(
                tmp_path, monkeypatch, "py-mismatch", "main.py", "def real():\n    pass\n", patch,
            )
            assert False, "expected a hard ValueError for a .py anchor mismatch"
        except ValueError as e:
            assert "AST pre-flight validation failed" in str(e)

    def test_unsafe_repo_relative_path_still_hard_fails(self, tmp_path, monkeypatch):
        patch = FilePatch(
            file_path="../../etc/passwd",
            original_code_snippet="root",
            updated_code_snippet="pwned",
            explanation="malicious",
        )
        try:
            _run_developer_node_with_patch(
                tmp_path, monkeypatch, "unsafe-path", "README.md", _REAL_README_CONTENT, patch,
            )
            assert False, "expected a hard ValueError for an unsafe repository-relative path"
        except ValueError as e:
            assert "not a safe repository-relative path" in str(e)


class TestRevisionNodeReceivesFailureFeedback:
    def test_revision_analyzes_the_recoverable_failure_summary(self, tmp_path, monkeypatch):
        from backend.graph.nodes import revision_node
        from backend.revision.analyzer import ErrorTraceAnalyzer

        output = _run_developer_node_with_patch(
            tmp_path, monkeypatch, "e2e-repro-5", "README.md", _REAL_README_CONTENT, _MISMATCHED_README_PATCH,
        )
        qa_result = output["qa_result"]
        analysis = ErrorTraceAnalyzer.analyze(qa_result.summary)
        assert "README.md" in analysis.error_traceback

        captured = {}

        def fake_generate_revision_patches(**kwargs):
            captured["error_analysis"] = kwargs["error_analysis"]
            return []

        monkeypatch.setattr("backend.agents.revision.generate_revision_patches", fake_generate_revision_patches)
        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda **kwargs: DeveloperResult(summary="revised", changes=[], requires_testing=True, notes=[]),
        )

        state = {
            "user_message": "Add E2E Test section",
            "project_id": "e2e-repro-5",
            "qa_result": qa_result,
            "generated_patches": [],
            "developer_result": DeveloperResult(summary="stub", changes=[], requires_testing=True, notes=[]),
            "repo_context": output["repo_context"],
            "revision_count": 0,
            "pre_patch_snapshots": output["pre_patch_snapshots"],
        }
        revision_node(state)
        assert "README.md" in captured["error_analysis"].error_traceback


class TestGraphWiring:
    def test_developer_has_conditional_edges_to_both_qa_and_revision(self):
        from backend.graph.runner import AgentRunner

        runner = AgentRunner()
        edges = {(e.source, e.target) for e in runner._graph.get_graph().edges}
        assert ("developer", "qa") in edges
        assert ("developer", "revision") in edges


class TestBoundedRetryPreserved:
    def test_route_after_developer_only_two_targets(self):
        """route_after_developer never needs a max_retries branch:
        developer_node runs exactly once per run (nothing loops back to
        it), so revision_count is always 0 - MAX_REVISIONS still applies
        normally afterwards, in qa_router, for every revision -> qa cycle."""
        qa_result = QAResult(
            status="FAIL",
            failure_category=FailureCategory.PATCH_APPLICATION_FAILURE.value,
            summary="Quality Gate FAILED: Proposed patch failed pre-flight validation.",
        )
        for count in (0, MAX_REVISIONS, MAX_REVISIONS + 5):
            assert route_after_developer({"qa_result": qa_result, "revision_count": count}) == "revision"

    def test_no_qa_result_routes_to_qa(self):
        assert route_after_developer({"revision_count": 0}) == "qa"

    def test_unrelated_fail_category_routes_to_qa(self):
        qa_result = QAResult(status="FAIL", failure_category=FailureCategory.TEST_FAILURE.value, summary="x")
        assert route_after_developer({"qa_result": qa_result, "revision_count": 0}) == "qa"


class TestNoOpPatchNotTreatedAsSuccess:
    def test_no_generated_patches_and_no_recoverable_failure_is_not_success(self, tmp_path, monkeypatch):
        """When the LLM returns zero patches and there is nothing to fall
        back on, generated_patches must stay empty - never silently
        reported as a successful, revision-worthy, or QA-worthy patch."""

        def fake_generate_code_changes(user_request, plan, knowledge):
            return DeveloperResult(summary="no-op", changes=[], requires_testing=False, notes=[])

        monkeypatch.setattr("backend.graph.nodes.generate_code_changes", fake_generate_code_changes)

        class _EmptyPatchResult:
            patches = []

        monkeypatch.setattr(
            "backend.graph.nodes.invoke_structured",
            lambda llm, schema_cls, prompt, *a, **k: _EmptyPatchResult(),
        )
        _real_git_workspace_with_file(tmp_path, monkeypatch, "no-op-repro", "README.md", _REAL_README_CONTENT)
        state = {
            "user_message": "Do nothing",
            "project_id": "no-op-repro",
            "repo_context": [
                CodeChunk(
                    file_path="README.md", content=_REAL_README_CONTENT,
                    start_line=1, end_line=1, chunk_type="module",
                )
            ],
        }
        output = developer_node(state)
        assert output["generated_patches"] == []
        assert "qa_result" not in output
        assert route_after_developer({**output, "revision_count": 0}) == "qa"
