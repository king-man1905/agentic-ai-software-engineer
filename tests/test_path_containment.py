"""
P0-1 regression tests: LLM-controlled FilePatch.file_path must never be
able to escape the workspace/project root it's supposed to be confined to,
at any of the four read/write sites that join a root path with it.

Covers:
- backend.policy.path_filter.safe_repo_relative_path (the shared primitive)
- backend.graph.nodes.developer_node (exact-snippet write path)
- backend.agents.revision.generate_revision_patches
- backend.qa.pipeline.QualityPipeline.check_ast
- backend.vcs.git_manager.GitWorkspaceManager.apply_patches
"""

import os
import subprocess
from pathlib import Path

import pytest

from backend.developer.models import FilePatch
from backend.policy.path_filter import safe_repo_relative_path
from backend.vcs.git_manager import GitWorkspaceManager


UNSAFE_PATHS = [
    "../../../evil.py",
    "../../../../tmp/evil.py",
    "/etc/passwd",
    "..\\..\\evil.py",
    "C:\\Windows\\System32\\evil.py",
    "C:/Windows/System32/evil.py",
]


class TestSafeRepoRelativePath:
    """Direct unit tests of the shared containment primitive."""

    def test_rejects_unsafe_paths(self, tmp_path):
        root = tmp_path / "workspace_root"
        root.mkdir()
        for unsafe in UNSAFE_PATHS:
            assert safe_repo_relative_path(root, unsafe) is None, (
                f"expected {unsafe!r} to be rejected"
            )

    def test_rejects_empty_and_none_like_input(self, tmp_path):
        root = tmp_path / "workspace_root"
        root.mkdir()
        assert safe_repo_relative_path(root, "") is None
        assert safe_repo_relative_path(root, "   ") is None

    def test_accepts_legitimate_nested_path(self, tmp_path):
        root = tmp_path / "workspace_root"
        root.mkdir()
        target = safe_repo_relative_path(root, "src/sub/module.py")
        assert target is not None
        assert target == (root / "src" / "sub" / "module.py").resolve()
        assert root.resolve() in target.parents

    def test_accepts_simple_top_level_file(self, tmp_path):
        root = tmp_path / "workspace_root"
        root.mkdir()
        target = safe_repo_relative_path(root, "README.md")
        assert target is not None
        assert target == (root / "README.md").resolve()

    @pytest.mark.skipif(os.name == "nt", reason="os.symlink requires elevated privileges on Windows")
    def test_rejects_symlink_that_resolves_outside_root(self, tmp_path):
        """A file_path targeting a symlink *tracked inside the repo* that
        points outside the workspace root must be rejected - the whole
        reason containment uses .resolve() (which follows symlinks)
        instead of a naive string-prefix check."""
        root = tmp_path / "workspace_root"
        root.mkdir()
        outside = tmp_path / "outside_secret.txt"
        outside.write_text("should never be reachable\n", encoding="utf-8")

        link = root / "innocuous_link.txt"
        link.symlink_to(outside)

        assert safe_repo_relative_path(root, "innocuous_link.txt") is None

    @pytest.mark.skipif(os.name == "nt", reason="os.symlink requires elevated privileges on Windows")
    def test_rejects_path_through_symlinked_directory(self, tmp_path):
        root = tmp_path / "workspace_root"
        root.mkdir()
        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()

        link_dir = root / "linked_dir"
        link_dir.symlink_to(outside_dir, target_is_directory=True)

        assert safe_repo_relative_path(root, "linked_dir/new_file.py") is None


class TestCheckAstContainment:
    """check_ast (backend/qa/pipeline.py) must never read outside repo_path."""

    def test_traversal_path_is_a_preflight_failure_not_a_read(self, tmp_path):
        from backend.qa.pipeline import QualityPipeline

        repo = tmp_path / "repo"
        repo.mkdir()
        outside = tmp_path / "outside_secret.py"
        outside.write_text("SECRET = 'do-not-read-me'\n", encoding="utf-8")

        patch = FilePatch(
            file_path="../outside_secret.py",
            original_code_snippet="",
            updated_code_snippet="x = 1\n",
            explanation="malicious traversal attempt",
        )
        check = QualityPipeline.check_ast(str(repo), [patch])
        assert check.status == "FAIL"
        assert "not a safe repository-relative path" in (check.stderr_summary or "")


class TestApplyPatchesContainment:
    """GitWorkspaceManager.apply_patches (backend/vcs/git_manager.py) must
    never write or create directories outside repo_path."""

    def test_traversal_path_raises_and_writes_nothing(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "existing.py").write_text("x = 1\n", encoding="utf-8")

        patch = FilePatch(
            file_path="../../outside/evil.py",
            original_code_snippet="",
            updated_code_snippet="import os\nos.system('rm -rf /')\n",
            explanation="malicious traversal attempt",
        )

        with pytest.raises(ValueError, match="not a safe repository-relative path"):
            GitWorkspaceManager.apply_patches(str(repo), [patch])

        # Nothing was created outside tmp_path's repo directory or anywhere else.
        assert not (tmp_path / "outside").exists()
        assert list(repo.iterdir()) == [repo / "existing.py"]

    def test_legitimate_nested_new_file_still_creates_parent_dirs(self, tmp_path):
        """Confirms the fix doesn't regress the legitimate case: a safe,
        newly-nested file_path still has its parent directories created
        and the file written, exactly as before."""
        repo = tmp_path / "repo"
        repo.mkdir()

        patch = FilePatch(
            file_path="src/new/module.py",
            original_code_snippet="",
            updated_code_snippet="def run():\n    return True\n",
            explanation="legitimate new nested file",
        )
        results = GitWorkspaceManager.apply_patches(str(repo), [patch])
        assert "src/new/module.py" in results
        written = repo / "src" / "new" / "module.py"
        assert written.exists()
        assert written.read_text(encoding="utf-8") == "def run():\n    return True\n"


class TestGenerateRevisionPatchesContainment:
    """generate_revision_patches (backend/agents/revision.py) must never
    read outside project_path."""

    def test_traversal_path_raises(self, tmp_path, monkeypatch):
        from backend.agents.revision import generate_revision_patches
        from backend.revision.models import ErrorTraceAnalysis
        from backend.schemas.planning import ExecutionPlan
        from backend.indexer.models import CodeChunk

        project_id = "revision-traversal-test"
        monkeypatch.chdir(tmp_path)
        (tmp_path / "workspace" / project_id).mkdir(parents=True)

        malicious_patch = FilePatch(
            file_path="../../evil.py",
            original_code_snippet="",
            updated_code_snippet="import os\nos.system('whoami')\n",
            explanation="malicious traversal attempt",
        )

        class _FakePatchResult:
            patches = [malicious_patch]

        monkeypatch.setattr(
            "backend.agents.revision.invoke_structured",
            lambda llm, schema_cls, prompt: _FakePatchResult(),
        )
        monkeypatch.setattr("backend.agents.revision.get_llm", lambda: object())

        with pytest.raises(ValueError, match="not a safe repository-relative path"):
            generate_revision_patches(
                user_request="fix the bug",
                plan=ExecutionPlan(goal="fix", steps=[], success_criteria="done"),
                error_analysis=ErrorTraceAnalysis(
                    failing_tests=[], error_traceback="", error_type="", root_cause_hint=""
                ),
                repo_context=[
                    CodeChunk(
                        file_path="main.py",
                        content="x = 1",
                        start_line=1,
                        end_line=1,
                        chunk_type="module",
                    )
                ],
                project_id=project_id,
            )

        assert not (tmp_path / "evil.py").exists()


class TestDeveloperNodeExactSnippetContainment:
    """developer_node's exact-snippet write path (backend/graph/nodes.py)
    is the most directly LLM-driven of the four sites - the patch's
    file_path comes straight from structured LLM output."""

    def _init_git_workspace(self, tmp_path, monkeypatch, project_id: str):
        # Namespaced under "default-org" to match state.get("organization_id",
        # "default-org") - the resolver P0-3 introduced (resolve_workspace_path).
        workspace_dir = tmp_path / "workspace" / "default-org" / project_id
        workspace_dir.mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(workspace_dir), capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=str(workspace_dir), capture_output=True, text=True
        )
        (workspace_dir / "README.md").write_text("# Test\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"], cwd=str(workspace_dir), capture_output=True, text=True, check=True
        )
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
        return workspace_dir

    def test_malicious_file_path_from_llm_is_rejected(self, tmp_path, monkeypatch):
        from backend.graph.nodes import developer_node
        from backend.indexer.models import CodeChunk
        from backend.schemas.developer import DeveloperResult

        project_id = "developer-traversal-test"
        self._init_git_workspace(tmp_path, monkeypatch, project_id)

        malicious_patch = FilePatch(
            file_path="../../../evil.py",
            original_code_snippet="",
            updated_code_snippet="import os\nos.system('whoami')\n",
            explanation="malicious traversal attempt",
        )

        class _FakePatchResult:
            patches = [malicious_patch]

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
            "user_message": "do something",
            "project_id": project_id,
            "repo_context": [
                CodeChunk(
                    file_path="README.md",
                    content="# Test\n",
                    start_line=1,
                    end_line=1,
                    chunk_type="module",
                )
            ],
        }

        with pytest.raises(ValueError, match="not a safe repository-relative path"):
            developer_node(state)

        # The traversal target must never have been created anywhere.
        assert not (tmp_path / "evil.py").exists()
        assert not (tmp_path / "workspace" / "evil.py").exists()

    def test_legitimate_nested_file_path_is_written_inside_workspace(self, tmp_path, monkeypatch):
        from backend.graph.nodes import developer_node
        from backend.indexer.models import CodeChunk
        from backend.schemas.developer import DeveloperResult

        project_id = "developer-legit-nested-test"
        workspace_dir = self._init_git_workspace(tmp_path, monkeypatch, project_id)

        legit_patch = FilePatch(
            file_path="src/new/module.py",
            original_code_snippet="",
            updated_code_snippet="def run():\n    return True\n",
            explanation="legitimate new nested file",
        )

        class _FakePatchResult:
            patches = [legit_patch]

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
            "user_message": "add a module",
            "project_id": project_id,
            "repo_context": [
                CodeChunk(
                    file_path="README.md",
                    content="# Test\n",
                    start_line=1,
                    end_line=1,
                    chunk_type="module",
                )
            ],
        }

        output = developer_node(state)
        assert len(output["generated_patches"]) == 1
        written = workspace_dir / "src" / "new" / "module.py"
        assert written.exists()
        assert written.read_text(encoding="utf-8") == "def run():\n    return True\n"
