import pytest
import os
import subprocess
from pathlib import Path

from backend.vcs.models import GitDiffSummary, ApprovalDecision
from backend.vcs.git_manager import GitWorkspaceManager
from backend.developer.models import FilePatch
from backend.graph.state import AgentState
from backend.graph.nodes import (
    git_prepare_node,
    approval_node,
    route_after_approval,
    git_commit_node,
    cleanup_node,
    qa_router,
)
from backend.schemas.qa import QAResult
from backend.schemas.developer import DeveloperResult


# ============================================================================
# 1. GIT DIFF MODEL TESTS
# ============================================================================

class TestGitDiffModels:
    def test_git_diff_summary_instantiation(self):
        diff = GitDiffSummary(
            branch_name="agent/task-abc12345",
            files_changed=["src/main.py", "src/utils.py"],
            lines_added=25,
            lines_deleted=10,
            unified_diff="--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-old\n+new",
            risk_score="LOW",
            risk_reasons=["Standard code change with low risk profile."],
        )
        assert diff.branch_name == "agent/task-abc12345"
        assert len(diff.files_changed) == 2
        assert diff.lines_added == 25
        assert diff.lines_deleted == 10
        assert diff.risk_score == "LOW"
        assert "--- a/src/main.py" in diff.unified_diff

    def test_git_diff_summary_defaults(self):
        diff = GitDiffSummary(branch_name="agent/task-x")
        assert diff.files_changed == []
        assert diff.lines_added == 0
        assert diff.lines_deleted == 0
        assert diff.unified_diff == ""
        assert diff.risk_score == "LOW"
        assert diff.risk_reasons == []

    def test_git_diff_summary_serialization(self):
        diff = GitDiffSummary(
            branch_name="agent/task-test1234",
            files_changed=["app.py"],
            lines_added=5,
            lines_deleted=2,
            unified_diff="+new line",
            risk_score="MEDIUM",
            risk_reasons=["Large changeset: 6 files modified"],
        )
        json_data = diff.model_dump_json()
        restored = GitDiffSummary.model_validate_json(json_data)
        assert restored.branch_name == "agent/task-test1234"
        assert restored.risk_score == "MEDIUM"
        assert restored.risk_reasons == ["Large changeset: 6 files modified"]

    def test_approval_decision_approved(self):
        decision = ApprovalDecision(approved=True, reviewer="alice")
        assert decision.approved is True
        assert decision.reviewer == "alice"
        assert decision.rejection_reason is None

    def test_approval_decision_rejected(self):
        decision = ApprovalDecision(
            approved=False,
            reviewer="bob",
            rejection_reason="Changes introduce a security vulnerability.",
        )
        assert decision.approved is False
        assert decision.rejection_reason == "Changes introduce a security vulnerability."

    def test_approval_decision_serialization(self):
        decision = ApprovalDecision(approved=True, reviewer="ci-bot")
        json_data = decision.model_dump_json()
        restored = ApprovalDecision.model_validate_json(json_data)
        assert restored.approved is True
        assert restored.reviewer == "ci-bot"


# ============================================================================
# 2. DIFF CALCULATION TESTS
# ============================================================================

class TestDiffCalculation:
    def test_single_file_diff_line_counts(self):
        file_changes = {
            "src/app.py": (
                "def hello():\n    return 'world'\n",
                "def hello():\n    return 'universe'\n    # updated\n",
            ),
        }
        unified_diff, added, deleted = GitWorkspaceManager.compute_diff(file_changes)

        assert added >= 1  # At least the new line
        assert deleted >= 1  # The replaced line
        assert "--- a/src/app.py" in unified_diff
        assert "+++ b/src/app.py" in unified_diff
        assert "+    return 'universe'" in unified_diff
        assert "-    return 'world'" in unified_diff

    def test_multi_file_diff(self):
        file_changes = {
            "a.py": ("line1\nline2\n", "line1\nline2\nline3\n"),
            "b.py": ("old\n", "new\n"),
        }
        unified_diff, added, deleted = GitWorkspaceManager.compute_diff(file_changes)
        assert added >= 2  # line3 in a.py + new in b.py
        assert deleted >= 1  # old in b.py
        assert "a/a.py" in unified_diff
        assert "a/b.py" in unified_diff

    def test_new_file_diff(self):
        file_changes = {
            "new_module.py": ("", "print('hello')\n"),
        }
        unified_diff, added, deleted = GitWorkspaceManager.compute_diff(file_changes)
        assert added >= 1
        assert deleted == 0
        assert "+print('hello')" in unified_diff

    def test_empty_diff_when_no_changes(self):
        file_changes = {
            "same.py": ("content\n", "content\n"),
        }
        unified_diff, added, deleted = GitWorkspaceManager.compute_diff(file_changes)
        assert added == 0
        assert deleted == 0
        assert unified_diff == ""


# ============================================================================
# 3. RISK EVALUATION TESTS
# ============================================================================

class TestRiskEvaluation:
    def test_low_risk_single_file_edit(self):
        score, reasons = GitWorkspaceManager.evaluate_risk(
            files_changed=["src/utils.py"],
            lines_added=5,
            lines_deleted=2,
            unified_diff="",
        )
        assert score == "LOW"
        assert len(reasons) >= 1

    def test_high_risk_config_file_yml(self):
        score, reasons = GitWorkspaceManager.evaluate_risk(
            files_changed=["config/settings.yml"],
            lines_added=1,
            lines_deleted=1,
            unified_diff="",
        )
        assert score == "HIGH"
        assert any("config" in r.lower() or "settings.yml" in r.lower() for r in reasons)

    def test_high_risk_env_file(self):
        score, reasons = GitWorkspaceManager.evaluate_risk(
            files_changed=[".env"],
            lines_added=1,
            lines_deleted=0,
            unified_diff="",
        )
        assert score == "HIGH"

    def test_high_risk_alembic_migration(self):
        score, reasons = GitWorkspaceManager.evaluate_risk(
            files_changed=["alembic/versions/001_init.py"],
            lines_added=20,
            lines_deleted=0,
            unified_diff="",
        )
        assert score == "HIGH"
        assert any("alembic" in r.lower() for r in reasons)

    def test_high_risk_requirements_txt(self):
        score, reasons = GitWorkspaceManager.evaluate_risk(
            files_changed=["requirements.txt"],
            lines_added=2,
            lines_deleted=1,
            unified_diff="",
        )
        assert score == "HIGH"

    def test_high_risk_deletion_ratio(self):
        score, reasons = GitWorkspaceManager.evaluate_risk(
            files_changed=["src/module.py"],
            lines_added=2,
            lines_deleted=10,
            unified_diff="",
        )
        assert score == "HIGH"
        assert any("deletion ratio" in r.lower() for r in reasons)

    def test_medium_risk_many_files(self):
        files = [f"src/module_{i}.py" for i in range(7)]
        score, reasons = GitWorkspaceManager.evaluate_risk(
            files_changed=files,
            lines_added=20,
            lines_deleted=5,
            unified_diff="",
        )
        assert score in ("MEDIUM", "HIGH")
        assert any("files" in r.lower() for r in reasons)


# ============================================================================
# 4. BRANCH NAMING TESTS
# ============================================================================

class TestBranchNaming:
    def test_standard_branch_name(self):
        name = GitWorkspaceManager.generate_branch_name("abc12345")
        assert name.startswith("agent/task-abc12345-")
        suffix = name.rsplit("-", 1)[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_long_task_id_truncated(self):
        name = GitWorkspaceManager.generate_branch_name("abcdefghijklmnop")
        assert name.startswith("agent/task-abcdefgh-")

    def test_short_task_id(self):
        name = GitWorkspaceManager.generate_branch_name("xy")
        assert name.startswith("agent/task-xy-")

    def test_special_chars_sanitized(self):
        name = GitWorkspaceManager.generate_branch_name("task 123!@")
        assert name.startswith("agent/task-")
        assert " " not in name
        assert "!" not in name
        assert "@" not in name

    def test_same_task_id_produces_unique_branch_names(self):
        # Regression test for the real collision: two runs sharing the
        # same 8-char task_id/project_id prefix used to generate the exact
        # same branch name and silently clobber each other's PR.
        first = GitWorkspaceManager.generate_branch_name("sandbox-ai-demo")
        second = GitWorkspaceManager.generate_branch_name("sandbox-ai-demo-v2")
        assert first != second

    def test_repeated_calls_with_identical_task_id_are_unique(self):
        names = {GitWorkspaceManager.generate_branch_name("same-project") for _ in range(20)}
        assert len(names) == 20


# ============================================================================
# 5. PATCH APPLICATION TESTS (tmp_path isolated)
# ============================================================================

class TestPatchApplication:
    def test_apply_patch_creates_and_writes_file(self, tmp_path):
        # Create a source file in the isolated workspace
        src_file = tmp_path / "src" / "app.py"
        src_file.parent.mkdir(parents=True)
        src_file.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

        patch = FilePatch(
            file_path="src/app.py",
            original_code_snippet="return 'hello'",
            updated_code_snippet="return 'hi there'",
            explanation="Update greeting",
        )

        results = GitWorkspaceManager.apply_patches(str(tmp_path), [patch])

        assert "src/app.py" in results
        original, patched = results["src/app.py"]
        assert "hello" in original
        assert "hi there" in patched

        # Verify file on disk was updated
        content = src_file.read_text(encoding="utf-8")
        assert "hi there" in content

    def test_apply_patch_new_file(self, tmp_path):
        patch = FilePatch(
            file_path="new_file.py",
            original_code_snippet="",
            updated_code_snippet="print('new')\n",
            explanation="Create new file",
        )
        results = GitWorkspaceManager.apply_patches(str(tmp_path), [patch])

        assert "new_file.py" in results
        new_file = tmp_path / "new_file.py"
        assert new_file.exists()
        assert "print('new')" in new_file.read_text(encoding="utf-8")

    def test_apply_patch_validation_failure_raises(self, tmp_path):
        src_file = tmp_path / "broken.py"
        src_file.write_text("x = 1\n", encoding="utf-8")

        patch = FilePatch(
            file_path="broken.py",
            original_code_snippet="x = 1",
            updated_code_snippet="def bad(:\n",
            explanation="Introduce syntax error",
        )

        with pytest.raises(ValueError, match="Patch validation failed"):
            GitWorkspaceManager.apply_patches(str(tmp_path), [patch])

    def test_apply_multiple_patches(self, tmp_path):
        (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("b = 2\n", encoding="utf-8")

        patches = [
            FilePatch(
                file_path="a.py",
                original_code_snippet="a = 1",
                updated_code_snippet="a = 10",
                explanation="Update a",
            ),
            FilePatch(
                file_path="b.py",
                original_code_snippet="b = 2",
                updated_code_snippet="b = 20",
                explanation="Update b",
            ),
        ]
        results = GitWorkspaceManager.apply_patches(str(tmp_path), [patches[0], patches[1]])
        assert len(results) == 2
        assert "a = 10" in (tmp_path / "a.py").read_text(encoding="utf-8")
        assert "b = 20" in (tmp_path / "b.py").read_text(encoding="utf-8")


# ============================================================================
# 6. GIT OPERATIONS IN ISOLATED TMP REPO
# ============================================================================

class TestGitOperationsIsolated:
    @pytest.fixture
    def git_repo(self, tmp_path):
        """Creates a real git repo in tmp_path for isolated testing."""
        subprocess.run(
            ["git", "init"], cwd=str(tmp_path),
            capture_output=True, text=True, check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(tmp_path), capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(tmp_path), capture_output=True, text=True,
        )
        # Create initial commit on main
        readme = tmp_path / "README.md"
        readme.write_text("# Test\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path),
            capture_output=True, text=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "initial commit"],
            cwd=str(tmp_path), capture_output=True, text=True, check=True,
        )
        # Rename default branch to 'main' if not already
        subprocess.run(
            ["git", "branch", "-M", "main"],
            cwd=str(tmp_path), capture_output=True, text=True,
        )
        return tmp_path

    def test_create_feature_branch(self, git_repo):
        success = GitWorkspaceManager.create_feature_branch(
            str(git_repo), "agent/task-test1234"
        )
        assert success is True

        # Verify we are on the new branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(git_repo), capture_output=True, text=True,
        )
        assert "agent/task-test1234" in result.stdout.strip()

    def test_stage_and_commit(self, git_repo):
        # Create feature branch and modify a file
        GitWorkspaceManager.create_feature_branch(str(git_repo), "agent/task-commit")
        (git_repo / "new_file.py").write_text("x = 42\n", encoding="utf-8")

        success = GitWorkspaceManager.stage_and_commit(
            str(git_repo), "agent: add new_file.py"
        )
        assert success is True

        # Verify commit exists
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=str(git_repo), capture_output=True, text=True,
        )
        assert "agent: add new_file.py" in log.stdout

    def test_cleanup_branch(self, git_repo):
        GitWorkspaceManager.create_feature_branch(str(git_repo), "agent/task-cleanup")
        (git_repo / "temp.py").write_text("pass\n", encoding="utf-8")
        GitWorkspaceManager.stage_and_commit(str(git_repo), "temp commit")

        success = GitWorkspaceManager.cleanup_branch(str(git_repo), "agent/task-cleanup")
        assert success is True

        # Verify we're back on main
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(git_repo), capture_output=True, text=True,
        )
        assert result.stdout.strip() == "main"

        # Verify feature branch is deleted
        branches = subprocess.run(
            ["git", "branch"],
            cwd=str(git_repo), capture_output=True, text=True,
        )
        assert "agent/task-cleanup" not in branches.stdout

    def test_prepare_diff_summary_end_to_end(self, git_repo):
        # Write source file
        src = git_repo / "app.py"
        src.write_text("def main():\n    pass\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(git_repo), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add app.py"],
            cwd=str(git_repo), capture_output=True,
        )

        patch = FilePatch(
            file_path="app.py",
            original_code_snippet="    pass",
            updated_code_snippet="    print('hello')",
            explanation="Implement main",
        )

        summary = GitWorkspaceManager.prepare_diff_summary(
            repo_path=str(git_repo),
            patches=[patch],
            task_id="e2e-test-id",
        )

        assert summary.branch_name.startswith("agent/task-e2e-test-")
        assert "app.py" in summary.files_changed
        assert summary.lines_added >= 1
        assert summary.lines_deleted >= 1
        assert summary.risk_score == "LOW"
        assert "print('hello')" in summary.unified_diff


# ============================================================================
# 7. APPROVAL NODE & ROUTING TESTS
# ============================================================================

class TestApprovalRouting:
    def test_route_after_approval_approved(self):
        state: AgentState = {
            "approval": ApprovalDecision(approved=True, reviewer="human"),
        }
        assert route_after_approval(state) == "git_commit"

    def test_route_after_approval_rejected(self):
        state: AgentState = {
            "approval": ApprovalDecision(
                approved=False,
                rejection_reason="Needs more tests",
            ),
        }
        assert route_after_approval(state) == "cleanup"

    def test_route_after_approval_missing_defaults_to_cleanup(self):
        state: AgentState = {}
        assert route_after_approval(state) == "cleanup"

    def test_approval_node_interrupt_payload_shape(self):
        """
        Verify the approval_node constructs the correct interrupt payload.
        We test by monkeypatching interrupt to capture the payload.
        """
        from backend.vcs.models import ApprovalDecision

        captured_payload = {}

        def mock_interrupt(payload):
            captured_payload.update(payload)
            return {"approved": True, "reviewer": "test-reviewer"}

        import backend.graph.nodes as nodes_module
        original_interrupt = nodes_module.interrupt

        try:
            nodes_module.interrupt = mock_interrupt

            diff = GitDiffSummary(
                branch_name="agent/task-test",
                files_changed=["app.py"],
                lines_added=3,
                lines_deleted=1,
                unified_diff="+new\n-old",
                risk_score="LOW",
                risk_reasons=["Standard change."],
            )
            dev_result = DeveloperResult(
                summary="Fix bug",
                changes=[],
                requires_testing=True,
            )

            state: AgentState = {
                "user_message": "Fix the bug",
                "git_diff": diff,
                "developer_result": dev_result,
            }

            output = approval_node(state)

            # Verify payload shape
            assert captured_payload["task"] == "approval_required"
            assert captured_payload["diff"] is not None
            assert captured_payload["diff"]["branch_name"] == "agent/task-test"
            assert captured_payload["developer_result"] is not None
            assert "message" in captured_payload

            # Verify output
            assert output["approval"].approved is True
            assert output["approval"].reviewer == "test-reviewer"
            assert output["approval_status"] == "APPROVED"

        finally:
            nodes_module.interrupt = original_interrupt

    def test_approval_node_boolean_true_decision(self):
        """Test backward compat: simple True decision produces APPROVED."""
        import backend.graph.nodes as nodes_module
        original_interrupt = nodes_module.interrupt
        try:
            nodes_module.interrupt = lambda payload: True

            state: AgentState = {
                "user_message": "Test",
                "git_diff": GitDiffSummary(branch_name="agent/task-t"),
                "developer_result": DeveloperResult(
                    summary="Test", changes=[], requires_testing=False,
                ),
            }
            output = approval_node(state)
            assert output["approval_status"] == "APPROVED"
            assert output["approval"].approved is True
        finally:
            nodes_module.interrupt = original_interrupt

    def test_approval_node_false_decision(self):
        """Test simple False decision produces REJECTED."""
        import backend.graph.nodes as nodes_module
        original_interrupt = nodes_module.interrupt
        try:
            nodes_module.interrupt = lambda payload: False

            state: AgentState = {
                "user_message": "Test",
                "git_diff": GitDiffSummary(branch_name="agent/task-t"),
                "developer_result": DeveloperResult(
                    summary="Test", changes=[], requires_testing=False,
                ),
            }
            output = approval_node(state)
            assert output["approval_status"] == "REJECTED"
            assert output["approval"].approved is False
        finally:
            nodes_module.interrupt = original_interrupt

    def test_qa_router_pass_routes_to_pass(self):
        """Verify qa_router still returns 'pass' for PASS status (graph maps this to git_prepare)."""
        state: AgentState = {
            "qa_result": QAResult(
                status="PASS",
                issues=[],
                test_cases=[],
                summary="All good.",
            ),
            "revision_count": 0,
        }
        assert qa_router(state) == "pass"


# ============================================================================
# 8. GRAPH COMPILATION TEST
# ============================================================================

class TestGraphCompilation:
    def test_graph_compiles_with_new_topology(self):
        """
        Verify the full LangGraph StateGraph compiles successfully
        with all new nodes and edges.
        """
        from langgraph.checkpoint.memory import MemorySaver
        from backend.graph.runner import _build_graph
        graph = _build_graph(MemorySaver())
        assert graph is not None

        # Verify all node names exist in the compiled graph
        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "router", "planner", "knowledge", "developer", "qa",
            "revision", "git_prepare", "approval", "git_commit", "cleanup",
        }
        # LangGraph may add __start__ / __end__ meta nodes
        for expected in expected_nodes:
            assert expected in node_names, f"Missing node: {expected}"
