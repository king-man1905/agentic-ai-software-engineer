import hashlib
import os
import tempfile
import pytest
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
)
from backend.schemas.developer import DeveloperResult


# ============================================================================
# 1. PATCH HASH GENERATION TESTS
# ============================================================================

class TestPatchHashGeneration:
    def test_compute_patch_hash_deterministic(self):
        diff_text = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-print(1)\n+print(2)"
        hash1 = GitWorkspaceManager.compute_patch_hash(diff_text)
        hash2 = GitWorkspaceManager.compute_patch_hash(diff_text)

        expected = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
        assert hash1 == expected
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_empty_diff_hash(self):
        empty_hash = GitWorkspaceManager.compute_patch_hash("")
        assert empty_hash == hashlib.sha256(b"").hexdigest()

    def test_prepare_diff_summary_populates_patch_hash(self, tmp_path):
        # Create a workspace with a target file
        file_a = tmp_path / "hello.py"
        file_a.write_text("def hello():\n    return 'old'\n", encoding="utf-8")

        patch = FilePatch(
            file_path="hello.py",
            original_code_snippet="return 'old'",
            updated_code_snippet="return 'new'",
            explanation="update return",
        )

        summary = GitWorkspaceManager.prepare_diff_summary(
            repo_path=str(tmp_path),
            patches=[patch],
            task_id="task-test1234",
        )

        assert summary.patch_hash != ""
        assert len(summary.patch_hash) == 64
        assert summary.patch_hash == hashlib.sha256(summary.unified_diff.encode("utf-8")).hexdigest()


# ============================================================================
# 2. APPROVAL NODE INTEGRITY TESTS
# ============================================================================

class TestApprovalNodeIntegrity:
    def test_approval_node_includes_patch_hash_in_interrupt(self):
        captured_payload = {}

        def mock_interrupt(payload):
            captured_payload.update(payload)
            return {"approved": True, "reviewer": "alice", "patch_hash": payload.get("patch_hash")}

        import backend.graph.nodes as nodes_module
        original_interrupt = nodes_module.interrupt

        try:
            nodes_module.interrupt = mock_interrupt

            diff_text = "+added line"
            diff_hash = GitWorkspaceManager.compute_patch_hash(diff_text)
            diff = GitDiffSummary(
                branch_name="agent/task-test",
                files_changed=["app.py"],
                lines_added=1,
                lines_deleted=0,
                unified_diff=diff_text,
                patch_hash=diff_hash,
                risk_score="LOW",
                risk_reasons=["Low risk."],
            )

            state: AgentState = {
                "user_message": "Add line",
                "git_diff": diff,
                "developer_result": DeveloperResult(summary="Added", changes=[], requires_testing=False),
            }

            output = approval_node(state)

            assert captured_payload["patch_hash"] == diff_hash
            assert output["approval_status"] == "APPROVED"
            assert output["approval"].approved is True
            assert output["approval"].patch_hash == diff_hash
        finally:
            nodes_module.interrupt = original_interrupt

    def test_approval_rejected_on_hash_mismatch(self):
        """Reviewer or client submits an approval with a mismatched patch hash."""
        def mock_interrupt(payload):
            return {
                "approved": True,
                "reviewer": "charlie",
                "patch_hash": "deadbeef" * 8,  # Mismatched hash!
            }

        import backend.graph.nodes as nodes_module
        original_interrupt = nodes_module.interrupt

        try:
            nodes_module.interrupt = mock_interrupt

            diff_text = "+line 1"
            diff_hash = GitWorkspaceManager.compute_patch_hash(diff_text)
            diff = GitDiffSummary(
                branch_name="agent/task-mismatch",
                files_changed=["main.py"],
                unified_diff=diff_text,
                patch_hash=diff_hash,
            )

            state: AgentState = {
                "user_message": "Change main",
                "git_diff": diff,
            }

            output = approval_node(state)

            assert output["approval_status"] == "PATCH_HASH_MISMATCH"
            assert output["approval"].approved is False
            assert "PATCH_HASH_MISMATCH" in output["approval"].rejection_reason
            assert route_after_approval(output) == "cleanup"
        finally:
            nodes_module.interrupt = original_interrupt

    def test_approval_backward_compatibility_none_hash(self):
        """Legacy client approval omitting patch_hash still succeeds."""
        def mock_interrupt(payload):
            return {"approved": True, "reviewer": "legacy-bot", "patch_hash": None}

        import backend.graph.nodes as nodes_module
        original_interrupt = nodes_module.interrupt

        try:
            nodes_module.interrupt = mock_interrupt

            diff_text = "+line 1"
            diff_hash = GitWorkspaceManager.compute_patch_hash(diff_text)
            diff = GitDiffSummary(
                branch_name="agent/task-legacy",
                files_changed=["main.py"],
                unified_diff=diff_text,
                patch_hash=diff_hash,
            )

            state: AgentState = {
                "user_message": "Change main",
                "git_diff": diff,
            }

            output = approval_node(state)

            assert output["approval_status"] == "APPROVED"
            assert output["approval"].approved is True
            assert route_after_approval(output) == "git_commit"
        finally:
            nodes_module.interrupt = original_interrupt


# ============================================================================
# 3. PRE-COMMIT DRIFT VERIFICATION TESTS
# ============================================================================

@pytest.fixture
def git_repo(tmp_path):
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), capture_output=True, text=True)
    f = tmp_path / "foo.py"
    f.write_text("def run():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True, text=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(tmp_path), capture_output=True, text=True, check=True)
    return tmp_path


class TestPreCommitDriftVerification:
    def test_verify_workspace_drift_valid(self, git_repo):
        patch = FilePatch(
            file_path="foo.py",
            original_code_snippet="return 1",
            updated_code_snippet="return 2",
            explanation="update return",
        )

        summary = GitWorkspaceManager.prepare_diff_summary(
            repo_path=str(git_repo),
            patches=[patch],
            task_id="task-drift-test",
        )

        is_valid, err = GitWorkspaceManager.verify_workspace_drift(
            repo_path=str(git_repo),
            expected_diff=summary.unified_diff,
            expected_hash=summary.patch_hash,
            files_changed=summary.files_changed,
        )
        assert is_valid is True
        assert err == ""

    def test_verify_workspace_drift_detects_tamper(self, git_repo):
        patch = FilePatch(
            file_path="foo.py",
            original_code_snippet="return 1",
            updated_code_snippet="return 2",
            explanation="update return",
        )

        summary = GitWorkspaceManager.prepare_diff_summary(
            repo_path=str(git_repo),
            patches=[patch],
            task_id="task-drift-test",
        )

        # Rogue process modifies foo.py after diff was generated
        (git_repo / "foo.py").write_text("def run():\n    return 'MALICIOUS_TAMPER'\n", encoding="utf-8")

        is_valid, err = GitWorkspaceManager.verify_workspace_drift(
            repo_path=str(git_repo),
            expected_diff=summary.unified_diff,
            expected_hash=summary.patch_hash,
            files_changed=summary.files_changed,
        )
        assert is_valid is False
        assert "Workspace drift detected" in err

    def test_git_commit_node_aborts_on_workspace_tamper(self, monkeypatch):
        diff_summary = GitDiffSummary(
            branch_name="agent/task-tamper",
            files_changed=["foo.py"],
            unified_diff="+tampered diff",
            patch_hash="abc" * 21 + "a",
        )

        state: AgentState = {
            "project_id": "test_project",
            "git_diff": diff_summary,
            "approval": ApprovalDecision(approved=True, patch_hash=diff_summary.patch_hash),
        }

        # Mock verify_workspace_drift to simulate tamper detection
        monkeypatch.setattr(
            GitWorkspaceManager,
            "verify_workspace_drift",
            lambda repo_path, expected_diff, expected_hash, files_changed: (False, "Workspace drift detected!"),
        )

        result = git_commit_node(state)
        assert result["approval_status"] == "PATCH_HASH_MISMATCH"


# ============================================================================
# 4. API RESUME WITH PATCH HASH INTEGRITY TESTS
# ============================================================================

class TestApiResumeWithPatchHash:
    def test_api_resume_passes_patch_hash(self):
        from fastapi.testclient import TestClient
        from backend.api.app import create_app
        from backend.graph.runner import AgentRunner
        from unittest.mock import MagicMock

        mock_runner = MagicMock(spec=AgentRunner)
        mock_runner.resume_run.return_value = MagicMock(
            run_id="run-123",
            status="COMPLETED",
            current_node=None,
            git_diff=None,
            error_summary=None,
            message=None,
        )

        app = create_app(runner=mock_runner)
        client = TestClient(app)

        test_hash = "a" * 64
        response = client.post(
            "/api/v1/runs/run-123/resume",
            json={
                "approved": True,
                "reviewer": "alice",
                "patch_hash": test_hash,
            },
        )

        assert response.status_code == 200
        mock_runner.resume_run.assert_called_once()
        passed_decision = mock_runner.resume_run.call_args[0][1]
        assert passed_decision.approved is True
        assert passed_decision.patch_hash == test_hash


