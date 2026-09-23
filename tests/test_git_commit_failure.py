"""
Regression tests for Git commit failure handling:
1. Git identity fallback ensures stage_and_commit succeeds in clean environments with no global git config.
2. Failed commit produces correct top-level run status (FAILED), commit_status (COMMIT_FAILED), and error_summary.
3. Failed commit does not publish PR (HTTP 409, no branch push, no PR creation).
4. Successful commit produces normal success state (COMPLETED, COMMITTED).
5. Existing GitHub PR workflow remains intact for successful commits.
"""

from unittest.mock import MagicMock
import os
import subprocess
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.graph.runner import AgentRunner
from backend.integrations.github_client import GitHubClient
from backend.integrations.github_models import GitHubPRResult
from backend.observability.collector import telemetry_collector
from backend.observability.store import telemetry_store
from backend.schemas.qa import QAResult
from backend.schemas.developer import DeveloperResult
from backend.schemas.routing import RoutingDecision, TaskType
from backend.vcs.git_manager import GitWorkspaceManager
from backend.vcs.models import GitDiffSummary


@pytest.fixture(autouse=True)
def isolated_telemetry(tmp_path, monkeypatch):
    """Isolates telemetry_store to a tmp_path db so these runs don't leak."""
    from backend.observability.store import TelemetryStore

    test_db = str(tmp_path / "test_commit_failure_telemetry.db")
    test_store = TelemetryStore(test_db)
    monkeypatch.setattr(telemetry_store, "db_path", test_db)
    monkeypatch.setattr(telemetry_collector, "store", test_store)


def _init_local_git_repo(path):
    subprocess.run(["git", "init"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(path), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(path), check=True)
    readme = path / "README.md"
    readme.write_text("# Test Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=str(path), check=True, capture_output=True)


def test_stage_and_commit_succeeds_in_unconfigured_git_environment(tmp_path, monkeypatch):
    """
    1. In an environment without global git config (~/.gitconfig unset / isolated),
    stage_and_commit provides fallback author/committer identity and succeeds.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    # Modify file
    readme = repo_dir / "README.md"
    readme.write_text("# Test Repo\nModified content\n", encoding="utf-8")

    # Isolate git config from user system / global config
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "nonexistent"))

    # Remove any repo-level user config to simulate unconfigured clone
    subprocess.run(["git", "config", "--unset", "user.name"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "--unset", "user.email"], cwd=str(repo_dir), check=True)

    success = GitWorkspaceManager.stage_and_commit(str(repo_dir), "agent: test unconfigured commit")
    assert success is True


def test_derive_status_commit_failed_returns_failed():
    """
    2. _derive_status returns 'FAILED' when approval_status is 'COMMIT_FAILED',
    even if QA status is 'PASS'.
    """
    runner = AgentRunner()
    snap = MagicMock()
    snap.next = ()
    snap.tasks = ()
    snap.values = {
        "qa_result": QAResult(status="PASS", summary="All checks passed."),
        "approval_status": "COMMIT_FAILED",
    }
    assert runner._derive_status(snap) == "FAILED"


def test_failed_commit_produces_failed_run_status_and_blocks_pr(tmp_path, monkeypatch):
    """
    3. When git_commit fails, the run status must be FAILED, commit_status must be COMMIT_FAILED,
    error_summary must be 'Git commit failed', and publish-pr must reject with HTTP 409.
    """
    # Force stage_and_commit to fail
    monkeypatch.setattr(GitWorkspaceManager, "stage_and_commit", lambda *a, **k: False)
    monkeypatch.setattr(GitWorkspaceManager, "create_feature_branch", lambda *a, **k: True)
    monkeypatch.setattr(GitWorkspaceManager, "verify_workspace_drift", lambda *a, **k: (True, ""))

    test_diff = GitDiffSummary(
        branch_name="agent/test-branch",
        files_changed=["README.md"],
        lines_added=1,
        lines_deleted=0,
        unified_diff="+test",
        patch_hash="abc123",
        risk_score="LOW",
        risk_reasons=[],
    )

    # Mock pipeline nodes BEFORE AgentRunner instantiates the graph
    monkeypatch.setattr(
        "backend.graph.nodes.route_task",
        lambda msg: RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            reasoning="test",
            requires_planning=False,
            requires_knowledge=False,
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.developer_node",
        lambda state: {
            "developer_result": DeveloperResult(summary="test change", changes=[], requires_testing=True, notes=[]),
            "generated_patches": [],
        },
    )
    monkeypatch.setattr(
        "backend.graph.nodes.qa_node",
        lambda state: {"qa_result": QAResult(status="PASS", summary="QA passed")},
    )
    monkeypatch.setattr(
        "backend.graph.nodes.git_prepare_node",
        lambda state: {
            "git_diff": test_diff,
            "patch_hash": test_diff.patch_hash,
        },
    )

    db_path = str(tmp_path / "checkpoints.db")
    runner = AgentRunner(checkpoint_db_path=db_path)
    app = create_app(runner=runner)
    client = TestClient(app)

    # Start run -> should pause at approval_node
    resp = client.post(
        "/api/v1/runs",
        json={"user_message": "Fix a bug", "project_id": "test_project"},
    )
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    init_status = client.get(f"/api/v1/runs/{run_id}").json()
    assert init_status["status"] == "WAITING_APPROVAL"

    # Approve run -> triggers git_commit_node -> stage_and_commit returns False
    resume_resp = client.post(
        f"/api/v1/runs/{run_id}/resume",
        json={"approved": True, "reviewer": "test-admin", "patch_hash": "abc123"},
    )
    assert resume_resp.status_code == 200
    resume_data = resume_resp.json()

    # Verify top-level status is FAILED, not COMPLETED
    assert resume_data["status"] == "FAILED"
    assert resume_data["error_summary"] == "Git commit failed"

    # Verify get_status endpoint also returns FAILED
    status_resp = client.get(f"/api/v1/runs/{run_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "FAILED"
    assert status_data["error_summary"] == "Git commit failed"

    # Verify publish-pr is BLOCKED (409 Conflict)
    push_spy = MagicMock()
    monkeypatch.setattr(GitWorkspaceManager, "push_branch", push_spy)

    publish_resp = client.post(
        f"/api/v1/runs/{run_id}/publish-pr",
        json={"repo_full_name": "owner/repo"},
    )
    assert publish_resp.status_code == 409
    push_spy.assert_not_called()


def test_successful_commit_produces_completed_status_and_allows_pr(tmp_path, monkeypatch):
    """
    4 & 5. When git_commit succeeds:
    - Status is COMPLETED
    - approval_status is COMMITTED
    - PR workflow succeeds
    """
    monkeypatch.setattr(GitWorkspaceManager, "stage_and_commit", lambda *a, **k: True)
    monkeypatch.setattr(GitWorkspaceManager, "create_feature_branch", lambda *a, **k: True)
    monkeypatch.setattr(GitWorkspaceManager, "verify_workspace_drift", lambda *a, **k: (True, ""))
    monkeypatch.setattr(GitWorkspaceManager, "push_branch", lambda *a, **k: True)
    monkeypatch.setattr(
        GitHubClient,
        "find_pull_request",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        GitHubClient,
        "create_pull_request",
        lambda *a, **k: GitHubPRResult(
            pr_number=42,
            pr_url="https://github.com/owner/repo/pull/42",
            head_branch="agent/test-success-branch",
            base_branch="main",
            title="fix",
            body="body",
            draft=False,
            created_at="2026-09-23T00:00:00Z",
        ),
    )

    test_diff = GitDiffSummary(
        branch_name="agent/test-success-branch",
        files_changed=["README.md"],
        lines_added=1,
        lines_deleted=0,
        unified_diff="+test",
        patch_hash="xyz789",
        risk_score="LOW",
        risk_reasons=[],
    )

    # Mock pipeline nodes BEFORE AgentRunner instantiates the graph
    monkeypatch.setattr(
        "backend.graph.nodes.route_task",
        lambda msg: RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            reasoning="test",
            requires_planning=False,
            requires_knowledge=False,
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.developer_node",
        lambda state: {
            "developer_result": DeveloperResult(summary="test change", changes=[], requires_testing=True, notes=[]),
            "generated_patches": [],
        },
    )
    monkeypatch.setattr(
        "backend.graph.nodes.qa_node",
        lambda state: {"qa_result": QAResult(status="PASS", summary="QA passed")},
    )
    monkeypatch.setattr(
        "backend.graph.nodes.git_prepare_node",
        lambda state: {
            "git_diff": test_diff,
            "patch_hash": test_diff.patch_hash,
        },
    )

    db_path = str(tmp_path / "checkpoints_success.db")
    runner = AgentRunner(checkpoint_db_path=db_path)
    monkeypatch.setattr(runner, "_ensure_workspace_provisioned", lambda *a, **k: None)
    app = create_app(runner=runner)
    client = TestClient(app)

    from backend.security.tenant import tenant_manager
    tenant_manager.register_repository("owner/repo", "default-org", "repo")

    resp = client.post(
        "/api/v1/runs",
        json={"user_message": "Fix a bug", "project_id": "test_project", "repository_id": "owner/repo"},
    )
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    init_status = client.get(f"/api/v1/runs/{run_id}").json()
    assert init_status["status"] == "WAITING_APPROVAL"

    resume_resp = client.post(
        f"/api/v1/runs/{run_id}/resume",
        json={"approved": True, "reviewer": "test-admin", "patch_hash": "xyz789"},
    )
    assert resume_resp.status_code == 200
    resume_data = resume_resp.json()

    assert resume_data["status"] == "COMPLETED"
    assert resume_data["error_summary"] is None

    # Status check
    status_resp = client.get(f"/api/v1/runs/{run_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "COMPLETED"

    # Publish PR should succeed
    publish_resp = client.post(
        f"/api/v1/runs/{run_id}/publish-pr",
        json={"repo_full_name": "owner/repo"},
    )
    assert publish_resp.status_code == 200
    pub_data = publish_resp.json()
    assert pub_data["status"] == "PUBLISHED"
    assert pub_data["pr_number"] == 42
    assert pub_data["pr_url"] == "https://github.com/owner/repo/pull/42"
