import os
from unittest.mock import MagicMock, patch
import httpx
import pytest

from backend.integrations.github_client import GitHubClient, GitHubApiError
from backend.integrations.github_models import GitHubIssuePayload, GitHubPRResult
from backend.integrations.run_github_bot import solve_issue_and_open_pr
from backend.sandbox.models import TestExecutionResult
from backend.vcs.models import GitDiffSummary, ApprovalDecision
from backend.api.models import RunStatusResponse
from backend.graph.runner import AgentRunner

# Prevent pytest from attempting to discover TestExecutionResult as a test case class
TestExecutionResult.__test__ = False


# ============================================================================
# 1. GITHUB DATA MODEL TESTS
# ============================================================================

class TestGitHubModels:
    def test_github_issue_payload_instantiation(self):
        issue = GitHubIssuePayload(
            repo_full_name="octocat/Hello-World",
            issue_number=42,
            title="Fix null pointer exception in user auth",
            body="When user logs in with null email, server crashes.",
            labels=["bug", "security", "p1"],
        )
        assert issue.repo_full_name == "octocat/Hello-World"
        assert issue.issue_number == 42
        assert issue.title == "Fix null pointer exception in user auth"
        assert len(issue.labels) == 3
        assert "security" in issue.labels

    def test_github_issue_payload_defaults(self):
        issue = GitHubIssuePayload(
            repo_full_name="org/repo",
            issue_number=1,
            title="Update README",
        )
        assert issue.body == ""
        assert issue.labels == []

    def test_github_pr_result_instantiation_and_serialization(self):
        pr = GitHubPRResult(
            pr_number=101,
            pr_url="https://github.com/octocat/Hello-World/pull/101",
            head_branch="agent/task-fix-auth",
            base_branch="main",
            is_draft=True,
        )
        assert pr.pr_number == 101
        assert pr.is_draft is True
        assert pr.base_branch == "main"

        # Pydantic serialization round-trip
        json_data = pr.model_dump_json()
        restored = GitHubPRResult.model_validate_json(json_data)
        assert restored.pr_number == pr.pr_number
        assert restored.pr_url == pr.pr_url
        assert restored.is_draft is True


# ============================================================================
# 2. GITHUB CLIENT TESTS (MOCKED HTTPX TRANSPORT)
# ============================================================================

class TestGitHubClient:
    def test_missing_token_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        client = GitHubClient(token=None)
        with pytest.raises(ValueError) as exc_info:
            client.fetch_issue("owner/repo", 1)
        assert "GITHUB_TOKEN" in str(exc_info.value)

    def test_fetch_issue_success(self):
        issue_data = {
            "number": 42,
            "title": "Authentication crash on empty token",
            "body": "Detailed reproduction steps here...",
            "labels": [{"name": "bug"}, {"name": "backend"}],
        }

        def mock_handler(request: httpx.Request):
            assert request.method == "GET"
            assert request.url.path == "/repos/octocat/Hello-World/issues/42"
            assert request.headers["authorization"] == "Bearer mock_token_123"
            return httpx.Response(200, json=issue_data)

        transport = httpx.MockTransport(mock_handler)
        mock_http = httpx.Client(transport=transport)

        client = GitHubClient(token="mock_token_123", http_client=mock_http)
        issue = client.fetch_issue("octocat/Hello-World", 42)

        assert issue.repo_full_name == "octocat/Hello-World"
        assert issue.issue_number == 42
        assert issue.title == "Authentication crash on empty token"
        assert issue.body == "Detailed reproduction steps here..."
        assert issue.labels == ["bug", "backend"]

    def test_fetch_issue_string_labels(self):
        issue_data = {
            "number": 10,
            "title": "Simple issue",
            "body": None,
            "labels": ["enhancement", "good-first-issue"],
        }

        def mock_handler(request: httpx.Request):
            return httpx.Response(200, json=issue_data)

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        client = GitHubClient(token="mock_token", http_client=mock_http)
        issue = client.fetch_issue("owner/repo", 10)

        assert issue.labels == ["enhancement", "good-first-issue"]
        assert issue.body == ""

    def test_fetch_issue_unauthorized_raises_permission_error(self):
        def mock_handler(request: httpx.Request):
            return httpx.Response(401, json={"message": "Bad credentials"})

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        client = GitHubClient(token="invalid_token", http_client=mock_http)

        with pytest.raises(PermissionError) as exc_info:
            client.fetch_issue("owner/repo", 1)
        assert "authorization failed (401)" in str(exc_info.value)

    def test_fetch_issue_not_found_raises_file_not_found_error(self):
        def mock_handler(request: httpx.Request):
            return httpx.Response(404, json={"message": "Not Found"})

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        client = GitHubClient(token="valid_token", http_client=mock_http)

        with pytest.raises(FileNotFoundError) as exc_info:
            client.fetch_issue("owner/repo", 9999)
        assert "was not found (404)" in str(exc_info.value)

    def test_create_pull_request_success(self):
        pr_data = {
            "number": 55,
            "html_url": "https://github.com/octocat/Hello-World/pull/55",
            "draft": True,
        }

        def mock_handler(request: httpx.Request):
            assert request.method == "POST"
            assert request.url.path == "/repos/octocat/Hello-World/pulls"
            assert request.headers["authorization"] == "Bearer mock_token_123"

            import json
            body = json.loads(request.content)
            assert body["title"] == "fix: resolve auth crash"
            assert body["head"] == "agent/task-fix-auth"
            assert body["base"] == "main"
            assert body["draft"] is True
            return httpx.Response(201, json=pr_data)

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        client = GitHubClient(token="mock_token_123", http_client=mock_http)

        result = client.create_pull_request(
            repo_full_name="octocat/Hello-World",
            title="fix: resolve auth crash",
            body="PR body description with diff",
            head_branch="agent/task-fix-auth",
            base_branch="main",
            draft=True,
        )

        assert result.pr_number == 55
        assert result.pr_url == "https://github.com/octocat/Hello-World/pull/55"
        assert result.head_branch == "agent/task-fix-auth"
        assert result.base_branch == "main"
        assert result.is_draft is True

    def test_create_pull_request_unprocessable_raises_value_error(self):
        def mock_handler(request: httpx.Request):
            return httpx.Response(
                422,
                json={"message": "Validation Failed", "errors": [{"message": "No commits between main and branch"}]},
            )

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        client = GitHubClient(token="valid_token", http_client=mock_http)

        with pytest.raises(ValueError) as exc_info:
            client.create_pull_request(
                repo_full_name="owner/repo",
                title="title",
                body="body",
                head_branch="head",
                base_branch="main",
            )
        assert "unprocessable (422)" in str(exc_info.value)

    def test_create_pull_request_not_found_raises_file_not_found_error(self):
        def mock_handler(request: httpx.Request):
            return httpx.Response(404, json={"message": "Not Found"})

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        client = GitHubClient(token="valid_token", http_client=mock_http)

        with pytest.raises(FileNotFoundError) as exc_info:
            client.create_pull_request(
                repo_full_name="nonexistent/repo",
                title="title",
                body="body",
                head_branch="head",
                base_branch="main",
            )
        assert "was not found (404)" in str(exc_info.value)

    def test_create_pull_request_malformed_missing_pr_number_raises_github_api_error(self):
        def mock_handler(request: httpx.Request):
            return httpx.Response(201, json={"html_url": "https://github.com/octocat/Hello-World/pull/55"})

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        client = GitHubClient(token="mock_token", http_client=mock_http)

        with pytest.raises(GitHubApiError) as exc_info:
            client.create_pull_request(
                repo_full_name="octocat/Hello-World",
                title="fix: bug",
                body="body",
                head_branch="head",
            )
        assert "missing valid PR number" in str(exc_info.value)

    def test_create_pull_request_malformed_missing_html_url_raises_github_api_error(self):
        def mock_handler(request: httpx.Request):
            return httpx.Response(201, json={"number": 55, "html_url": ""})

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        client = GitHubClient(token="mock_token", http_client=mock_http)

        with pytest.raises(GitHubApiError) as exc_info:
            client.create_pull_request(
                repo_full_name="octocat/Hello-World",
                title="fix: bug",
                body="body",
                head_branch="head",
            )
        assert "missing valid html_url" in str(exc_info.value)

    def test_find_pull_request_malformed_pr_data_raises_github_api_error(self):
        def mock_handler(request: httpx.Request):
            return httpx.Response(200, json=[{
                "head": {"ref": "agent/task-fix"},
                "base": {"ref": "main"},
                "number": 0,
                "html_url": "https://github.com/octocat/Hello-World/pull/0",
            }])

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        client = GitHubClient(token="mock_token", http_client=mock_http)

        with pytest.raises(GitHubApiError) as exc_info:
            client.find_pull_request(
                repo_full_name="octocat/Hello-World",
                head_branch="agent/task-fix",
                base_branch="main",
            )
        assert "missing valid PR number" in str(exc_info.value)

    def test_find_pull_request_valid_pr_success(self):
        def mock_handler(request: httpx.Request):
            return httpx.Response(200, json=[{
                "head": {"ref": "agent/task-fix"},
                "base": {"ref": "main"},
                "number": 88,
                "html_url": "https://github.com/octocat/Hello-World/pull/88",
                "draft": False,
            }])

        mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
        client = GitHubClient(token="mock_token", http_client=mock_http)

        res = client.find_pull_request(
            repo_full_name="octocat/Hello-World",
            head_branch="agent/task-fix",
            base_branch="main",
        )
        assert res is not None
        assert res.pr_number == 88
        assert res.pr_url == "https://github.com/octocat/Hello-World/pull/88"
        assert res.is_draft is False


# ============================================================================
# 3. PR DESCRIPTION MARKDOWN GENERATION TESTS
# ============================================================================

class TestPRDescriptionFormatter:
    @pytest.fixture
    def sample_issue(self):
        return GitHubIssuePayload(
            repo_full_name="octocat/Hello-World",
            issue_number=42,
            title="Fix null pointer exception in user auth",
            body="Repro: Login with null email",
            labels=["bug"],
        )

    def test_format_pr_description_full(self, sample_issue):
        diff = GitDiffSummary(
            branch_name="agent/task-fix-auth",
            files_changed=["backend/auth.py", "backend/models.py"],
            lines_added=15,
            lines_deleted=3,
            unified_diff="--- a/backend/auth.py\n+++ b/backend/auth.py\n@@ -1 +1 @@\n-old\n+new",
            risk_score="LOW",
            risk_reasons=["Small isolated patch to auth validation."],
        )
        test_res = TestExecutionResult(
            success=True,
            exit_code=0,
            passed_count=8,
            failed_count=0,
            stdout="8 passed in 1.45s",
            stderr="",
            duration_seconds=1.45,
            error_summary=None,
        )
        metrics = {
            "prompt_tokens": 1250,
            "completion_tokens": 320,
            "total_tokens": 1570,
            "estimated_cost_usd": 0.0031,
        }

        md = GitHubClient.format_pr_description(sample_issue, diff, test_res, metrics)

        # 1. Closing issue reference
        assert "Closes #42" in md
        assert "Fix null pointer exception in user auth" in md

        # 2. Risk Assessment table
        assert "🟢 `LOW`" in md
        assert "`2`" in md  # 2 files changed
        assert "`+15 / -3`" in md
        assert "Small isolated patch to auth validation." in md

        # 3. Sandbox Verification proof
        assert "✅ `PASSED`" in md
        assert "`8`" in md
        assert "`0`" in md
        assert "1.45s" in md

        # 4. Telemetry metrics
        assert "1,250" in md
        assert "320" in md
        assert "1,570" in md
        assert "$0.0031" in md

        # 5. Collapsible diff
        assert "<details>" in md
        assert "<summary><b>View Unified Diff (2 files changed)</b></summary>" in md
        assert "--- a/backend/auth.py" in md

    def test_format_pr_description_high_risk_and_test_failure(self, sample_issue):
        diff = GitDiffSummary(
            branch_name="agent/task-risky",
            files_changed=["config.yml", "alembic/env.py", "auth.py", "db.py"],
            lines_added=120,
            lines_deleted=85,
            unified_diff="--- a/config.yml\n+++ b/config.yml",
            risk_score="HIGH",
            risk_reasons=["Modifies configuration file: config.yml", "Database migration detected."],
        )
        test_res = TestExecutionResult(
            success=False,
            exit_code=1,
            passed_count=5,
            failed_count=2,
            stdout="2 failed, 5 passed in 2.10s",
            stderr="AssertionError in test_auth.py",
            duration_seconds=2.10,
            error_summary="AssertionError: assert False is True in test_auth.py:22",
        )

        md = GitHubClient.format_pr_description(sample_issue, diff, test_res)

        assert "Closes #42" in md
        assert "🔴 `HIGH`" in md
        assert "❌ `FAILED`" in md
        assert "Modifies configuration file: config.yml" in md
        assert "Database migration detected." in md
        assert "AssertionError: assert False is True in test_auth.py:22" in md

    def test_format_pr_description_null_inputs(self, sample_issue):
        # Graceful handling when diff and test_result are None
        md = GitHubClient.format_pr_description(sample_issue, None, None, None)

        assert "Closes #42" in md
        assert "🟢 `LOW`" in md
        assert "⚪ `SKIPPED`" in md
        assert "No changes recorded." in md
        # metrics=None means "not measured", not "$0 measured cost" - the
        # PR body should say so plainly rather than implying free/verified.
        assert "N/A" in md
        assert "$0.0000" not in md


# ============================================================================
# 4. CLI RUNNER DISPATCHER TESTS
# ============================================================================

class TestRunGitHubBotCLI:
    def test_solve_issue_and_open_pr_end_to_end_mocked(self):
        # Mock GitHub Client
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.fetch_issue.return_value = GitHubIssuePayload(
            repo_full_name="octocat/Hello-World",
            issue_number=12,
            title="Fix memory leak in buffer",
            body="Buffer never gets cleared after flush.",
            labels=["bug"],
        )
        mock_client.create_pull_request.return_value = GitHubPRResult(
            pr_number=77,
            pr_url="https://github.com/octocat/Hello-World/pull/77",
            head_branch="agent/task-fix-buffer",
            base_branch="main",
            is_draft=True,
        )

        # Mock AgentRunner
        mock_runner = MagicMock(spec=AgentRunner)
        diff = GitDiffSummary(
            branch_name="agent/task-fix-buffer",
            files_changed=["buffer.py"],
            lines_added=4,
            lines_deleted=1,
            unified_diff="+buffer.clear()",
            risk_score="LOW",
            risk_reasons=["Small buffer cleanup."],
        )
        # 1. start_run returns WAITING_APPROVAL
        mock_runner.start_run.return_value = RunStatusResponse(
            run_id="gh_12_abc12345",
            status="WAITING_APPROVAL",
            current_node="approval",
            git_diff=diff,
        )
        mock_runner.get_state_values.return_value = {
            "git_diff": diff,
            "test_result": TestExecutionResult(
                success=True,
                exit_code=0,
                passed_count=3,
                failed_count=0,
                stdout="3 passed",
                stderr="",
                duration_seconds=0.5,
            ),
            "metrics": {
                "prompt_tokens": 1200,
                "completion_tokens": 300,
                "total_tokens": 1500,
                "estimated_cost_usd": 0.0123,
            },
        }
        # 2. resume_run approves and completes
        mock_runner.resume_run.return_value = RunStatusResponse(
            run_id="gh_12_abc12345",
            status="COMPLETED",
            current_node=None,
            git_diff=diff,
        )

        result = solve_issue_and_open_pr(
            repo="octocat/Hello-World",
            issue_number=12,
            base_branch="main",
            auto_approve=True,
            draft=True,
            runner=mock_runner,
            client=mock_client,
        )

        assert result is not None
        assert result.pr_number == 77
        assert result.pr_url == "https://github.com/octocat/Hello-World/pull/77"
        assert result.is_draft is True

        # Verify calls
        mock_client.fetch_issue.assert_called_once_with("octocat/Hello-World", 12)
        mock_runner.start_run.assert_called_once()
        mock_runner.resume_run.assert_called_once()
        mock_client.create_pull_request.assert_called_once()

        call_args = mock_client.create_pull_request.call_args[1]
        assert call_args["repo_full_name"] == "octocat/Hello-World"
        assert "Closes #12" in call_args["body"]
        assert call_args["head_branch"] == "agent/task-fix-buffer"
        assert call_args["draft"] is True

        # Real per-run telemetry from get_state_values reaches the PR body,
        # instead of the previously hardcoded metrics=None.
        assert "1,200" in call_args["body"]
        assert "1,500" in call_args["body"]
        assert "$0.0123" in call_args["body"]

    def test_solve_issue_aborts_pr_when_diff_is_no_op(self):
        """The developer agent found nothing to change (e.g. the issue was
        already fixed, or patch validation failed): no PR should be opened."""
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.fetch_issue.return_value = GitHubIssuePayload(
            repo_full_name="octocat/Hello-World",
            issue_number=7,
            title="Already-fixed bug",
        )

        mock_runner = MagicMock(spec=AgentRunner)
        no_op_diff = GitDiffSummary(
            branch_name="agent/task-noop",
            files_changed=[],
            lines_added=0,
            lines_deleted=0,
            unified_diff="",
            risk_score="LOW",
            risk_reasons=["No file patches to apply."],
        )
        mock_runner.start_run.return_value = RunStatusResponse(
            run_id="gh_7_noop1234",
            status="WAITING_APPROVAL",
            current_node="approval",
            git_diff=no_op_diff,
        )
        mock_runner.get_state_values.return_value = {"git_diff": no_op_diff}
        mock_runner.resume_run.return_value = RunStatusResponse(
            run_id="gh_7_noop1234",
            status="COMPLETED",
            current_node=None,
            git_diff=no_op_diff,
        )

        result = solve_issue_and_open_pr(
            repo="octocat/Hello-World",
            issue_number=7,
            auto_approve=True,
            runner=mock_runner,
            client=mock_client,
        )

        assert result is None
        mock_client.create_pull_request.assert_not_called()

    def test_solve_issue_when_auto_approve_disabled_exits_early(self):
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.fetch_issue.return_value = GitHubIssuePayload(
            repo_full_name="octocat/Hello-World",
            issue_number=15,
            title="Update dependencies",
        )

        mock_runner = MagicMock(spec=AgentRunner)
        mock_runner.start_run.return_value = RunStatusResponse(
            run_id="gh_15_xyz987",
            status="WAITING_APPROVAL",
            current_node="approval",
            git_diff=GitDiffSummary(branch_name="agent/task-deps"),
        )
        mock_runner.get_state_values.return_value = {}

        result = solve_issue_and_open_pr(
            repo="octocat/Hello-World",
            issue_number=15,
            auto_approve=False,
            runner=mock_runner,
            client=mock_client,
        )

        # Should pause and exit without opening PR
        assert result is None
        mock_runner.resume_run.assert_not_called()
        mock_client.create_pull_request.assert_not_called()

    def test_solve_issue_failed_run_returns_none(self):
        mock_client = MagicMock(spec=GitHubClient)
        mock_client.fetch_issue.return_value = GitHubIssuePayload(
            repo_full_name="octocat/Hello-World",
            issue_number=99,
            title="Broken task",
        )

        mock_runner = MagicMock(spec=AgentRunner)
        mock_runner.start_run.return_value = RunStatusResponse(
            run_id="gh_99_fail",
            status="FAILED",
            error_summary="Syntax error could not be recovered.",
        )
        mock_runner.get_state_values.return_value = {}

        result = solve_issue_and_open_pr(
            repo="octocat/Hello-World",
            issue_number=99,
            runner=mock_runner,
            client=mock_client,
        )

        assert result is None
        mock_client.create_pull_request.assert_not_called()


# ============================================================================
# 5. RUN STATUS PR PUBLICATION ENDPOINT TESTS
# ============================================================================

class TestRunStatusPRPublication:
    def test_get_run_status_completed_without_pr_returns_null_pr_fields(self):
        from fastapi.testclient import TestClient
        from backend.api.app import create_app

        runner = MagicMock(spec=AgentRunner)
        runner.get_status.return_value = RunStatusResponse(
            run_id="run_comp_no_pr",
            status="COMPLETED",
            current_node=None,
        )
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.get("/api/v1/runs/run_comp_no_pr")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "COMPLETED"
        assert data["pr_number"] is None
        assert data["pr_url"] is None
        assert data["pr_status"] is None

    def test_get_run_status_with_published_pr_exposes_pr_fields(self):
        from fastapi.testclient import TestClient
        from backend.api.app import create_app
        from backend.observability.collector import telemetry_collector

        run_id = "run_with_pr_123"
        telemetry_collector.on_pr_published(
            run_id=run_id,
            organization_id="default-org",
            pr_number=77,
            pr_url="https://github.com/octocat/Hello-World/pull/77",
        )

        runner = MagicMock(spec=AgentRunner)
        runner.get_status.return_value = RunStatusResponse(
            run_id=run_id,
            status="COMPLETED",
            current_node=None,
        )
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.get(f"/api/v1/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "COMPLETED"
        assert data["pr_number"] == 77
        assert data["pr_url"] == "https://github.com/octocat/Hello-World/pull/77"
        assert data["pr_status"] == "PUBLISHED"

    def test_publish_pr_unauthorized_unregistered_repo_remains_blocked(self):
        from fastapi.testclient import TestClient
        from backend.api.app import create_app
        from backend.api.models import PublishPRRequest

        diff = GitDiffSummary(
            branch_name="agent/test-unauth",
            files_changed=["auth.py"],
            lines_added=2,
            lines_deleted=0,
            unified_diff="+test",
            patch_hash="patch_hash_123",
        )
        runner = MagicMock(spec=AgentRunner)
        runner.get_status.return_value = RunStatusResponse(
            run_id="run_unauth_repo",
            status="COMPLETED",
            current_node=None,
            git_diff=diff,
        )
        decision = ApprovalDecision(
            approved=True,
            reviewer="admin",
            timestamp="2026-09-08T00:00:00Z",
        )
        runner.get_state_values.return_value = {
            "approval": decision,
            "approval_status": "COMMITTED",
            "git_diff": diff,
        }
        app = create_app(runner=runner)
        client = TestClient(app)

        req = PublishPRRequest(repo_full_name="unregistered-org/unregistered-repo")
        resp = client.post("/api/v1/runs/run_unauth_repo/publish-pr", json=req.model_dump())
        assert resp.status_code == 403
        assert "REPO_ACCESS_DENIED" in resp.text
