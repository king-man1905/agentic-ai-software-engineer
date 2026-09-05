import os
from typing import Any, Dict, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

from backend.integrations.github_models import GitHubIssuePayload, GitHubPRResult
from backend.sandbox.models import TestExecutionResult
from backend.vcs.models import GitDiffSummary


class GitHubError(Exception):
    """Base exception for structured GitHub integration failures."""
    def __init__(self, message: str, code: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class GitHubAuthError(GitHubError, PermissionError, ValueError):
    """Raised when GitHub authentication fails (bad or missing token)."""
    def __init__(self, message: str = "GitHub API authentication failed.", status_code: int = 401):
        super().__init__(message, code="GITHUB_AUTH_FAILURE", status_code=status_code)


class GitHubPermissionError(GitHubError, PermissionError):
    """Raised when GitHub permissions are insufficient."""
    def __init__(self, message: str = "GitHub API permission denied.", status_code: int = 403):
        super().__init__(message, code="GITHUB_PERMISSION_DENIED", status_code=status_code)


class GitHubNotFoundError(GitHubError, FileNotFoundError):
    """Raised when target repository or issue is not found."""
    def __init__(self, message: str = "GitHub resource not found.", status_code: int = 404):
        super().__init__(message, code="GITHUB_REPO_NOT_FOUND", status_code=status_code)


class GitHubBranchConflictError(GitHubError):
    """Raised when branch collision or merge conflict occurs."""
    def __init__(self, message: str = "GitHub branch conflict.", status_code: int = 409):
        super().__init__(message, code="GITHUB_BRANCH_CONFLICT", status_code=status_code)


class GitHubRateLimitError(GitHubError, PermissionError):
    """Raised when GitHub rate limits or secondary abuse limits are exceeded."""
    def __init__(self, message: str = "GitHub API rate limit exceeded.", status_code: int = 429):
        super().__init__(message, code="GITHUB_RATE_LIMIT", status_code=status_code)


class GitHubApiError(GitHubError):
    """Raised on internal GitHub service errors or network failures."""
    def __init__(self, message: str = "GitHub API service error.", status_code: int = 500):
        super().__init__(message, code="GITHUB_API_FAILURE", status_code=status_code)


class GitHubPRCreationError(GitHubError, ValueError):
    """Raised when PR payload is unprocessable or PR creation fails."""
    def __init__(self, message: str = "GitHub PR creation unprocessable.", status_code: int = 422):
        super().__init__(message, code="PR_CREATION_FAILURE", status_code=status_code)


class GitHubClient:
    """
    Client for interacting with GitHub REST APIs (Issues, Pull Requests)
    using httpx with structured error handling, token isolation, and PR formatting.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = "https://api.github.com",
        http_client: Optional[httpx.Client] = None,
    ):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.base_url = base_url.rstrip("/")
        self._client = http_client

    def _get_headers(self, token_override: Optional[str] = None) -> Dict[str, str]:
        active_token = token_override or self.token
        if not active_token:
            raise GitHubAuthError(
                "GITHUB_TOKEN environment variable or token parameter is required."
            )
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {active_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def fetch_issue(
        self,
        repo_full_name: str,
        issue_number: int,
        token_override: Optional[str] = None,
    ) -> GitHubIssuePayload:
        """
        Fetches an issue by repository and issue number.
        """
        headers = self._get_headers(token_override=token_override)
        url = f"{self.base_url}/repos/{repo_full_name}/issues/{issue_number}"

        client = self._client or httpx.Client(timeout=30.0)
        try:
            response = client.get(url, headers=headers)
            if response.status_code == 401:
                raise GitHubAuthError(
                    f"GitHub API authorization failed (401): {response.text}",
                    status_code=401,
                )
            if response.status_code == 403:
                txt = response.text.lower()
                if "rate limit" in txt or "secondary rate" in txt:
                    raise GitHubRateLimitError(
                        f"GitHub API rate limit exceeded (403): {response.text}",
                        status_code=403,
                    )
                raise GitHubPermissionError(
                    f"GitHub API authorization failed (403): {response.text}",
                    status_code=403,
                )
            if response.status_code == 404:
                raise GitHubNotFoundError(
                    f"Issue #{issue_number} in repository '{repo_full_name}' was not found (404).",
                    status_code=404,
                )
            if response.status_code == 429:
                raise GitHubRateLimitError(
                    f"GitHub API rate limit exceeded (429): {response.text}",
                    status_code=429,
                )
            if response.status_code >= 500:
                raise GitHubApiError(
                    f"GitHub API service error ({response.status_code}): {response.text}",
                    status_code=response.status_code,
                )
            response.raise_for_status()

            data = response.json()
            raw_labels = data.get("labels", [])
            labels = []
            for item in raw_labels:
                if isinstance(item, dict) and "name" in item:
                    labels.append(item["name"])
                elif isinstance(item, str):
                    labels.append(item)

            return GitHubIssuePayload(
                repo_full_name=repo_full_name,
                issue_number=data.get("number", issue_number),
                title=data.get("title", ""),
                body=data.get("body") or "",
                labels=labels,
            )
        finally:
            if self._client is None:
                client.close()

    def create_pull_request(
        self,
        repo_full_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        draft: bool = True,
        token_override: Optional[str] = None,
    ) -> GitHubPRResult:
        """
        Creates a new Pull Request on the target repository.
        """
        headers = self._get_headers(token_override=token_override)
        url = f"{self.base_url}/repos/{repo_full_name}/pulls"

        payload = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
            "draft": draft,
        }

        client = self._client or httpx.Client(timeout=30.0)
        try:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code == 401:
                raise GitHubAuthError(
                    f"GitHub API authorization failed (401): {response.text}",
                    status_code=401,
                )
            if response.status_code == 403:
                txt = response.text.lower()
                if "rate limit" in txt or "secondary rate" in txt:
                    raise GitHubRateLimitError(
                        f"GitHub API rate limit exceeded (403): {response.text}",
                        status_code=403,
                    )
                raise GitHubPermissionError(
                    f"GitHub API authorization failed (403): {response.text}",
                    status_code=403,
                )
            if response.status_code == 404:
                raise GitHubNotFoundError(
                    f"Repository '{repo_full_name}' or branch '{head_branch}' was not found (404).",
                    status_code=404,
                )
            if response.status_code == 409:
                raise GitHubBranchConflictError(
                    f"GitHub branch conflict (409): {response.text}",
                    status_code=409,
                )
            if response.status_code == 422:
                raise GitHubPRCreationError(
                    f"GitHub PR creation unprocessable (422): {response.text}",
                    status_code=422,
                )
            if response.status_code == 429:
                raise GitHubRateLimitError(
                    f"GitHub API rate limit exceeded (429): {response.text}",
                    status_code=429,
                )
            if response.status_code >= 500:
                raise GitHubApiError(
                    f"GitHub API service error ({response.status_code}): {response.text}",
                    status_code=response.status_code,
                )
            response.raise_for_status()

            data = response.json()
            return GitHubPRResult(
                pr_number=data.get("number", 0),
                pr_url=data.get("html_url", ""),
                head_branch=head_branch,
                base_branch=base_branch,
                is_draft=data.get("draft", draft),
            )
        finally:
            if self._client is None:
                client.close()

    @staticmethod
    def format_pr_description(
        issue: GitHubIssuePayload,
        diff_summary: Optional[GitDiffSummary],
        test_result: Optional[TestExecutionResult],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Formats a structured Markdown description for an autonomous Pull Request.
        """
        # 1. Issue closing reference
        closing_ref = f"Closes #{issue.issue_number}"

        # 2. Risk Evaluation
        if diff_summary:
            risk_score = diff_summary.risk_score.upper()
            files_count = len(diff_summary.files_changed)
            lines_added = diff_summary.lines_added
            lines_deleted = diff_summary.lines_deleted
            reasons = "<br>".join(diff_summary.risk_reasons) if diff_summary.risk_reasons else "Standard automated changeset."
            unified_diff = diff_summary.unified_diff or "No unified diff changes."
        else:
            risk_score = "LOW"
            files_count = 0
            lines_added = 0
            lines_deleted = 0
            reasons = "No diff summary available."
            unified_diff = "No changes recorded."

        # Risk badge color style
        badge_style = {
            "LOW": "🟢 `LOW`",
            "MEDIUM": "🟡 `MEDIUM`",
            "HIGH": "🔴 `HIGH`",
        }.get(risk_score, f"`{risk_score}`")

        # 3. Sandbox Verification Proof
        if test_result:
            test_status = "✅ `PASSED`" if test_result.success else "❌ `FAILED`"
            passed_count = test_result.passed_count
            failed_count = test_result.failed_count
            duration_str = f"{test_result.duration_seconds:.2f}s"
            exit_code = str(test_result.exit_code)
            error_block = ""
            if test_result.error_summary:
                error_block = f"\n> **Failure Summary:**\n> ```\n> {test_result.error_summary}\n> ```\n"
        else:
            test_status = "⚪ `SKIPPED`"
            passed_count = 0
            failed_count = 0
            duration_str = "0.00s"
            exit_code = "0"
            error_block = ""

        # 4. Telemetry & Cost Metrics
        m = metrics or {}
        # `tracked` distinguishes "genuinely measured zero usage" from
        # "usage wasn't available to measure" (e.g. ChatNVIDIA's
        # with_structured_output doesn't support include_raw, so no call
        # made through it ever reports token counts). Absent the flag,
        # a caller-supplied metrics dict is assumed tracked for backward
        # compatibility; metrics=None is treated as untracked.
        is_tracked = m.get("tracked", metrics is not None)
        prompt_tokens = m.get("prompt_tokens", 0)
        completion_tokens = m.get("completion_tokens", 0)
        total_tokens = m.get("total_tokens", prompt_tokens + completion_tokens)
        cost_usd = m.get("estimated_cost_usd", 0.0)

        prompt_tokens_str = f"{prompt_tokens:,}" if is_tracked else "N/A"
        completion_tokens_str = f"{completion_tokens:,}" if is_tracked else "N/A"
        total_tokens_str = f"{total_tokens:,}" if is_tracked else "N/A"
        cost_str = f"${cost_usd:.4f}" if is_tracked else "N/A"

        # 5. Build PR markdown
        markdown = f"""## Summary of Changes
{closing_ref}

**Issue:** {issue.title}

### 🛡️ Risk Assessment
| Metric | Assessment |
| :--- | :--- |
| **Risk Level** | {badge_style} |
| **Files Changed** | `{files_count}` |
| **Lines Added / Deleted** | `+{lines_added} / -{lines_deleted}` |
| **Risk Reasons** | {reasons} |

### 🧪 Sandbox Verification Proof
| Check | Status | Passed | Failed | Duration | Exit Code |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pytest Suite** | {test_status} | `{passed_count}` | `{failed_count}` | `{duration_str}` | `{exit_code}` |
{error_block}
### 📊 Telemetry & Resource Consumption
| Metric | Value |
| :--- | :--- |
| **Prompt Tokens** | `{prompt_tokens_str}` |
| **Completion Tokens** | `{completion_tokens_str}` |
| **Total Tokens** | `{total_tokens_str}` |
| **Estimated Cost (USD)** | `{cost_str}` |

<details>
<summary><b>View Unified Diff ({files_count} files changed)</b></summary>

```diff
{unified_diff}
```

</details>

---
*Generated autonomously by [Agentic AI Software Engineer](https://github.com).*
"""
        return markdown
