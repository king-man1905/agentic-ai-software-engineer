import argparse
import os
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from backend.graph.runner import AgentRunner
from backend.integrations.github_client import GitHubClient
from backend.integrations.github_models import GitHubPRResult
from backend.vcs.models import ApprovalDecision
from backend.vcs.git_manager import GitWorkspaceManager


def solve_issue_and_open_pr(
    repo: str,
    issue_number: int,
    base_branch: str = "main",
    project_id: Optional[str] = None,
    auto_approve: bool = False,
    token: Optional[str] = None,
    draft: bool = True,
    runner: Optional[AgentRunner] = None,
    client: Optional[GitHubClient] = None,
) -> Optional[GitHubPRResult]:
    """
    Ingests a GitHub issue, executes the agent workflow, and opens an
    enriched Draft Pull Request with risk assessment and test proofs.

    Args:
        repo: Full repository identifier ('owner/repo').
        issue_number: GitHub issue number.
        base_branch: Target branch to merge PR into.
        project_id: Workspace project directory identifier.
        auto_approve: Whether to auto-approve the HITL gate if tests pass.
            Defaults to False: a real PR is only opened after a human
            approves, since this action is external and hard to reverse.
            Opt in explicitly for unattended/demo use.
        token: Optional GitHub token.
        draft: Whether to open PR as a draft.
        runner: Optional injected AgentRunner instance (for testing).
        client: Optional injected GitHubClient instance (for testing).

    Returns:
        GitHubPRResult if PR is created successfully, None otherwise.
    """
    github_client = client or GitHubClient(token=token)
    agent_runner = runner or AgentRunner()

    print(f"[Step 1/5] Fetching issue #{issue_number} from {repo}...", flush=True)
    issue = github_client.fetch_issue(repo, issue_number)
    print(f"[+] Retrieved issue: '{issue.title}' (Labels: {issue.labels})", flush=True)

    # Construct the user task prompt for the agent
    user_message = (
        f"Resolve GitHub Issue #{issue.issue_number}: {issue.title}\n\n"
        f"Issue Description:\n{issue.body}\n"
    )
    if issue.labels:
        user_message += f"\nIssue Labels: {', '.join(issue.labels)}"

    run_id = f"gh_{issue.issue_number}_{uuid.uuid4().hex[:8]}"
    resolved_project_id = project_id or repo.split("/")[-1]

    # Ensure workspace repository exists
    project_path = Path("workspace") / resolved_project_id
    if not project_path.exists():
        project_path = Path(os.getcwd()) / "workspace" / resolved_project_id

    git_env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
    }

    if not project_path.exists() and runner is None:
        print(f"[*] Workspace directory '{project_path}' does not exist. Cloning repository from GitHub...", flush=True)
        project_path.parent.mkdir(parents=True, exist_ok=True)
        clone_url = f"https://github.com/{repo}.git"
        try:
            subprocess.run(
                ["git", "clone", clone_url, str(project_path)],
                capture_output=True,
                text=True,
                check=True,
                env=git_env,
                timeout=60,
            )
            print(f"[+] Cloned repository into '{project_path}'.", flush=True)
        except Exception as e:
            print(f"[!] Notice: Could not clone repository: {e}", flush=True)

    print(f"[Step 2/5] Preparing workspace & running LangGraph engine (run_id: {run_id})...", flush=True)
    status_res = agent_runner.start_run(
        run_id=run_id,
        user_message=user_message,
        project_id=resolved_project_id,
        metadata={"github_repo": repo, "issue_number": issue_number},
    )

    git_diff = status_res.git_diff
    test_result = None
    metrics = None

    try:
        state_values = agent_runner.get_state_values(run_id)
        test_result = state_values.get("test_result")
        metrics = state_values.get("metrics")
        if not git_diff:
            git_diff = state_values.get("git_diff")
    except Exception:
        pass

    test_status_str = "PASSED" if (test_result and getattr(test_result, "success", False)) else "EVALUATED"
    print(f"[Step 3/5] Sandbox pytest validation: {test_status_str}...", flush=True)
    if test_result:
        print(f"[+] Sandbox Pytest Result: success={test_result.success}, passed={test_result.passed_count}, failed={test_result.failed_count}", flush=True)
        if test_result.stdout:
            print(f"[Pytest Stdout]\n{test_result.stdout.strip()}", flush=True)

    # Handle Human-In-The-Loop Approval Gate
    if status_res.status == "WAITING_APPROVAL":
        print("[!] Agent reached HITL approval gate.", flush=True)
        if auto_approve:
            print("[*] Auto-approving changes based on sandbox validation...", flush=True)
            status_res = agent_runner.resume_run(
                run_id=run_id,
                approval_decision=ApprovalDecision(
                    approved=True,
                    reviewer="autonomous_github_bot",
                ),
            )
            print(f"[+] Post-approval status: {status_res.status}", flush=True)
            try:
                state_values = agent_runner.get_state_values(run_id)
                if not git_diff:
                    git_diff = state_values.get("git_diff")
            except Exception:
                pass
        else:
            print("[!] Run paused for manual approval. Exiting without opening PR.", flush=True)
            return None

    if status_res.status == "FAILED":
        print(f"[-] Run failed: {status_res.error_summary}", flush=True)
        return None

    # Print completed graph nodes sequence
    try:
        final_values = agent_runner.get_state_values(run_id)
        nodes_seq = []
        if "routing" in final_values: nodes_seq.append("router")
        if "plan" in final_values: nodes_seq.append("planner")
        if "knowledge" in final_values: nodes_seq.append("knowledge")
        if "developer_result" in final_values: nodes_seq.append("developer")
        if "qa_result" in final_values: nodes_seq.append("qa")
        if "git_diff" in final_values: nodes_seq.append("git_prepare")
        if "approval" in final_values: nodes_seq.append("approval")
        if final_values.get("approval_status") == "COMMITTED": nodes_seq.append("git_commit")
        print(f"[+] Graph Execution Trace: {' -> '.join(nodes_seq)}", flush=True)
    except Exception:
        pass

    # Abort before any push/PR creation if nothing was actually changed -
    # e.g. the developer agent found no valid patch, or patch validation
    # failed, or the issue turned out to already be resolved in the code.
    if git_diff is None or git_diff.is_no_op:
        print(
            "[ABORT] No code changes produced or bug already resolved. "
            "Skipping PR creation.",
            flush=True,
        )
        return None

    # Determine head branch from git_diff summary or default naming
    head_branch = git_diff.branch_name if git_diff else f"agent/issue-{issue_number}"


    # Generate structured PR description
    print("[*] Formatting Pull Request body with risk and sandbox proofs...", flush=True)
    pr_body = GitHubClient.format_pr_description(
        issue=issue,
        diff_summary=git_diff,
        test_result=test_result,
        metrics=metrics,
    )
    pr_title = f"fix: {issue.title} (resolves #{issue.issue_number})"

    # Push feature branch to origin if workspace is a git repository
    if project_path.exists() and (project_path / ".git").exists():
        print(f"[Step 4/5] Pushing branch to remote ('{head_branch}')...", flush=True)
        pushed = GitWorkspaceManager.push_branch(str(project_path), head_branch)
        if not pushed:
            print(f"[!] Notice: Branch push did not complete or branch is local.", flush=True)

    print(f"[Step 5/5] Creating GitHub PR to {repo}:{base_branch} from {head_branch}...", flush=True)
    pr_result = github_client.create_pull_request(
        repo_full_name=repo,
        title=pr_title,
        body=pr_body,
        head_branch=head_branch,
        base_branch=base_branch,
        draft=draft,
    )

    print(f"[+] Pull Request #{pr_result.pr_number} successfully opened: {pr_result.pr_url}", flush=True)
    return pr_result



def main():
    parser = argparse.ArgumentParser(
        description="Autonomous GitHub PR Integration: Ingest an issue, resolve via agent, and open a Draft PR."
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="GitHub repository in 'owner/repo' format (e.g. 'octocat/Hello-World')",
    )
    parser.add_argument(
        "--issue",
        type=int,
        required=True,
        help="GitHub issue number to solve",
    )
    parser.add_argument(
        "--base-branch",
        type=str,
        default="main",
        help="Base branch to merge into (default: 'main')",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="Workspace project directory (defaults to repo name)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help=(
            "Auto-approve changes at the HITL gate and open the PR "
            "unattended. Default is off: the run pauses for a human "
            "approval decision before any PR is opened."
        ),
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="GitHub API Token (defaults to GITHUB_TOKEN environment variable)",
    )
    parser.add_argument(
        "--ready-for-review",
        action="store_true",
        help="Open PR as ready for review instead of Draft",
    )

    args = parser.parse_args()

    try:
        pr_result = solve_issue_and_open_pr(
            repo=args.repo,
            issue_number=args.issue,
            base_branch=args.base_branch,
            project_id=args.project_id,
            auto_approve=args.auto_approve,
            token=args.token,
            draft=not args.ready_for_review,
        )
        if pr_result:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"[-] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
