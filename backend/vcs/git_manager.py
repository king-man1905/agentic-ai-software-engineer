import difflib
import os
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.developer.models import FilePatch
from backend.developer.patcher import SafePatcher
from backend.vcs.models import GitDiffSummary


# File patterns that trigger elevated risk assessment
HIGH_RISK_PATTERNS = [
    ".env", ".env.", "alembic/", "migrations/", "migration/",
    "docker-compose", "Dockerfile", "Makefile",
    ".github/", ".gitlab-ci", "Jenkinsfile",
    "setup.py", "setup.cfg", "pyproject.toml",
    "requirements.txt", "package.json", "yarn.lock",
    "poetry.lock", "Pipfile",
]

CONFIG_EXTENSIONS = [
    ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf", ".json",
]


def _git_env() -> Dict[str, str]:
    """Ensures non-interactive Git operations without GUI or terminal prompts."""
    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
    }


class GitWorkspaceManager:
    """
    Manages isolated Git workspace operations: branch creation, patch application,
    diff computation, risk evaluation, staging/committing, and cleanup.
    """

    @staticmethod
    def generate_branch_name(task_id: str) -> str:
        """
        Produces a feature branch name from a task identifier: a short
        sanitized slug plus a random entropy suffix, so two runs whose
        task_id/project_id happen to share the same first 8 characters
        (e.g. "sandbox-ai-demo" and "sandbox-ai-demo-v2") never collide on
        the same branch - a real collision hit in production, where it
        silently clobbered an unrelated PR.
        Format: agent/task-{slug}-{short_uuid}
        """
        slug_source = task_id[:8] if len(task_id) > 8 else task_id
        # Sanitize: replace spaces/special chars with hyphens
        sanitized = "".join(c if c.isalnum() or c == "-" else "-" for c in slug_source)
        slug = sanitized.strip("-") or "task"
        suffix = uuid.uuid4().hex[:8]
        return f"agent/task-{slug}-{suffix}"

    @staticmethod
    def create_feature_branch(repo_path: str, branch_name: str) -> bool:
        """
        Creates and checks out a new feature branch in the given repository.
        If the branch already exists, checks it out.
        Returns True on success, False on failure.
        """
        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                env=_git_env(),
                timeout=60,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            try:
                subprocess.run(
                    ["git", "checkout", branch_name],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                    env=_git_env(),
                    timeout=60,
                )
                return True
            except Exception:
                return False



    @staticmethod
    def _read_head_content(repo_path: str, file_path: str) -> str:
        """
        Reads a file's content as of the last commit (HEAD). Used as the
        "before" baseline instead of the current working-tree file, since
        the working tree may already have been mutated by an earlier step
        (e.g. the developer agent writes its patch directly to disk so QA
        can test the real fix) - the committed blob is unaffected by that.
        Returns "" if there's no commit history yet or the file is new.
        """
        try:
            result = subprocess.run(
                ["git", "show", f"HEAD:{file_path}"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                env=_git_env(),
                timeout=60,
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return ""

    @staticmethod
    def apply_patches(
        repo_path: str,
        patches: List[FilePatch],
    ) -> Dict[str, Tuple[str, str]]:
        """
        Applies a list of FilePatch items to files in repo_path using SafePatcher.
        Returns a dict mapping file_path -> (original_content, patched_content)
        for each successfully applied patch.
        Raises ValueError if any patch fails AST pre-flight validation.

        If a file's working-tree content already differs from its committed
        (HEAD) content - i.e. it was already patched directly on disk by an
        earlier step - that working-tree content is trusted as the final
        result instead of re-running SafePatcher against it: re-applying an
        already-applied patch either fails to match (the "before" snippet
        is gone) or, worse, re-inserts it a second time.
        """
        results: Dict[str, Tuple[str, str]] = {}

        for patch in patches:
            abs_path = Path(repo_path) / patch.file_path

            original_content = GitWorkspaceManager._read_head_content(repo_path, patch.file_path)

            current_content = ""
            if abs_path.exists():
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        current_content = f.read()
                except Exception:
                    pass

            if not original_content:
                # No committed version to diff against (new file, or no git
                # history yet) - fall back to the working tree as baseline,
                # same as the previous behavior.
                original_content = current_content

            if current_content and current_content != original_content:
                # Already applied directly to disk - trust it.
                patched_content = current_content
            else:
                validation = SafePatcher.apply_patch(original_content, patch)

                if not validation.is_valid:
                    raise ValueError(
                        f"Patch validation failed for {patch.file_path}: "
                        f"{validation.syntax_errors}"
                    )

                patched_content = validation.applied_content or ""

                abs_path.parent.mkdir(parents=True, exist_ok=True)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(patched_content)

            results[patch.file_path] = (original_content, patched_content)

        return results

    @staticmethod
    def compute_diff(
        file_changes: Dict[str, Tuple[str, str]],
    ) -> Tuple[str, int, int]:
        """
        Generates unified diff text from before/after file content pairs.
        Returns (unified_diff_text, lines_added, lines_deleted).
        """
        all_diff_lines: List[str] = []
        total_added = 0
        total_deleted = 0

        for file_path, (original, patched) in sorted(file_changes.items()):
            original_lines = original.splitlines(keepends=True)
            patched_lines = patched.splitlines(keepends=True)

            diff = difflib.unified_diff(
                original_lines,
                patched_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm="",
            )

            for line in diff:
                all_diff_lines.append(line)
                stripped = line.rstrip("\n\r")
                if stripped.startswith("+") and not stripped.startswith("+++"):
                    total_added += 1
                elif stripped.startswith("-") and not stripped.startswith("---"):
                    total_deleted += 1

        unified_diff = "\n".join(all_diff_lines)
        return unified_diff, total_added, total_deleted

    @staticmethod
    def evaluate_risk(
        files_changed: List[str],
        lines_added: int,
        lines_deleted: int,
        unified_diff: str,
    ) -> Tuple[str, List[str]]:
        """
        Evaluates the risk level of a set of changes using heuristics.
        Returns (risk_score, risk_reasons).
        """
        reasons: List[str] = []
        is_high = False
        is_medium = False

        # Check for high-risk file patterns
        for fpath in files_changed:
            normalized = fpath.replace("\\", "/").lower()

            for pattern in HIGH_RISK_PATTERNS:
                if pattern in normalized:
                    reasons.append(
                        f"High-risk file modified: {fpath} (matches pattern '{pattern}')"
                    )
                    is_high = True
                    break

            # Check config extensions
            for ext in CONFIG_EXTENSIONS:
                if normalized.endswith(ext):
                    reasons.append(
                        f"Configuration file modified: {fpath}"
                    )
                    is_high = True
                    break

        # Check deletion ratio
        total_changed = lines_added + lines_deleted
        if total_changed > 0:
            deletion_ratio = lines_deleted / total_changed
            if deletion_ratio > 0.5:
                reasons.append(
                    f"High deletion ratio: {lines_deleted}/{total_changed} "
                    f"({deletion_ratio:.0%} of changes are deletions)"
                )
                is_high = True

        # Check number of files changed
        if len(files_changed) > 5:
            reasons.append(
                f"Large changeset: {len(files_changed)} files modified"
            )
            is_medium = True

        if is_high:
            return "HIGH", reasons
        if is_medium:
            return "MEDIUM", reasons

        if not reasons:
            reasons.append("Standard code change with low risk profile.")

        return "LOW", reasons

    @staticmethod
    def stage_and_commit(repo_path: str, message: str) -> bool:
        """
        Stages all changes and creates a commit. Returns True on success.
        """
        try:
            subprocess.run(
                ["git", "add", "."],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                env=_git_env(),
                timeout=60,
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                env=_git_env(),
                timeout=60,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def push_branch(repo_path: str, branch_name: str, remote: str = "origin") -> bool:
        """
        Pushes a local branch to the remote repository.
        Returns True on success, False on failure.
        """
        try:
            subprocess.run(
                ["git", "push", "-u", remote, branch_name],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                env=_git_env(),
                timeout=60,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            if hasattr(e, "stderr") and e.stderr:
                print(f"[!] Push error: {e.stderr.strip()}")
            return False

    @staticmethod
    def cleanup_branch(repo_path: str, branch_name: str) -> bool:
        """
        Checks out the previous branch (main/master) and deletes the feature branch.
        Returns True on success.
        """
        try:
            # Try 'main' first, fallback to 'master'
            fallback_branch = "main"
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "main"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                env=_git_env(),
                timeout=60,
            )
            if result.returncode != 0:
                fallback_branch = "master"

            subprocess.run(
                ["git", "checkout", fallback_branch],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                env=_git_env(),
                timeout=60,
            )
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                env=_git_env(),
                timeout=60,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False


    @classmethod
    def prepare_diff_summary(
        cls,
        repo_path: str,
        patches: List[FilePatch],
        task_id: str,
    ) -> GitDiffSummary:
        """
        Orchestrates the full preparation pipeline:
        1. Generate branch name
        2. Create feature branch (best-effort, non-blocking if git unavailable)
        3. Apply patches to workspace
        4. Compute unified diff
        5. Evaluate risk
        6. Return structured GitDiffSummary
        """
        branch_name = cls.generate_branch_name(task_id)

        # Attempt to create feature branch (non-blocking if git is unavailable)
        cls.create_feature_branch(repo_path, branch_name)

        # Apply patches and collect before/after content
        file_changes = cls.apply_patches(repo_path, patches)

        # Compute diff
        files_changed = list(file_changes.keys())
        unified_diff, lines_added, lines_deleted = cls.compute_diff(file_changes)

        # Evaluate risk
        risk_score, risk_reasons = cls.evaluate_risk(
            files_changed, lines_added, lines_deleted, unified_diff
        )

        return GitDiffSummary(
            branch_name=branch_name,
            files_changed=files_changed,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            unified_diff=unified_diff,
            risk_score=risk_score,
            risk_reasons=risk_reasons,
        )
