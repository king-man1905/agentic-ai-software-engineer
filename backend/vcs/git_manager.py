import difflib
import hashlib
import os
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from backend.developer.models import FilePatch
from backend.developer.patcher import SafePatcher
from backend.policy.patterns import (
    DEPENDENCY_MANIFEST_PATTERNS,
    CI_CD_PATTERNS,
    INFRA_CONFIG_PATTERNS,
)
from backend.vcs.models import GitDiffSummary


# File patterns that trigger elevated risk assessment. Reuses the same
# dependency-manifest/CI/infra-config patterns the policy engine recognizes
# (backend/policy/patterns.py) instead of re-typing them, restricted to the
# subset this heuristic has always flagged - the shared list is a superset
# covering a few ecosystems (package-lock.json, Gemfile, pom.xml,
# build.gradle, .circleci/, .travis.yml, azure-pipelines.yml, alembic.ini)
# this risk heuristic never scored as elevated risk, and widening that
# scope here is a policy-adjacent behavior change out of scope for this
# refactor - plus two migration-path patterns unique to this module.
_RISK_HEURISTIC_SUBSET = {
    "requirements.txt", "package.json", "yarn.lock", "poetry.lock", "Pipfile",
    "setup.py", "setup.cfg", "pyproject.toml",
    ".github/", ".gitlab-ci", "Jenkinsfile",
    "docker-compose", "Dockerfile", "Makefile", ".env", ".env.",
}
HIGH_RISK_PATTERNS = [
    p
    for p in DEPENDENCY_MANIFEST_PATTERNS + CI_CD_PATTERNS + INFRA_CONFIG_PATTERNS
    if p in _RISK_HEURISTIC_SUBSET
] + ["alembic/", "migrations/", "migration/"]

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


def _run_git(
    args: List[str],
    repo_path: str,
    check: bool = True,
    timeout: float = 60,
) -> subprocess.CompletedProcess:
    """
    Runs a `git` subprocess with the standard non-interactive environment,
    capturing text output. Centralizes the invocation shape every git
    operation in this module previously repeated individually - behavior
    (env, timeout, capture_output, text mode) is unchanged, only the
    repetition is removed.
    """
    return subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=check,
        env=_git_env(),
        timeout=timeout,
    )


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
            _run_git(["checkout", "-b", branch_name], repo_path)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            try:
                _run_git(["checkout", branch_name], repo_path)
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
            result = _run_git(["show", f"HEAD:{file_path}"], repo_path, check=False)
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
            _run_git(["add", "."], repo_path)
            _run_git(["commit", "-m", message], repo_path)
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
            _run_git(["push", "-u", remote, branch_name], repo_path)
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
            result = _run_git(["rev-parse", "--verify", "main"], repo_path, check=False)
            if result.returncode != 0:
                fallback_branch = "master"

            _run_git(["checkout", fallback_branch], repo_path)
            _run_git(["branch", "-D", branch_name], repo_path)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False


    @staticmethod
    def compute_patch_hash(unified_diff: str) -> str:
        """
        Computes the SHA-256 cryptographic digest of the exact unified diff text.
        Returns a deterministic 64-character hexadecimal string.
        """
        return hashlib.sha256((unified_diff or "").encode("utf-8")).hexdigest()

    @classmethod
    def verify_workspace_drift(
        cls,
        repo_path: str,
        expected_diff: str,
        expected_hash: str,
        files_changed: List[str],
    ) -> Tuple[bool, str]:
        """
        Verifies that the workspace has not drifted from the approved diff before committing.
        Returns (is_valid, error_message).
        """
        if not expected_hash:
            return True, ""

        # Re-compute hash of expected_diff to ensure internal consistency
        computed_expected = cls.compute_patch_hash(expected_diff)
        if expected_hash != computed_expected:
            return False, (
                f"Stored patch_hash '{expected_hash}' does not match hash of expected diff '{computed_expected}'."
            )

        # If workspace does not exist, nothing to drift
        repo = Path(repo_path)
        if not repo.exists():
            return True, ""

        file_changes: Dict[str, Tuple[str, str]] = {}
        has_baseline = False
        for fpath in files_changed:
            abs_path = repo / fpath
            original = cls._read_head_content(repo_path, fpath)
            if original:
                has_baseline = True
            current = ""
            if abs_path.exists():
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        current = f.read()
                except Exception:
                    pass
            file_changes[fpath] = (original, current)

        if has_baseline:
            recalculated_diff, _, _ = cls.compute_diff(file_changes)
            current_hash = cls.compute_patch_hash(recalculated_diff)

            if current_hash != expected_hash:
                return False, (
                    f"Workspace drift detected: current diff hash '{current_hash}' "
                    f"does not match approved patch hash '{expected_hash}'."
                )

        return True, ""

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
        4. Compute unified diff and cryptographic patch_hash
        5. Evaluate risk
        6. Return structured GitDiffSummary
        """
        branch_name = cls.generate_branch_name(task_id)

        # Attempt to create feature branch (non-blocking if git is unavailable)
        cls.create_feature_branch(repo_path, branch_name)

        # Apply patches and collect before/after content
        file_changes = cls.apply_patches(repo_path, patches)

        # Compute diff and cryptographic hash
        files_changed = list(file_changes.keys())
        unified_diff, lines_added, lines_deleted = cls.compute_diff(file_changes)
        patch_hash = cls.compute_patch_hash(unified_diff)

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
            patch_hash=patch_hash,
            risk_score=risk_score,
            risk_reasons=risk_reasons,
        )

