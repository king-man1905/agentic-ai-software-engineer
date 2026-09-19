import base64
import difflib
import hashlib
import os
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.developer.models import FilePatch
from backend.developer.patcher import SafePatcher
from backend.policy.patterns import (
    DEPENDENCY_MANIFEST_PATTERNS,
    CI_CD_PATTERNS,
    INFRA_CONFIG_PATTERNS,
)
from backend.policy.path_filter import safe_repo_relative_path
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


def _is_own_git_repo(repo_path: str) -> bool:
    """
    True only when repo_path is itself a git repository root (repo_path/.git
    exists). Every git-mutating call in this module must check this before
    running any git subprocess with cwd=repo_path: a plain `git` command run
    from a directory that isn't a repo root of its own doesn't fail just
    because that directory lacks history - git walks up to the nearest
    ANCESTOR repository instead (e.g. this tool's own checkout, for a
    project workspace nested under it) and silently operates there instead,
    with a normal success exit code. Confirmed in production: this let
    `git show HEAD:<path>` return an unrelated file from the wrong
    repository (see _read_head_content), and would equally let
    `git checkout -b`/`git add .`/`git commit` create branches and commits
    in that unrelated repository - never a code path that should be
    reachable for a project workspace that was never actually git-cloned.
    """
    return (Path(repo_path) / ".git").is_dir()


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


def build_github_auth_header(token: str) -> str:
    """
    Builds an in-memory HTTP Basic Authorization header value for GitHub's
    HTTPS Git-over-HTTP endpoint, using the standard "x-access-token:<token>"
    convention (the same userinfo scheme previously embedded directly in
    clone URLs). Callers pass the result as `auth_header` to
    clone_repository()/push_branch() instead of embedding the token in a
    URL - it's supplied to git via a process-scoped `-c http.extraHeader=...`
    flag, never written to any file. Never log, print, or persist the
    return value anywhere.
    """
    encoded = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
    return f"Authorization: Basic {encoded}"


def _with_auth_config(args: List[str], auth_header: Optional[str]) -> List[str]:
    """
    Prepends a process-scoped `-c http.extraHeader=<auth_header>` flag to a
    git argv when an auth header is supplied - never written to `.git/config`
    or any other file, and never part of the remote URL itself, unlike the
    previous embedded-credential-URL approach.
    """
    if not auth_header:
        return args
    return ["-c", f"http.extraHeader={auth_header}"] + args


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
    def clone_repository(
        clone_url: str,
        project_path: str,
        timeout: float = 60,
        auth_header: Optional[str] = None,
    ) -> bool:
        """
        Clones `clone_url` into `project_path` via the same non-interactive
        git subprocess convention every other operation in this module
        uses. Idempotent: if `project_path` is ALREADY a git repository in
        its own right (has its own .git), returns True immediately without
        touching it or attempting to clone - callers never need their own
        validity check first. A `project_path` that merely exists as a
        plain directory (never actually cloned - e.g. a stale or manually
        created leftover) is NOT treated as already-provisioned: git
        itself will refuse to clone into a non-empty directory, so this
        correctly falls through to an attempted (and failing, `False`)
        clone rather than silently claiming success over content that was
        never actually verified to be the right repository. Returns False
        (never raises) on any clone failure, so callers decide how to
        surface that (e.g. as an explicit run failure).

        This function is intentionally URL-agnostic - it doesn't know
        about GitHub, tokens, or authorization; a caller that needs
        authenticated access builds a pre-formed HTTP Authorization header
        value (see build_github_auth_header()) and passes it as
        `auth_header`, rather than embedding credentials in `clone_url`
        itself. When set, the header is supplied to git via a process-
        scoped `-c http.extraHeader=...` flag - never written to any file
        - so `clone_url` should always be a plain, credential-free URL;
        the resulting clone's `remote.origin.url` (and `git remote -v`)
        will only ever contain whatever `clone_url` itself was.
        SECURITY: this function never logs, prints, or returns `clone_url`
        or `auth_header`, nor any subprocess stdout/stderr (which could
        otherwise echo a credential embedded in the URL back verbatim on
        failure) - callers must uphold the same rule with whatever they build.

        `project_path` is always resolved to an absolute path before use
        (run_777a478d62df): a relative `project_path` combined with the
        relative subprocess `cwd` used below (`dest.parent`) let git
        resolve the destination argument a SECOND time against its own
        cwd, cloning into a doubled/nested directory while still exiting
        0 - defense in depth against any caller (present or future)
        passing a CWD-relative destination, on top of the fix in
        resolve_workspace_path() itself.
        """
        dest = Path(project_path).resolve()
        if _is_own_git_repo(str(dest)):
            return True
        args = _with_auth_config(["clone", clone_url, str(dest)], auth_header)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            _run_git(args, repo_path=str(dest.parent), timeout=timeout)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    @staticmethod
    def create_feature_branch(repo_path: str, branch_name: str) -> bool:
        """
        Creates and checks out a new feature branch in the given repository.
        If the branch already exists, checks it out.
        Returns True on success, False on failure.

        Fails closed (False, no git command run at all) when repo_path
        isn't a git repository in its own right - see _is_own_git_repo.
        Without this, `git checkout -b` for such a path would silently
        create and check out the branch in the nearest ANCESTOR
        repository instead (confirmed in production: this tool's own
        checkout), rather than failing or no-op'ing for a project that
        was never actually git-cloned.
        """
        if not _is_own_git_repo(repo_path):
            return False
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

        Requires repo_path to be a git repository ROOT in its own right
        (i.e. repo_path/.git exists) before ever running `git show`. A
        project workspace that was never `git init`/cloned (common for ad
        hoc test projects) has no .git of its own; running a bare `git`
        command with cwd=repo_path in that case doesn't fail - git walks
        up to the nearest ANCESTOR repository (e.g. this tool's own
        checkout, if the workspace happens to live under it) and
        `git show HEAD:<path>` resolves <path> relative to THAT repo's
        root, silently returning a completely unrelated file's content
        with exit code 0. That produced a real, confirmed production bug:
        risk/diff scoring compared an unrelated 134-line file against the
        actual patch, reporting a bogus ~79% "deletion" that had nothing
        to do with what was actually written to the target project.
        """
        if not _is_own_git_repo(repo_path):
            return ""
        try:
            result = _run_git(["show", f"HEAD:{file_path}"], repo_path, check=False)
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return ""

    @staticmethod
    def get_remote_url(repo_path: str, remote: str = "origin") -> Optional[str]:
        """
        Returns the configured URL for `remote` in repo_path's own git
        config, or None if repo_path isn't a git repository in its own
        right or the remote isn't configured. Used to verify an existing,
        already-cloned workspace actually corresponds to the repository
        currently being authorized for it before trusting it as
        "already provisioned" - see
        AgentRunner._ensure_workspace_provisioned, which must never
        silently reuse a workspace that was cloned from a DIFFERENT
        repository under the same project_id.
        """
        if not _is_own_git_repo(repo_path):
            return None
        try:
            result = _run_git(["remote", "get-url", remote], repo_path, check=False)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

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

        A patch with an empty original_code_snippet is developer_node's
        full-file-write convention (its no-repo-context fallback path):
        updated_code_snippet IS the file's complete intended content, and
        that path already writes it directly to disk in the same call that
        builds the FilePatch. That case is detected up front (before the
        general "no committed version" fallback below could substitute
        current_content in as the baseline and make a brand-new file's
        diff look like nothing changed) and always trusts the already-
        written current_content as the result, diffed against the true
        prior committed content (empty for a genuinely new/untracked
        file). A real snippet patch (non-empty original_code_snippet)
        takes the general path, unchanged from before.
        """
        results: Dict[str, Tuple[str, str]] = {}
        repo_root = Path(repo_path)

        for patch in patches:
            # SECURITY: patch.file_path is LLM-controlled - contained
            # before it can be joined into a filesystem path. This is the
            # most sensitive of the read/write sites since the write below
            # creates missing parent directories; an unsafe path fails the
            # whole diff-preparation step rather than writing anywhere or
            # creating directories outside repo_root.
            abs_path = safe_repo_relative_path(repo_root, patch.file_path)
            if abs_path is None:
                raise ValueError(
                    f"Refusing to apply patch: "
                    f"'{patch.file_path}' is not a safe repository-relative path."
                )

            original_content = GitWorkspaceManager._read_head_content(repo_path, patch.file_path)

            current_content = ""
            if abs_path.exists():
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        current_content = f.read()
                except Exception:
                    pass

            if not patch.original_code_snippet and current_content:
                patched_content = current_content
            else:
                if not original_content:
                    # No committed version to diff against (new file, or no
                    # git history yet) - fall back to the working tree as
                    # baseline, same as the previous behavior.
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

        Fails closed (False, no git command run at all) when repo_path
        isn't a git repository in its own right - see _is_own_git_repo.
        Without this, `git add .` + `git commit` for such a path would
        silently stage and commit whatever is in the nearest ANCESTOR
        repository's working tree (confirmed in production: this tool's
        own checkout) instead of failing for a project that was never
        actually git-cloned.
        """
        if not _is_own_git_repo(repo_path):
            return False
        try:
            _run_git(["add", "."], repo_path)
            _run_git(["commit", "-m", message], repo_path)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def push_branch(
        repo_path: str,
        branch_name: str,
        remote: str = "origin",
        auth_header: Optional[str] = None,
    ) -> bool:
        """
        Pushes a local branch to the remote repository.
        Returns True on success, False on failure.

        Accepts the same optional pre-built HTTP Authorization header as
        clone_repository() (see build_github_auth_header()), supplied via a
        process-scoped `-c http.extraHeader=...` flag - push no longer
        relies on a credential embedded in the remote's stored URL.

        Fails closed (False, no git command run at all) when repo_path
        isn't a git repository in its own right - see _is_own_git_repo.
        The highest-severity instance of this class of bug: pushing a
        branch_name that happens to already exist in the nearest ANCESTOR
        repository (e.g. this tool's own checkout) would push straight to
        its real, unrelated remote.
        """
        if not _is_own_git_repo(repo_path):
            return False
        args = _with_auth_config(["push", "-u", remote, branch_name], auth_header)
        try:
            _run_git(args, repo_path)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            if hasattr(e, "stderr") and e.stderr:
                # SECURITY: with auth supplied via a header that's never
                # embedded in the remote URL, git's own stderr for an
                # auth/network failure has no credential left to echo - the
                # remote URL it reports is always the clean, token-free
                # one. As defense in depth (e.g. a future remote that
                # reintroduces a URL-embedded credential), any known
                # auth_header value is also redacted before printing.
                msg = e.stderr.strip()
                if auth_header:
                    msg = msg.replace(auth_header, "[REDACTED]")
                print(f"[!] Push error: {msg}")
            return False

    @staticmethod
    def cleanup_branch(repo_path: str, branch_name: str) -> bool:
        """
        Checks out the previous branch (main/master), discards any
        uncommitted working-tree changes, and deletes the feature branch.
        Returns True on success.

        developer_node/git_prepare_node write proposed patch content
        directly to disk (so a diff can be computed) before the HITL
        approval gate ever runs. When a run is rejected or policy-blocked,
        no commit is ever made and no feature branch is ever created
        (git_commit_node is the only place that creates one, reached only
        after approval) - so without discarding those uncommitted edits
        here, they sit on disk and the next run against the same project
        workspace mistakes them for the repository's real state.

        `checkout -- .` resets tracked files to the just-checked-out
        commit and `clean -fd` removes untracked files/dirs - neither
        touches committed history, so any real prior commits on the
        fallback branch are left exactly as they were.

        No-op (returns True immediately, no git subprocess invoked) when
        `repo_path` doesn't exist at all - reachable now that a no-op diff
        (backend/graph/nodes.py route_after_policy) routes here even for a
        run whose project_id was never provisioned with a real workspace;
        there is nothing to clean up for a workspace that was never
        created.
        """
        if not Path(repo_path).exists():
            return True

        try:
            # Try 'main' first, fallback to 'master'
            fallback_branch = "main"
            result = _run_git(["rev-parse", "--verify", "main"], repo_path, check=False)
            if result.returncode != 0:
                fallback_branch = "master"

            _run_git(["checkout", fallback_branch], repo_path)

            # Discard uncommitted changes left behind by patch generation.
            # Never touches committed history - only the working tree/index.
            _run_git(["checkout", "--", "."], repo_path, check=False)
            _run_git(["clean", "-fd"], repo_path, check=False)

            # Delete the feature branch if it was actually created (only
            # true once a run reaches git_commit_node); a no-op rather than
            # a failure when it wasn't, since rejection/policy-block always
            # precede branch creation in the current graph.
            _run_git(["branch", "-D", branch_name], repo_path, check=False)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
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

