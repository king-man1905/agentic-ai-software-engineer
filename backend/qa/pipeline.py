import ast
import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from backend.developer.models import FilePatch
from backend.developer.patcher import SafePatcher
from backend.sandbox.models import TestExecutionResult
from backend.sandbox.runner import SandboxRunner, get_sandbox_env
from backend.schemas.qa import FailureCategory, QualityCheck, QualityCheckStatus


# Dangerous patterns for static AST security checks
DANGEROUS_CALLS = {"eval", "exec", "compile", "__import__"}
DANGEROUS_OS_CALLS = {"system", "popen", "popen2", "popen3", "popen4"}
SECRET_PATTERNS = [
    r"(?i)(?:api_key|apikey|secret_key|private_key|auth_token|access_token|password)\s*=\s*['\"][A-Za-z0-9_\-\.]{16,}['\"]",
    r"ghp_[A-Za-z0-9]{36}",  # GitHub personal access token
    r"sk-[A-Za-z0-9]{32,}",   # OpenAI / standard secret key
    r"AIza[0-9A-Za-z-_]{35}", # Google API Key
]


class QualityPipeline:
    """
    Orchestrates the multi-check quality assurance pipeline in the isolated sandbox.
    Detects repository environment and executes only applicable checks without blindly
    failing if optional tooling is absent.
    """

    @classmethod
    def check_ast(
        cls,
        repo_path: str,
        patches: List[FilePatch],
    ) -> QualityCheck:
        """
        Validates Python AST structure and syntax of all file patches before and after application.
        """
        start = time.time()
        if not patches:
            return QualityCheck(
                name="ast",
                status=QualityCheckStatus.PASS.value,
                exit_code=0,
                duration_ms=int((time.time() - start) * 1000),
                stdout_summary="No patches to validate.",
            )

        repo = Path(repo_path)
        syntax_errors = []
        # SafePatcher.apply_patch enforces two distinct things behind one
        # result: the anchor/snippet must exist in the target file (every
        # file type, mandatory), and - only for .py files - the patched
        # content must parse as valid Python. Track which of those actually
        # failed so the failure can be reported/categorized accurately
        # instead of unconditionally calling every failure a Python/AST
        # error (a README.md snippet mismatch is not a syntax error).
        has_python_syntax_failure = False
        has_patch_preflight_failure = False

        for patch in patches:
            abs_path = repo / patch.file_path
            source = ""
            if abs_path.exists():
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        source = f.read()
                except Exception:
                    pass

            result = SafePatcher.apply_patch(source, patch)
            if not result.is_valid:
                err_msg = f"{patch.file_path}: {', '.join(result.syntax_errors or ['AST validation failed'])}"
                syntax_errors.append(err_msg)
                if patch.file_path.lower().endswith(".py"):
                    has_python_syntax_failure = True
                else:
                    has_patch_preflight_failure = True

        duration_ms = int((time.time() - start) * 1000)

        if syntax_errors:
            summary = "\n".join(syntax_errors)
            if has_python_syntax_failure and has_patch_preflight_failure:
                reason = (
                    f"AST syntax validation failed and patch pre-flight validation "
                    f"failed, {len(syntax_errors)} patch(es) total."
                )
                category = "MIXED"
            elif has_python_syntax_failure:
                reason = f"AST syntax validation failed on {len(syntax_errors)} patch(es)."
                category = FailureCategory.AST_FAILURE.value
            else:
                reason = (
                    f"Patch pre-flight validation failed on {len(syntax_errors)} "
                    f"patch(es) (target snippet not found; not a Python syntax error)."
                )
                category = FailureCategory.PATCH_APPLICATION_FAILURE.value
            return QualityCheck(
                name="ast",
                status=QualityCheckStatus.FAIL.value,
                exit_code=1,
                duration_ms=duration_ms,
                stderr_summary=summary,
                reason=reason,
                category=category,
            )

        return QualityCheck(
            name="ast",
            status=QualityCheckStatus.PASS.value,
            exit_code=0,
            duration_ms=duration_ms,
            stdout_summary=f"AST pre-flight passed for {len(patches)} patch(es).",
        )

    @classmethod
    def check_pytest(
        cls,
        repo_path: str,
        timeout: float = 30.0,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[QualityCheck, Optional[TestExecutionResult]]:
        """
        Executes pytest in the sandbox. Returns QualityCheck and TestExecutionResult.
        """
        start = time.time()
        repo = Path(repo_path)
        if not repo.exists():
            duration_ms = int((time.time() - start) * 1000)
            return (
                QualityCheck(
                    name="pytest",
                    status=QualityCheckStatus.SKIPPED.value,
                    exit_code=0,
                    duration_ms=duration_ms,
                    reason="Workspace directory does not exist.",
                ),
                None,
            )

        try:
            cmd = ["python", "-m", "pytest"]
            test_result = SandboxRunner.run_command(
                cmd, cwd=str(repo), timeout=timeout, cancel_check=cancel_check
            )
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            test_result = TestExecutionResult(
                success=False,
                exit_code=-1,
                passed_count=0,
                failed_count=0,
                stdout="",
                stderr=str(e),
                duration_seconds=duration_ms / 1000.0,
                error_summary=str(e),
            )

        duration_ms = int((time.time() - start) * 1000)

        # Pytest exit code 5 means "no tests collected", treated as PASS (no failing tests)
        passed = test_result.success or test_result.exit_code == 5
        status = QualityCheckStatus.PASS.value if passed else QualityCheckStatus.FAIL.value

        stdout_summary = (
            f"Passed: {test_result.passed_count}, Failed: {test_result.failed_count}"
            if test_result.passed_count > 0 or test_result.failed_count > 0
            else (test_result.stdout[:200] if test_result.stdout else "Pytest completed.")
        )
        stderr_summary = test_result.error_summary or (test_result.stderr[:300] if test_result.stderr else "")

        check = QualityCheck(
            name="pytest",
            status=status,
            exit_code=test_result.exit_code,
            duration_ms=duration_ms,
            stdout_summary=stdout_summary,
            stderr_summary=stderr_summary,
            reason=stderr_summary if not passed else None,
        )

        return check, test_result

    @classmethod
    def check_lint(
        cls,
        repo_path: str,
        files_to_check: Optional[List[str]] = None,
    ) -> QualityCheck:
        """
        Executes ruff or flake8 if available in the sandbox environment.
        Gracefully returns SKIPPED if neither linter is installed or configured.
        """
        start = time.time()
        env = get_sandbox_env()
        path_var = env.get("PATH", "")

        # Find linter executable
        linter = None
        for candidate in ["ruff", "flake8"]:
            if shutil.which(candidate, path=path_var):
                linter = candidate
                break

        if not linter:
            return QualityCheck(
                name="lint",
                status=QualityCheckStatus.SKIPPED.value,
                exit_code=0,
                duration_ms=int((time.time() - start) * 1000),
                reason="NOT_AVAILABLE: Neither ruff nor flake8 is installed in environment.",
            )

        cmd = [linter]
        if linter == "ruff":
            cmd.extend(["check"])

        if files_to_check:
            # Filter to existing python files
            existing = [f for f in files_to_check if (Path(repo_path) / f).exists() and f.endswith(".py")]
            if existing:
                cmd.extend(existing)
            else:
                return QualityCheck(
                    name="lint",
                    status=QualityCheckStatus.SKIPPED.value,
                    exit_code=0,
                    duration_ms=int((time.time() - start) * 1000),
                    reason="NOT_APPLICABLE: No Python files modified to lint.",
                )

        try:
            res = SandboxRunner.run_command(cmd, cwd=repo_path, timeout=20.0)
            duration_ms = int((time.time() - start) * 1000)

            if res.success:
                return QualityCheck(
                    name="lint",
                    status=QualityCheckStatus.PASS.value,
                    exit_code=res.exit_code,
                    duration_ms=duration_ms,
                    stdout_summary=f"{linter} passed with 0 errors.",
                )
            else:
                summary = res.error_summary or res.stdout[:300] or res.stderr[:300]
                return QualityCheck(
                    name="lint",
                    status=QualityCheckStatus.FAIL.value,
                    exit_code=res.exit_code,
                    duration_ms=duration_ms,
                    stderr_summary=summary,
                    reason=f"{linter} detected style/lint violations.",
                )
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            return QualityCheck(
                name="lint",
                status=QualityCheckStatus.SKIPPED.value,
                exit_code=-1,
                duration_ms=duration_ms,
                reason=f"Linter execution skipped: {e}",
            )

    @classmethod
    def check_typecheck(
        cls,
        repo_path: str,
        files_to_check: Optional[List[str]] = None,
    ) -> QualityCheck:
        """
        Executes mypy if installed and configured. Returns SKIPPED if not available.
        """
        start = time.time()
        env = get_sandbox_env()
        mypy_bin = shutil.which("mypy", path=env.get("PATH", ""))

        if not mypy_bin:
            return QualityCheck(
                name="typecheck",
                status=QualityCheckStatus.SKIPPED.value,
                exit_code=0,
                duration_ms=int((time.time() - start) * 1000),
                reason="NOT_AVAILABLE: mypy type checker is not installed in environment.",
            )

        cmd = ["mypy", "--ignore-missing-imports"]
        if files_to_check:
            existing = [f for f in files_to_check if (Path(repo_path) / f).exists() and f.endswith(".py")]
            if existing:
                cmd.extend(existing)
            else:
                return QualityCheck(
                    name="typecheck",
                    status=QualityCheckStatus.SKIPPED.value,
                    exit_code=0,
                    duration_ms=int((time.time() - start) * 1000),
                    reason="NOT_APPLICABLE: No Python files to typecheck.",
                )

        try:
            res = SandboxRunner.run_command(cmd, cwd=repo_path, timeout=30.0)
            duration_ms = int((time.time() - start) * 1000)
            if res.success:
                return QualityCheck(
                    name="typecheck",
                    status=QualityCheckStatus.PASS.value,
                    exit_code=0,
                    duration_ms=duration_ms,
                    stdout_summary="mypy type check passed with 0 errors.",
                )
            else:
                return QualityCheck(
                    name="typecheck",
                    status=QualityCheckStatus.FAIL.value,
                    exit_code=res.exit_code,
                    duration_ms=duration_ms,
                    stderr_summary=res.error_summary or res.stdout[:300],
                    reason="mypy reported type errors.",
                )
        except Exception as e:
            return QualityCheck(
                name="typecheck",
                status=QualityCheckStatus.SKIPPED.value,
                exit_code=-1,
                duration_ms=int((time.time() - start) * 1000),
                reason=f"Typecheck execution skipped: {e}",
            )

    @classmethod
    def check_security(
        cls,
        repo_path: str,
        patches: List[FilePatch],
    ) -> QualityCheck:
        """
        Static AST security analyzer that inspects patch code for dangerous execution,
        insecure subprocess usage (shell=True), and leaked secrets/tokens.
        """
        start = time.time()
        findings = []

        for patch in patches:
            code = patch.updated_code_snippet or ""
            fpath = patch.file_path

            # 1. Regex check for hardcoded secrets
            for pattern in SECRET_PATTERNS:
                matches = re.findall(pattern, code)
                if matches:
                    findings.append(f"Hardcoded credential or token detected in {fpath}.")
                    break

            # 2. AST check for dangerous function calls and shell=True
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    # Check Call nodes
                    if isinstance(node, ast.Call):
                        # Detect direct calls to eval, exec, compile
                        if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALLS:
                            findings.append(f"Dangerous builtin '{node.func.id}()' called in {fpath}.")

                        # Detect os.system, os.popen
                        elif isinstance(node.func, ast.Attribute):
                            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                                if node.func.attr in DANGEROUS_OS_CALLS:
                                    findings.append(f"Insecure OS command execution 'os.{node.func.attr}()' in {fpath}.")

                        # Detect shell=True in subprocess calls
                        for kw in node.keywords:
                            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                findings.append(f"Prohibited 'shell=True' execution detected in {fpath}.")

            except Exception:
                # If code is a partial snippet that cannot parse standalone, scan textually
                if "shell=True" in code:
                    findings.append(f"Prohibited 'shell=True' execution detected in {fpath}.")
                for d in DANGEROUS_CALLS:
                    if f"{d}(" in code:
                        findings.append(f"Dangerous builtin '{d}()' detected in {fpath}.")

        duration_ms = int((time.time() - start) * 1000)

        if findings:
            return QualityCheck(
                name="security",
                status=QualityCheckStatus.FAIL.value,
                exit_code=1,
                duration_ms=duration_ms,
                stderr_summary="\n".join(findings),
                reason=f"Security analysis flagged {len(findings)} blocking issue(s).",
            )

        return QualityCheck(
            name="security",
            status=QualityCheckStatus.PASS.value,
            exit_code=0,
            duration_ms=duration_ms,
            stdout_summary="Static security scan passed with 0 violations.",
        )

    @classmethod
    def run_all(
        cls,
        repo_path: str,
        patches: List[FilePatch],
        timeout: float = 30.0,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[List[QualityCheck], Optional[TestExecutionResult]]:
        """
        Executes all configured quality checks in sequence.
        Returns the list of QualityCheck items and the raw TestExecutionResult.
        """
        files = [p.file_path for p in patches]

        # 1. AST Validation
        ast_check = cls.check_ast(repo_path, patches)

        # 2. Pytest Execution
        pytest_check, test_result = cls.check_pytest(repo_path, timeout=timeout, cancel_check=cancel_check)

        # 3. Static Security Scan
        security_check = cls.check_security(repo_path, patches)

        # 4. Optional Linting (ruff / flake8)
        lint_check = cls.check_lint(repo_path, files)

        # 5. Optional Type Checking (mypy)
        type_check = cls.check_typecheck(repo_path, files)

        all_checks = [ast_check, pytest_check, security_check, lint_check, type_check]
        return all_checks, test_result
