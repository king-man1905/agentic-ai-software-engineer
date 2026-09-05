import os
import re
import sys
import time
import subprocess
from typing import List, Tuple, Optional
from backend.sandbox.models import TestExecutionResult


def validate_command(cmd: List[str]):
    """
    Validates a command against the security allowlist.
    Only allows pytest, python -m pytest, ruff, and flake8.
    """
    if not cmd:
        raise PermissionError("Empty command is not allowed.")

    executable = cmd[0]
    if executable not in ["pytest", "python", "ruff", "flake8"]:
        raise PermissionError(f"Command executable '{executable}' is not allowed.")

    if executable == "python":
        for arg in cmd[1:]:
            if arg.startswith("-") and arg not in ["-m", "-v", "-q", "--tb", "--verbose"]:
                raise PermissionError(f"Python flag '{arg}' is not allowed.")

        if "-m" in cmd:
            idx = cmd.index("-m")
            if idx + 1 < len(cmd):
                module = cmd[idx + 1]
                if module not in ["pytest", "ruff", "flake8"]:
                    raise PermissionError(
                        f"Python module '{module}' is not allowed to be executed."
                    )
            else:
                raise PermissionError("Invalid python -m command syntax.")


import contextlib
import shutil
import tempfile


SENSITIVE_ENV_SUBSTRINGS = [
    "TOKEN",
    "SECRET",
    "KEY",
    "PASSWORD",
    "CREDENTIAL",
    "PRIVATE",
    "AUTH",
]

# Essential system/runtime environment variables permitted into the sandbox
PERMITTED_SYS_VARS = {
    "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "TMPDIR",
    "USER", "USERNAME", "HOME", "USERPROFILE",
    "PYTHONPATH", "PYTHONHOME", "PYTHONIOENCODING", "PYTHONUTF8",
    "VIRTUAL_ENV", "LANG", "LC_ALL", "LC_CTYPE",
    "COMSPEC", "PATHEXT", "OS", "WINDIR", "APPDATA", "LOCALAPPDATA",
    "ALLUSERSPROFILE", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "COMMONPROGRAMFILES", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS",
}


def get_sandbox_env(allow_network: bool = False) -> dict:
    """
    Constructs a hardened, isolated environment for sandbox execution.
    Strips host credentials, API keys, and secrets to ensure child processes
    cannot access GitHub tokens, OpenAI/NVIDIA API keys, or cloud credentials.
    """
    env = {}

    for k, v in os.environ.items():
        k_upper = k.upper()
        # Strictly exclude any variable containing sensitive markers
        if any(sub in k_upper for sub in SENSITIVE_ENV_SUBSTRINGS):
            continue

        # Retain permitted system or runtime configuration
        if k_upper in PERMITTED_SYS_VARS or k_upper.startswith("PYTEST"):
            env[k] = v

    # Prepend virtual environment bin/Scripts directory to PATH
    venv_dir = sys.prefix
    if os.path.exists(os.path.join(venv_dir, "Scripts")):
        venv_bin = os.path.join(venv_dir, "Scripts")
    elif os.path.exists(os.path.join(venv_dir, "bin")):
        venv_bin = os.path.join(venv_dir, "bin")
    else:
        venv_bin = None

    if venv_bin:
        path_sep = ";" if os.name == "nt" else ":"
        env["PATH"] = venv_bin + path_sep + env.get("PATH", "")

    if not allow_network:
        env["PIP_NO_INDEX"] = "1"
        env["PIP_FIND_LINKS"] = ""

    return env


@contextlib.contextmanager
def isolated_workspace(source_dir: str):
    """
    Creates an ephemeral copy of a workspace directory for sandboxed test execution.
    Automatically cleans up the temporary directory upon exit.
    """
    temp_dir = tempfile.mkdtemp(prefix="agy_sandbox_")
    try:
        if os.path.exists(source_dir):
            shutil.copytree(source_dir, temp_dir, dirs_exist_ok=True)
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_pytest_summary(stdout: str) -> Tuple[int, int]:
    """
    Parses pytest stdout to extract passed and failed test counts.
    """
    passed = 0
    failed = 0
    lines = stdout.splitlines()

    for line in reversed(lines):
        # Match lines like "=== 2 failed, 41 passed, 1 warning in 1.23s ==="
        if "in" in line and ("passed" in line or "failed" in line):
            passed_match = re.search(r"(\d+)\s+passed", line)
            if passed_match:
                passed = int(passed_match.group(1))

            failed_match = re.search(r"(\d+)\s+failed", line)
            if failed_match:
                failed = int(failed_match.group(1))
            break

    return passed, failed


def extract_failures_and_errors(stdout: str) -> Optional[str]:
    """
    Extracts concise failure traceback and errors sections from pytest output.
    """
    lines = stdout.splitlines()
    captured_lines = []
    capture = False

    for line in lines:
        is_header = line.startswith("===") and line.endswith("===")
        if is_header and ("FAILURES" in line or "ERRORS" in line):
            capture = True
            captured_lines.append(line)
            continue

        if capture:
            # Stop if we hit test summary or summary lines
            if is_header and (
                "short test summary info" in line
                or "passed" in line
                or "failed" in line
                or "warnings" in line
                or "error" in line
            ):
                break
            captured_lines.append(line)

    if captured_lines:
        return "\n".join(captured_lines)
    return None


class SandboxRunner:
    """
    An isolated sandbox runner that secure-executes command suites.
    """

    @staticmethod
    def run_command(
        cmd: List[str],
        cwd: str,
        timeout: float = 30.0,
        allow_network: bool = False,
    ) -> TestExecutionResult:
        # Validate command against the allowlist
        validate_command(cmd)

        env = get_sandbox_env(allow_network=allow_network)
        start_time = time.time()

        try:
            # Enforce shell=False (Strict requirement)
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                shell=False,
                timeout=timeout,
            )

            duration = time.time() - start_time
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            exit_code = result.returncode

            # Extract metrics
            passed_count, failed_count = parse_pytest_summary(stdout)
            error_summary = extract_failures_and_errors(stdout)

            # Fallback if no pytest failures block is parsed but code is non-zero
            if not error_summary and exit_code != 0:
                error_summary = stderr.strip() or stdout.strip()
                if len(error_summary) > 500:
                    error_summary = error_summary[:500] + "... [truncated]"

            success = exit_code == 0

            return TestExecutionResult(
                success=success,
                exit_code=exit_code,
                passed_count=passed_count,
                failed_count=failed_count,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                error_summary=error_summary if error_summary else None,
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            return TestExecutionResult(
                success=False,
                exit_code=-1,
                passed_count=0,
                failed_count=0,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                duration_seconds=duration,
                error_summary=f"Timeout expired after {timeout} seconds.",
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestExecutionResult(
                success=False,
                exit_code=-1,
                passed_count=0,
                failed_count=0,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_summary=str(e),
            )
