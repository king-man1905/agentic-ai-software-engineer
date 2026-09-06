import os
import re
import sys
import time
import subprocess
from typing import Callable, List, Tuple, Optional
from backend.sandbox.models import TestExecutionResult

# How often to poll the subprocess for completion, and how it notices a
# cancellation or timeout without blocking indefinitely on communicate().
_POLL_INTERVAL_SECONDS = 0.2
# Grace period after terminate() before escalating to kill() - long enough
# for pytest/python to flush and exit cleanly, short enough to never hang
# a cancellation or timeout indefinitely.
_TERMINATE_GRACE_SECONDS = 5.0


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
    def _terminate_safely(proc: subprocess.Popen) -> Tuple[str, str]:
        """
        Terminates a subprocess, waiting a bounded grace period before
        escalating to kill(). Always waits for the process to actually
        exit so it's never left as an orphan.
        """
        try:
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        return stdout or "", stderr or ""

    @staticmethod
    def run_command(
        cmd: List[str],
        cwd: str,
        timeout: float = 30.0,
        allow_network: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> TestExecutionResult:
        """
        Runs `cmd` under the sandbox allowlist and environment, polling for
        completion so a cancellation request (checked via `cancel_check`)
        can terminate the process instead of only a fixed timeout. Uses
        Popen rather than subprocess.run specifically so the process can be
        signaled mid-execution - shell=False and the command allowlist are
        unchanged from before.
        """
        # Validate command against the allowlist
        validate_command(cmd)

        env = get_sandbox_env(allow_network=allow_network)
        start_time = time.time()

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
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

        cancelled = False
        timed_out = False
        stdout = stderr = ""
        try:
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=_POLL_INTERVAL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        timed_out = True
                        stdout, stderr = SandboxRunner._terminate_safely(proc)
                        break
                    if cancel_check is not None and cancel_check():
                        cancelled = True
                        stdout, stderr = SandboxRunner._terminate_safely(proc)
                        break
                    continue
        finally:
            # Belt-and-suspenders: never return with the process still
            # alive, whatever exit path was taken above.
            if proc.poll() is None:
                stdout, stderr = SandboxRunner._terminate_safely(proc)

        duration = time.time() - start_time
        stdout = stdout or ""
        stderr = stderr or ""
        exit_code = proc.returncode if proc.returncode is not None else -1

        if cancelled:
            return TestExecutionResult(
                success=False,
                exit_code=-1,
                passed_count=0,
                failed_count=0,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                error_summary="Cancelled by user request.",
            )

        if timed_out:
            return TestExecutionResult(
                success=False,
                exit_code=-1,
                passed_count=0,
                failed_count=0,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                error_summary=f"Timeout expired after {timeout} seconds.",
            )

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
