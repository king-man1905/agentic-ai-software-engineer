import time
import pytest
from backend.sandbox.runner import SandboxRunner


def test_successful_pytest_execution(tmp_path):
    """
    Test that a passing test suite yields success=True, exit_code=0, and correct pass counts.
    """
    test_file = tmp_path / "test_success.py"
    test_file.write_text(
        """def test_one():
    assert 1 == 1

def test_two():
    assert True
""",
        encoding="utf-8",
    )

    res = SandboxRunner.run_command(["python", "-m", "pytest", "test_success.py"], cwd=str(tmp_path))
    assert res.success is True
    assert res.exit_code == 0
    assert res.passed_count == 2
    assert res.failed_count == 0
    assert res.error_summary is None


def test_failed_pytest_execution(tmp_path):
    """
    Test that a failing test suite yields success=False, non-zero exit_code,
    correct failed counts, and captures tracebacks in error_summary.
    """
    test_file = tmp_path / "test_fail.py"
    test_file.write_text(
        """def test_pass():
    assert 1 == 1

def test_fail():
    assert 1 == 2
""",
        encoding="utf-8",
    )

    res = SandboxRunner.run_command(["python", "-m", "pytest", "test_fail.py"], cwd=str(tmp_path))
    assert res.success is False
    assert res.exit_code == 1
    assert res.passed_count == 1
    assert res.failed_count == 1
    assert res.error_summary is not None
    assert "test_fail" in res.error_summary
    assert "assert 1 == 2" in res.error_summary


def test_security_rejection_of_unapproved_commands():
    """
    Verify that unapproved executables or module flags raise PermissionError.
    """
    # 1. Blocked executables
    for cmd in [
        ["bash", "-c", "echo hello"],
        ["sh", "script.sh"],
        ["rm", "-rf", "/"],
        ["curl", "https://google.com"],
    ]:
        with pytest.raises(PermissionError) as exc_info:
            SandboxRunner.run_command(cmd, cwd=".")
        assert "not allowed" in str(exc_info.value)

    # 2. Blocked python modules
    for cmd in [
        ["python", "-m", "pip", "install", "requests"],
        ["python", "-m", "http.server"],
        ["python", "-c", "import os; os.system('echo dangerous')"],
    ]:
        with pytest.raises(PermissionError) as exc_info:
            SandboxRunner.run_command(cmd, cwd=".")
        assert "not allowed" in str(exc_info.value) or "python -m" in str(exc_info.value)


def test_execution_timeout_handling(tmp_path):
    """
    Verify that test runs exceeding the specified timeout are terminated
    and return a failure result with timeout error summary.
    """
    test_file = tmp_path / "test_timeout.py"
    test_file.write_text(
        """import time
def test_long():
    time.sleep(5)
    assert True
""",
        encoding="utf-8",
    )

    res = SandboxRunner.run_command(
        ["python", "-m", "pytest", "test_timeout.py"], cwd=str(tmp_path), timeout=1.0
    )
    assert res.success is False
    assert res.exit_code == -1
    assert "Timeout expired" in res.error_summary
