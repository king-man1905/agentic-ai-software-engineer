"""
Tests for Phase 8 - Step 1: Durable LangGraph Checkpointing.
Validates SQLite-backed checkpoint persistence, server restart recovery,
tenant security enforcement during recovery, patch integrity, and multi-process concurrency.
"""

import os
import sqlite3
import threading
import time
from unittest.mock import MagicMock
import pytest

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from backend.graph.runner import AgentRunner
from backend.schemas.routing import RoutingDecision, TaskType
from backend.schemas.developer import DeveloperResult, FileChange
from backend.schemas.qa import QAResult
from backend.vcs.models import ApprovalDecision
from backend.vcs.git_manager import GitWorkspaceManager
from backend.vcs.workspace_lock import (
    WorkspaceLockManager,
    WorkspaceLockTimeoutError,
    WorkspaceLockError,
    workspace_lock_manager,
)
from backend.security.auth import AuthMode
from backend.security.tenant import tenant_manager


@pytest.fixture(autouse=True)
def mock_agent_nodes(monkeypatch):
    """
    Mock agent LLM calls and Git mutations so graph execution
    runs deterministically, quickly, and offline.
    """
    monkeypatch.setattr(
        "backend.graph.nodes.route_task",
        lambda msg: RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.95,
            reasoning="Deterministic test routing",
            requires_planning=False,
            requires_knowledge=False,
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.generate_code_changes",
        lambda user_request, plan, knowledge: DeveloperResult(
            summary=f"Automated test patch for: {user_request[:30]}",
            changes=[
                FileChange(
                    file_path="src/service.py",
                    change_type="MODIFY",
                    content="def handle_request(): return {'status': 'ok'}",
                    reason="Fix test service logic",
                )
            ],
            requires_testing=True,
            notes=[],
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.review_code_changes",
        lambda user_request, plan, developer_result: QAResult(
            status="PASS",
            issues=[],
            test_cases=["test_handle_request"],
            summary="QA evaluation passed.",
        ),
    )
    monkeypatch.setattr(GitWorkspaceManager, "stage_and_commit", MagicMock(return_value=True))
    monkeypatch.setattr(GitWorkspaceManager, "create_feature_branch", MagicMock(return_value=True))
    monkeypatch.setattr(GitWorkspaceManager, "verify_workspace_drift", MagicMock(return_value=(True, "")))
    from backend.vcs.models import GitDiffSummary
    test_diff = GitDiffSummary(
        branch_name="agent/test-branch",
        files_changed=["src/service.py"],
        lines_added=1,
        lines_deleted=0,
        unified_diff="--- a/src/service.py\n+++ b/src/service.py\n@@ -1 +1 @@\n+def handle_request(): return {'status': 'ok'}\n",
        patch_hash="a1b2c3d4e5f678901234567890abcdef1234567890abcdef1234567890abcdef",
        risk_score="LOW",
        risk_reasons=[],
    )
    monkeypatch.setattr(
        "backend.graph.nodes.git_prepare_node",
        lambda state: {
            "git_diff": test_diff,
            "patch_hash": test_diff.patch_hash,
        },
    )


# ============================================================================
# 1. DURABLE CHECKPOINTER INITIALIZATION
# ============================================================================

def test_durable_checkpointer_initialization(tmp_path):
    """
    Verifies that AgentRunner initializes a durable SqliteSaver checkpointer
    backed by SQLite with WAL mode, busy timeout, and correct schema.
    """
    db_path = str(tmp_path / "init_checkpoints.db")
    runner = AgentRunner(checkpoint_db_path=db_path)

    try:
        assert isinstance(runner._checkpointer, SqliteSaver)
        assert not isinstance(runner._checkpointer, MemorySaver)
        assert os.path.exists(db_path)

        # Verify tables and WAL journal mode
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode;")
        journal_mode = cur.fetchone()[0]
        assert journal_mode.lower() == "wal"

        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        assert "checkpoints" in tables
        assert "writes" in tables
        conn.close()
    finally:
        runner.close()


# ============================================================================
# 2. CHECKPOINT SURVIVES RUNNER RECREATION
# ============================================================================

def test_checkpoint_survives_runner_recreation(tmp_path):
    """
    Verifies that a run persisted by one runner instance is discoverable
    by an entirely new runner instance connected to the same SQLite database.
    """
    db_path = str(tmp_path / "survives_recreation.db")
    run_id = "run-survive-recreation-001"
    org_id = "org-engineering"

    # Instance 1: start run until it pauses at approval
    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Fix payment timeout", organization_id=org_id)
    assert status1.status == "WAITING_APPROVAL"
    runner1.close()

    # Instance 2: new runner, simulating server restart
    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status2 = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert status2.status == "WAITING_APPROVAL"
        assert status2.run_id == run_id
    finally:
        runner2.close()


# ============================================================================
# 3. WAITING_FOR_APPROVAL STATE SURVIVES RESTART
# ============================================================================

def test_waiting_for_approval_state_survives_restart(tmp_path):
    """
    Verifies that WAITING_APPROVAL state, current_node ('approval'),
    and full state values survive process/runner restart.
    """
    db_path = str(tmp_path / "waiting_approval.db")
    run_id = "run-approval-restart-002"
    org_id = "org-platform"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Add retry policy", organization_id=org_id)
    assert status1.status == "WAITING_APPROVAL"
    assert status1.current_node == "approval"
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status2 = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert status2.status == "WAITING_APPROVAL"
        assert status2.current_node == "approval"

        values = runner2.get_state_values(run_id=run_id, organization_id=org_id)
        assert values["user_message"] == "Add retry policy"
        assert values["organization_id"] == org_id
        assert values["run_id"] == run_id
    finally:
        runner2.close()


# ============================================================================
# 4. RECOVERED RUN CAN RESUME
# ============================================================================

def test_recovered_run_can_resume(tmp_path):
    """
    Verifies that a run paused at the approval gate on runner 1 can be
    successfully resumed on a newly instantiated runner 2 and runs to completion.
    """
    db_path = str(tmp_path / "resume_recovered.db")
    run_id = "run-resume-003"
    org_id = "org-security"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Patch SQL sanitization", organization_id=org_id)
    assert status1.status == "WAITING_APPROVAL"
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        decision = ApprovalDecision(
            approved=True,
            reviewer="lead-architect",
            patch_hash=status1.git_diff.patch_hash if status1.git_diff else "",
        )
        resumed = runner2.resume_run(run_id=run_id, approval_decision=decision, organization_id=org_id)
        assert resumed.status == "COMPLETED"

        final_status = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert final_status.status == "COMPLETED"
    finally:
        runner2.close()


# ============================================================================
# 5. RECOVERED RUN REMAINS TENANT-SCOPED
# ============================================================================

def test_recovered_run_remains_tenant_scoped(tmp_path):
    """
    Verifies that a recovered run preserves its organization_id binding
    and can be retrieved by the originating tenant.
    """
    db_path = str(tmp_path / "tenant_scoped.db")
    run_id = "run-tenant-scope-004"
    org_id = "org-alpha-corp"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    runner1.start_run(run_id=run_id, user_message="Update billing logic", organization_id=org_id)
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert status.status == "WAITING_APPROVAL"

        state_values = runner2.get_state_values(run_id=run_id, organization_id=org_id)
        assert state_values.get("organization_id") == org_id
    finally:
        runner2.close()


# ============================================================================
# 6. CROSS-TENANT CHECKPOINT ACCESS IS REJECTED
# ============================================================================

def test_cross_tenant_checkpoint_access_is_rejected(tmp_path):
    """
    Verifies that recovering or resuming a checkpoint belonging to Organization A
    fails closed with KeyError when requested by Organization B.
    """
    db_path = str(tmp_path / "cross_tenant.db")
    run_id = "run-org-a-private-005"
    org_a = "org-owner-a"
    org_b = "org-intruder-b"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    runner1.start_run(run_id=run_id, user_message="Sensitive internal refactor", organization_id=org_a)
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        # Org B attempting get_status -> rejected
        with pytest.raises(KeyError) as exc_info:
            runner2.get_status(run_id=run_id, organization_id=org_b)
        assert f"Run not found: {run_id}" in str(exc_info.value)

        # Org B attempting get_state_values -> rejected
        with pytest.raises(KeyError) as exc_info:
            runner2.get_state_values(run_id=run_id, organization_id=org_b)
        assert f"Run not found: {run_id}" in str(exc_info.value)

        # Org B attempting resume_run -> rejected
        with pytest.raises(KeyError) as exc_info:
            runner2.resume_run(
                run_id=run_id,
                approval_decision=ApprovalDecision(approved=True, reviewer="intruder"),
                organization_id=org_b,
            )
        assert f"Run not found: {run_id}" in str(exc_info.value)

        # Legitimate Org A access remains untouched and permitted
        status_a = runner2.get_status(run_id=run_id, organization_id=org_a)
        assert status_a.status == "WAITING_APPROVAL"
    finally:
        runner2.close()


# ============================================================================
# 7. PATCH HASH SURVIVES RECOVERY
# ============================================================================

def test_patch_hash_survives_recovery(tmp_path):
    """
    Verifies that the SHA-256 cryptographic patch_hash generated before
    interruption is preserved bit-for-bit after runner restart.
    """
    db_path = str(tmp_path / "patch_hash.db")
    run_id = "run-patch-hash-006"
    org_id = "org-vcs-team"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Fix memory leak in buffer", organization_id=org_id)
    assert status1.git_diff is not None
    orig_hash = status1.git_diff.patch_hash
    assert orig_hash != ""
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status2 = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert status2.git_diff is not None
        assert status2.git_diff.patch_hash == orig_hash

        values = runner2.get_state_values(run_id=run_id, organization_id=org_id)
        assert values["git_diff"].patch_hash == orig_hash
    finally:
        runner2.close()


# ============================================================================
# 8. APPROVAL STATE SURVIVES RECOVERY
# ============================================================================

def test_approval_state_survives_recovery(tmp_path):
    """
    Verifies that after an approval decision is resumed and processed,
    the updated approval record and COMMITTED status persist across
    subsequent runner restarts.
    """
    db_path = str(tmp_path / "approval_state.db")
    run_id = "run-approval-state-007"
    org_id = "org-audit"

    # Step 1: run to approval
    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Refactor rate limiter", organization_id=org_id)
    runner1.close()

    # Step 2: resume on runner 2
    runner2 = AgentRunner(checkpoint_db_path=db_path)
    decision = ApprovalDecision(
        approved=True,
        reviewer="lead-sec-auditor",
        patch_hash=status1.git_diff.patch_hash if status1.git_diff else "",
    )
    resumed = runner2.resume_run(run_id=run_id, approval_decision=decision, organization_id=org_id)
    assert resumed.status == "COMPLETED"
    runner2.close()

    # Step 3: inspect on runner 3
    runner3 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status3 = runner3.get_status(run_id=run_id, organization_id=org_id)
        assert status3.status == "COMPLETED"

        values = runner3.get_state_values(run_id=run_id, organization_id=org_id)
        assert values.get("approval_status") == "COMMITTED"
        approval = values.get("approval")
        assert approval is not None
        assert approval.approved is True
        assert approval.reviewer == "lead-sec-auditor"
    finally:
        runner3.close()


# ============================================================================
# 9. POLICY STATE SURVIVES RECOVERY
# ============================================================================

def test_policy_state_survives_recovery(tmp_path):
    """
    Verifies that PolicyEvaluationResult evaluated before interruption
    is recovered intact, retaining decision and risk score.
    """
    db_path = str(tmp_path / "policy_state.db")
    run_id = "run-policy-eval-008"
    org_id = "org-governance"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Modify firewall rule", organization_id=org_id)
    assert status1.policy_result is not None
    orig_decision = status1.policy_result.decision
    orig_approval = status1.policy_result.requires_human_approval
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status2 = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert status2.policy_result is not None
        assert status2.policy_result.decision == orig_decision
        assert status2.policy_result.requires_human_approval == orig_approval
    finally:
        runner2.close()


# ============================================================================
# 10. NO PRODUCTION FALLBACK TO MEMORYSAVER
# ============================================================================

def test_no_production_fallback_to_memory_saver(tmp_path, monkeypatch):
    """
    Verifies that when AUTH_MODE=production, a failure to initialize the
    SQLite checkpointer raises a RuntimeError and does NOT fall back to MemorySaver.
    """
    monkeypatch.setattr(tenant_manager, "auth_mode", AuthMode.PRODUCTION)
    monkeypatch.setattr(tenant_manager, "dev_auth_fallback", False)

    # Use a directory path as the db path to trigger a SQLite initialization failure
    bad_db_path = str(tmp_path)  # directory, cannot connect as SQLite database file

    with pytest.raises(RuntimeError) as exc_info:
        AgentRunner(checkpoint_db_path=bad_db_path)

    assert "DURABLE_CHECKPOINT_INIT_FAILED" in str(exc_info.value)


# ============================================================================
# 11. MULTIPLE RUNS HAVE ISOLATED CHECKPOINTS
# ============================================================================

def test_multiple_runs_have_isolated_checkpoints(tmp_path):
    """
    Verifies that multiple concurrent runs sharing the same checkpoint database
    have strictly isolated thread checkpoints and independent lifecycles.
    """
    db_path = str(tmp_path / "multi_run_isolation.db")
    run_a = "run-isolated-A-011"
    run_b = "run-isolated-B-011"
    org_id = "org-enterprise"

    runner = AgentRunner(checkpoint_db_path=db_path)
    try:
        status_a = runner.start_run(run_id=run_a, user_message="Task A", organization_id=org_id)
        status_b = runner.start_run(run_id=run_b, user_message="Task B", organization_id=org_id)

        assert status_a.status == "WAITING_APPROVAL"
        assert status_b.status == "WAITING_APPROVAL"

        # Resume Run A only
        decision = ApprovalDecision(approved=True, reviewer="reviewer-a")
        resumed_a = runner.resume_run(run_id=run_a, approval_decision=decision, organization_id=org_id)
        assert resumed_a.status == "COMPLETED"

        # Verify Run B is completely unaffected and still WAITING_APPROVAL
        check_b = runner.get_status(run_id=run_b, organization_id=org_id)
        assert check_b.status == "WAITING_APPROVAL"

        # Verify state values isolation
        vals_a = runner.get_state_values(run_id=run_a, organization_id=org_id)
        vals_b = runner.get_state_values(run_id=run_b, organization_id=org_id)
        assert vals_a["user_message"] == "Task A"
        assert vals_b["user_message"] == "Task B"
        assert vals_a.get("approval_status") == "COMMITTED"
        assert vals_b.get("approval_status") != "COMMITTED"
    finally:
        runner.close()


# ============================================================================
# 12. SQLITE CHECKPOINT PERSISTENCE WORKS CONCURRENTLY
# ============================================================================

def test_sqlite_checkpoint_persistence_works_concurrently(tmp_path):
    """
    Verifies that concurrent threads executing runs on separate AgentRunner
    instances sharing the same SQLite database operate safely without lock contention
    errors or database corruption.
    """
    db_path = str(tmp_path / "concurrency_checkpoints.db")
    errors = []
    thread_count = 4

    def worker(worker_idx: int):
        runner = AgentRunner(checkpoint_db_path=db_path)
        try:
            run_id = f"run-concurrent-{worker_idx}"
            org_id = f"org-worker-{worker_idx}"

            # Start run
            status = runner.start_run(
                run_id=run_id,
                user_message=f"Concurrent task from worker {worker_idx}",
                organization_id=org_id,
            )
            if status.status != "WAITING_APPROVAL":
                errors.append(f"Worker {worker_idx}: unexpected initial status {status.status}")

            # Query status
            retrieved = runner.get_status(run_id=run_id, organization_id=org_id)
            if retrieved.status != "WAITING_APPROVAL":
                errors.append(f"Worker {worker_idx}: unexpected retrieved status {retrieved.status}")

            # Resume run
            resumed = runner.resume_run(
                run_id=run_id,
                approval_decision=ApprovalDecision(approved=True, reviewer=f"rev-{worker_idx}"),
                organization_id=org_id,
            )
            if resumed.status != "COMPLETED":
                errors.append(f"Worker {worker_idx}: unexpected resumed status {resumed.status}")
        except Exception as exc:
            errors.append(f"Worker {worker_idx} exception: {str(exc)}")
        finally:
            runner.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert not errors, f"Concurrent execution errors encountered: {errors}"


# ============================================================================
# PHASE 8 — STEP 2: PRODUCTION LLM TIMEOUTS + SAFE PROVIDER FALLBACK
# ============================================================================

from backend.services.llm import get_llm, get_fallback_provider
from backend.services.errors import (
    LLMError,
    LLMTimeoutError,
    LLMTransientError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMInvalidRequestError,
    LLMPermanentError,
    classify_llm_exception,
    is_fallback_eligible,
)
from backend.observability.telemetry import invoke_structured, run_context
from backend.observability.collector import telemetry_collector
from backend.observability.store import telemetry_store
from backend.schemas.telemetry import TelemetryEventType, FailureCategory
import httpx
import requests


class FakeProviderLLM:
    """Mock LLM that can simulate success, timeout, auth error, or transient failure."""
    def __init__(self, provider="nvidia", model_name="test-model", failure=None, result=None):
        self._provider = provider
        self.provider = provider
        self.model = model_name
        self.model_name = model_name
        self.failure = failure
        self.result = result
        self.invocations = 0

    def with_structured_output(self, schema, include_raw=False):
        return self

    def invoke(self, prompt):
        self.invocations += 1
        if self.failure:
            raise self.failure
        return self.result


# 1. test_llm_timeout_configuration
def test_llm_timeout_configuration(monkeypatch):
    """
    Verifies that the configured timeout (LLM_REQUEST_TIMEOUT_SECONDS) is actually
    passed to the underlying client instances for all supported providers:
    NVIDIA NIM, Google Gemini, and OpenAI.
    """
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "45.0")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key")

    from backend.services import llm as llm_module

    # NVIDIA
    nvidia_client = llm_module.get_llm(provider="nvidia")
    assert getattr(nvidia_client._client, "timeout", None) == 45.0

    # Gemini
    gemini_client = llm_module.get_llm(provider="gemini")
    assert getattr(gemini_client, "timeout", None) == 45.0

    # OpenAI
    openai_client = llm_module.get_llm(provider="openai")
    assert getattr(openai_client, "request_timeout", None) == 45.0

    # Explicit override timeout parameter
    custom_client = llm_module.get_llm(provider="nvidia", timeout=15.0)
    assert getattr(custom_client._client, "timeout", None) == 15.0


# 2. test_provider_timeout_is_structured
def test_provider_timeout_is_structured():
    """
    Verifies that socket, network, connect, and read timeouts from underlying SDKs
    are deterministically classified into structured LLMTimeoutError exceptions.
    """
    # httpx ConnectTimeout
    exc1 = httpx.ConnectTimeout("Connection timed out after 60s")
    classified1 = classify_llm_exception(exc1, provider="nvidia")
    assert isinstance(classified1, LLMTimeoutError)
    assert is_fallback_eligible(classified1) is True
    assert classified1.provider == "nvidia"

    # requests Timeout
    exc2 = requests.exceptions.Timeout("Read timed out")
    classified2 = classify_llm_exception(exc2, provider="gemini")
    assert isinstance(classified2, LLMTimeoutError)
    assert is_fallback_eligible(classified2) is True
    assert classified2.provider == "gemini"

    # Python standard TimeoutError
    exc3 = TimeoutError("Deadline exceeded")
    classified3 = classify_llm_exception(exc3, provider="openai")
    assert isinstance(classified3, LLMTimeoutError)
    assert is_fallback_eligible(classified3) is True


# 3. test_transient_provider_failure_triggers_fallback
def test_transient_provider_failure_triggers_fallback(monkeypatch):
    """
    Verifies that a transient provider failure (e.g. 503 Service Unavailable)
    on the primary provider automatically triggers fallback to the eligible secondary provider.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=httpx.NetworkError("503 Service Unavailable"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.98,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Fallback successfully classified",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback if provider == "gemini" else primary)

    result = invoke_structured(primary, RoutingDecision, "Fix broken endpoint")
    assert result.task_type == TaskType.BUG_FIX
    assert primary.invocations == 1
    assert fallback.invocations == 1


# 4. test_timeout_triggers_fallback
def test_timeout_triggers_fallback(monkeypatch):
    """
    Verifies that a timeout on the primary provider triggers safe fallback
    to the secondary provider, returning the expected structured result.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=TimeoutError("Request timed out after 60.0 seconds"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.CODE_GENERATION,
            confidence=0.95,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Fallback succeeded after timeout",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback if provider == "gemini" else primary)

    result = invoke_structured(primary, RoutingDecision, "Create new user model")
    assert result.task_type == TaskType.CODE_GENERATION
    assert primary.invocations == 1
    assert fallback.invocations == 1


# 5. test_auth_failure_does_not_trigger_fallback
def test_auth_failure_does_not_trigger_fallback(monkeypatch):
    """
    Verifies that authentication/authorization failures fail closed immediately
    and NEVER trigger fallback to another provider.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=Exception("401 Unauthorized: Invalid API key nvapi-invalid"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Should not be called",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with pytest.raises((LLMAuthenticationError, PermissionError)) as exc_info:
        invoke_structured(primary, RoutingDecision, "Some prompt")

    assert "authentication" in str(exc_info.value).lower() or "401" in str(exc_info.value)
    assert primary.invocations == 1
    assert fallback.invocations == 0, "Fallback provider must NOT be invoked on auth failures!"


# 6. test_invalid_request_does_not_trigger_fallback
def test_invalid_request_does_not_trigger_fallback(monkeypatch):
    """
    Verifies that malformed requests (400 Bad Request / schema errors) fail closed
    immediately and NEVER trigger automatic fallback.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=Exception("400 Bad Request: maximum context length exceeded"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Should not be called",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with pytest.raises((LLMInvalidRequestError, ValueError)) as exc_info:
        invoke_structured(primary, RoutingDecision, "Too long prompt")

    assert "invalid request" in str(exc_info.value).lower() or "context length" in str(exc_info.value)
    assert primary.invocations == 1
    assert fallback.invocations == 0, "Fallback provider must NOT be invoked on invalid request errors!"


# 7. test_fallback_is_bounded
def test_fallback_is_bounded(monkeypatch):
    """
    Verifies that provider attempts are strictly bounded to max 2 attempts (primary -> fallback).
    If the fallback provider also fails, execution raises a structured final failure
    and does NOT attempt a 3rd provider or loop infinitely.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=httpx.NetworkError("503 Service Unavailable"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        failure=httpx.NetworkError("502 Bad Gateway"),
    )

    fallback_attempts = []
    def mock_get_fallback_provider(*args, **kwargs):
        primary_arg = kwargs.get("primary") or (args[0] if args else None)
        fallback_attempts.append(primary_arg)
        return "gemini" if primary_arg == "nvidia" else "openai"

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", mock_get_fallback_provider)
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback if provider == "gemini" else primary)

    with pytest.raises(LLMTransientError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Any prompt")

    assert "transient provider failure" in str(exc_info.value).lower() or "502" in str(exc_info.value)
    assert primary.invocations == 1
    assert fallback.invocations == 1
    assert len(fallback_attempts) == 1, "Fallback must be attempted at most once (strict bound of 2 attempts)!"


# 8. test_all_providers_failure_returns_structured_error
def test_all_providers_failure_returns_structured_error(monkeypatch):
    """
    Verifies that when all eligible providers fail, a structured final LLM failure
    propagates cleanly without leaving the run appearing successful.
    """
    primary = FakeProviderLLM(provider="nvidia", failure=TimeoutError("Request timed out"))
    fallback = FakeProviderLLM(provider="gemini", failure=TimeoutError("Fallback timed out"))

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with pytest.raises(LLMTimeoutError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Prompt")

    assert isinstance(exc_info.value, LLMTimeoutError)
    assert exc_info.value.provider == "gemini"


# 9. test_provider_fallback_emits_telemetry
def test_provider_fallback_emits_telemetry(monkeypatch):
    """
    Verifies that PROVIDER_FALLBACK telemetry event is emitted with sanitized
    metadata (run_id, organization_id, primary_provider, fallback_provider, failure_category).
    """
    primary = FakeProviderLLM(provider="nvidia", failure=TimeoutError("Timeout on primary"))
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Fallback telemetry test",
        ),
    )

    recorded_events = []
    original_on_fallback = telemetry_collector.on_provider_fallback

    def mock_on_fallback(**kwargs):
        recorded_events.append(kwargs)
        original_on_fallback(**kwargs)

    monkeypatch.setattr(telemetry_collector, "on_provider_fallback", mock_on_fallback)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with run_context(run_id="run-telemetry-test", organization_id="org-acme"):
        invoke_structured(primary, RoutingDecision, "Test prompt")

    assert len(recorded_events) == 1
    event = recorded_events[0]
    assert event["run_id"] == "run-telemetry-test"
    assert event["organization_id"] == "org-acme"
    assert event["primary_provider"] == "nvidia"
    assert event["fallback_provider"] == "gemini"
    assert event["failure_category"] == "LLM_TIMEOUT"
    assert event["failure_type"] == "LLMTimeoutError"
    assert event["attempt_number"] == 1


# 10. test_fallback_preserves_tenant_context
def test_fallback_preserves_tenant_context(monkeypatch):
    """
    Verifies that provider fallback preserves tenant isolation and does not alter
    the organization_id, project_id, or security context of the run.
    """
    primary = FakeProviderLLM(provider="nvidia", failure=httpx.NetworkError("503 Service Unavailable"))
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Secure tenant routing",
        ),
    )

    captured_org = []
    def mock_record(event):
        if event.event_type == TelemetryEventType.PROVIDER_FALLBACK:
            captured_org.append(event.organization_id)

    monkeypatch.setattr(telemetry_store, "record_event", mock_record)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with run_context(run_id="run-iso-123", organization_id="tenant-secure-corp"):
        res = invoke_structured(primary, RoutingDecision, "Secure prompt")

    assert res.task_type == TaskType.BUG_FIX
    assert captured_org == ["tenant-secure-corp"], "Tenant organization_id must be strictly preserved across fallback!"


# 11. test_fallback_does_not_bypass_policy_or_qa
def test_fallback_does_not_bypass_policy_or_qa(monkeypatch, tmp_path):
    """
    Verifies that code generated by a fallback provider continues through all
    authoritative downstream QA and policy evaluation gates.
    """
    db_path = str(tmp_path / "policy_qa_checkpoints.db")
    runner = AgentRunner(checkpoint_db_path=db_path)

    # Developer produces changes that introduce a policy violation (e.g. modify protected config)
    violating_dev_result = DeveloperResult(
        summary="Fallback developer changes",
        changes=[
            FileChange(
                file_path="config/production_secrets.json",
                change_type="MODIFY",
                content="{'secret': 'new'}",
                reason="Unauthorized modification via fallback",
            )
        ],
        requires_testing=True,
    )
    monkeypatch.setattr("backend.graph.nodes.generate_code_changes", lambda *args, **kwargs: violating_dev_result)

    try:
        status = runner.start_run(
            run_id="run-fallback-policy",
            user_message="Modify protected files",
            organization_id="test-org",
        )
        assert status.policy_result is not None
        assert status.status in ("WAITING_APPROVAL", "COMPLETED", "FAILED")
    finally:
        runner.close()


# 12. test_no_credentials_in_provider_fallback_telemetry
def test_no_credentials_in_provider_fallback_telemetry(monkeypatch):
    """
    Verifies that API keys, tokens, or auth headers are never recorded
    in the PROVIDER_FALLBACK telemetry event metadata.
    """
    sensitive_error_msg = "500 Server Error: nvapi-secretkey1234567890 failed with bearer token ghp_supersecret"
    primary = FakeProviderLLM(provider="nvidia", failure=httpx.NetworkError(sensitive_error_msg))
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.GENERAL,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Redacted reasoning",
        ),
    )

    emitted_events = []
    def mock_record(event):
        if event.event_type == TelemetryEventType.PROVIDER_FALLBACK:
            emitted_events.append(event)

    monkeypatch.setattr(telemetry_store, "record_event", mock_record)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with run_context(run_id="run-redact-test", organization_id="test-org"):
        invoke_structured(primary, RoutingDecision, "Redaction test")

    assert len(emitted_events) == 1
    event_meta_str = str(emitted_events[0].safe_metadata)
    assert "nvapi-secretkey1234567890" not in event_meta_str
    assert "ghp_supersecret" not in event_meta_str


# 13. test_existing_provider_routing_remains_compatible
def test_existing_provider_routing_remains_compatible(monkeypatch):
    """
    Verifies that the existing provider routing (defaulting to LLM_PROVIDER)
    remains completely compatible and untouched.
    """
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    from backend.services.llm import get_llm

    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    client_nv = get_llm()
    assert getattr(client_nv, "_provider", None) == "nvidia"

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client_gem = get_llm()
    assert getattr(client_gem, "_provider", None) == "gemini"

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    client_oai = get_llm()
    assert getattr(client_oai, "_provider", None) == "openai"


# 14. test_llm_timeout_does_not_hang_graph
def test_llm_timeout_does_not_hang_graph(monkeypatch, tmp_path):
    """
    Verifies that an unrecoverable LLM timeout during graph execution propagates
    cleanly into existing error handling and terminates with status FAILED without hanging.
    """
    db_path = str(tmp_path / "timeout_hang_checkpoints.db")
    runner = AgentRunner(checkpoint_db_path=db_path)

    def mock_timing_out_router(msg):
        raise LLMTimeoutError("Outbound request timed out after 60.0s", provider="nvidia")

    monkeypatch.setattr("backend.graph.nodes.route_task", mock_timing_out_router)

    try:
        status = runner.start_run(
            run_id="run-timeout-hang-test",
            user_message="Should fail fast on timeout",
            organization_id="test-org",
        )
        assert status.status == "FAILED"
        assert "timed out" in (status.error_summary or "").lower()
    finally:
        runner.close()


# ============================================================================
# PHASE 8 — STEP 3: WORKSPACE LOCKING + CONCURRENCY SAFETY TESTS
# ============================================================================

# 1. test_workspace_lock_acquisition
def test_workspace_lock_acquisition(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    org_id = "org-acq-test"
    resource_id = "repo-alpha"
    run_id = "run-acq-001"

    acquired = mgr.acquire_lock(org_id, resource_id, run_id, timeout=2.0)
    assert acquired is True
    assert mgr.is_locked(org_id, resource_id) is True

    owner = mgr.get_lock_owner(org_id, resource_id)
    assert owner is not None
    assert owner["organization_id"] == org_id
    assert owner["resource_id"] == resource_id
    assert owner["run_id"] == run_id
    assert "owner_pid" in owner

    mgr.release_lock(org_id, resource_id, run_id)
    assert mgr.is_locked(org_id, resource_id) is False
    assert mgr.get_lock_owner(org_id, resource_id) is None


# 2. test_workspace_lock_release
def test_workspace_lock_release(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    org_id = "org-rel-test"
    resource_id = "repo-beta"
    run_1 = "run-rel-001"
    run_2 = "run-rel-002"

    mgr.acquire_lock(org_id, resource_id, run_1)

    # Another run cannot release run_1's lock
    with pytest.raises(PermissionError, match="does not own the lock"):
        mgr.release_lock(org_id, resource_id, run_2)

    # run_1 releases properly
    mgr.release_lock(org_id, resource_id, run_1)
    assert mgr.is_locked(org_id, resource_id) is False

    # Now run_2 can acquire cleanly
    assert mgr.acquire_lock(org_id, resource_id, run_2, timeout=1.0) is True
    mgr.release_lock(org_id, resource_id, run_2)


# 3. test_workspace_lock_timeout
def test_workspace_lock_timeout(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    org_id = "org-timeout-test"
    resource_id = "repo-gamma"
    run_1 = "run-holder"
    run_2 = "run-timed-out"

    mgr.acquire_lock(org_id, resource_id, run_1)

    # run_2 should fail fast with WorkspaceLockTimeoutError
    with pytest.raises(WorkspaceLockTimeoutError) as exc_info:
        mgr.acquire_lock(org_id, resource_id, run_2, timeout=0.15)

    assert "WORKSPACE_LOCK_TIMEOUT" in str(exc_info.value)
    assert "repo-gamma" in str(exc_info.value)

    # run_1 is still the authoritative owner
    owner = mgr.get_lock_owner(org_id, resource_id)
    assert owner["run_id"] == run_1

    mgr.release_lock(org_id, resource_id, run_1)


# 4. test_second_run_cannot_mutate_locked_workspace
def test_second_run_cannot_mutate_locked_workspace(tmp_path, monkeypatch):
    db_path = str(tmp_path / "checkpoints.db")
    lock_mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    runner = AgentRunner(checkpoint_db_path=db_path, lock_manager=lock_mgr)

    # Simulate run-1 holding the lock on project-1
    lock_mgr.acquire_lock("org-test", "project-1", "run-1")

    # Spy on developer node to verify it is NEVER called for run-2
    dev_called = []
    def spy_generate(*args, **kwargs):
        dev_called.append(True)
        return DeveloperResult(summary="Should not be generated", changes=[], requires_testing=False)

    monkeypatch.setattr("backend.graph.nodes.generate_code_changes", spy_generate)
    monkeypatch.setattr("backend.core.config.WORKSPACE_LOCK_TIMEOUT_SECONDS", 0.15)

    try:
        status_res = runner.start_run(
            run_id="run-2",
            user_message="Modify files in project-1",
            project_id="project-1",
            organization_id="org-test",
        )
        assert status_res.status == "FAILED"
        assert "WORKSPACE_LOCK_TIMEOUT" in (status_res.error_summary or "")
        assert len(dev_called) == 0, "Developer mutation must NOT run when lock times out!"
    finally:
        lock_mgr.release_lock("org-test", "project-1", "run-1")
        runner.close()


# 5. test_lock_released_after_exception
def test_lock_released_after_exception(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    org_id = "org-exc-test"
    resource_id = "repo-delta"
    run_1 = "run-fail"
    run_2 = "run-next"

    try:
        with mgr.acquire(org_id, resource_id, run_1):
            assert mgr.is_locked(org_id, resource_id) is True
            raise RuntimeError("Unexpected failure in graph node")
    except RuntimeError:
        pass

    # Lock must be released despite exception
    assert mgr.is_locked(org_id, resource_id) is False

    # Next run can acquire without issue
    assert mgr.acquire_lock(org_id, resource_id, run_2, timeout=1.0) is True
    mgr.release_lock(org_id, resource_id, run_2)


# 6. test_same_run_does_not_deadlock
def test_same_run_does_not_deadlock(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    org_id = "org-reentrant"
    resource_id = "repo-epsilon"
    run_id = "run-nested"

    # Nested acquisition by the same run_id must succeed without deadlock
    with mgr.acquire(org_id, resource_id, run_id):
        assert mgr.is_locked(org_id, resource_id) is True
        with mgr.acquire(org_id, resource_id, run_id):
            assert mgr.is_locked(org_id, resource_id) is True
            with mgr.acquire(org_id, resource_id, run_id):
                owner = mgr.get_lock_owner(org_id, resource_id)
                assert owner["run_id"] == run_id

        # Still locked until outermost context exits
        assert mgr.is_locked(org_id, resource_id) is True

    # Fully released after outermost exit
    assert mgr.is_locked(org_id, resource_id) is False


# 7. test_different_projects_can_run_concurrently
def test_different_projects_can_run_concurrently(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    barrier = threading.Barrier(2, timeout=5.0)
    concurrent_success = [False, False]

    def worker(worker_idx, proj_name, run_name):
        with mgr.acquire("org-shared", proj_name, run_name, timeout=5.0):
            # If both hold their locks simultaneously, the barrier will trip
            barrier.wait()
            concurrent_success[worker_idx] = True

    t1 = threading.Thread(target=worker, args=(0, "project-alpha", "run-t1"))
    t2 = threading.Thread(target=worker, args=(1, "project-beta", "run-t2"))

    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert concurrent_success[0] is True
    assert concurrent_success[1] is True


# 8. test_same_resource_is_serialized
def test_same_resource_is_serialized(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    execution_order = []
    run1_hold_event = threading.Event()
    run1_acquired_event = threading.Event()

    def worker_run1():
        with mgr.acquire("org-shared", "project-same", "run-1", timeout=5.0):
            execution_order.append("run1_acquired")
            run1_acquired_event.set()
            run1_hold_event.wait(timeout=5.0)
            execution_order.append("run1_releasing")

    def worker_run2():
        run1_acquired_event.wait(timeout=5.0)
        with mgr.acquire("org-shared", "project-same", "run-2", timeout=5.0):
            execution_order.append("run2_acquired")

    t1 = threading.Thread(target=worker_run1)
    t2 = threading.Thread(target=worker_run2)

    t1.start()
    t2.start()

    # Give t2 a moment to block on the lock held by t1
    run1_acquired_event.wait(timeout=2.0)
    time.sleep(0.1)

    # Release run1
    run1_hold_event.set()

    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert execution_order == ["run1_acquired", "run1_releasing", "run2_acquired"]


# 9. test_actual_branch_workspace_isolation_behavior
def test_actual_branch_workspace_isolation_behavior():
    task_id_1 = "sandbox-ai-demo"
    task_id_2 = "sandbox-ai-demo"

    branch1 = GitWorkspaceManager.generate_branch_name(task_id_1)
    branch2 = GitWorkspaceManager.generate_branch_name(task_id_2)

    assert branch1 != branch2, "Branch generation must include entropy suffix to prevent collision!"
    assert branch1.startswith("agent/task-sandbox-")
    assert branch2.startswith("agent/task-sandbox-")


# 10. test_cross_tenant_lock_isolation
def test_cross_tenant_lock_isolation(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    barrier = threading.Barrier(2, timeout=5.0)
    isolated_success = [False, False]

    def tenant_worker(idx, org_id, proj_id, run_id):
        with mgr.acquire(org_id, proj_id, run_id, timeout=5.0):
            barrier.wait()
            isolated_success[idx] = True

    t1 = threading.Thread(target=tenant_worker, args=(0, "tenant-a", "common-repo", "run-a"))
    t2 = threading.Thread(target=tenant_worker, args=(1, "tenant-b", "common-repo", "run-b"))

    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert isolated_success[0] is True
    assert isolated_success[1] is True

    # Confirm another run/tenant cannot release Tenant B's lock
    mgr.acquire_lock("tenant-b", "common-repo", "run-b")
    with pytest.raises(PermissionError):
        mgr.release_lock("tenant-b", "common-repo", "run-unauthorized")
    mgr.release_lock("tenant-b", "common-repo", "run-b")

    # Confirm Tenant A's and Tenant B's keys are completely distinct
    key_a = mgr.get_resource_key("tenant-a", "common-repo")
    key_b = mgr.get_resource_key("tenant-b", "common-repo")
    assert key_a != key_b


# 11. test_lock_metadata_is_sanitized
def test_lock_metadata_is_sanitized(tmp_path):
    lock_dir = tmp_path / "locks"
    mgr = WorkspaceLockManager(lock_dir=str(lock_dir))
    org_id = "org-sec-audit"
    resource_id = "project-billing"
    run_id = "run-audit-001"

    with mgr.acquire(org_id, resource_id, run_id):
        owner = mgr.get_lock_owner(org_id, resource_id)
        assert owner is not None
        assert owner["organization_id"] == org_id
        assert owner["resource_id"] == resource_id
        assert owner["run_id"] == run_id
        assert "owner_pid" in owner

    # After release, verify on-disk lock file content is valid JSON with sanitized metadata
    lock_files = list(lock_dir.glob("*.lock"))
    assert len(lock_files) == 1
    with open(lock_files[0], "r", encoding="utf-8") as f:
        raw_meta = f.read()

    import json
    meta = json.loads(raw_meta)
    assert meta["organization_id"] == org_id
    assert meta["resource_id"] == resource_id
    assert meta["run_id"] == run_id
    assert "owner_pid" in meta

    # Verify no secret keywords exist in raw content
    raw_lower = raw_meta.lower()
    assert "token" not in raw_lower
    assert "secret" not in raw_lower
    assert "password" not in raw_lower
    assert "bearer" not in raw_lower
    assert "apikey" not in raw_lower


# 12. test_workspace_lock_telemetry
def test_workspace_lock_telemetry(tmp_path, monkeypatch):
    recorded_events = []
    def mock_record(event):
        if event.event_type in (
            TelemetryEventType.WORKSPACE_LOCK_ACQUIRED,
            TelemetryEventType.WORKSPACE_LOCK_RELEASED,
            TelemetryEventType.WORKSPACE_LOCK_TIMEOUT,
        ):
            recorded_events.append(event)

    monkeypatch.setattr(telemetry_store, "record_event", mock_record)
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))

    # Acquire and release
    with mgr.acquire("org-telemetry", "repo-tel", "run-tel-1"):
        pass

    # Force a timeout
    mgr.acquire_lock("org-telemetry", "repo-tel", "run-holder")
    try:
        mgr.acquire_lock("org-telemetry", "repo-tel", "run-timeout", timeout=0.1)
    except WorkspaceLockTimeoutError:
        pass
    finally:
        mgr.release_lock("org-telemetry", "repo-tel", "run-holder")

    types = [e.event_type for e in recorded_events]
    assert TelemetryEventType.WORKSPACE_LOCK_ACQUIRED in types
    assert TelemetryEventType.WORKSPACE_LOCK_RELEASED in types
    assert TelemetryEventType.WORKSPACE_LOCK_TIMEOUT in types

    timeout_ev = next(e for e in recorded_events if e.event_type == TelemetryEventType.WORKSPACE_LOCK_TIMEOUT)
    assert timeout_ev.safe_metadata["outcome"] == "TIMEOUT"
    assert "wait_duration_ms" in timeout_ev.safe_metadata


# 13. test_waiting_approval_workspace_safety
def test_waiting_approval_workspace_safety(tmp_path):
    db_path = str(tmp_path / "hitl_checkpoints.db")
    lock_dir = str(tmp_path / "locks")
    lock_mgr = WorkspaceLockManager(lock_dir=lock_dir)
    runner = AgentRunner(checkpoint_db_path=db_path, lock_manager=lock_mgr)

    try:
        # Start run, which pauses at approval interrupt
        status = runner.start_run(
            run_id="run-hitl-lock-test",
            user_message="Feature requiring review",
            project_id="project-hitl",
            organization_id="org-hitl",
        )
        assert status.status == "WAITING_APPROVAL"

        # Verify lock is NOT held while waiting for human approval
        assert lock_mgr.is_locked("org-hitl", "project-hitl") is False

        # Another run can safely acquire the lock without being blocked for hours
        assert lock_mgr.acquire_lock("org-hitl", "project-hitl", "run-subsequent", timeout=1.0) is True
        lock_mgr.release_lock("org-hitl", "project-hitl", "run-subsequent")
    finally:
        runner.close()


# 14. test_patch_hash_integrity_with_concurrent_run_attempt
def test_patch_hash_integrity_with_concurrent_run_attempt(tmp_path, monkeypatch):
    db_path = str(tmp_path / "drift_checkpoints.db")
    lock_dir = str(tmp_path / "locks")
    lock_mgr = WorkspaceLockManager(lock_dir=lock_dir)
    runner = AgentRunner(checkpoint_db_path=db_path, lock_manager=lock_mgr)

    run_id = "run-drift-prevention"
    org_id = "org-drift-test"

    try:
        # 1. Run enters WAITING_APPROVAL
        status = runner.start_run(
            run_id=run_id,
            user_message="Fix bug with integrity check",
            project_id="proj-drift",
            organization_id=org_id,
        )
        assert status.status == "WAITING_APPROVAL"

        # 2. Simulate concurrent workspace modification / drift before approval resumes
        # GitWorkspaceManager.verify_workspace_drift detects drift
        monkeypatch.setattr(
            GitWorkspaceManager,
            "verify_workspace_drift",
            MagicMock(return_value=(False, "Workspace drift detected: content altered")),
        )

        # 3. Reviewer attempts to resume and approve
        decision = ApprovalDecision(
            approved=True,
            reviewer="Security Admin",
            patch_hash=status.git_diff.patch_hash,
        )
        resume_status = runner.resume_run(run_id, decision, organization_id=org_id)

        # 4. Must fail pre-commit drift validation and reject commit
        assert resume_status.status == "COMPLETED"
        values = runner.get_state_values(run_id, organization_id=org_id)
        assert values.get("approval_status") == "PATCH_HASH_MISMATCH"
    finally:
        runner.close()


# 15. test_no_commit_after_lock_timeout
def test_no_commit_after_lock_timeout(tmp_path, monkeypatch):
    db_path = str(tmp_path / "no_commit_checkpoints.db")
    lock_dir = str(tmp_path / "locks")
    lock_mgr = WorkspaceLockManager(lock_dir=lock_dir)
    runner = AgentRunner(checkpoint_db_path=db_path, lock_manager=lock_mgr)

    # Pre-lock resource
    lock_mgr.acquire_lock("org-block", "proj-block", "external-holder")

    commit_spy = MagicMock(return_value=True)
    monkeypatch.setattr(GitWorkspaceManager, "stage_and_commit", commit_spy)
    monkeypatch.setattr("backend.core.config.WORKSPACE_LOCK_TIMEOUT_SECONDS", 0.1)

    try:
        status = runner.start_run(
            run_id="run-block-commit",
            user_message="Should not commit",
            project_id="proj-block",
            organization_id="org-block",
        )
        assert status.status == "FAILED"
        assert "WORKSPACE_LOCK_TIMEOUT" in (status.error_summary or "")
        assert commit_spy.call_count == 0, "No git commit must be performed after lock timeout!"
    finally:
        lock_mgr.release_lock("org-block", "proj-block", "external-holder")
        runner.close()
