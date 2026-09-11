"""
Comprehensive test suite for Phase 4: Organization Policy Engine & Enterprise Guardrails.

Covers all 26 requirements:
1. default policy
2. allowed repository
3. blocked repository
4. allowed branch
5. protected branch
6. protected path
7. path normalization
8. max files changed
9. max lines added
10. max lines deleted
11. max revisions
12. required check failure
13. LOW risk
14. MEDIUM risk
15. HIGH risk
16. CRITICAL risk
17. policy BLOCK routing
18. policy REVIEW routing
19. policy ALLOW routing
20. policy telemetry
21. sandbox timeout
22. sandbox environment isolation
23. sandbox cleanup
24. Git mutation blocked by policy
25. existing HITL still required
26. existing patch-hash integrity still works
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from backend.schemas.policy import (
    PolicyConfig,
    PolicyDecision,
    PolicyViolation,
    PolicyEvaluationResult,
    PolicyTelemetry,
)
from backend.policy.path_filter import (
    normalize_path,
    is_traversal_attack,
)
from backend.policy.evaluator import PolicyEvaluator
from backend.schemas.qa import QAResult, QualityCheck
from backend.sandbox.runner import SandboxRunner, get_sandbox_env, isolated_workspace
from backend.graph.state import AgentState
from backend.vcs.models import GitDiffSummary, ApprovalDecision
from backend.vcs.git_manager import GitWorkspaceManager
from backend.graph.nodes import (
    policy_node,
    route_after_policy,
    approval_node,
    route_after_approval,
)


# ============================================================================
# 1-6: Policy Configuration, Repositories, Branches, & Paths
# ============================================================================


def test_default_policy():
    """1. Verify default PolicyConfig enforces secure enterprise defaults."""
    policy = PolicyConfig()
    assert policy.max_files_changed == 10
    assert policy.max_lines_added == 500
    assert policy.max_lines_deleted == 200
    assert policy.max_revisions == 3
    assert policy.allow_network is False
    assert policy.allow_dependency_changes is False
    assert policy.allow_config_changes is False
    assert policy.allow_ci_changes is False
    assert policy.policy_version == "1.0.0"
    assert "main" in policy.protected_branches
    assert ".env*" in policy.protected_paths


def test_allowed_repository():
    """2. Verify allowed repository pattern permits compliant repos."""
    policy = PolicyConfig(allowed_repositories=["acme-corp/agentic-*", "acme-corp/core"])
    res = PolicyEvaluator.evaluate(
        policy=policy,
        repository="acme-corp/agentic-ai",
        branch="agent/task-001",
        changed_files=["src/app.py"],
    )
    assert res.checks["repository"] == "PASS"
    assert not any(v.rule == "ALLOWED_REPOSITORIES" for v in res.violations)


def test_blocked_repository():
    """3. Verify unlisted repository triggers BLOCK with violation."""
    policy = PolicyConfig(allowed_repositories=["acme-corp/*"])
    res = PolicyEvaluator.evaluate(
        policy=policy,
        repository="malicious-org/target-repo",
        branch="agent/task-001",
        changed_files=["src/app.py"],
    )
    assert res.decision == PolicyDecision.BLOCK
    assert res.checks["repository"] == "FAIL"
    assert any(v.rule == "ALLOWED_REPOSITORIES" for v in res.violations)


def test_allowed_branch():
    """4. Verify branch allowlist permits valid feature branches and rejects unlisted."""
    policy = PolicyConfig(allowed_branches=["agent/*", "feature/*"])
    # Allowed
    res_pass = PolicyEvaluator.evaluate(
        policy=policy,
        branch="agent/task-100",
        changed_files=["src/app.py"],
    )
    assert res_pass.checks["branch"] == "PASS"

    # Blocked
    res_block = PolicyEvaluator.evaluate(
        policy=policy,
        branch="hotfix/unauthorized",
        changed_files=["src/app.py"],
    )
    assert res_block.decision == PolicyDecision.BLOCK
    assert res_block.checks["branch"] == "FAIL"
    assert any(v.rule == "ALLOWED_BRANCHES" for v in res_block.violations)


def test_protected_branch():
    """5. Verify targeting protected branches (main, master, production) triggers BLOCK."""
    policy = PolicyConfig()
    for protected in ["main", "master", "release/v1.0", "production"]:
        res = PolicyEvaluator.evaluate(
            policy=policy,
            branch=protected,
            changed_files=["src/app.py"],
        )
        assert res.decision == PolicyDecision.BLOCK
        assert res.checks["branch"] == "FAIL"
        assert any(v.rule == "PROTECTED_BRANCH" for v in res.violations)


def test_protected_path():
    """6. Verify modifying protected paths triggers BLOCK."""
    policy = PolicyConfig(protected_paths=[".env*", "secrets/**", "auth/**"])

    for pfile in [".env", ".env.local", "secrets/key.pem", "auth/service.py"]:
        res = PolicyEvaluator.evaluate(
            policy=policy,
            branch="agent/task-1",
            changed_files=[pfile],
        )
        assert res.decision == PolicyDecision.BLOCK
        assert res.checks["protected_paths"] == "FAIL"
        assert any(v.rule == "PROTECTED_PATH" for v in res.violations)


# ============================================================================
# 7-11: Path Normalization & Patch/Revision Limits
# ============================================================================


def test_path_normalization():
    """7. Verify path normalization resists directory traversal attacks."""
    attack_paths = [
        "src/../../.env",
        "foo/bar/../../../secrets/master.key",
        ".\\auth\\..\\secrets\\token.txt",
        "../database/migrations/001.py",
    ]

    for p in attack_paths:
        assert is_traversal_attack(p) is True
        norm = normalize_path(p)
        assert not norm.startswith("../")
        assert not norm.startswith("./")

    # Evaluate traversal in PolicyEvaluator
    res = PolicyEvaluator.evaluate(
        changed_files=["src/../../.env"],
    )
    assert res.decision == PolicyDecision.BLOCK
    assert any(v.rule == "PATH_TRAVERSAL" for v in res.violations)


def test_max_files_changed():
    """8. Verify exceeding max_files_changed triggers BLOCK."""
    policy = PolicyConfig(max_files_changed=3)
    files = ["a.py", "b.py", "c.py", "d.py"]
    res = PolicyEvaluator.evaluate(
        policy=policy,
        changed_files=files,
    )
    assert res.decision == PolicyDecision.BLOCK
    assert res.checks["patch_limits"] == "FAIL"
    assert any(v.rule == "MAX_FILES_EXCEEDED" for v in res.violations)


def test_max_lines_added():
    """9. Verify exceeding max_lines_added triggers BLOCK."""
    policy = PolicyConfig(max_lines_added=100)
    res = PolicyEvaluator.evaluate(
        policy=policy,
        changed_files=["a.py"],
        lines_added=150,
    )
    assert res.decision == PolicyDecision.BLOCK
    assert res.checks["patch_limits"] == "FAIL"
    assert any(v.rule == "MAX_LINES_ADDED_EXCEEDED" for v in res.violations)


def test_max_lines_deleted():
    """10. Verify exceeding max_lines_deleted triggers BLOCK."""
    policy = PolicyConfig(max_lines_deleted=50)
    res = PolicyEvaluator.evaluate(
        policy=policy,
        changed_files=["a.py"],
        lines_deleted=80,
    )
    assert res.decision == PolicyDecision.BLOCK
    assert res.checks["patch_limits"] == "FAIL"
    assert any(v.rule == "MAX_LINES_DELETED_EXCEEDED" for v in res.violations)


def test_max_revisions():
    """11. Verify exceeding max_revisions triggers BLOCK."""
    policy = PolicyConfig(max_revisions=2)
    res = PolicyEvaluator.evaluate(
        policy=policy,
        changed_files=["a.py"],
        revision_count=3,
    )
    assert res.decision == PolicyDecision.BLOCK
    assert res.checks["patch_limits"] == "FAIL"
    assert any(v.rule == "MAX_REVISIONS_EXCEEDED" for v in res.violations)


# ============================================================================
# 12-16: Required QA Checks & Risk Level Thresholds
# ============================================================================


def test_required_check_failure():
    """12. Verify failing required quality check triggers BLOCK and cannot be bypassed."""
    policy = PolicyConfig(required_quality_checks=["AST", "PYTEST", "SECURITY"])
    failing_qa = QAResult(
        status="FAIL",
        failure_category="TEST_FAILURE",
        summary="2 tests failed in pytest.",
        checks=[
            QualityCheck(name="AST", status="PASS"),
            QualityCheck(name="PYTEST", status="FAIL"),
            QualityCheck(name="SECURITY", status="PASS"),
        ],
    )
    res = PolicyEvaluator.evaluate(
        policy=policy,
        changed_files=["a.py"],
        quality_results=failing_qa,
    )
    assert res.decision == PolicyDecision.BLOCK
    assert res.checks["required_checks"] == "FAIL"
    assert any(v.rule == "REQUIRED_CHECK_FAILED" for v in res.violations)


def test_low_risk():
    """13. Verify standard LOW risk changeset yields ALLOW decision."""
    policy = PolicyConfig()
    res = PolicyEvaluator.evaluate(
        policy=policy,
        branch="agent/task-low",
        changed_files=["src/helper.py"],
        lines_added=10,
        lines_deleted=2,
        risk_score="LOW",
    )
    assert res.decision == PolicyDecision.ALLOW
    assert res.checks["risk_policy"] == "PASS"
    assert len(res.violations) == 0


def test_medium_risk():
    """14. Verify MEDIUM risk yields REVIEW decision with warning."""
    policy = PolicyConfig()
    res = PolicyEvaluator.evaluate(
        policy=policy,
        branch="agent/task-med",
        changed_files=["src/helper.py"],
        lines_added=10,
        lines_deleted=2,
        risk_score="MEDIUM",
        risk_reasons=["Changeset spans multiple components."],
    )
    assert res.decision == PolicyDecision.REVIEW
    assert res.checks["risk_policy"] == "REVIEW"
    assert len(res.warnings) > 0


def test_high_risk():
    """15. Verify HIGH risk yields REVIEW decision with elevated warnings."""
    policy = PolicyConfig()
    res = PolicyEvaluator.evaluate(
        policy=policy,
        branch="agent/task-high",
        changed_files=["src/helper.py"],
        lines_added=10,
        lines_deleted=40,
        risk_score="HIGH",
        risk_reasons=["High deletion ratio > 50%."],
    )
    assert res.decision == PolicyDecision.REVIEW
    assert res.checks["risk_policy"] == "REVIEW"
    assert len(res.warnings) > 0


def test_critical_risk():
    """16. Verify CRITICAL risk unconditionally triggers BLOCK."""
    policy = PolicyConfig()
    res = PolicyEvaluator.evaluate(
        policy=policy,
        branch="agent/task-crit",
        changed_files=["src/core.py"],
        risk_score="CRITICAL",
        risk_reasons=["Zero-day vulnerability detected in security AST check."],
    )
    assert res.decision == PolicyDecision.BLOCK
    assert res.checks["risk_policy"] == "BLOCK"
    assert any(v.rule == "CRITICAL_RISK_BLOCKED" for v in res.violations)


# ============================================================================
# 17-20: Graph Routing & Telemetry
# ============================================================================


def test_policy_block_routing():
    """17. Verify policy decision BLOCK routes to cleanup and updates state."""
    state: AgentState = {
        "user_message": "Modify core",
        "git_diff": GitDiffSummary(
            branch_name="main",  # Protected branch -> BLOCK
            files_changed=["core.py"],
            lines_added=1,
            lines_deleted=0,
            unified_diff="+pass",
            patch_hash="abc",
            risk_score="LOW",
        ),
    }

    out = policy_node(state)
    assert out["policy_result"].decision == PolicyDecision.BLOCK
    assert out["approval_status"] == "POLICY_BLOCKED"

    # Route test
    state["policy_result"] = out["policy_result"]
    assert route_after_policy(state) == "cleanup"


def test_policy_review_routing():
    """18. Verify policy decision REVIEW routes to approval gate."""
    state: AgentState = {
        "policy_result": PolicyEvaluationResult(
            decision=PolicyDecision.REVIEW,
            violations=[],
            warnings=["Requires reviewer sign-off."],
            checks={"risk_policy": "REVIEW"},
            policy_version="1.0.0",
            evaluated_at="2026-09-05T00:00:00Z",
        )
    }
    assert route_after_policy(state) == "approval"


def test_policy_allow_routing():
    """19. Verify policy decision ALLOW routes to approval gate."""
    state: AgentState = {
        "policy_result": PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW,
            violations=[],
            warnings=[],
            checks={"risk_policy": "PASS"},
            policy_version="1.0.0",
            evaluated_at="2026-09-05T00:00:00Z",
        )
    }
    assert route_after_policy(state) == "approval"


def test_policy_telemetry():
    """20. Verify policy telemetry object populates all audit fields."""
    eval_result = PolicyEvaluationResult(
        decision=PolicyDecision.REVIEW,
        violations=[],
        warnings=["Elevated risk."],
        checks={"repository": "PASS", "risk_policy": "REVIEW"},
        policy_version="1.0.0",
        evaluated_at="2026-09-05T12:00:00Z",
    )
    telem = PolicyEvaluator.create_telemetry(
        result=eval_result,
        risk_score="MEDIUM",
        repository="org/repo",
        branch="agent/task-001",
        changed_files=["src/app.py"],
        lines_added=5,
        lines_deleted=1,
        required_checks=["AST", "PYTEST", "SECURITY"],
        failed_checks=[],
    )
    assert isinstance(telem, PolicyTelemetry)
    assert telem.decision == "REVIEW"
    assert telem.risk_score == "MEDIUM"
    assert telem.repository == "org/repo"
    assert telem.branch == "agent/task-001"
    assert telem.patch_size["lines_added"] == 5
    assert telem.patch_size["lines_deleted"] == 1
    assert "AST" in telem.required_checks


# ============================================================================
# 21-23: Sandbox Hardening & Isolation
# ============================================================================


def test_sandbox_timeout(tmp_path):
    """21. Verify hardened sandbox terminates commands exceeding timeout."""
    test_file = tmp_path / "test_hang.py"
    test_file.write_text(
        "import time\ndef test_slow():\n    time.sleep(5)\n    assert True\n",
        encoding="utf-8",
    )
    res = SandboxRunner.run_command(
        ["python", "-m", "pytest", "test_hang.py"],
        cwd=str(tmp_path),
        timeout=0.5,
    )
    assert res.success is False
    assert res.exit_code == -1
    assert "Timeout expired" in (res.error_summary or "")


def test_sandbox_environment_isolation():
    """22. Verify host secrets are stripped from sandbox environment."""
    secret_vars = {
        "GITHUB_TOKEN": "ghp_1234567890abcdef",
        "OPENAI_API_KEY": "sk-secret-openai-key",
        "NVIDIA_API_KEY": "nv-secret-nvidia-key",
        "GOOGLE_API_KEY": "AIzaSySecretGoogleKey",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "MY_DATABASE_PASSWORD": "super_secret_db_password",
    }

    with patch.dict(os.environ, secret_vars, clear=False):
        env = get_sandbox_env()
        # Verify none of the sensitive variables leaked into the sandbox env
        for k in secret_vars:
            assert k not in env, f"Secret {k} leaked into sandbox environment!"

        # Verify essential system variables remain intact
        assert "PATH" in env
        if os.name == "nt":
            assert "SYSTEMROOT" in env


def test_sandbox_cleanup(tmp_path):
    """23. Verify isolated_workspace copies source and completely cleans up temporary workspace."""
    src = tmp_path / "src_repo"
    src.mkdir()
    f = src / "calc.py"
    f.write_text("def add(a, b): return a + b", encoding="utf-8")

    captured_tmp_path = None
    with isolated_workspace(str(src)) as isolated_dir:
        captured_tmp_path = Path(isolated_dir)
        assert captured_tmp_path.exists()
        assert (captured_tmp_path / "calc.py").exists()

    # After exiting context manager, ephemeral workspace must be completely removed
    assert not captured_tmp_path.exists()


# ============================================================================
# 24-26: Pre-Commit Policy & Approval Invariants
# ============================================================================


def test_git_mutation_blocked_by_policy():
    """24. Verify Git commit is completely blocked when policy is BLOCK even if approved=True."""
    state: AgentState = {
        "policy_result": PolicyEvaluationResult(
            decision=PolicyDecision.BLOCK,
            violations=[
                PolicyViolation(rule="PROTECTED_PATH", message="Cannot edit .env")
            ],
            warnings=[],
            checks={"protected_paths": "FAIL"},
            policy_version="1.0.0",
            evaluated_at="2026-09-05T00:00:00Z",
        ),
        "approval": ApprovalDecision(approved=True, reviewer="alice"),
    }

    # route_after_approval must return cleanup and reject git mutation
    dest = route_after_approval(state)
    assert dest == "cleanup"


def test_existing_hitl_still_required():
    """25. Verify approval_node pauses execution and provides policy evaluation to reviewer."""
    captured_payload = {}

    def mock_interrupt(payload):
        captured_payload.update(payload)
        return {"approved": True, "reviewer": "bob", "patch_hash": payload.get("patch_hash")}

    diff_text = "+new line\n"
    diff_hash = GitWorkspaceManager.compute_patch_hash(diff_text)
    diff = GitDiffSummary(
        branch_name="agent/task-001",
        files_changed=["app.py"],
        lines_added=1,
        lines_deleted=0,
        unified_diff=diff_text,
        patch_hash=diff_hash,
        risk_score="LOW",
    )
    pol_res = PolicyEvaluationResult(
        decision=PolicyDecision.REVIEW,
        violations=[],
        warnings=["Human check."],
        checks={"risk_policy": "REVIEW"},
        policy_version="1.0.0",
        evaluated_at="2026-09-05T00:00:00Z",
    )

    state: AgentState = {
        "git_diff": diff,
        "policy_result": pol_res,
    }

    with patch("backend.graph.nodes.interrupt", side_effect=mock_interrupt):
        out = approval_node(state)
        assert out["approval_status"] == "APPROVED"
        assert captured_payload["policy"] is not None
        assert captured_payload["policy_decision"] == "REVIEW"
        assert captured_payload["patch_hash"] == diff_hash


def test_existing_patch_hash_integrity_still_works():
    """26. Verify patch hash mismatch in approval continues to reject approval."""
    captured_payload = {}

    def mock_interrupt(payload):
        captured_payload.update(payload)
        # Reviewer approved a DIFFERENT patch hash (tampered or outdated diff)
        return {
            "approved": True,
            "reviewer": "charlie",
            "patch_hash": "different_hash_00000000000000000000000000000000000000000000000000000000",
        }

    diff_text = "+line\n"
    real_hash = GitWorkspaceManager.compute_patch_hash(diff_text)
    diff = GitDiffSummary(
        branch_name="agent/task-001",
        files_changed=["app.py"],
        lines_added=1,
        lines_deleted=0,
        unified_diff=diff_text,
        patch_hash=real_hash,
        risk_score="LOW",
    )

    state: AgentState = {
        "git_diff": diff,
    }

    with patch("backend.graph.nodes.interrupt", side_effect=mock_interrupt):
        out = approval_node(state)
        assert out["approval_status"] == "PATCH_HASH_MISMATCH"
        assert out["approval"].approved is False
        assert "PATCH_HASH_MISMATCH" in out["approval"].rejection_reason
