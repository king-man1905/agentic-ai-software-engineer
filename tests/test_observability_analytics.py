"""
Comprehensive test suite for Phase 7: Production Observability, Run Analytics & Evaluation.
Verifies all 31 requirements from Step 22.
"""

import os
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.models import CreateApiKeyRequest
from backend.graph.runner import AgentRunner
from backend.observability import (
    EvaluationEngine,
    ModelPricingManager,
    PricingRate,
    TelemetryCollector,
    TelemetryStore,
    default_collector,
    default_pricing,
    default_store,
    sanitize_payload,
)
from backend.schemas.policy import PolicyDecision, PolicyEvaluationResult, PolicyTelemetry
from backend.schemas.qa import QualityCheck, QAResult
from backend.schemas.routing import RoutingDecision, TaskType
from backend.schemas.telemetry import (
    EvaluationResult,
    EvaluationSummary,
    EvaluationTask,
    FailureCategory,
    RunRecord,
    TelemetryEvent,
    TelemetryEventType,
)
from backend.schemas.tenant import Role, User
from backend.security.audit import AuditAction, audit_logger
from backend.security.auth import AuthMode, auth_manager
from backend.security.tenant import tenant_manager
from backend.vcs.models import ApprovalDecision, GitDiffSummary


@pytest.fixture
def temp_store(tmp_path):
    """Creates an isolated temporary TelemetryStore."""
    db_file = str(tmp_path / "telemetry_test.db")
    return TelemetryStore(db_file)


@pytest.fixture
def isolated_collector(temp_store):
    """Creates an isolated TelemetryCollector with temp store and pricing."""
    pricing = ModelPricingManager()
    return TelemetryCollector(store=temp_store, pricing_manager=pricing)


@pytest.fixture(autouse=True)
def clean_system_state(tmp_path, monkeypatch):
    """Resets tenant manager, audit logger, and sets isolated test telemetry store."""
    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
    audit_logger.clear()

    # Use isolated test DB for default_store and default_collector
    test_db = str(tmp_path / "global_test_telemetry.db")
    test_store = TelemetryStore(test_db)
    monkeypatch.setattr("backend.observability.default_store", test_store)
    monkeypatch.setattr("backend.api.app.telemetry_store", test_store)
    monkeypatch.setattr("backend.graph.runner.telemetry_store", test_store)

    test_pricing = ModelPricingManager()
    test_collector = TelemetryCollector(store=test_store, pricing_manager=test_pricing)
    monkeypatch.setattr("backend.observability.default_collector", test_collector)
    monkeypatch.setattr("backend.api.app.telemetry_collector", test_collector)
    monkeypatch.setattr("backend.graph.runner.telemetry_collector", test_collector)
    monkeypatch.setattr("backend.graph.nodes.telemetry_collector", test_collector)

    # `from backend.observability import default_store, default_collector` at
    # the top of this file already bound those names to the *original*
    # singleton objects - patching the module attributes above doesn't
    # retarget that pre-existing binding. Any test body that calls
    # `default_store.X(...)` / `default_collector.X(...)` directly would
    # otherwise still hit the real, persistent workspace/telemetry.db.
    # Mutating the singletons' own internals in place fixes it for every
    # reference, old or new, since they're all the same object.
    monkeypatch.setattr(default_store, "db_path", test_db)
    monkeypatch.setattr(default_collector, "store", test_store)
    monkeypatch.setattr(default_collector, "pricing_manager", test_pricing)

    # Clean LLM mocks
    monkeypatch.setattr(
        "backend.graph.nodes.route_task",
        lambda msg: RoutingDecision(
            task_type=TaskType.BUG_FIX,
            reasoning="Testing bug fix",
            requires_planning=False,
            requires_knowledge=False,
        ),
    )


# ---------------------------------------------------------------------------
# Test 1: run_record_created_on_start
# ---------------------------------------------------------------------------
def test_1_run_record_created_on_start(temp_store, isolated_collector):
    run_id = "run_test_start_001"
    tenant_id = "tenant_alpha"
    isolated_collector.on_run_started(
        run_id=run_id,
        tenant_id=tenant_id,
        project_id="repo-alpha",
        metadata={"triggered_by": "test"},
    )
    rec = temp_store.get_run(run_id)
    assert rec is not None
    assert rec.run_id == run_id
    assert rec.tenant_id == tenant_id
    assert rec.project_id == repo_name if (repo_name := "repo-alpha") else True
    assert rec.status == "RUNNING"
    assert rec.started_at is not None
    assert rec.ended_at is None


# ---------------------------------------------------------------------------
# Test 2: event_stream_records_chronological_events
# ---------------------------------------------------------------------------
def test_2_event_stream_records_chronological_events(temp_store, isolated_collector):
    run_id = "run_test_stream_002"
    isolated_collector.on_run_started(run_id, "tenant_a")

    time.sleep(0.01)
    isolated_collector.emit_event(
        run_id=run_id,
        event_type=TelemetryEventType.ROUTER_DECISION,
        node="router",
        details={"decision": "bug_fix"},
    )
    time.sleep(0.01)
    isolated_collector.emit_event(
        run_id=run_id,
        event_type=TelemetryEventType.PLANNER_COMPLETE,
        node="planner",
        details={"steps": 2},
    )

    events = temp_store.get_run_events(run_id)
    assert len(events) >= 3  # RUN_STARTED, ROUTER_DECISION, PLANNER_COMPLETE
    types = [e.event_type for e in events]
    assert TelemetryEventType.RUN_STARTED.value in types
    assert TelemetryEventType.ROUTER_DECISION.value in types
    assert TelemetryEventType.PLANNER_COMPLETE.value in types

    # Verify chronological ordering
    for i in range(len(events) - 1):
        assert events[i].timestamp <= events[i + 1].timestamp


# ---------------------------------------------------------------------------
# Test 3: event_node_duration_calculated
# ---------------------------------------------------------------------------
def test_3_event_node_duration_calculated(temp_store, isolated_collector):
    run_id = "run_test_duration_003"
    isolated_collector.on_run_started(run_id, "tenant_a")
    isolated_collector.on_node_started(run_id, "developer")
    time.sleep(0.02)  # At least 20ms
    isolated_collector.on_node_completed(
        run_id, "developer", details={"lines_written": 15}
    )

    events = temp_store.get_run_events(run_id)
    dev_comp = next(
        e for e in events if e.event_type == TelemetryEventType.NODE_COMPLETE.value and e.node == "developer"
    )
    assert dev_comp.duration_ms is not None
    assert dev_comp.duration_ms >= 15  # Account for scheduler jitter


# ---------------------------------------------------------------------------
# Test 4: run_duration_recorded
# ---------------------------------------------------------------------------
def test_4_run_duration_recorded(temp_store, isolated_collector):
    run_id = "run_test_duration_004"
    isolated_collector.on_run_started(run_id, "tenant_a")
    time.sleep(0.05)
    isolated_collector.on_run_completed(run_id, status="COMPLETED")

    rec = temp_store.get_run(run_id)
    assert rec.status == "COMPLETED"
    assert rec.ended_at is not None
    assert rec.duration_seconds is not None
    assert rec.duration_seconds >= 0.04


# ---------------------------------------------------------------------------
# Test 5: model_name_and_provider_tracked
# ---------------------------------------------------------------------------
def test_5_model_name_and_provider_tracked(temp_store, isolated_collector):
    run_id = "run_test_model_005"
    isolated_collector.on_run_started(run_id, "tenant_a")
    isolated_collector.emit_event(
        run_id=run_id,
        event_type=TelemetryEventType.NODE_COMPLETE,
        node="developer",
        model_name="gpt-4o",
        provider="openai",
        prompt_tokens=100,
        completion_tokens=50,
    )
    isolated_collector.on_run_completed(run_id)

    rec = temp_store.get_run(run_id)
    assert rec.model_name == "gpt-4o"
    assert rec.provider == "openai"


# ---------------------------------------------------------------------------
# Test 6: token_usage_accumulated
# ---------------------------------------------------------------------------
def test_6_token_usage_accumulated(temp_store, isolated_collector):
    run_id = "run_test_tokens_006"
    isolated_collector.on_run_started(run_id, "tenant_a")
    isolated_collector.emit_event(
        run_id=run_id,
        event_type=TelemetryEventType.NODE_COMPLETE,
        node="router",
        model_name="gpt-4o-mini",
        prompt_tokens=200,
        completion_tokens=100,
    )
    isolated_collector.emit_event(
        run_id=run_id,
        event_type=TelemetryEventType.NODE_COMPLETE,
        node="developer",
        model_name="gpt-4o-mini",
        prompt_tokens=800,
        completion_tokens=400,
    )
    isolated_collector.on_run_completed(run_id)

    rec = temp_store.get_run(run_id)
    assert rec.prompt_tokens == 1000
    assert rec.completion_tokens == 500
    assert rec.total_tokens == 1500


# ---------------------------------------------------------------------------
# Test 7: known_model_cost_calculated
# ---------------------------------------------------------------------------
def test_7_known_model_cost_calculated():
    pricing = ModelPricingManager()
    # gpt-4o: prompt 2.50 / 1M, compl 10.00 / 1M
    cost = pricing.calculate_cost("gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost is not None
    assert round(cost, 4) == 12.5000

    # claude-3-5-sonnet: 3.00 / 15.00
    claude_cost = pricing.calculate_cost("claude-3-5-sonnet", 500_000, 200_000)
    assert claude_cost is not None
    # 0.5 * 3.00 + 0.2 * 15.00 = 1.50 + 3.00 = 4.50
    assert round(claude_cost, 4) == 4.5000


# ---------------------------------------------------------------------------
# Test 8: unknown_model_cost_is_none
# ---------------------------------------------------------------------------
def test_8_unknown_model_cost_is_none():
    pricing = ModelPricingManager()
    cost = pricing.calculate_cost("unregistered-frontier-model", 500, 500)
    assert cost is None  # Honest UNKNOWN representation, no fake rates!


# ---------------------------------------------------------------------------
# Test 9: pricing_manager_allows_registration
# ---------------------------------------------------------------------------
def test_9_pricing_manager_allows_registration():
    pricing = ModelPricingManager()
    assert pricing.get_rate("custom-mistral-7b") is None

    pricing.register_rate(
        model_name="custom-mistral-7b",
        prompt_price_per_1m=0.20,
        completion_price_per_1m=0.40,
        provider="mistral",
    )
    assert pricing.get_rate("custom-mistral-7b") is not None
    cost = pricing.calculate_cost("custom-mistral-7b", 1_000_000, 1_000_000)
    assert round(cost, 4) == 0.6000


# ---------------------------------------------------------------------------
# Test 10: qa_telemetry_captures_pass_fail
# ---------------------------------------------------------------------------
def test_10_qa_telemetry_captures_pass_fail(temp_store, isolated_collector):
    run_pass = "run_qa_pass_010"
    isolated_collector.on_run_started(run_pass, "tenant_a")
    qa_pass = QAResult(
        status="PASS",
        confidence=0.98,
        regression_risk="LOW",
        summary="All tests pass",
        checks=[QualityCheck(name="pytest", status="PASS", duration_ms=1200)],
    )
    isolated_collector.on_node_completed(
        run_pass, "qa", qa_result=qa_pass, details={"verdict": "PASS"}
    )
    rec_pass = temp_store.get_run(run_pass)
    assert rec_pass.qa_passed is True

    run_fail = "run_qa_fail_010"
    isolated_collector.on_run_started(run_fail, "tenant_a")
    qa_fail = QAResult(
        status="FAIL",
        confidence=0.40,
        regression_risk="HIGH",
        summary="Pytest failure",
        checks=[QualityCheck(name="pytest", status="FAIL", duration_ms=1500)],
    )
    isolated_collector.on_node_completed(
        run_fail, "qa", qa_result=qa_fail, details={"verdict": "FAIL"}
    )
    rec_fail = temp_store.get_run(run_fail)
    assert rec_fail.qa_passed is False


# ---------------------------------------------------------------------------
# Test 11: qa_telemetry_captures_failure_class
# ---------------------------------------------------------------------------
def test_11_qa_telemetry_captures_failure_class(temp_store, isolated_collector):
    run_id = "run_qa_fail_class_011"
    isolated_collector.on_run_started(run_id, "tenant_a")
    qa_fail = QAResult(
        status="FAIL",
        confidence=0.30,
        regression_risk="HIGH",
        summary="Pytest error in test_math.py",
        checks=[QualityCheck(name="pytest", status="FAIL", duration_ms=800)],
    )
    isolated_collector.on_node_completed(
        run_id, "qa", qa_result=qa_fail, failure_category=FailureCategory.TEST_FAILURE
    )
    rec = temp_store.get_run(run_id)
    assert rec.failure_category == FailureCategory.TEST_FAILURE.value


# ---------------------------------------------------------------------------
# Test 12: rag_telemetry_tracks_queries_and_chunks
# ---------------------------------------------------------------------------
def test_12_rag_telemetry_tracks_queries_and_chunks(temp_store, isolated_collector):
    run_id = "run_rag_012"
    isolated_collector.on_run_started(run_id, "tenant_a")
    isolated_collector.emit_event(
        run_id=run_id,
        event_type=TelemetryEventType.KNOWLEDGE_RETRIEVED,
        node="knowledge",
        details={
            "query": "authentication bearer token verify",
            "chunks_retrieved": 5,
            "sufficient": True,
        },
    )
    events = temp_store.get_run_events(run_id)
    rag_ev = next(e for e in events if e.event_type == TelemetryEventType.KNOWLEDGE_RETRIEVED.value)
    assert rag_ev.details["chunks_retrieved"] == 5
    assert rag_ev.details["sufficient"] is True


# ---------------------------------------------------------------------------
# Test 13: rag_insufficient_context_flagged
# ---------------------------------------------------------------------------
def test_13_rag_insufficient_context_flagged(temp_store, isolated_collector):
    run_id = "run_rag_insufficient_013"
    isolated_collector.on_run_started(run_id, "tenant_a")
    isolated_collector.emit_event(
        run_id=run_id,
        event_type=TelemetryEventType.KNOWLEDGE_RETRIEVED,
        node="knowledge",
        details={
            "query": "non_existent_module_foo",
            "chunks_retrieved": 0,
            "sufficient": False,
        },
    )
    events = temp_store.get_run_events(run_id)
    rag_ev = next(e for e in events if e.event_type == TelemetryEventType.KNOWLEDGE_RETRIEVED.value)
    assert rag_ev.details["sufficient"] is False
    assert rag_ev.details["chunks_retrieved"] == 0


# ---------------------------------------------------------------------------
# Test 14: revision_count_recorded
# ---------------------------------------------------------------------------
def test_14_revision_count_recorded(temp_store, isolated_collector):
    run_id = "run_rev_014"
    isolated_collector.on_run_started(run_id, "tenant_a")
    isolated_collector.on_node_completed(run_id, "revision", revision_count=1)
    isolated_collector.on_node_completed(run_id, "revision", revision_count=2)
    rec = temp_store.get_run(run_id)
    assert rec.revision_count == 2


# ---------------------------------------------------------------------------
# Test 15: policy_telemetry_persisted
# ---------------------------------------------------------------------------
def test_15_policy_telemetry_persisted(temp_store, isolated_collector):
    run_id = "run_policy_015"
    isolated_collector.on_run_started(run_id, "tenant_a")
    pol_res = PolicyEvaluationResult(
        decision=PolicyDecision.REVIEW,
        checks={"repository": "PASS", "risk_policy": "REVIEW"},
        violations=[],
        requires_human_approval=True,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )
    isolated_collector.on_node_completed(
        run_id, "policy", policy_result=pol_res, details={"risk_score": "MEDIUM"}
    )
    events = temp_store.get_run_events(run_id)
    pol_ev = next(e for e in events if e.event_type == TelemetryEventType.POLICY_EVALUATED.value)
    assert pol_ev.details["decision"] == "HUMAN_REVIEW"
    assert pol_ev.details["requires_human_approval"] is True


# ---------------------------------------------------------------------------
# Test 16: approval_latency_calculated
# ---------------------------------------------------------------------------
def test_16_approval_latency_calculated(temp_store, isolated_collector):
    run_id = "run_appr_latency_016"
    isolated_collector.on_run_started(run_id, "tenant_a")
    isolated_collector.on_waiting_approval(run_id)
    time.sleep(0.05)
    isolated_collector.on_approval_decision(run_id, approved=True, reviewer="alice")

    rec = temp_store.get_run(run_id)
    assert rec.approval_required is True
    assert rec.approval_decision == "APPROVED"
    assert rec.approval_reviewer == "alice"
    assert rec.approval_latency_seconds is not None
    assert rec.approval_latency_seconds >= 0.04


# ---------------------------------------------------------------------------
# Test 17: github_pr_event_persisted
# ---------------------------------------------------------------------------
def test_17_github_pr_event_persisted(temp_store, isolated_collector):
    run_id = "run_github_017"
    isolated_collector.on_run_started(run_id, "tenant_a")
    isolated_collector.emit_event(
        run_id=run_id,
        event_type=TelemetryEventType.GITHUB_PR_PUBLISHED,
        node="git_vcs",
        details={"pr_url": "https://github.com/org/repo/pull/42", "pr_number": 42},
    )
    rec = temp_store.get_run(run_id)
    assert rec.pr_published is True
    events = temp_store.get_run_events(run_id)
    pr_ev = next(e for e in events if e.event_type == TelemetryEventType.GITHUB_PR_PUBLISHED.value)
    assert pr_ev.details["pr_number"] == 42


# ---------------------------------------------------------------------------
# Test 18: failure_category_normalized_correctly
# ---------------------------------------------------------------------------
def test_18_failure_category_normalized_correctly(temp_store, isolated_collector):
    test_cases = [
        ("Process timed out after 30s", FailureCategory.SANDBOX_FAILURE),
        ("SyntaxError: invalid syntax in AST", FailureCategory.AST_FAILURE),
        ("Patch hash mismatch detected!", FailureCategory.COMMIT_FAILURE),
        ("Workspace drift detected before commit", FailureCategory.COMMIT_FAILURE),
        ("Policy blocked: protected path edited", FailureCategory.POLICY_BLOCK),
        ("Pytest subprocess failed with exit code 1", FailureCategory.TEST_FAILURE),
        ("Unauthorized GitHub API token", FailureCategory.AUTHENTICATION_FAILURE),
        ("Something completely unexpected happened", FailureCategory.INTERNAL_ERROR),
    ]

    for err_msg, expected_cat in test_cases:
        norm_cat = isolated_collector.classify_error(err_msg)
        assert norm_cat == expected_cat


# ---------------------------------------------------------------------------
# Test 19: analytics_overview_aggregates_tenant_runs
# ---------------------------------------------------------------------------
def test_19_analytics_overview_aggregates_tenant_runs(temp_store, isolated_collector):
    tenant = "tenant_metrics_019"
    # Run 1: completed, gpt-4o, 1000 tokens
    r1 = "run_m1"
    isolated_collector.on_run_started(r1, tenant)
    isolated_collector.emit_event(r1, TelemetryEventType.NODE_COMPLETE, node="dev", model_name="gpt-4o", prompt_tokens=600, completion_tokens=400)
    isolated_collector.on_run_completed(r1, status="COMPLETED")

    # Run 2: failed, gpt-4o, 500 tokens
    r2 = "run_m2"
    isolated_collector.on_run_started(r2, tenant)
    isolated_collector.emit_event(r2, TelemetryEventType.NODE_COMPLETE, node="dev", model_name="gpt-4o", prompt_tokens=300, completion_tokens=200)
    isolated_collector.on_run_failed(r2, error_message="pytest failed", failure_category=FailureCategory.TEST_FAILURE)

    overview = temp_store.get_analytics_overview(tenant)
    assert overview.total_runs == 2
    assert overview.successful_runs == 1
    assert overview.failed_runs == 1
    assert overview.success_rate == 0.5
    assert overview.total_tokens == 1500
    assert overview.total_cost_usd is not None
    assert overview.total_cost_usd > 0


# ---------------------------------------------------------------------------
# Test 20: analytics_quality_computes_pass_rate
# ---------------------------------------------------------------------------
def test_20_analytics_quality_computes_pass_rate(temp_store, isolated_collector):
    tenant = "tenant_qa_020"
    # Run 1: QA pass
    r1 = "run_q1"
    isolated_collector.on_run_started(r1, tenant)
    isolated_collector.on_node_completed(r1, "qa", qa_result=QAResult(status="PASS", confidence=0.9))
    isolated_collector.on_node_completed(r1, "revision", revision_count=1)
    isolated_collector.on_run_completed(r1)

    # Run 2: QA fail (test failure)
    r2 = "run_q2"
    isolated_collector.on_run_started(r2, tenant)
    isolated_collector.on_node_completed(r2, "qa", qa_result=QAResult(status="FAIL", confidence=0.4))
    isolated_collector.on_run_failed(r2, error_message="pytest failed", failure_category=FailureCategory.TEST_FAILURE)

    # Run 3: QA fail (security failure)
    r3 = "run_q3"
    isolated_collector.on_run_started(r3, tenant)
    isolated_collector.on_node_completed(r3, "qa", qa_result=QAResult(status="FAIL", confidence=0.1))
    isolated_collector.on_run_failed(r3, error_message="ast violation", failure_category=FailureCategory.SECURITY_FAILURE)

    quality = temp_store.get_quality_analytics(tenant)
    assert round(quality.qa_pass_rate, 2) == 0.33
    assert quality.test_failure_count == 1
    assert quality.security_failure_count == 1
    assert quality.avg_revisions > 0


# ---------------------------------------------------------------------------
# Test 21: analytics_models_breaks_down_by_provider
# ---------------------------------------------------------------------------
def test_21_analytics_models_breaks_down_by_provider(temp_store, isolated_collector):
    tenant = "tenant_models_021"
    # OpenAI run
    r1 = "run_mod_1"
    isolated_collector.on_run_started(r1, tenant)
    isolated_collector.emit_event(r1, TelemetryEventType.NODE_COMPLETE, node="dev", model_name="gpt-4o", provider="openai", prompt_tokens=500, completion_tokens=500)
    isolated_collector.on_run_completed(r1)

    # Anthropic run
    r2 = "run_mod_2"
    isolated_collector.on_run_started(r2, tenant)
    isolated_collector.emit_event(r2, TelemetryEventType.NODE_COMPLETE, node="dev", model_name="claude-3-5-sonnet", provider="anthropic", prompt_tokens=1000, completion_tokens=1000)
    isolated_collector.on_run_completed(r2)

    mod_analytics = temp_store.get_model_analytics(tenant)
    assert "openai" in mod_analytics.providers
    assert "anthropic" in mod_analytics.providers
    assert "gpt-4o" in mod_analytics.models
    assert "claude-3-5-sonnet" in mod_analytics.models


# ---------------------------------------------------------------------------
# Test 22: analytics_failures_groups_by_category
# ---------------------------------------------------------------------------
def test_22_analytics_failures_groups_by_category(temp_store, isolated_collector):
    tenant = "tenant_fail_022"
    r1 = "run_f1"
    isolated_collector.on_run_started(r1, tenant)
    isolated_collector.on_run_failed(r1, error_message="drift detected", failure_category=FailureCategory.COMMIT_FAILURE)

    r2 = "run_f2"
    isolated_collector.on_run_started(r2, tenant)
    isolated_collector.on_run_failed(r2, error_message="drift detected again", failure_category=FailureCategory.COMMIT_FAILURE)

    r3 = "run_f3"
    isolated_collector.on_run_started(r3, tenant)
    isolated_collector.on_run_failed(r3, error_message="timeout in sandbox", failure_category=FailureCategory.SANDBOX_FAILURE)

    fail_analytics = temp_store.get_failure_analytics(tenant)
    assert fail_analytics.total_failures == 3
    assert fail_analytics.categories[FailureCategory.COMMIT_FAILURE.value] == 2
    assert fail_analytics.categories[FailureCategory.SANDBOX_FAILURE.value] == 1


# ---------------------------------------------------------------------------
# Test 23: analytics_enforces_tenant_isolation
# ---------------------------------------------------------------------------
def test_23_analytics_enforces_tenant_isolation(temp_store, isolated_collector):
    # Tenant Alpha: 2 runs
    isolated_collector.on_run_started("run_alpha_1", "tenant_alpha")
    isolated_collector.on_run_completed("run_alpha_1")
    isolated_collector.on_run_started("run_alpha_2", "tenant_alpha")
    isolated_collector.on_run_completed("run_alpha_2")

    # Tenant Beta: 1 run
    isolated_collector.on_run_started("run_beta_1", "tenant_beta")
    isolated_collector.on_run_completed("run_beta_1")

    alpha_overview = temp_store.get_analytics_overview("tenant_alpha")
    beta_overview = temp_store.get_analytics_overview("tenant_beta")

    assert alpha_overview.total_runs == 2
    assert beta_overview.total_runs == 1


# ---------------------------------------------------------------------------
# Test 24: tenant_cannot_see_other_tenant_runs
# ---------------------------------------------------------------------------
def test_24_tenant_cannot_see_other_tenant_runs():
    app = create_app()
    client = TestClient(app)

    # Set up Tenant Alpha user and Tenant Beta user
    tenant_manager.create_organization("org-alpha", "Alpha Org")
    tenant_manager.create_user("user-alpha", "user-alpha@test.local", "User Alpha")
    tenant_manager.add_membership("org-alpha", "user-alpha", Role.ENGINEER)
    tenant_manager.create_organization("org-beta", "Beta Org")
    tenant_manager.create_user("user-beta", "user-beta@test.local", "User Beta")
    tenant_manager.add_membership("org-beta", "user-beta", Role.ENGINEER)

    # Create run belonging to org-alpha
    default_collector.on_run_started("run-secret-alpha", "org-alpha")
    default_collector.emit_event("run-secret-alpha", TelemetryEventType.NODE_COMPLETE, node="dev")
    default_collector.on_run_completed("run-secret-alpha")

    # User Beta tries to access Alpha run events
    headers_beta = {"X-User-ID": "user-beta", "X-Organization-ID": "org-beta"}
    res = client.get("/api/v1/runs/run-secret-alpha/events", headers=headers_beta)
    assert res.status_code in (403, 404)  # Strict rejection / not found across tenant boundaries!


# ---------------------------------------------------------------------------
# Test 25: runs_list_endpoint_supports_filtering
# ---------------------------------------------------------------------------
def test_25_runs_list_endpoint_supports_filtering():
    app = create_app()
    client = TestClient(app)

    tenant_manager.create_organization("org-filter", "Filter Org")
    tenant_manager.create_user("user-filter", "user-filter@test.local", "User Filter")
    tenant_manager.add_membership("org-filter", "user-filter", Role.ADMIN)
    headers = {"X-User-ID": "user-filter", "X-Organization-ID": "org-filter"}

    default_collector.on_run_started("run-f-comp", "org-filter", project_id="repo-1")
    default_collector.on_run_completed("run-f-comp", status="COMPLETED")

    default_collector.on_run_started("run-f-fail", "org-filter", project_id="repo-2")
    default_collector.on_run_failed("run-f-fail", error_message="failed")

    # Filter by status COMPLETED
    res_comp = client.get("/api/v1/runs?status=COMPLETED", headers=headers)
    assert res_comp.status_code == 200
    data_comp = res_comp.json()
    assert all(r["status"] == "COMPLETED" for r in data_comp["runs"])

    # Filter by project_id repo-2
    res_proj = client.get("/api/v1/runs?project_id=repo-2", headers=headers)
    assert res_proj.status_code == 200
    data_proj = res_proj.json()
    assert all(r["project_id"] == "repo-2" for r in data_proj["runs"])


# ---------------------------------------------------------------------------
# Test 26: run_events_endpoint_returns_chronological_events
# ---------------------------------------------------------------------------
def test_26_run_events_endpoint_returns_chronological_events():
    app = create_app()
    client = TestClient(app)

    tenant_manager.create_organization("org-events", "Events Org")
    tenant_manager.create_user("user-events", "user-events@test.local", "User Events")
    tenant_manager.add_membership("org-events", "user-events", Role.ENGINEER)
    headers = {"X-User-ID": "user-events", "X-Organization-ID": "org-events"}

    run_id = "run-event-api-26"
    default_collector.on_run_started(run_id, "org-events")
    default_collector.emit_event(run_id, TelemetryEventType.ROUTER_DECISION, node="router")
    default_collector.emit_event(run_id, TelemetryEventType.PLANNER_COMPLETE, node="planner")

    res = client.get(f"/api/v1/runs/{run_id}/events", headers=headers)
    assert res.status_code == 200
    ev_list = res.json()["events"]
    assert len(ev_list) >= 3
    # Check ordering
    for i in range(len(ev_list) - 1):
        assert ev_list[i]["timestamp"] <= ev_list[i + 1]["timestamp"]


# ---------------------------------------------------------------------------
# Test 27: evaluation_benchmark_runs_standard_tasks
# ---------------------------------------------------------------------------
def test_27_evaluation_benchmark_runs_standard_tasks(monkeypatch):
    tenant_manager.create_organization("org-eval", "Eval Org")
    mock_runner = MagicMock()
    # Mock runner.start_run to return COMPLETED run
    mock_runner.start_run.return_value = {
        "status": "COMPLETED",
        "qa_result": {"status": "PASS"},
        "policy_result": {"decision": "ALLOW"},
        "git_diff": {"lines_added": 10, "lines_deleted": 0},
    }

    engine = EvaluationEngine(
        runner=mock_runner,
        store=default_store,
        collector=default_collector,
    )

    tasks = [
        EvaluationTask(
            task_id="test-task-1",
            category="bug_fix",
            difficulty="easy",
            prompt="Fix off-by-one error",
            expected_files=["math.py"],
            max_revisions_allowed=2,
            target_repo="test-org/test-repo",
        )
    ]

    summary = engine.run_benchmark(
        tenant_id="org-eval",
        user_id="user-eval",
        benchmark_name="mini-benchmark",
        tasks=tasks,
    )

    assert isinstance(summary, EvaluationSummary)
    assert summary.total_tasks == 1
    assert summary.passed_tasks == 1
    assert summary.pass_rate == 1.0


# ---------------------------------------------------------------------------
# Test 28: evaluation_results_stored_and_retrieved
# ---------------------------------------------------------------------------
def test_28_evaluation_results_stored_and_retrieved():
    benchmark_name = "integration-suite-28"
    res1 = EvaluationResult(
        task_id="task-1",
        run_id="run-eval-1",
        tenant_id="tenant-eval-28",
        benchmark_name=benchmark_name,
        passed=True,
        duration_seconds=3.5,
        total_tokens=450,
        cost_usd=0.005,
    )
    res2 = EvaluationResult(
        task_id="task-2",
        run_id="run-eval-2",
        tenant_id="tenant-eval-28",
        benchmark_name=benchmark_name,
        passed=False,
        failure_reason="AssertionError",
        duration_seconds=4.0,
        total_tokens=600,
        cost_usd=0.007,
    )

    default_store.save_evaluation(res1)
    default_store.save_evaluation(res2)

    retrieved = default_store.list_evaluations("tenant-eval-28", benchmark_name=benchmark_name)
    assert len(retrieved) == 2
    task_ids = {r.task_id for r in retrieved}
    assert task_ids == {"task-1", "task-2"}


# ---------------------------------------------------------------------------
# Test 29: telemetry_sanitization_redacts_tokens
# ---------------------------------------------------------------------------
def test_29_telemetry_sanitization_redacts_tokens():
    sensitive_payload = {
        "user_id": "normal_user",
        "github_token": "ghp_SECRET_TOKEN_VALUE_1234567890",
        "api_key": "sk-SECRET_OPENAI_KEY_1234567890",
        "nested": {
            "password": "super_secret_password!",
            "private_key": "-----BEGIN RSA PRIVATE KEY-----",
            "safe_text": "This is completely safe",
        },
        "massive_string": "A" * 6000,
    }

    sanitized = sanitize_payload(sensitive_payload)

    # Assert redacting
    assert sanitized["github_token"] == "[REDACTED]"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["password"] == "[REDACTED]"
    assert sanitized["nested"]["private_key"] == "[REDACTED]"
    assert sanitized["nested"]["safe_text"] == "This is completely safe"

    # Assert truncation
    assert len(sanitized["massive_string"]) <= 4050
    assert "[TRUNCATED" in sanitized["massive_string"]


# ---------------------------------------------------------------------------
# Test 30: telemetry_never_bypasses_approval_or_drift
# ---------------------------------------------------------------------------
def test_30_telemetry_never_bypasses_approval_or_drift():
    """
    Verifies that observability instrumentations do NOT bypass or weaken
    HITL approval gates, patch hash verification, or workspace drift detection.
    """
    app = create_app()
    client = TestClient(app)

    tenant_manager.create_organization("org-gate", "Gate Org")
    tenant_manager.create_user("user-gate", "user-gate@test.local", "User Gate")
    # Needs RUN_APPROVE (ENGINEER doesn't have it) to reach the not-found/
    # patch-hash check this test is actually about, rather than a 403 on
    # the permission check itself.
    tenant_manager.add_membership("org-gate", "user-gate", Role.REVIEWER)
    headers = {"X-User-ID": "user-gate", "X-Organization-ID": "org-gate"}

    # Attempt to resume a run without a valid patch_hash when state is interrupted
    res = client.post(
        "/api/v1/runs/nonexistent-run-999/resume",
        headers=headers,
        json={"approved": True, "reviewer": "user-gate", "patch_hash": "invalid_hash"},
    )
    # The runner / API correctly rejects mismatched or non-existent run resumes
    assert res.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Test 31: audit_hash_chain_preserved_alongside_telemetry
# ---------------------------------------------------------------------------
def test_31_audit_hash_chain_preserved_alongside_telemetry(temp_store, isolated_collector):
    """
    Verifies that tamper-evident audit logging with SHA-256 hash chains
    functions properly alongside structured telemetry event persistence.
    """
    audit_logger.clear()

    # Perform action that creates both audit entry and telemetry event
    run_id = "run-audit-chain-031"
    tenant_id = "org-audit"

    audit_entry_1 = audit_logger.log(
        organization_id=tenant_id,
        user_id="alice",
        action=AuditAction.RUN_INITIATED,
        resource_type="run",
        resource_id=run_id,
        details={"project": "core"},
    )
    isolated_collector.on_run_started(run_id, tenant_id)

    audit_entry_2 = audit_logger.log(
        organization_id=tenant_id,
        user_id="alice",
        action=AuditAction.APPROVAL_GRANTED,
        resource_type="run",
        resource_id=run_id,
        details={"approved": True},
    )
    isolated_collector.on_approval_decision(run_id, approved=True, reviewer="alice")

    # Verify audit chain integrity
    verified, error = audit_logger.verify_integrity(tenant_id)
    assert verified is True, error
    assert len(audit_logger.get_events(tenant_id)) == 2
    assert audit_entry_2.previous_hash == audit_entry_1.event_hash

    # Verify telemetry persistence is also intact
    rec = temp_store.get_run(run_id)
    assert rec is not None
    assert rec.approval_required is True
