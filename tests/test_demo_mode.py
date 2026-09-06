"""
Tests for Free Render Demo Mode isolation and safety invariants.

Verifies:
1. Production mode is the default and fails closed.
2. Demo mode is only activated when DEPLOYMENT_MODE is explicitly set to 'demo'.
3. /health and /health/ready accurately disclose deployment mode and storage durability.
4. Checkpoint failure behavior remains fail-closed in production.
5. All security boundaries (authentication, RBAC, tenant isolation) remain active in demo mode.
"""

import os
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.core.config import is_demo_mode
from backend.graph.runner import AgentRunner
from backend.observability.store import telemetry_store
from backend.security.auth import AuthMode
from backend.security.tenant import tenant_manager


@pytest.fixture(autouse=True)
def reset_security_and_env(monkeypatch):
    """Ensures clean tenant manager and deployment mode state before and after each test."""
    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("ALLOW_VOLATILE_CHECKPOINTER", raising=False)
    if hasattr(telemetry_store, "reopen"):
        telemetry_store.reopen()
    yield
    if hasattr(telemetry_store, "reopen"):
        telemetry_store.reopen()


def test_default_deployment_mode_is_production():
    """Confirms production mode is the default and fails closed when env is unset."""
    assert is_demo_mode() is False


def test_explicit_demo_mode_activation(monkeypatch):
    """Verifies is_demo_mode() only returns True when explicitly set to 'demo'."""
    monkeypatch.setenv("DEPLOYMENT_MODE", "demo")
    assert is_demo_mode() is True

    monkeypatch.setenv("DEPLOYMENT_MODE", "DEMO")
    assert is_demo_mode() is True

    monkeypatch.setenv("DEPLOYMENT_MODE", " Demo ")
    assert is_demo_mode() is True

    monkeypatch.setenv("DEPLOYMENT_MODE", "production")
    assert is_demo_mode() is False

    monkeypatch.setenv("DEPLOYMENT_MODE", "staging")
    assert is_demo_mode() is False


def test_production_fail_closed_checkpointing_preserved(tmp_path, monkeypatch):
    """
    Confirms that in production mode (even if someone attempts volatile fallback),
    corrupted or uncreatable SQLite paths fail closed with DURABLE_CHECKPOINT_INIT_FAILED.
    """
    monkeypatch.setattr(tenant_manager, "auth_mode", AuthMode.PRODUCTION)
    monkeypatch.setattr(tenant_manager, "dev_auth_fallback", False)
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)

    bad_db_path = str(tmp_path)  # directory cannot be opened as SQLite database file

    with pytest.raises(RuntimeError) as exc_info:
        AgentRunner(checkpoint_db_path=bad_db_path)

    assert "DURABLE_CHECKPOINT_INIT_FAILED" in str(exc_info.value)


def test_demo_mode_checkpoint_behavior(tmp_path, monkeypatch):
    """
    Verifies that in demo mode:
    1. Valid SQLite path initializes a functional checkpointer with is_ephemeral == True.
    2. Failed SQLite path fails closed unless ALLOW_VOLATILE_CHECKPOINTER is explicitly true.
    """
    monkeypatch.setenv("DEPLOYMENT_MODE", "demo")

    # 1. Normal SQLite in demo mode
    db_path = str(tmp_path / "demo_checkpoints.db")
    runner = AgentRunner(checkpoint_db_path=db_path)
    try:
        assert runner.is_ready() is True
        assert runner.is_ephemeral is True
    finally:
        runner.close()

    # 2. Broken path without volatile permission fails closed
    bad_db_path = str(tmp_path)
    with pytest.raises(RuntimeError) as exc_info:
        AgentRunner(checkpoint_db_path=bad_db_path)
    assert "DEMO_CHECKPOINT_INIT_FAILED" in str(exc_info.value)

    # 3. Broken path WITH explicit volatile permission falls back safely
    monkeypatch.setenv("ALLOW_VOLATILE_CHECKPOINTER", "true")
    volatile_runner = AgentRunner(checkpoint_db_path=bad_db_path)
    try:
        assert volatile_runner.is_ready() is True
        assert volatile_runner.is_ephemeral is True
    finally:
        volatile_runner.close()


def test_health_endpoint_reports_deployment_mode(monkeypatch):
    """Verifies that /health discloses deployment_mode and ephemeral storage notice in demo mode."""
    # Production
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    app_prod = create_app()
    client_prod = TestClient(app_prod)
    resp_prod = client_prod.get("/health")
    assert resp_prod.status_code == 200
    data_prod = resp_prod.json()
    assert data_prod["status"] == "healthy"
    assert data_prod["deployment_mode"] == "production"
    assert "storage_notice" not in data_prod

    # Demo
    monkeypatch.setenv("DEPLOYMENT_MODE", "demo")
    app_demo = create_app()
    client_demo = TestClient(app_demo)
    resp_demo = client_demo.get("/health")
    assert resp_demo.status_code == 200
    data_demo = resp_demo.json()
    assert data_demo["status"] == "healthy"
    assert data_demo["deployment_mode"] == "demo"
    assert "DEMO_EPHEMERAL_STORAGE" in data_demo["storage_notice"]


def test_readiness_endpoint_storage_disclosure(monkeypatch):
    """Verifies that /health/ready includes storage_durability disclosure."""
    # Production
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    app_prod = create_app()
    client_prod = TestClient(app_prod)
    resp_prod = client_prod.get("/health/ready")
    assert resp_prod.status_code == 200
    report_prod = resp_prod.json()
    assert report_prod["deployment_mode"] == "production"
    assert report_prod["checks"]["storage_durability"]["persistent"] is True
    assert "DURABLE_STORAGE" in report_prod["checks"]["storage_durability"]["notice"]

    # Demo
    monkeypatch.setenv("DEPLOYMENT_MODE", "demo")
    app_demo = create_app()
    client_demo = TestClient(app_demo)
    resp_demo = client_demo.get("/health/ready")
    assert resp_demo.status_code == 200
    report_demo = resp_demo.json()
    assert report_demo["deployment_mode"] == "demo"
    assert report_demo["checks"]["storage_durability"]["persistent"] is False
    assert "DEMO_EPHEMERAL_STORAGE" in report_demo["checks"]["storage_durability"]["notice"]


def test_demo_mode_retains_authentication_and_tenant_boundaries(monkeypatch):
    """
    Confirms that in demo mode with AUTH_MODE=production, authentication is NOT bypassed:
    unauthenticated requests are rejected with 401, while authenticated requests succeed.
    """
    monkeypatch.setenv("DEPLOYMENT_MODE", "demo")
    monkeypatch.setattr(tenant_manager, "auth_mode", AuthMode.PRODUCTION)
    monkeypatch.setattr(tenant_manager, "dev_auth_fallback", False)

    app = create_app()
    client = TestClient(app)
    # 1. Unauthenticated request must be rejected
    unauth_resp = client.get("/api/v1/tenant/context")
    assert unauth_resp.status_code == 401
    assert "AUTHENTICATION_REQUIRED" in unauth_resp.json()["detail"]

    # 2. Authenticated request with valid sandbox key must succeed
    auth_resp = client.get(
        "/api/v1/tenant/context",
        headers={"Authorization": "Bearer default-sandbox-key"},
    )
    assert auth_resp.status_code == 200
    ctx = auth_resp.json()
    assert ctx["organization_id"] == "default-org"
    assert ctx["user_id"] == "default-user"
    assert ctx["role"] == "OWNER"
