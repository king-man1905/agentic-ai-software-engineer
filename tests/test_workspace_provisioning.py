"""
Regression tests for repository workspace provisioning: a run whose
project_id has no local workspace directory, but whose repository_id
matches a registered, authorized repository, should have that repository
cloned into place before the graph runs - rather than silently falling
through to RAG_INSUFFICIENT_CONTEXT with no explanation of why.

Root cause: POST /api/v1/runs -> AgentRunner.start_run never provisioned a
local workspace for a registered repository; only the standalone
run_github_bot.py CLI script had clone logic, unreachable from the API.
"""

import shutil
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.graph.runner import AgentRunner
from backend.observability.collector import telemetry_collector
from backend.observability.store import telemetry_store
from backend.schemas.developer import DeveloperResult
from backend.schemas.qa import QAResult
from backend.schemas.routing import RoutingDecision, TaskType
from backend.security.auth import AuthMode
from backend.security.tenant import tenant_manager


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)

    from backend.observability.store import TelemetryStore

    test_db = str(tmp_path / "test_workspace_provisioning_telemetry.db")
    test_store = TelemetryStore(test_db)
    monkeypatch.setattr(telemetry_store, "db_path", test_db)
    monkeypatch.setattr(telemetry_collector, "store", test_store)

    yield

    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)


# ---------------------------------------------------------------------------
# Unit-level: AgentRunner._ensure_workspace_provisioned
# ---------------------------------------------------------------------------


class TestEnsureWorkspaceProvisioned:
    def test_clones_when_missing_and_authorized(self, tmp_path, monkeypatch):
        """A. Missing workspace + registered/authorized repo -> clone is
        attempted with a URL identifying the correct repository."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets",
            full_name="acme/widgets", github_token="secret-token-abc",
        )
        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(return_value=True)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
        )

        runner = AgentRunner()
        runner._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
        )

        clone_spy.assert_called_once()
        clone_url, project_path = clone_spy.call_args[0][:2]
        assert "acme/widgets" in clone_url
        assert project_path.endswith("widgets")

    def test_skips_clone_when_workspace_exists(self, tmp_path, monkeypatch):
        """B. Existing workspace -> clone is never attempted (idempotent)."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
        )
        monkeypatch.chdir(tmp_path)
        (tmp_path / "workspace" / "widgets").mkdir(parents=True)
        clone_spy = MagicMock(return_value=True)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
        )

        runner = AgentRunner()
        runner._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
        )

        clone_spy.assert_not_called()

    def test_never_clones_unauthorized_or_unregistered_repo(self, tmp_path, monkeypatch):
        """C. No matching registered repository -> clone is never
        attempted; this is a silent no-op (existing insufficient-context
        behavior takes over unchanged), not a hard failure."""
        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(return_value=True)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
        )

        runner = AgentRunner()
        runner._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
        )

        clone_spy.assert_not_called()

    def test_raises_explicit_error_on_clone_failure(self, tmp_path, monkeypatch):
        """D. Clone auth/network failure -> an explicit RuntimeError, never
        a silent fall-through."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository",
            lambda *a, **k: False,
        )

        runner = AgentRunner()
        with pytest.raises(RuntimeError, match="WORKSPACE_PROVISIONING_FAILED"):
            runner._ensure_workspace_provisioned(
                project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
            )

    def test_token_never_leaks_into_raised_error(self, tmp_path, monkeypatch):
        """E. The tenant-scoped GitHub token must never appear in the
        explicit failure raised on a clone failure."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets",
            full_name="acme/widgets", github_token="super-secret-token-xyz",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository",
            lambda *a, **k: False,
        )

        runner = AgentRunner()
        with pytest.raises(RuntimeError) as exc_info:
            runner._ensure_workspace_provisioned(
                project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
            )

        assert "super-secret-token-xyz" not in str(exc_info.value)

    def test_noop_without_repository_id(self, tmp_path, monkeypatch):
        """G (precondition): no repository_id supplied -> provisioning is a
        complete no-op, preserving the existing behavior for runs with no
        associated GitHub repository."""
        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(return_value=True)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
        )

        runner = AgentRunner()
        runner._ensure_workspace_provisioned(
            project_id="widgets", repository_id=None, organization_id="default-org",
        )

        clone_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Full-graph: real AgentRunner/TestClient, LLM nodes mocked
# ---------------------------------------------------------------------------


@pytest.fixture
def source_repo_fixture(tmp_path):
    """A tiny 'remote' repository to stand in for GitHub: a directory with
    a real README.md, used as the clone source when clone_repository is
    faked to copy instead of shelling out to git/network."""
    source = tmp_path / "source_repo"
    source.mkdir()
    (source / "README.md").write_text(
        "# Widgets\n\nThis project builds widgets for the acme corporation.\n",
        encoding="utf-8",
    )
    return source


@pytest.fixture
def mocked_router_and_developer(monkeypatch):
    monkeypatch.setattr(
        "backend.graph.nodes.route_task",
        lambda msg: RoutingDecision(
            task_type=TaskType.DOCUMENTATION,
            confidence=0.9,
            reasoning="test",
            requires_planning=False,
            requires_knowledge=True,
        ),
    )
    # knowledge_node's own retrieval/scan (BM25 + chunk_file over the
    # provisioned workspace) is real and is exactly what these tests
    # verify; only its final LLM summarization call is mocked, to keep
    # the tests fast/offline without touching what's actually under test.
    from backend.schemas.knowledge import KnowledgeAnswer as _KnowledgeAnswer

    monkeypatch.setattr(
        "backend.graph.nodes.answer_from_project",
        lambda project_id, question, k=4, documents=None: _KnowledgeAnswer(
            answer="stub knowledge answer",
            sources=[],
            sufficient_context=bool(documents),
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.generate_code_changes",
        lambda user_request, plan, knowledge: DeveloperResult(
            summary="test change", changes=[], requires_testing=True, notes=[]
        ),
    )
    # PASS (not the point of these tests - a FAIL status would drive the
    # graph into revision_node, which does its own real, unmocked LLM patch
    # generation) so the graph completes in a single developer/QA pass.
    monkeypatch.setattr(
        "backend.graph.nodes.review_code_changes",
        lambda user_request, plan, developer_result: QAResult(
            status="PASS", summary="stub - not the focus of this test"
        ),
    )
    monkeypatch.setattr(
        "backend.agents.developer.revise_code_changes",
        lambda user_request, plan, previous_result, qa_result: DeveloperResult(
            summary="revised", changes=[], requires_testing=True, notes=[]
        ),
    )
    # qa_node's QualityPipeline.run_all shells out to real pytest/mypy/
    # bandit/lint subprocesses against the provisioned repo. None of that
    # is what these tests are about, so it's stubbed out too, to keep the
    # tests fast/offline.
    monkeypatch.setattr(
        "backend.qa.pipeline.QualityPipeline.run_all",
        lambda repo_path, patches, timeout=30.0, cancel_check=None: ([], None),
    )
    # developer_node does its OWN inline real LLM call (get_llm +
    # invoke_structured for a locally-defined PatchResponse schema) to turn
    # non-empty repo_context into patches - separate from the
    # generate_code_changes stub above, and only reached when a workspace
    # was actually provisioned with real files. Stub it generically (by
    # whatever schema class is requested) rather than hardcoding
    # PatchResponse, since nodes.py's invoke_structured is a single shared
    # entry point.
    monkeypatch.setattr(
        "backend.graph.nodes.invoke_structured",
        lambda llm, schema_cls, prompt, *a, **k: schema_cls(patches=[]),
    )


def test_api_run_provisions_missing_repository_and_rag_sees_it(
    tmp_path, monkeypatch, source_repo_fixture, mocked_router_and_developer
):
    """
    F + I. A normal API-driven run (POST /api/v1/runs) whose project_id has
    no local workspace, but whose repository_id is a registered, authorized
    repository, provisions that workspace before the graph runs - and
    knowledge_node's RAG step then actually finds the repository's files
    (docs_count > 0), instead of RAG_INSUFFICIENT_CONTEXT with docs_count=0.
    """
    monkeypatch.chdir(tmp_path)
    tenant_manager.register_repository(
        "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
    )

    def fake_clone(clone_url, project_path, timeout=60):
        shutil.copytree(str(source_repo_fixture), project_path)
        return True

    monkeypatch.setattr(
        "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", fake_clone
    )

    runner = AgentRunner()
    app = create_app(runner=runner)
    client = TestClient(app)

    assert not (tmp_path / "workspace" / "widgets").exists()

    resp = client.post(
        "/api/v1/runs",
        json={
            "user_message": "Explain what this project builds",
            "project_id": "widgets",
            "repository_id": "acme/widgets",
        },
    )
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    # Workspace now exists, populated from the "cloned" source.
    assert (tmp_path / "workspace" / "widgets" / "README.md").exists()

    state_values = runner.get_state_values(run_id, organization_id="default-org")
    assert state_values.get("rag_status") != "RAG_INSUFFICIENT_CONTEXT"
    repo_context = state_values.get("repo_context") or []
    assert len(repo_context) > 0
    assert any("README" in c.file_path for c in repo_context)


def test_api_run_clone_failure_produces_explicit_failed_run(
    tmp_path, monkeypatch, mocked_router_and_developer
):
    """D (end-to-end). A clone failure for an authorized repository must
    surface as an explicit FAILED run with a clear provisioning error, not
    a misleading QA-failure trail - and the token must not appear in the
    recorded telemetry error message."""
    monkeypatch.chdir(tmp_path)
    tenant_manager.register_repository(
        "acme/widgets", "default-org", "widgets",
        full_name="acme/widgets", github_token="super-secret-token-xyz",
    )
    monkeypatch.setattr(
        "backend.vcs.git_manager.GitWorkspaceManager.clone_repository",
        lambda *a, **k: False,
    )

    runner = AgentRunner()
    app = create_app(runner=runner)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/runs",
        json={
            "user_message": "Explain what this project builds",
            "project_id": "widgets",
            "repository_id": "acme/widgets",
        },
    )
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    status_resp = client.get(f"/api/v1/runs/{run_id}")
    data = status_resp.json()
    assert data["status"] == "FAILED"
    assert "WORKSPACE_PROVISIONING_FAILED" in (data.get("error_summary") or "")
    assert "super-secret-token-xyz" not in (data.get("error_summary") or "")


def test_api_run_without_repository_id_keeps_existing_insufficient_context_behavior(
    tmp_path, monkeypatch, mocked_router_and_developer
):
    """
    G + H (regression). When no repository_id is supplied at all (the
    existing, pre-fix shape of a run), provisioning never triggers and
    knowledge_node/developer_node behave exactly as before: RAG reports
    insufficient context for a nonexistent workspace, and Developer
    produces zero changes rather than guessing.
    """
    monkeypatch.chdir(tmp_path)
    clone_spy = MagicMock(return_value=True)
    monkeypatch.setattr(
        "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
    )

    runner = AgentRunner()
    app = create_app(runner=runner)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/runs",
        json={"user_message": "Explain what this project builds", "project_id": "no_such_project"},
    )
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    clone_spy.assert_not_called()

    state_values = runner.get_state_values(run_id, organization_id="default-org")
    assert state_values.get("rag_status") == "RAG_INSUFFICIENT_CONTEXT"
    dev_result = state_values.get("developer_result")
    assert dev_result is not None
    assert dev_result.changes == []
