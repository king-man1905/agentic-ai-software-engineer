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
import subprocess
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
from backend.schemas.tenant import Role
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
    def test_clones_when_missing_and_authorized(self, tmp_path, monkeypatch, fake_clone_creates_real_git_repo):
        """A. Missing workspace + registered/authorized repo -> clone is
        attempted with a URL identifying the correct repository."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets",
            full_name="acme/widgets", github_token="secret-token-abc",
        )
        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(side_effect=fake_clone_creates_real_git_repo)
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

    def test_skips_clone_when_valid_git_workspace_for_same_repo_exists(self, tmp_path, monkeypatch):
        """B. Existing, valid git workspace already cloned from the SAME
        repository -> clone is never attempted (idempotent reuse)."""
        import subprocess

        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
        )
        monkeypatch.chdir(tmp_path)
        project_path = tmp_path / "workspace" / "default-org" / "widgets"
        project_path.mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=str(project_path), capture_output=True, text=True, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/acme/widgets.git"],
            cwd=str(project_path), capture_output=True, text=True, check=True,
        )
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
        attempted, and since repository_id was explicitly supplied by the
        caller, this now fails closed (an explicit RuntimeError) rather
        than silently degrading to a context-less/blind run - the silent
        no-op was exactly what let a later Git operation run against an
        unprovisioned workspace directory."""
        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(return_value=True)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
        )

        runner = AgentRunner()
        with pytest.raises(RuntimeError, match="WORKSPACE_PROVISIONING_FAILED"):
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

    def test_clone_url_is_clean_and_token_passed_separately(self, tmp_path, monkeypatch, fake_clone_creates_real_git_repo):
        """F. _ensure_workspace_provisioned must pass a clean,
        credential-free URL to clone_repository() and the resolved token
        only via the separate auth_header parameter - never embedded in
        the URL itself, unlike the previous
        "https://x-access-token:<token>@github.com/..." construction."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets",
            full_name="acme/widgets", github_token="fake-provisioning-token",
        )
        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(side_effect=fake_clone_creates_real_git_repo)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
        )

        runner = AgentRunner()
        runner._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
        )

        clone_spy.assert_called_once()
        clone_url, project_path = clone_spy.call_args[0][:2]
        auth_header = clone_spy.call_args.kwargs.get("auth_header")

        assert clone_url == "https://github.com/acme/widgets.git"
        assert "fake-provisioning-token" not in clone_url
        assert project_path.endswith("widgets")
        assert auth_header is not None
        assert auth_header.startswith("Authorization: Basic ")
        assert "fake-provisioning-token" not in auth_header

    def test_provisioning_failure_never_leaks_token_in_telemetry(self, tmp_path, monkeypatch):
        """H. A failed clone during workspace provisioning must never leak
        the resolved token into RUN_FAILED (or any other) telemetry event
        - only the project/repo names are recorded, matching what the
        raised RuntimeError itself is already proven to omit."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets",
            full_name="acme/widgets", github_token="fake-leak-check-token",
        )
        monkeypatch.chdir(tmp_path)
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
                "user_message": "Explain this project",
                "project_id": "widgets-fail-case",
                "repository_id": "acme/widgets",
            },
        )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        rec = telemetry_store.get_run(run_id, "default-org")
        assert rec is not None
        assert rec.status == "FAILED"

        events = telemetry_store.list_events(run_id, "default-org")
        assert len(events) > 0
        for event in events:
            assert "fake-leak-check-token" not in str(event.safe_metadata)


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
        lambda project_id, question, k=4, documents=None, organization_id=None: _KnowledgeAnswer(
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
        lambda repo_path, patches, timeout=30.0, cancel_check=None, user_request="", original_file_snapshots=None, true_original_snapshots=None: ([], None),
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

    def fake_clone(clone_url, project_path, timeout=60, auth_header=None):
        shutil.copytree(str(source_repo_fixture), project_path)
        subprocess.run(["git", "init"], cwd=project_path, capture_output=True, text=True, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", clone_url],
            cwd=project_path, capture_output=True, text=True, check=True,
        )
        return True

    monkeypatch.setattr(
        "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", fake_clone
    )

    runner = AgentRunner()
    app = create_app(runner=runner)
    client = TestClient(app)

    assert not (tmp_path / "workspace" / "default-org" / "widgets").exists()

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

    # Workspace now exists, namespaced under the caller's organization_id
    # (dev-mode fallback: "default-org"), populated from the "cloned" source.
    assert (tmp_path / "workspace" / "default-org" / "widgets" / "README.md").exists()

    state_values = runner.get_state_values(run_id, organization_id="default-org")
    assert state_values.get("rag_status") != "RAG_INSUFFICIENT_CONTEXT"
    repo_context = state_values.get("repo_context") or []
    assert len(repo_context) > 0
    assert any("README" in c.file_path for c in repo_context)


def test_repository_registration_endpoint_enables_fresh_workspace_provisioning(
    tmp_path, monkeypatch, source_repo_fixture, mocked_router_and_developer
):
    """
    Repository registration verification: unlike
    test_api_run_provisions_missing_repository_and_rag_sees_it (which
    registers the repository directly via tenant_manager, bypassing the
    HTTP API), this exercises the actual production path end-to-end for a
    genuinely fresh project_id/workspace that has never existed:

        POST /api/v1/repositories -> repository registered for the tenant
        -> POST /api/v1/runs -> _ensure_workspace_provisioned authorizes
        the repository -> workspace/<project_id> gets cloned/provisioned.

    This is exactly the path the read-only investigation found was never
    actually exercised by the "fresh" E2E runs, because their workspace
    directory already existed from an earlier, unrelated provisioning.
    GitHub credentials/network stay mocked throughout (fake_clone copies
    from a local fixture directory instead of shelling out to git).
    """
    monkeypatch.chdir(tmp_path)

    def fake_clone(clone_url, project_path, timeout=60, auth_header=None):
        shutil.copytree(str(source_repo_fixture), project_path)
        subprocess.run(["git", "init"], cwd=project_path, capture_output=True, text=True, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", clone_url],
            cwd=project_path, capture_output=True, text=True, check=True,
        )
        return True

    monkeypatch.setattr(
        "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", fake_clone
    )

    runner = AgentRunner()
    app = create_app(runner=runner)
    client = TestClient(app)

    org = tenant_manager.create_organization("org-fresh", "Org Fresh")
    user = tenant_manager.create_user("user-fresh", "fresh@org.com", "Fresh User")
    tenant_manager.add_membership(org.id, user.id, Role.ENGINEER)
    headers = {"X-Organization-ID": "org-fresh", "X-User-ID": "user-fresh"}

    # Nothing registered yet - the API is the only path exercised here.
    assert tenant_manager.get_repository("acme/fresh-widgets") is None
    assert not (tmp_path / "workspace" / "org-fresh" / "fresh-widgets").exists()

    register_resp = client.post(
        "/api/v1/repositories",
        json={"repo_full_name": "acme/fresh-widgets", "github_token": "fake-test-token"},
        headers=headers,
    )
    assert register_resp.status_code == 201
    assert register_resp.json()["is_authorized"] is True
    assert "github_token" not in register_resp.json()

    run_resp = client.post(
        "/api/v1/runs",
        json={
            "user_message": "Explain what this project builds",
            "project_id": "fresh-widgets",
            "repository_id": "acme/fresh-widgets",
        },
        headers=headers,
    )
    assert run_resp.status_code == 202
    run_id = run_resp.json()["run_id"]

    # The workspace that did not exist before the registration call now
    # does, namespaced under the requesting tenant's organization_id,
    # populated from the (faked) clone.
    assert (tmp_path / "workspace" / "org-fresh" / "fresh-widgets" / "README.md").exists()

    state_values = runner.get_state_values(run_id, organization_id="org-fresh")
    assert state_values.get("rag_status") != "RAG_INSUFFICIENT_CONTEXT"
    repo_context = state_values.get("repo_context") or []
    assert len(repo_context) > 0


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


# ============================================================================
# run_26511e289d84 investigation: workspace/default-org/e2e-test existed as
# a plain directory (no .git) forever, because _ensure_workspace_provisioned
# treated ANY existing path as "already provisioned" - skipping both
# cloning AND the authorization check below it - rather than verifying it
# was actually a valid, correctly-authorized git clone.
# ============================================================================

class TestWorkspaceProvisioningGitValidityCheck:
    def _real_git_repo_with_remote(self, path, remote_url):
        import subprocess
        path.mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=str(path), capture_output=True, text=True, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", remote_url],
            cwd=str(path), capture_output=True, text=True, check=True,
        )

    def test_A_missing_workspace_clones(self, tmp_path, monkeypatch, fake_clone_creates_real_git_repo):
        """A. No workspace directory at all -> clone is attempted."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
        )
        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(side_effect=fake_clone_creates_real_git_repo)
        monkeypatch.setattr("backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy)

        AgentRunner()._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
        )
        clone_spy.assert_called_once()

    def test_B_valid_git_workspace_for_same_repo_is_reused(self, tmp_path, monkeypatch):
        """B. Existing, valid git workspace already cloned from the SAME
        repository -> reused, clone is never attempted."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
        )
        monkeypatch.chdir(tmp_path)
        self._real_git_repo_with_remote(
            tmp_path / "workspace" / "default-org" / "widgets", "https://github.com/acme/widgets.git"
        )
        clone_spy = MagicMock(return_value=True)
        monkeypatch.setattr("backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy)

        AgentRunner()._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
        )
        clone_spy.assert_not_called()

    def test_C_existing_non_git_directory_is_not_silently_reused(self, tmp_path, monkeypatch, fake_clone_creates_real_git_repo):
        """C. Existing directory WITHOUT its own .git (exactly the
        production incident: a stale/manually-created leftover, never
        actually cloned) -> must NOT be silently treated as provisioned.
        It reauthorizes and attempts a (re-)clone - the safe, existing
        lifecycle behavior for a workspace that turns out to need
        provisioning - rather than permanently skipping both steps."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
        )
        monkeypatch.chdir(tmp_path)
        project_path = tmp_path / "workspace" / "default-org" / "widgets"
        project_path.mkdir(parents=True)
        (project_path / "some_stale_file.txt").write_text("leftover, never git-cloned\n", encoding="utf-8")
        clone_spy = MagicMock(side_effect=fake_clone_creates_real_git_repo)
        monkeypatch.setattr("backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy)

        AgentRunner()._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
        )
        # Falls through to reauthorize + attempt (re-)clone, exactly as if
        # the directory had never existed - never a silent, permanent skip.
        clone_spy.assert_called_once()

    def test_C_existing_non_git_directory_fails_closed_when_clone_actually_runs(self, tmp_path, monkeypatch):
        """C (end-to-end, real clone_repository - not mocked): a stale
        non-git directory causes an explicit WORKSPACE_PROVISIONING_FAILED
        error (git refuses to clone into a non-empty directory), never a
        silent success with unverified stale content left in place."""
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
        )
        monkeypatch.chdir(tmp_path)
        project_path = tmp_path / "workspace" / "default-org" / "widgets"
        project_path.mkdir(parents=True)
        (project_path / "some_stale_file.txt").write_text("leftover, never git-cloned\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="WORKSPACE_PROVISIONING_FAILED"):
            AgentRunner()._ensure_workspace_provisioned(
                project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
            )
        # The stale content must never be silently deleted or overwritten.
        assert (project_path / "some_stale_file.txt").exists()
        assert not (project_path / ".git").exists()

    def test_D_same_tenant_different_repository_reauthorizes_not_silently_reused(self, tmp_path, monkeypatch, fake_clone_creates_real_git_repo):
        """D. Same tenant, but the existing git workspace was cloned from a
        DIFFERENT repository than repository_id now requests -> must
        reauthorize (and attempt a fresh clone) rather than silently
        continuing to serve the old repository's stale clone."""
        tenant_manager.register_repository(
            "acme/other-repo", "default-org", "widgets", full_name="acme/other-repo",
        )
        monkeypatch.chdir(tmp_path)
        self._real_git_repo_with_remote(
            tmp_path / "workspace" / "default-org" / "widgets",
            "https://github.com/acme/widgets.git",  # the OLD, different repository
        )
        clone_spy = MagicMock(side_effect=fake_clone_creates_real_git_repo)
        monkeypatch.setattr("backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy)

        AgentRunner()._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/other-repo", organization_id="default-org",
        )
        # Reauthorized for the NEW repository and attempted a fresh clone -
        # never silently kept serving the old repo's content.
        clone_spy.assert_called_once()
        clone_url = clone_spy.call_args[0][0]
        assert clone_url == "https://github.com/acme/other-repo.git"

    def test_E_cross_tenant_access_remains_rejected(self, tmp_path, monkeypatch):
        """E. A repository registered for a DIFFERENT organization must
        remain rejected for this tenant - unaffected by the validity-check
        fix, and now fails closed (repository_id was explicitly supplied)
        rather than silently proceeding without it."""
        tenant_manager.create_organization("org-other-tenant", "Other Tenant")
        tenant_manager.register_repository(
            "acme/widgets", "org-other-tenant", "widgets", full_name="acme/widgets",
        )
        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(return_value=True)
        monkeypatch.setattr("backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy)

        with pytest.raises(RuntimeError, match="WORKSPACE_PROVISIONING_FAILED"):
            AgentRunner()._ensure_workspace_provisioned(
                project_id="widgets", repository_id="acme/widgets", organization_id="default-org",
            )
        clone_spy.assert_not_called()


# ============================================================================
# run_e40550efb53d investigation: a run whose routing decided
# requires_knowledge=False still needs its repository_id workspace
# provisioned before developer_node runs - provisioning must never be
# coupled to whether knowledge_node happens to execute.
# ============================================================================

class TestProvisioningIndependentOfRequiresKnowledge:
    def test_repository_id_provisions_even_when_requires_knowledge_is_false(
        self, tmp_path, monkeypatch, fake_clone_creates_real_git_repo
    ):
        """A run whose router decision has requires_knowledge=False (so
        knowledge_node never executes) must still have its repository_id
        workspace cloned before developer_node runs - provisioning happens
        unconditionally in start_run, never gated on the routing
        decision. Confirms this remains true regardless of what routing
        decides."""
        monkeypatch.chdir(tmp_path)
        tenant_manager.register_repository(
            "acme/widgets", "default-org", "widgets", full_name="acme/widgets",
        )
        monkeypatch.setattr(
            "backend.graph.nodes.route_task",
            lambda msg: RoutingDecision(
                task_type=TaskType.DOCUMENTATION,
                confidence=0.9,
                reasoning="test",
                requires_planning=False,
                requires_knowledge=False,
            ),
        )
        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary="test change", changes=[], requires_testing=True, notes=[]
            ),
        )
        monkeypatch.setattr(
            "backend.graph.nodes.review_code_changes",
            lambda user_request, plan, developer_result: QAResult(
                status="PASS", summary="stub"
            ),
        )
        clone_spy = MagicMock(side_effect=fake_clone_creates_real_git_repo)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
        )

        runner = AgentRunner()
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/runs",
            json={
                "user_message": "Do something that needs no repository knowledge",
                "project_id": "widgets",
                "repository_id": "acme/widgets",
            },
        )
        assert resp.status_code == 202

        clone_spy.assert_called_once()
        project_path = tmp_path / "workspace" / "default-org" / "widgets"
        assert (project_path / ".git").is_dir()


class TestNoFallthroughToAncestorGitRepository:
    """The actual observed production failure: a workspace directory
    nested inside another git repository (e.g. workspace/<org>/<project>
    living inside the application's own checkout) must never let a git
    command scoped to that workspace silently resolve against the
    ancestor's .git when the workspace itself was never a valid clone."""

    def test_unprovisioned_workspace_never_falls_through_to_ancestor_repo(
        self, tmp_path, monkeypatch
    ):
        """No authorized repository -> _ensure_workspace_provisioned fails
        closed BEFORE any directory is created at the workspace path, so
        there is nothing for a later git command to run against at all -
        eliminating the ancestor-.git-fallback risk structurally, not just
        by chance."""
        import subprocess

        monkeypatch.chdir(tmp_path)
        # A real git repository at the ANCESTOR of where the workspace
        # would live - mirrors production's workspace/ directory nested
        # inside the application's own repository checkout.
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, text=True, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/king-man1905/agentic-ai-software-engineer.git"],
            cwd=str(tmp_path), capture_output=True, text=True, check=True,
        )

        runner = AgentRunner()
        with pytest.raises(RuntimeError, match="WORKSPACE_PROVISIONING_FAILED"):
            runner._ensure_workspace_provisioned(
                project_id="e2e-test", repository_id="king-man1905/agentic-ai-test-repo",
                organization_id="default-org",
            )

        workspace_dir = tmp_path / "workspace" / "default-org" / "e2e-test"
        assert not workspace_dir.exists()
        # Proves the ancestor .git is genuinely reachable via upward search
        # from anywhere under tmp_path (i.e. a git command run from a
        # workspace path that existed but lacked its own .git would have
        # silently resolved here, exactly like production) - reinforcing
        # that failing BEFORE any such directory is ever created is what
        # actually prevents the fallthrough, not mere chance.
        probe_dir = tmp_path / "workspace" / "default-org"
        probe_dir.mkdir(parents=True)
        remote = subprocess.run(
            ["git", "-C", str(probe_dir), "remote", "get-url", "origin"],
            capture_output=True, text=True,
        )
        assert "agentic-ai-software-engineer" in remote.stdout

    def test_provisioned_workspace_has_its_own_git_shadowing_ancestor(
        self, tmp_path, monkeypatch, fake_clone_creates_real_git_repo
    ):
        """When provisioning DOES succeed, the workspace's own .git
        correctly shadows the ancestor's for any git command scoped to
        that exact path - proving a successfully-provisioned workspace is
        never at risk of the fallthrough, even nested inside another repo."""
        import subprocess

        monkeypatch.chdir(tmp_path)
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, text=True, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/king-man1905/agentic-ai-software-engineer.git"],
            cwd=str(tmp_path), capture_output=True, text=True, check=True,
        )
        tenant_manager.register_repository(
            "king-man1905/agentic-ai-test-repo", "default-org", "e2e-test",
            full_name="king-man1905/agentic-ai-test-repo",
        )
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository",
            MagicMock(side_effect=fake_clone_creates_real_git_repo),
        )

        runner = AgentRunner()
        runner._ensure_workspace_provisioned(
            project_id="e2e-test", repository_id="king-man1905/agentic-ai-test-repo",
            organization_id="default-org",
        )

        project_path = tmp_path / "workspace" / "default-org" / "e2e-test"
        remote = subprocess.run(
            ["git", "-C", str(project_path), "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True,
        )
        assert remote.stdout.strip() == "https://github.com/king-man1905/agentic-ai-test-repo.git"
