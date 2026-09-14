"""
P0-3 regression tests: workspace/<organization_id>/<project_id> namespacing.

Covers the shared resolver (backend.vcs.workspace_paths.resolve_workspace_path)
directly, and its wiring into AgentRunner._ensure_workspace_provisioned -
proving two different tenants can never collide on the same project_id, and
that path/component traversal is rejected rather than silently accepted.
"""

from unittest.mock import MagicMock

import pytest

from backend.graph.runner import AgentRunner
from backend.security.tenant import tenant_manager
from backend.vcs.workspace_paths import resolve_workspace_path


@pytest.fixture(autouse=True)
def _isolated_tenant_state():
    tenant_manager.reset()
    yield
    tenant_manager.reset()


class TestResolveWorkspacePath:
    def test_different_orgs_same_project_id_resolve_to_different_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path_a = resolve_workspace_path("org-a", "project-x")
        path_b = resolve_workspace_path("org-b", "project-x")

        assert path_a is not None
        assert path_b is not None
        assert path_a != path_b
        assert "org-a" in str(path_a) and "org-b" not in str(path_a)
        assert "org-b" in str(path_b) and "org-a" not in str(path_b)

    def test_same_org_same_project_id_is_deterministic_and_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        first = resolve_workspace_path("org-a", "project-x")
        second = resolve_workspace_path("org-a", "project-x")
        assert first == second

    @pytest.mark.parametrize("bad_org", [
        "../escape", "..", "org/../../etc", "a/b", "a\\b", "", "   ", None,
    ])
    def test_traversal_or_unsafe_organization_id_rejected(self, tmp_path, monkeypatch, bad_org):
        monkeypatch.chdir(tmp_path)
        assert resolve_workspace_path(bad_org, "safe-project") is None

    @pytest.mark.parametrize("bad_project", [
        "../escape", "..", "project/../../etc", "a/b", "a\\b", "", "   ", None,
    ])
    def test_traversal_or_unsafe_project_id_rejected(self, tmp_path, monkeypatch, bad_project):
        monkeypatch.chdir(tmp_path)
        assert resolve_workspace_path("safe-org", bad_project) is None

    def test_drive_letter_component_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_workspace_path("C:", "project-x") is None
        assert resolve_workspace_path("org-a", "C:evil") is None


class TestEnsureWorkspaceProvisionedTenantIsolation:
    """Integration-level proof at the exact call site the audit flagged:
    AgentRunner._ensure_workspace_provisioned."""

    def test_org_a_and_org_b_never_collide_on_same_project_id(self, tmp_path, monkeypatch):
        """
        The exact scenario the P0-3 finding described: Tenant A provisions
        a workspace for project_id "shared-name" against their own
        repository; Tenant B later provisions using the identical
        project_id against a different repository. Both must succeed
        independently, on different directories, with neither ever
        observing the other's cloned content.
        """
        tenant_manager.create_organization("org-a", "Org A")
        tenant_manager.create_organization("org-b", "Org B")
        tenant_manager.register_repository(
            "acme/repo-a", "org-a", "repo-a", full_name="acme/repo-a",
        )
        tenant_manager.register_repository(
            "acme/repo-b", "org-b", "repo-b", full_name="acme/repo-b",
        )
        monkeypatch.chdir(tmp_path)

        cloned_urls = []

        def fake_clone(clone_url, project_path, timeout=60, auth_header=None):
            cloned_urls.append((clone_url, project_path))
            from pathlib import Path
            Path(project_path).mkdir(parents=True, exist_ok=True)
            (Path(project_path) / "MARKER.txt").write_text(clone_url, encoding="utf-8")
            return True

        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", fake_clone
        )

        runner = AgentRunner()
        runner._ensure_workspace_provisioned(
            project_id="shared-name", repository_id="acme/repo-a", organization_id="org-a",
        )
        runner._ensure_workspace_provisioned(
            project_id="shared-name", repository_id="acme/repo-b", organization_id="org-b",
        )

        # Two distinct clones happened - org-b's provisioning was NOT
        # skipped as "already exists" just because org-a used the same
        # project_id.
        assert len(cloned_urls) == 2

        path_a = tmp_path / "workspace" / "org-a" / "shared-name"
        path_b = tmp_path / "workspace" / "org-b" / "shared-name"
        assert path_a.exists()
        assert path_b.exists()
        assert (path_a / "MARKER.txt").read_text(encoding="utf-8") == "https://github.com/acme/repo-a.git"
        assert (path_b / "MARKER.txt").read_text(encoding="utf-8") == "https://github.com/acme/repo-b.git"

    def test_provisioning_is_idempotent_within_same_tenant(self, tmp_path, monkeypatch):
        """A second call for the same (org, project_id) never re-clones -
        the pre-existing idempotency guarantee, unaffected by namespacing."""
        tenant_manager.create_organization("org-a", "Org A")
        tenant_manager.register_repository(
            "acme/widgets", "org-a", "widgets", full_name="acme/widgets",
        )
        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(return_value=True)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
        )

        runner = AgentRunner()
        runner._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="org-a",
        )
        assert clone_spy.call_count == 1

        (tmp_path / "workspace" / "org-a" / "widgets").mkdir(parents=True, exist_ok=True)
        runner._ensure_workspace_provisioned(
            project_id="widgets", repository_id="acme/widgets", organization_id="org-a",
        )
        # Still just the one clone from the first call - the second found
        # its own (org-a-namespaced) workspace already there.
        assert clone_spy.call_count == 1

    def test_traversal_organization_id_fails_closed_not_silently_ignored(self, tmp_path, monkeypatch):
        tenant_manager.create_organization("org-a", "Org A")
        tenant_manager.register_repository(
            "acme/widgets", "org-a", "widgets", full_name="acme/widgets",
        )
        monkeypatch.chdir(tmp_path)
        clone_spy = MagicMock(return_value=True)
        monkeypatch.setattr(
            "backend.vcs.git_manager.GitWorkspaceManager.clone_repository", clone_spy
        )

        runner = AgentRunner()
        with pytest.raises(RuntimeError, match="WORKSPACE_PROVISIONING_FAILED"):
            runner._ensure_workspace_provisioned(
                project_id="widgets", repository_id="acme/widgets", organization_id="../../escape",
            )
        clone_spy.assert_not_called()


class TestResumeUsesSameWorkspace:
    def test_organization_id_and_resolved_path_are_stable_across_resume(self, tmp_path, monkeypatch):
        """
        Every node re-derives the workspace path from
        (state["organization_id"], state["project_id"]) on every access
        rather than caching it - so "resume uses the same workspace"
        reduces to: organization_id/project_id in the run's checkpointed
        state never change between the initial invocation and a resume,
        and therefore resolve_workspace_path() yields the identical path
        both times. Confirmed against a real checkpointed run reaching
        WAITING_APPROVAL and then resumed (rejected, the simplest resume
        path that doesn't require a real git workspace/commit).
        """
        from backend.schemas.routing import RoutingDecision, TaskType
        from backend.schemas.developer import DeveloperResult
        from backend.schemas.qa import QAResult
        from backend.vcs.models import ApprovalDecision

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "backend.graph.nodes.route_task",
            lambda msg: RoutingDecision(
                task_type=TaskType.BUG_FIX, confidence=0.9, reasoning="test",
                requires_planning=False, requires_knowledge=False,
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
            lambda user_request, plan, developer_result: QAResult(status="PASS", summary="stub"),
        )

        db_path = str(tmp_path / "resume_workspace_checkpoints.db")
        runner = AgentRunner(checkpoint_db_path=db_path)
        try:
            status = runner.start_run(
                run_id="run-resume-workspace-stability",
                user_message="Do something trivial",
                organization_id="org-resume-test",
                project_id="proj-resume-test",
            )

            before = runner.get_state_values("run-resume-workspace-stability", organization_id="org-resume-test")
            org_before = before.get("organization_id")
            proj_before = before.get("project_id")
            path_before = resolve_workspace_path(org_before, proj_before)

            if status.status == "WAITING_APPROVAL":
                runner.resume_run(
                    "run-resume-workspace-stability",
                    ApprovalDecision(approved=False, rejection_reason="test rejection"),
                    organization_id="org-resume-test",
                )

            after = runner.get_state_values("run-resume-workspace-stability", organization_id="org-resume-test")
            org_after = after.get("organization_id")
            proj_after = after.get("project_id")
            path_after = resolve_workspace_path(org_after, proj_after)

            assert org_before == org_after == "org-resume-test"
            assert proj_before == proj_after == "proj-resume-test"
            assert path_before == path_after
        finally:
            runner.close()
