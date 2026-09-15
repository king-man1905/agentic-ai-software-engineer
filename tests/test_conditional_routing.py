import pytest
from backend.graph.state import AgentState
from backend.graph.nodes import (
    route_after_router,
    route_after_planner,
    route_after_knowledge,
    qa_router,
    MAX_REVISIONS,
)
from backend.schemas.routing import RoutingDecision, TaskType
from backend.schemas.qa import QAResult


class TestRouteAfterRouter:
    def test_general_task_routes_to_end(self):
        state: AgentState = {
            "user_message": "Hello",
            "routing": RoutingDecision(
                task_type=TaskType.GENERAL,
                requires_planning=False,
                requires_knowledge=False,
                reasoning="General conversation",
            ),
        }
        assert route_after_router(state) == "end"

    def test_general_task_with_planning_still_routes_to_end(self):
        state: AgentState = {
            "user_message": "Hello",
            "routing": RoutingDecision(
                task_type=TaskType.GENERAL,
                requires_planning=True,
                requires_knowledge=False,
                reasoning="General conversation",
            ),
        }
        assert route_after_router(state) == "end"

    def test_requires_planning_routes_to_planner(self):
        state: AgentState = {
            "user_message": "Build a new feature",
            "routing": RoutingDecision(
                task_type=TaskType.CODE_GENERATION,
                requires_planning=True,
                requires_knowledge=True,
                reasoning="Complex feature implementation",
            ),
        }
        assert route_after_router(state) == "planner"

    def test_knowledge_search_without_planning_routes_to_knowledge(self):
        state: AgentState = {
            "user_message": "Where is auth implemented?",
            "routing": RoutingDecision(
                task_type=TaskType.KNOWLEDGE_SEARCH,
                requires_planning=False,
                requires_knowledge=True,
                reasoning="Find auth code in project",
            ),
        }
        assert route_after_router(state) == "knowledge"

    def test_requires_knowledge_without_planning_routes_to_knowledge(self):
        state: AgentState = {
            "user_message": "Explain how auth works in this project",
            "routing": RoutingDecision(
                task_type=TaskType.CODE_EXPLANATION,
                requires_planning=False,
                requires_knowledge=True,
                reasoning="Explanation requires project context",
            ),
        }
        assert route_after_router(state) == "knowledge"

    def test_direct_developer_route(self):
        state: AgentState = {
            "user_message": "Fix typo in variable name",
            "routing": RoutingDecision(
                task_type=TaskType.BUG_FIX,
                requires_planning=False,
                requires_knowledge=False,
                reasoning="Simple fix",
            ),
        }
        assert route_after_router(state) == "developer"


class TestRouteAfterPlanner:
    def test_routes_to_knowledge_when_required_and_project_id_present(self):
        state: AgentState = {
            "user_message": "Refactor auth logic",
            "project_id": "proj_123",
            "routing": RoutingDecision(
                task_type=TaskType.CODE_GENERATION,
                requires_planning=True,
                requires_knowledge=True,
                reasoning="Needs context",
            ),
        }
        assert route_after_planner(state) == "knowledge"

    def test_routes_to_developer_when_knowledge_required_but_no_project_id(self):
        state: AgentState = {
            "user_message": "Refactor auth logic",
            "project_id": None,
            "routing": RoutingDecision(
                task_type=TaskType.CODE_GENERATION,
                requires_planning=True,
                requires_knowledge=True,
                reasoning="Needs context",
            ),
        }
        assert route_after_planner(state) == "developer"

    def test_routes_to_developer_when_knowledge_not_required(self):
        state: AgentState = {
            "user_message": "Write a standalone script",
            "project_id": "proj_123",
            "routing": RoutingDecision(
                task_type=TaskType.CODE_GENERATION,
                requires_planning=True,
                requires_knowledge=False,
                reasoning="No project context needed",
            ),
        }
        assert route_after_planner(state) == "developer"

    def test_routes_to_developer_when_no_routing_in_state(self):
        state: AgentState = {
            "user_message": "Write a script",
        }
        assert route_after_planner(state) == "developer"


class TestRouteAfterKnowledge:
    def test_knowledge_search_without_planning_routes_to_end(self):
        state: AgentState = {
            "user_message": "Where is auth defined?",
            "routing": RoutingDecision(
                task_type=TaskType.KNOWLEDGE_SEARCH,
                requires_planning=False,
                requires_knowledge=True,
                reasoning="Search query",
            ),
        }
        assert route_after_knowledge(state) == "end"

    def test_knowledge_search_with_planning_routes_to_developer(self):
        state: AgentState = {
            "user_message": "Find and fix auth bug",
            "routing": RoutingDecision(
                task_type=TaskType.KNOWLEDGE_SEARCH,
                requires_planning=True,
                requires_knowledge=True,
                reasoning="Search and fix",
            ),
        }
        assert route_after_knowledge(state) == "developer"

    def test_other_task_routes_to_developer(self):
        state: AgentState = {
            "user_message": "Refactor login",
            "routing": RoutingDecision(
                task_type=TaskType.CODE_GENERATION,
                requires_planning=True,
                requires_knowledge=True,
                reasoning="Feature build",
            ),
        }
        assert route_after_knowledge(state) == "developer"

    def test_routes_to_developer_when_no_routing(self):
        state: AgentState = {
            "user_message": "Hello",
        }
        assert route_after_knowledge(state) == "developer"


class TestQARouter:
    def test_pass_status_routes_to_pass(self):
        state: AgentState = {
            "qa_result": QAResult(
                status="PASS",
                issues=[],
                test_cases=["test_login"],
                summary="All checks passed.",
            ),
            "revision_count": 0,
        }
        assert qa_router(state) == "pass"

    def test_pass_status_with_high_revision_count_still_passes(self):
        state: AgentState = {
            "qa_result": QAResult(
                status="PASS",
                issues=[],
                test_cases=["test_login"],
                summary="All checks passed.",
            ),
            "revision_count": MAX_REVISIONS,
        }
        assert qa_router(state) == "pass"

    def test_fail_status_routes_to_fail_when_below_max_revisions(self):
        state: AgentState = {
            "qa_result": QAResult(
                status="FAIL",
                issues=[],
                test_cases=[],
                summary="Syntax error found.",
            ),
            "revision_count": 0,
        }
        assert qa_router(state) == "fail"

    def test_fail_status_routes_to_fail_on_intermediate_revision(self):
        state: AgentState = {
            "qa_result": QAResult(
                status="FAIL",
                issues=[],
                test_cases=[],
                summary="Edge case failed.",
            ),
            "revision_count": MAX_REVISIONS - 1,
        }
        assert qa_router(state) == "fail"

    def test_fail_status_routes_to_max_retries_at_limit(self):
        state: AgentState = {
            "qa_result": QAResult(
                status="FAIL",
                issues=[],
                test_cases=[],
                summary="Still failing.",
            ),
            "revision_count": MAX_REVISIONS,
        }
        assert qa_router(state) == "max_retries"

    def test_fail_status_routes_to_max_retries_above_limit(self):
        state: AgentState = {
            "qa_result": QAResult(
                status="FAIL",
                issues=[],
                test_cases=[],
                summary="Still failing.",
            ),
            "revision_count": MAX_REVISIONS + 1,
        }
        assert qa_router(state) == "max_retries"

    def test_fail_status_without_revision_count_in_state(self):
        state: AgentState = {
            "qa_result": QAResult(
                status="FAIL",
                issues=[],
                test_cases=[],
                summary="Failed.",
            ),
        }
        assert qa_router(state) == "fail"

    @pytest.mark.parametrize("status_variant", ["pass", "Pass", "PASS", "pAsS", " Pass ", "PASS\n", "\tpass\t"])
    def test_qa_router_status_case_insensitive_pass(self, status_variant: str):
        state: AgentState = {
            "qa_result": QAResult(
                status=status_variant,
                issues=[],
                test_cases=[],
                summary="Passed with variant.",
            ),
            "revision_count": 0,
        }
        assert qa_router(state) == "pass"

    @pytest.mark.parametrize("status_variant", ["fail", "Fail", "FAIL", " FAIL ", "FAIL\n", ""])
    def test_qa_router_status_case_insensitive_fail(self, status_variant: str):
        state: AgentState = {
            "qa_result": QAResult(
                status=status_variant,
                issues=[],
                test_cases=[],
                summary="Failed with variant.",
            ),
            "revision_count": 0,
        }
        assert qa_router(state) == "fail"

    def test_qa_router_status_none_handled_safely(self):
        qa = QAResult.model_construct(
            status=None,
            issues=[],
            test_cases=[],
            summary="None status.",
        )
        state: AgentState = {
            "qa_result": qa,
            "revision_count": 0,
        }
        assert qa_router(state) == "fail"


class TestNodeFallbacks:
    def test_developer_node_persists_fallback_plan(self, monkeypatch):
        from backend.graph.nodes import developer_node
        from backend.schemas.developer import DeveloperResult
        from backend.schemas.planning import ExecutionPlan

        mock_dev_result = DeveloperResult(
            summary="Mock implementation",
            changes=[],
            requires_testing=True,
            notes=[],
        )

        def mock_generate_code_changes(user_request, plan, knowledge):
            assert isinstance(plan, ExecutionPlan)
            assert plan.goal == user_request
            return mock_dev_result

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            mock_generate_code_changes,
        )

        state: AgentState = {
            "user_message": "Fix login bug",
        }

        output = developer_node(state)
        assert output["developer_result"] == mock_dev_result
        assert "plan" in output
        assert isinstance(output["plan"], ExecutionPlan)
        assert output["plan"].goal == "Fix login bug"

    def test_developer_node_gives_patch_prompt_one_contiguous_readme_block(self, tmp_path, monkeypatch):
        """
        Integration check for the fragmentation fix: developer_node's
        exact-snippet patch-generation prompt must contain README.md
        exactly once, as its full verbatim content - not split across
        multiple overlapping "FILE: README.md (Lines ...)" fragments.
        """
        import os
        from pathlib import Path
        from backend.graph.nodes import developer_node
        from backend.schemas.developer import DeveloperResult
        from backend.schemas.planning import ExecutionPlan

        project_id = "consolidation_test_proj_xyz"
        # Namespaced under "default-org" to match state.get("organization_id",
        # "default-org") - the resolver P0-3 introduced (resolve_workspace_path).
        workspace_dir = tmp_path / "workspace" / "default-org" / project_id
        workspace_dir.mkdir(parents=True)
        content = "\n".join(f"## Section {i}\nBody text for section {i}.\n" for i in range(1, 60))
        (workspace_dir / "README.md").write_text(content, encoding="utf-8")

        # Force nodes.py's os.getcwd()-based fallback path to resolve into
        # this tmp workspace instead of the real repo's workspace/.
        assert not (Path("workspace") / "default-org" / project_id).exists()
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary="x", changes=[], requires_testing=True, notes=[]
            ),
        )
        monkeypatch.setattr("backend.services.llm.get_llm", lambda: object())

        captured_prompts = []

        class _FakePatchResult:
            patches = []

        def fake_invoke_structured(llm, schema, prompt):
            captured_prompts.append(prompt)
            return _FakePatchResult()

        monkeypatch.setattr("backend.graph.nodes.invoke_structured", fake_invoke_structured)

        state: AgentState = {
            "user_message": "Update README",
            "project_id": project_id,
            "plan": ExecutionPlan(goal="Update README", steps=[], success_criteria="Done"),
        }

        developer_node(state)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert prompt.count("FILE: README.md") == 1
        assert content in prompt

    def test_developer_node_tiny_readme_exact_match_patch_applies_successfully(self, tmp_path, monkeypatch):
        """
        FIX 3 regression test: a genuinely tiny README (matching the real
        repository that triggered the investigation - a single line, no
        trailing newline) is shown to the patch-generation prompt as a
        [COMPLETE FILE CONTENT] block, and when the LLM returns a patch
        whose original_code_snippet is copied verbatim from that exact
        content, SafePatcher's exact-match AST pre-flight validation
        succeeds (not weakened - it still requires a byte-exact match) and
        the patch is actually applied to disk.
        """
        import os
        from pathlib import Path
        from backend.developer.models import FilePatch
        from backend.graph.nodes import developer_node
        from backend.schemas.developer import DeveloperResult
        from backend.schemas.planning import ExecutionPlan

        project_id = "tiny_readme_proj"
        # Namespaced under "default-org" to match state.get("organization_id",
        # "default-org") - the resolver P0-3 introduced (resolve_workspace_path).
        workspace_dir = tmp_path / "workspace" / "default-org" / project_id
        workspace_dir.mkdir(parents=True)
        # Exactly the real repository's README: 22 bytes, one line, no
        # trailing newline.
        original_content = "# agentic-ai-test-repo"
        (workspace_dir / "README.md").write_text(original_content, encoding="utf-8")

        assert not (Path("workspace") / "default-org" / project_id).exists()
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary="x", changes=[], requires_testing=True, notes=[]
            ),
        )
        monkeypatch.setattr("backend.services.llm.get_llm", lambda: object())

        captured_prompts = []
        updated_content = "# agentic-ai-test-repo\n\n## E2E Test\n\nDescription.\n"

        class _FakePatchResult:
            patches = [
                FilePatch(
                    file_path="README.md",
                    original_code_snippet=original_content,
                    updated_code_snippet=updated_content,
                    explanation="Add E2E Test section",
                )
            ]

        def fake_invoke_structured(llm, schema, prompt):
            captured_prompts.append(prompt)
            return _FakePatchResult()

        monkeypatch.setattr("backend.graph.nodes.invoke_structured", fake_invoke_structured)

        state: AgentState = {
            "user_message": "Add an E2E Test section to README.md",
            "project_id": project_id,
            "plan": ExecutionPlan(goal="Add E2E Test section", steps=[], success_criteria="Done"),
        }

        out = developer_node(state)

        # The prompt gave the LLM the file marked as complete/verbatim, and
        # contained the real content the LLM's snippet must match.
        prompt = captured_prompts[0]
        assert "[COMPLETE FILE CONTENT - verbatim, nothing omitted]" in prompt
        assert original_content in prompt

        # SafePatcher's exact-match validation was not weakened - it still
        # ran, and passed because the snippet genuinely matched byte-exact.
        assert len(out["generated_patches"]) == 1
        assert out["generated_patches"][0].original_code_snippet == original_content

        # The patch was actually applied to the real workspace file.
        assert (workspace_dir / "README.md").read_text(encoding="utf-8") == updated_content

    def test_developer_node_prompt_instructs_empty_snippet_for_whole_file(self, tmp_path, monkeypatch):
        """
        Root-cause fix regression test: the patch-generation prompt must
        explicitly tell the model to use the empty-original_code_snippet
        whole-file-replace convention for [COMPLETE FILE CONTENT] blocks -
        never ask it to quote/copy the existing content as an anchor,
        which is what produced non-matching snippets in practice.
        """
        import os
        from pathlib import Path
        from backend.graph.nodes import developer_node
        from backend.schemas.developer import DeveloperResult
        from backend.schemas.planning import ExecutionPlan

        project_id = "whole_file_prompt_proj"
        # Namespaced under "default-org" to match state.get("organization_id",
        # "default-org") - the resolver P0-3 introduced (resolve_workspace_path).
        workspace_dir = tmp_path / "workspace" / "default-org" / project_id
        workspace_dir.mkdir(parents=True)
        (workspace_dir / "README.md").write_text("# agentic-ai-test-repo", encoding="utf-8")

        assert not (Path("workspace") / "default-org" / project_id).exists()
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary="x", changes=[], requires_testing=True, notes=[]
            ),
        )
        monkeypatch.setattr("backend.services.llm.get_llm", lambda: object())

        captured_prompts = []

        class _FakePatchResult:
            patches = []

        def fake_invoke_structured(llm, schema, prompt):
            captured_prompts.append(prompt)
            return _FakePatchResult()

        monkeypatch.setattr("backend.graph.nodes.invoke_structured", fake_invoke_structured)

        state: AgentState = {
            "user_message": "Add an E2E Test section to README.md",
            "project_id": project_id,
            "plan": ExecutionPlan(goal="Add E2E Test section", steps=[], success_criteria="Done"),
        }

        developer_node(state)

        prompt = captured_prompts[0]
        assert "[COMPLETE FILE CONTENT - verbatim, nothing omitted]" in prompt
        assert 'set original_code_snippet to an empty string ("")' in prompt
        assert 'replace the entire file with updated_code_snippet' in prompt
        assert "Do NOT quote, copy, or paraphrase" in prompt

    def test_developer_node_whole_file_empty_snippet_patch_applies_successfully(self, tmp_path, monkeypatch):
        """
        Proves the required behavior itself: a FilePatch that follows the
        new instruction - original_code_snippet="" against a
        [COMPLETE FILE CONTENT] file - is accepted by the existing,
        unmodified full-file-write convention in SafePatcher.apply_patch
        and fully replaces the file's content, without ever needing a
        byte-exact snippet match.
        """
        import os
        from pathlib import Path
        from backend.developer.models import FilePatch
        from backend.graph.nodes import developer_node
        from backend.schemas.developer import DeveloperResult
        from backend.schemas.planning import ExecutionPlan

        project_id = "whole_file_empty_snippet_proj"
        # Namespaced under "default-org" to match state.get("organization_id",
        # "default-org") - the resolver P0-3 introduced (resolve_workspace_path).
        workspace_dir = tmp_path / "workspace" / "default-org" / project_id
        workspace_dir.mkdir(parents=True)
        original_content = "# agentic-ai-test-repo"
        (workspace_dir / "README.md").write_text(original_content, encoding="utf-8")

        assert not (Path("workspace") / "default-org" / project_id).exists()
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary="x", changes=[], requires_testing=True, notes=[]
            ),
        )
        monkeypatch.setattr("backend.services.llm.get_llm", lambda: object())

        updated_content = "# agentic-ai-test-repo\n\n## E2E Test\n\nDescription.\n"

        class _FakePatchResult:
            patches = [
                FilePatch(
                    file_path="README.md",
                    original_code_snippet="",
                    updated_code_snippet=updated_content,
                    explanation="Add E2E Test section (whole-file replacement)",
                )
            ]

        monkeypatch.setattr(
            "backend.graph.nodes.invoke_structured",
            lambda llm, schema, prompt: _FakePatchResult(),
        )

        state: AgentState = {
            "user_message": "Add an E2E Test section to README.md",
            "project_id": project_id,
            "plan": ExecutionPlan(goal="Add E2E Test section", steps=[], success_criteria="Done"),
        }

        out = developer_node(state)

        assert len(out["generated_patches"]) == 1
        assert out["generated_patches"][0].original_code_snippet == ""
        assert (workspace_dir / "README.md").read_text(encoding="utf-8") == updated_content

    # ------------------------------------------------------------------
    # FIX 2 regression tests: developer_node's no-repo-context fallback
    # write path must never silently drop an effective DeveloperResult
    # change - it must either become a FilePatch or the run must fail
    # explicitly (ValueError), never a silent generated_patches=[].
    # ------------------------------------------------------------------

    def _empty_workspace_state(self, tmp_path, monkeypatch, project_id="fallback_write_proj"):
        """An existing-but-empty workspace directory, so developer_node's
        self-scan finds zero chunks (repo_context stays falsy) and the
        exact-snippet LLM-patch branch is skipped entirely - isolating the
        no-repo-context fallback write path under test."""
        import os
        from pathlib import Path

        # Namespaced under "default-org" to match state.get("organization_id",
        # "default-org") - the resolver P0-3 introduced (resolve_workspace_path).
        workspace_dir = tmp_path / "workspace" / "default-org" / project_id
        workspace_dir.mkdir(parents=True)
        assert not (Path("workspace") / "default-org" / project_id).exists()
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
        return workspace_dir

    def test_developer_node_fallback_creates_missing_parent_directory(self, tmp_path, monkeypatch):
        from backend.graph.nodes import developer_node
        from backend.schemas.developer import DeveloperResult, FileChange
        from backend.schemas.planning import ExecutionPlan

        workspace_dir = self._empty_workspace_state(tmp_path, monkeypatch)
        assert not (workspace_dir / "docs" / "sub").exists()

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary="add notes",
                changes=[
                    FileChange(
                        file_path="docs/sub/NOTES.md",
                        change_type="CREATE",
                        content="# Notes\n",
                        reason="New doc",
                    )
                ],
                requires_testing=False,
                notes=[],
            ),
        )

        state: AgentState = {
            "user_message": "Add notes",
            "project_id": "fallback_write_proj",
            "plan": ExecutionPlan(goal="Add notes", steps=[], success_criteria="Done"),
        }

        out = developer_node(state)

        assert len(out["generated_patches"]) == 1
        written = workspace_dir / "docs" / "sub" / "NOTES.md"
        assert written.exists()
        assert written.read_text(encoding="utf-8") == "# Notes\n"

    def test_developer_node_fallback_valid_nested_path_writes_file(self, tmp_path, monkeypatch):
        from backend.graph.nodes import developer_node
        from backend.schemas.developer import DeveloperResult, FileChange
        from backend.schemas.planning import ExecutionPlan

        workspace_dir = self._empty_workspace_state(tmp_path, monkeypatch)

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary="add module",
                changes=[
                    FileChange(
                        file_path="src/pkg/module.py",
                        change_type="CREATE",
                        content="x = 1\n",
                        reason="New module",
                    )
                ],
                requires_testing=False,
                notes=[],
            ),
        )

        state: AgentState = {
            "user_message": "Add module",
            "project_id": "fallback_write_proj",
            "plan": ExecutionPlan(goal="Add module", steps=[], success_criteria="Done"),
        }

        out = developer_node(state)

        assert len(out["generated_patches"]) == 1
        patch = out["generated_patches"][0]
        assert patch.file_path == "src/pkg/module.py"
        assert patch.updated_code_snippet == "x = 1\n"
        assert (workspace_dir / "src" / "pkg" / "module.py").read_text(encoding="utf-8") == "x = 1\n"

    @pytest.mark.parametrize(
        "unsafe_path",
        [
            "../outside.py",
            "../../etc/passwd",
            "/etc/passwd",
            "C:\\Windows\\evil.txt",
        ],
    )
    def test_developer_node_fallback_rejects_unsafe_path(self, tmp_path, monkeypatch, unsafe_path):
        from backend.graph.nodes import developer_node
        from backend.schemas.developer import DeveloperResult, FileChange
        from backend.schemas.planning import ExecutionPlan

        self._empty_workspace_state(tmp_path, monkeypatch)

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary="malicious change",
                changes=[
                    FileChange(
                        file_path=unsafe_path,
                        change_type="CREATE",
                        content="pwned\n",
                        reason="unsafe",
                    )
                ],
                requires_testing=False,
                notes=[],
            ),
        )

        state: AgentState = {
            "user_message": "Do something",
            "project_id": "fallback_write_proj",
            "plan": ExecutionPlan(goal="Do something", steps=[], success_criteria="Done"),
        }

        # The invariant: an unsafe path must never be silently skipped
        # (generated_patches=[]) - it must fail the run explicitly.
        with pytest.raises(ValueError, match="not a safe repository-relative path"):
            developer_node(state)

        # Nothing was ever written outside the workspace.
        assert not (tmp_path / "etc" / "passwd").exists()
        assert not (tmp_path / "outside.py").exists()

    def test_developer_node_fallback_non_empty_change_produces_filepatch(self, tmp_path, monkeypatch):
        """The core invariant's happy path: an effective (non-empty-content)
        DeveloperResult change always becomes a FilePatch, never silently
        vanishes - the bug behind the read-only investigation's
        WAITING_APPROVAL/0-diff finding."""
        from backend.graph.nodes import developer_node
        from backend.schemas.developer import DeveloperResult, FileChange
        from backend.schemas.planning import ExecutionPlan

        workspace_dir = self._empty_workspace_state(tmp_path, monkeypatch)

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: DeveloperResult(
                summary="update readme",
                changes=[
                    FileChange(
                        file_path="README.md",
                        change_type="MODIFY",
                        content="# Project\n\n## E2E Test\n\nDescription.\n",
                        reason="Add section",
                    )
                ],
                requires_testing=False,
                notes=[],
            ),
        )

        state: AgentState = {
            "user_message": "Update README",
            "project_id": "fallback_write_proj",
            "plan": ExecutionPlan(goal="Update README", steps=[], success_criteria="Done"),
        }

        out = developer_node(state)

        assert len(out["generated_patches"]) == 1
        assert out["generated_patches"][0].file_path == "README.md"
        assert (workspace_dir / "README.md").exists()

    def test_revision_node_handles_missing_plan_safely(self, monkeypatch):
        from backend.graph.nodes import revision_node
        from backend.schemas.developer import DeveloperResult
        from backend.schemas.planning import ExecutionPlan

        mock_revised = DeveloperResult(
            summary="Revised implementation",
            changes=[],
            requires_testing=True,
            notes=[],
        )

        received_plan = []

        def mock_revise_code_changes(user_request, plan, previous_result, qa_result):
            assert isinstance(plan, ExecutionPlan)
            received_plan.append(plan)
            return mock_revised

        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            mock_revise_code_changes,
        )

        state: AgentState = {
            "user_message": "Fix broken query",
            # Note: "plan" is completely omitted
            "developer_result": DeveloperResult(
                summary="Original",
                changes=[],
                requires_testing=True,
            ),
            "qa_result": QAResult(
                status="FAIL",
                issues=[],
                test_cases=[],
                summary="QA failed.",
            ),
            "revision_count": 0,
        }

        output = revision_node(state)
        assert output["developer_result"] == mock_revised
        assert output["revision_count"] == 1
        assert "plan" in output
        assert isinstance(output["plan"], ExecutionPlan)
        assert len(received_plan) == 1
        assert received_plan[0].goal == "Fix broken query"

    def test_unplanned_task_entering_full_revision_flow(self, monkeypatch):
        from backend.graph.nodes import developer_node, qa_node, qa_router, revision_node
        from backend.schemas.developer import DeveloperResult
        from backend.schemas.planning import ExecutionPlan
        from backend.schemas.qa import QAResult, QAIssue

        initial_dev_result = DeveloperResult(
            summary="Initial fix",
            changes=[],
            requires_testing=True,
            notes=[],
        )
        revised_dev_result = DeveloperResult(
            summary="Revised fix",
            changes=[],
            requires_testing=True,
            notes=[],
        )
        failing_qa_result = QAResult(
            status="FAIL",
            issues=[QAIssue(file_path="main.py", issue="Syntax error", severity="HIGH")],
            test_cases=[],
            summary="QA failed on syntax.",
        )
        passing_qa_result = QAResult(
            status="PASS",
            issues=[],
            test_cases=["test_main"],
            summary="QA passed.",
        )

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: initial_dev_result,
        )
        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda user_request, plan, previous_result, qa_result: revised_dev_result,
        )

        qa_calls = [failing_qa_result, passing_qa_result]
        monkeypatch.setattr(
            "backend.graph.nodes.review_code_changes",
            lambda user_request, plan, developer_result: qa_calls.pop(0),
        )

        # 1. Unplanned task starts with no 'plan' in state
        state: AgentState = {
            "user_message": "Fix broken query in database helper",
        }

        # 2. Developer node generates initial code and persists fallback plan
        dev_out = developer_node(state)
        state.update(dev_out)
        assert "plan" in state
        assert isinstance(state["plan"], ExecutionPlan)
        assert state["developer_result"] == initial_dev_result

        # 3. QA node reviews code changes
        qa_out = qa_node(state)
        state.update(qa_out)
        assert state["qa_result"].status == "FAIL"

        # 4. QA router evaluates status and routes to revision
        next_route = qa_router(state)
        assert next_route == "fail"

        # 5. Revision node safely consumes state (with fallback safety) and updates result
        rev_out = revision_node(state)
        state.update(rev_out)
        assert state["developer_result"] == revised_dev_result
        assert state["revision_count"] == 1
        assert state["plan"].goal == "Fix broken query in database helper"

        # 6. Subsequent QA pass
        qa_pass_out = qa_node(state)
        state.update(qa_pass_out)
        assert state["qa_result"].status == "PASS"

        # 7. QA router routes to approval
        final_route = qa_router(state)
        assert final_route == "pass"


