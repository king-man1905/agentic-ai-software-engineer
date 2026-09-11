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
        workspace_dir = tmp_path / "workspace" / project_id
        workspace_dir.mkdir(parents=True)
        content = "\n".join(f"## Section {i}\nBody text for section {i}.\n" for i in range(1, 60))
        (workspace_dir / "README.md").write_text(content, encoding="utf-8")

        # Force nodes.py's os.getcwd()-based fallback path to resolve into
        # this tmp workspace instead of the real repo's workspace/.
        assert not (Path("workspace") / project_id).exists()
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


