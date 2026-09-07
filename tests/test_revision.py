from backend.graph.state import AgentState
from backend.graph.nodes import (
    revision_node,
    qa_router,
    MAX_REVISIONS,
)
from backend.revision.models import (
    RevisionAttempt,
    RevisionHistory,
    ParsedFailure,
    ErrorTraceAnalysis,
)
from backend.revision.analyzer import ErrorTraceAnalyzer, analyze_error_trace
from backend.developer.models import FilePatch
from backend.sandbox.models import TestExecutionResult
from backend.schemas.planning import ExecutionPlan
from backend.schemas.developer import DeveloperResult
from backend.schemas.qa import QAResult, QAIssue

# Prevent pytest from attempting to discover TestExecutionResult as a test case class
TestExecutionResult.__test__ = False


# ============================================================================
# 1. ERROR TRACE ANALYZER TESTS
# ============================================================================

class TestErrorTraceAnalyzer:
    def test_parse_pytest_single_assertion_failure(self):
        trace = """
=================================== FAILURES ===================================
_____________________________ test_addition_failure _____________________________

    def test_addition_failure():
>       assert add(2, 2) == 5
E       assert 4 == 5

tests/test_calculator.py:12: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_calculator.py::test_addition_failure - assert 4 == 5
============================== 1 failed in 0.12s ===============================
"""
        analysis = analyze_error_trace(trace)

        assert len(analysis.failing_tests) >= 1
        assert "tests/test_calculator.py::test_addition_failure" in analysis.failing_tests[0] or "test_addition_failure" in analysis.failing_tests[0]
        assert len(analysis.failures) >= 1

        primary = analysis.failures[0]
        assert "test_addition_failure" in primary.test_name
        assert "test_calculator.py" in primary.test_file
        assert 12 in primary.target_lines
        assert "assert 4 == 5" in primary.error_message
        assert primary.exception_type == "AssertionError"
        assert "test_addition_failure" in analysis.diagnosis

    def test_parse_pytest_multiple_failures(self):
        trace = """
=================================== FAILURES ===================================
_________________________________ test_divide __________________________________

    def test_divide():
>       assert divide(10, 0) == 0

tests/test_math.py:45: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
backend/math_utils.py:10: in divide
    return a / b
E   ZeroDivisionError: division by zero
__________________________ TestUser.test_authenticate __________________________

    def test_authenticate(self):
>       assert auth.login("admin", "wrong") is True
E       AssertionError: assert False is True

tests/test_auth.py:88: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_math.py::test_divide - ZeroDivisionError: division by zero
FAILED tests/test_auth.py::TestUser::test_authenticate - AssertionError: assert False is True
============================== 2 failed in 0.45s ===============================
"""
        analysis = ErrorTraceAnalyzer.analyze(trace)

        assert len(analysis.failing_tests) == 2
        assert len(analysis.failures) == 2

        f1 = analysis.failures[0]
        assert "test_divide" in f1.test_name
        assert "ZeroDivisionError" in (f1.exception_type or "") or "ZeroDivisionError" in f1.error_message
        assert 45 in f1.target_lines or 10 in f1.target_lines

        f2 = analysis.failures[1]
        assert "test_authenticate" in f2.test_name
        assert "AssertionError" in (f2.exception_type or "")
        assert 88 in f2.target_lines

    def test_parse_pytest_collection_error(self):
        trace = """
==================================== ERRORS ====================================
_____________ ERROR collecting tests/test_conditional_routing.py ______________
ImportError while importing test module 'tests/test_conditional_routing.py'.
Traceback:
tests/test_conditional_routing.py:2: in <module>
    from backend.graph.state import AgentState
E   ModuleNotFoundError: No module named 'backend'
=========================== short test summary info ===========================
ERROR tests/test_conditional_routing.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!
"""
        analysis = ErrorTraceAnalyzer.analyze(trace)

        assert len(analysis.failures) >= 1
        f = analysis.failures[0]
        assert "test_conditional_routing.py" in f.test_file
        assert 2 in f.target_lines
        assert "ModuleNotFoundError" in (f.exception_type or "") or "ModuleNotFoundError" in f.error_message

    def test_parse_standard_python_traceback(self):
        trace = """
Traceback (most recent call last):
  File "backend/server.py", line 42, in handle_request
    response = process_payload(data)
  File "backend/processor.py", line 18, in process_payload
    raise ValueError("Invalid payload encoding")
ValueError: Invalid payload encoding
"""
        analysis = ErrorTraceAnalyzer.analyze(trace)

        assert len(analysis.failures) == 1
        f = analysis.failures[0]
        assert f.test_file == "backend/processor.py" or "processor.py" in f.test_file
        assert f.test_name == "process_payload"
        assert 42 in f.target_lines and 18 in f.target_lines
        assert f.exception_type == "ValueError"
        assert "Invalid payload encoding" in f.error_message

    def test_analyze_empty_or_whitespace_trace(self):
        analysis = ErrorTraceAnalyzer.analyze("   \n\t  ")
        assert analysis.failing_tests == []
        assert analysis.failures == []
        assert "No test failures detected" in analysis.diagnosis

    def test_analyze_test_result_none_handled_safely(self):
        analysis = ErrorTraceAnalyzer.analyze_test_result(None)
        assert analysis.failing_tests == []
        assert analysis.failures == []
        assert "No test execution result provided" in analysis.diagnosis

    def test_analyze_test_result_with_error_summary(self):
        result = TestExecutionResult(
            success=False,
            exit_code=1,
            passed_count=5,
            failed_count=1,
            stdout="",
            stderr="",
            duration_seconds=1.2,
            error_summary="""
=================================== FAILURES ===================================
__________________________________ test_json ___________________________________
    def test_json():
>       assert parse("{}") == {"ok": True}
E       AssertionError: assert {} == {'ok': True}
tests/test_parser.py:30: AssertionError
""",
        )
        analysis = ErrorTraceAnalyzer.analyze_test_result(result)
        assert len(analysis.failing_tests) >= 1
        assert "test_json" in analysis.failing_tests[0]


# ============================================================================
# 2. REVISION STATE & TRACKING SCHEMA TESTS
# ============================================================================

class TestRevisionModels:
    def test_revision_attempt_instantiation(self):
        patch = FilePatch(
            file_path="src/main.py",
            original_code_snippet="return 1",
            updated_code_snippet="return 2",
            explanation="Fix return value",
        )
        attempt = RevisionAttempt(
            attempt_number=1,
            failing_tests=["tests/test_main.py::test_return"],
            error_traceback="AssertionError: assert 1 == 2",
            applied_patch=patch,
            diagnosis="Function returned 1 instead of 2.",
        )

        assert attempt.attempt_number == 1
        assert attempt.failing_tests == ["tests/test_main.py::test_return"]
        assert attempt.applied_patch == patch
        assert "Function returned" in attempt.diagnosis

    def test_revision_history_defaults_and_progression(self):
        history = RevisionHistory(max_retries=3)
        assert history.attempts == []
        assert history.max_retries == 3
        assert history.is_exhausted is False
        assert history.total_attempts == 0
        assert history.latest_attempt is None

        # Attempt 1
        att1 = RevisionAttempt(attempt_number=1, failing_tests=["test_1"])
        history.add_attempt(att1)
        assert history.total_attempts == 1
        assert history.is_exhausted is False
        assert history.latest_attempt == att1

        # Attempt 2
        att2 = RevisionAttempt(attempt_number=2, failing_tests=["test_2"])
        history.add_attempt(att2)
        assert history.total_attempts == 2
        assert history.is_exhausted is False

        # Attempt 3 (Reaches max_retries)
        att3 = RevisionAttempt(attempt_number=3, failing_tests=["test_3"])
        history.add_attempt(att3)
        assert history.total_attempts == 3
        assert history.is_exhausted is True
        assert history.latest_attempt == att3

    def test_revision_models_pydantic_serialization(self):
        history = RevisionHistory(
            max_retries=2,
            attempts=[
                RevisionAttempt(
                    attempt_number=1,
                    failing_tests=["tests/test_api.py::test_get"],
                    error_traceback="404 != 200",
                    diagnosis="Route not found",
                )
            ],
            is_exhausted=False,
        )

        json_data = history.model_dump_json()
        assert "tests/test_api.py::test_get" in json_data
        assert "Route not found" in json_data

        restored = RevisionHistory.model_validate_json(json_data)
        assert restored.total_attempts == 1
        assert restored.attempts[0].attempt_number == 1
        assert restored.attempts[0].failing_tests == ["tests/test_api.py::test_get"]


# ============================================================================
# 3. REVISION NODE & STATE PROGRESSION TESTS
# ============================================================================

class TestRevisionNodeProgression:
    def test_revision_node_increments_count_and_creates_history(self, monkeypatch):
        mock_dev_result = DeveloperResult(
            summary="Fixed syntax bug",
            changes=[],
            requires_testing=True,
            notes=["Corrected operator precedence"],
        )

        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda user_request, plan, previous_result, qa_result: mock_dev_result,
        )

        state: AgentState = {
            "user_message": "Fix broken query in database helper",
            "developer_result": DeveloperResult(
                summary="Initial attempt",
                changes=[],
                requires_testing=True,
            ),
            "qa_result": QAResult(
                status="FAIL",
                issues=[QAIssue(file_path="db.py", issue="SyntaxError", severity="HIGH")],
                test_cases=[],
                summary="Pytest failed on syntax error.",
            ),
            "revision_count": 0,
        }

        output = revision_node(state)

        assert output["revision_count"] == 1
        assert output["developer_result"] == mock_dev_result
        assert "revision_history" in output
        assert isinstance(output["revision_history"], RevisionHistory)
        assert output["revision_history"].total_attempts == 1
        assert output["revision_history"].attempts[0].attempt_number == 1
        assert output["revision_history"].is_exhausted is False

    def test_revision_node_with_test_result_traceback(self, monkeypatch):
        mock_dev_result = DeveloperResult(
            summary="Fixed ZeroDivisionError in math utils",
            changes=[],
            requires_testing=True,
        )

        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda user_request, plan, previous_result, qa_result: mock_dev_result,
        )

        test_result = TestExecutionResult(
            success=False,
            exit_code=1,
            passed_count=2,
            failed_count=1,
            stdout="",
            stderr="",
            duration_seconds=0.5,
            error_summary="""
=================================== FAILURES ===================================
_________________________________ test_divide __________________________________
tests/test_math.py:20: in test_divide
    assert divide(10, 0) == 0
E   ZeroDivisionError: division by zero
=========================== short test summary info ===========================
FAILED tests/test_math.py::test_divide - ZeroDivisionError: division by zero
""",
        )

        state: AgentState = {
            "user_message": "Handle zero division safely",
            "developer_result": DeveloperResult(
                summary="Initial",
                changes=[],
                requires_testing=True,
            ),
            "qa_result": QAResult(
                status="FAIL",
                issues=[QAIssue(file_path="tests/test_math.py", issue="ZeroDivisionError", severity="HIGH")],
                test_cases=[],
                summary="Tests failed.",
            ),
            "test_result": test_result,
            "revision_count": 1,
        }

        output = revision_node(state)

        assert output["revision_count"] == 2
        history = output["revision_history"]
        assert history.total_attempts == 1
        latest = history.latest_attempt
        assert latest.attempt_number == 2
        assert any("test_divide" in t for t in latest.failing_tests)
        assert "ZeroDivisionError" in latest.error_traceback

    def test_revision_node_preserves_existing_history(self, monkeypatch):
        mock_dev_result = DeveloperResult(
            summary="Second revision",
            changes=[],
            requires_testing=True,
        )
        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda user_request, plan, previous_result, qa_result: mock_dev_result,
        )

        existing_history = RevisionHistory(
            max_retries=3,
            attempts=[
                RevisionAttempt(
                    attempt_number=1,
                    failing_tests=["test_a"],
                    error_traceback="AssertionError",
                    diagnosis="Attempt 1 failed",
                )
            ],
            is_exhausted=False,
        )

        state: AgentState = {
            "user_message": "Fix login bug",
            "revision_count": 1,
            "revision_history": existing_history,
            "developer_result": DeveloperResult(summary="A1", changes=[], requires_testing=True),
            "qa_result": QAResult(status="FAIL", issues=[], test_cases=[], summary="Still failing"),
        }

        output = revision_node(state)

        assert output["revision_count"] == 2
        history = output["revision_history"]
        assert history.total_attempts == 2
        assert history.attempts[0].attempt_number == 1
        assert history.attempts[1].attempt_number == 2
        assert history.is_exhausted is False


# ============================================================================
# 4. CIRCUIT BREAKER TERMINATION TESTS
# ============================================================================

class TestCircuitBreaker:
    def test_qa_router_routes_to_fail_when_below_max_revisions(self):
        for count in range(MAX_REVISIONS):
            state: AgentState = {
                "qa_result": QAResult(
                    status="FAIL",
                    issues=[QAIssue(file_path="main.py", issue="Bug", severity="HIGH")],
                    test_cases=[],
                    summary="Failed.",
                ),
                "revision_count": count,
            }
            assert qa_router(state) == "fail", f"Expected 'fail' at revision_count={count}"

    def test_qa_router_circuit_breaks_at_max_revisions(self):
        state: AgentState = {
            "qa_result": QAResult(
                status="FAIL",
                issues=[QAIssue(file_path="main.py", issue="Bug", severity="HIGH")],
                test_cases=[],
                summary="Failed after multiple attempts.",
            ),
            "revision_count": MAX_REVISIONS,
        }
        assert qa_router(state) == "max_retries"

    def test_qa_router_circuit_breaks_above_max_revisions(self):
        state: AgentState = {
            "qa_result": QAResult(
                status="FAIL",
                issues=[],
                test_cases=[],
                summary="Failed.",
            ),
            "revision_count": MAX_REVISIONS + 5,
        }
        assert qa_router(state) == "max_retries"

    def test_qa_router_passes_even_at_high_revision_count(self):
        state: AgentState = {
            "qa_result": QAResult(
                status="PASS",
                issues=[],
                test_cases=["test_login"],
                summary="Fixed after revisions.",
            ),
            "revision_count": MAX_REVISIONS,
        }
        assert qa_router(state) == "pass"

    def test_qa_router_none_status_handled_safely(self):
        qa = QAResult.model_construct(
            status=None,
            issues=[],
            test_cases=[],
            summary="No status.",
        )
        state: AgentState = {
            "qa_result": qa,
            "revision_count": MAX_REVISIONS,
        }
        assert qa_router(state) == "max_retries"


# ============================================================================
# 5. EDGE CASES & FALLBACK PERSISTENCE TESTS
# ============================================================================

class TestEdgeCasesAndFallbacks:
    def test_revision_node_handles_omitted_plan_persisting_fallback(self, monkeypatch):
        mock_dev = DeveloperResult(
            summary="Fallback revised",
            changes=[],
            requires_testing=True,
        )
        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda user_request, plan, previous_result, qa_result: mock_dev,
        )

        state: AgentState = {
            "user_message": "Fix unhandled exception in background worker",
            # "plan" is deliberately omitted
            "developer_result": DeveloperResult(summary="Init", changes=[], requires_testing=True),
            "qa_result": QAResult(status="FAIL", issues=[], test_cases=[], summary="QA failed"),
            "revision_count": 0,
        }

        output = revision_node(state)

        assert "plan" in output
        assert isinstance(output["plan"], ExecutionPlan)
        assert output["plan"].goal == "Fix unhandled exception in background worker"
        assert output["revision_count"] == 1

    def test_revision_node_handles_missing_developer_result_safely(self, monkeypatch):
        mock_dev = DeveloperResult(
            summary="Handled missing dev result",
            changes=[],
            requires_testing=True,
        )
        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda user_request, plan, previous_result, qa_result: mock_dev,
        )

        state: AgentState = {
            "user_message": "Fix math division",
            # "developer_result" is omitted
            "qa_result": QAResult(status="FAIL", issues=[], test_cases=[], summary="QA failed"),
            "revision_count": 0,
        }

        output = revision_node(state)
        assert output["developer_result"] == mock_dev
        assert output["revision_count"] == 1

    def test_full_self_correction_cycle_simulation(self, monkeypatch):
        from backend.graph.nodes import developer_node, qa_node

        initial_dev = DeveloperResult(
            summary="Initial buggy code",
            changes=[],
            requires_testing=True,
        )
        revised_dev = DeveloperResult(
            summary="Fixed code in revision",
            changes=[],
            requires_testing=True,
        )

        qa_fail = QAResult(
            status="FAIL",
            issues=[QAIssue(file_path="math.py", issue="AssertionError: 2 != 4", severity="HIGH")],
            test_cases=[],
            summary="Test test_double failed with AssertionError.",
        )
        qa_pass = QAResult(
            status="PASS",
            issues=[],
            test_cases=["test_double"],
            summary="All tests passed successfully.",
        )

        monkeypatch.setattr(
            "backend.graph.nodes.generate_code_changes",
            lambda user_request, plan, knowledge: initial_dev,
        )
        monkeypatch.setattr(
            "backend.agents.developer.revise_code_changes",
            lambda user_request, plan, previous_result, qa_result: revised_dev,
        )

        qa_sequence = [qa_fail, qa_pass]
        monkeypatch.setattr(
            "backend.graph.nodes.review_code_changes",
            lambda user_request, plan, developer_result: qa_sequence.pop(0),
        )

        # 1. Start state
        state: AgentState = {
            "user_message": "Implement double function correctly",
        }

        # 2. Developer Node runs
        dev_out = developer_node(state)
        state.update(dev_out)
        assert state["developer_result"] == initial_dev

        # 3. QA Node runs -> FAILS
        qa_out = qa_node(state)
        state.update(qa_out)
        assert state["qa_result"].status == "FAIL"

        # 4. QA Router routes to 'fail' (triggering revision)
        assert qa_router(state) == "fail"

        # 5. Revision Node executes -> Analyzes error, updates history and revision_count
        rev_out = revision_node(state)
        state.update(rev_out)
        assert state["revision_count"] == 1
        assert state["developer_result"] == revised_dev
        assert state["revision_history"].total_attempts == 1

        # 6. Re-testing via QA Node -> PASSES
        qa_out2 = qa_node(state)
        state.update(qa_out2)
        assert state["qa_result"].status == "PASS"

        # 7. QA Router routes to 'pass' -> Approval
        assert qa_router(state) == "pass"
