"""
Comprehensive test suite for Phase 3: AST-Aware, Repository-Aware Code RAG & Retrieval Quality Layer.

Covers all 20 requirements from Step 11:
1. AST extraction
2. Function extraction
3. Class extraction
4. Nested method extraction
5. Import extraction
6. Line range metadata
7. Source hash
8. Syntax-error file handling
9. FAISS metadata preservation
10. Relevant retrieval
11. Irrelevant retrieval
12. Retrieval evaluator
13. Insufficient context
14. Query rewriting
15. Maximum retrieval attempts
16. Planner evidence integration
17. Developer structured context
18. RAG telemetry
19. Existing RAG backward compatibility
20. Existing HITL / QA / approval integrity
"""

import ast
import hashlib
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from backend.indexer.ast_chunker import (
    chunk_file_with_error,
    chunk_python_code,
    compute_sha256,
    extract_imports,
)
from backend.indexer.models import CodeChunk
from backend.rag.evaluator import (
    MAX_RETRIEVAL_ATTEMPTS,
    QueryRewriter,
    RetrievalEvaluator,
)
from backend.rag.indexer import build_project_index
from backend.rag.retriever import (
    retrieve_project_context,
    retrieve_structured_context,
)
from backend.schemas.planning import ExecutionPlan, PlanStep
from backend.schemas.rag import (
    RAGTelemetry,
    RetrievalEvaluationStatus,
)


# ---------------------------------------------------------------------------
# 1-7: AST & Metadata Extraction Tests
# ---------------------------------------------------------------------------


def test_ast_extraction():
    """Verify AST extraction processes valid python source without errors."""
    code = "x = 1\ny = 2\n"
    chunks = chunk_python_code(code, "sample.py")
    assert isinstance(chunks, list)


def test_function_extraction():
    """Verify module-level functions are correctly extracted as 'function' symbols."""
    code = """
def calculate_tax(amount, rate):
    \"\"\"Calculates tax on an amount.\"\"\"
    return amount * rate
"""
    chunks = chunk_python_code(code, "billing/tax.py")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.symbol_name == "calculate_tax"
    assert chunk.chunk_type == "function"
    assert chunk.symbol_type == "function"
    assert chunk.function_name == "calculate_tax"
    assert chunk.class_name is None
    assert chunk.parent_symbol is None
    assert "return amount * rate" in chunk.content


def test_class_extraction():
    """Verify classes are correctly extracted with 'class' symbol_type."""
    code = """
class TokenManager:
    \"\"\"Manages user session tokens.\"\"\"
    def __init__(self):
        self.tokens = {}
"""
    chunks = chunk_python_code(code, "auth/token.py")
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) == 1
    cls_chunk = class_chunks[0]
    assert cls_chunk.symbol_name == "TokenManager"
    assert cls_chunk.symbol_type == "class"
    assert cls_chunk.class_name == "TokenManager"
    assert cls_chunk.parent_symbol is None


def test_nested_method_extraction():
    """Verify nested methods within a class retain their class_name and parent_symbol."""
    code = """
class AuthService:
    def validate_token(self, token: str) -> bool:
        return len(token) == 32
"""
    chunks = chunk_python_code(code, "auth/service.py")
    method_chunks = [c for c in chunks if c.symbol_name == "AuthService.validate_token"]
    assert len(method_chunks) == 1
    m = method_chunks[0]
    assert m.chunk_type == "function"
    assert m.symbol_type == "method"
    assert m.class_name == "AuthService"
    assert m.function_name == "validate_token"
    assert m.parent_symbol == "AuthService"
    assert m.module == "auth.service"


def test_import_extraction():
    """Verify standard, aliased, and relative imports are extracted."""
    code = """
import os
import sys as system
from pathlib import Path
from ..utils import helper as h
"""
    tree = ast.parse(code)
    imports = extract_imports(tree)
    assert "import os" in imports
    assert "import sys as system" in imports
    assert "from pathlib import Path" in imports
    assert "from ..utils import helper as h" in imports


def test_line_range_metadata():
    """Verify start_line and end_line accurately reflect code location including decorators."""
    code = """@router.get('/health')
@limiter.limit('100/minute')
def health_check():
    return {'status': 'ok'}
"""
    chunks = chunk_python_code(code, "api/health.py")
    assert len(chunks) == 1
    c = chunks[0]
    assert c.start_line == 1
    assert c.end_line == 4
    assert len(c.decorators) == 2


def test_source_hash():
    """Verify SHA-256 source hash is computed and matches content fingerprint."""
    content = "def sample(): pass"
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert compute_sha256(content) == expected_hash

    chunks = chunk_python_code(content, "test.py")
    assert len(chunks) == 1
    assert chunks[0].source_hash == expected_hash


# ---------------------------------------------------------------------------
# 8-9: Resilience & FAISS Metadata Preservation
# ---------------------------------------------------------------------------


def test_syntax_error_file_handling(tmp_path):
    """Verify syntax-error files fallback to line chunks and record an IndexingError."""
    broken_file = tmp_path / "broken.py"
    broken_file.write_text("class Broken:\n  def oops(\n", encoding="utf-8")

    chunks, err = chunk_file_with_error(str(broken_file), "broken.py")
    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "fallback"
    assert err is not None
    assert err.error_type == "syntax_error"
    assert err.file_path == "broken.py"


def test_faiss_metadata_preservation(tmp_path):
    """Verify FAISS index retains file, symbol, line numbers, and score in retrieved views."""
    src_file = tmp_path / "service.py"
    src_file.write_text(
        "class OrderService:\n"
        "    def create_order(self, item, qty):\n"
        "        return {'item': item, 'qty': qty}\n",
        encoding="utf-8",
    )

    # Use FakeEmbeddings to avoid external API calls during vector test
    mock_embeddings = FakeEmbeddings(size=128)

    with patch("backend.rag.indexer.get_embeddings", return_value=mock_embeddings), \
         patch("backend.rag.retriever.get_embeddings", return_value=mock_embeddings):
        res = build_project_index(str(tmp_path), "test_proj_meta")
        assert res["chunks"] >= 1

        structured_views = retrieve_structured_context("test_proj_meta", "create_order", k=2)
        assert len(structured_views) > 0
        v = structured_views[0]
        assert v.file == "service.py"
        assert v.line_start >= 1
        assert v.line_end >= v.line_start
        assert isinstance(v.score, float)


# ---------------------------------------------------------------------------
# 10-15: Retrieval Evaluation, Rewriting & Bounded Iteration
# ---------------------------------------------------------------------------


def test_relevant_retrieval():
    """Verify exact target file and symbol retrieval is classified as RELEVANT."""
    chunks = [
        CodeChunk(
            file_path="backend/auth/service.py",
            chunk_type="function",
            symbol_name="AuthService.validate_token",
            content="def validate_token(self, token): return True",
            start_line=10,
            end_line=20,
        )
    ]
    issue = "Fix token validation in backend/auth/service.py in AuthService.validate_token"
    evaluation = RetrievalEvaluator.evaluate(issue, "validate_token", chunks, scores=[0.95])

    assert evaluation.status == RetrievalEvaluationStatus.RELEVANT.value
    assert evaluation.confidence >= 0.85
    assert len(evaluation.relevant_documents) == 1
    assert len(evaluation.missing_context) == 0


def test_irrelevant_retrieval():
    """Verify unrelated retrieval is classified as INSUFFICIENT."""
    chunks = [
        CodeChunk(
            file_path="backend/ui/styles.css",
            chunk_type="fallback",
            symbol_name=None,
            content="body { color: red; }",
            start_line=1,
            end_line=5,
        )
    ]
    issue = "Fix database query in backend/database/connection.py with ConnectionPool"
    evaluation = RetrievalEvaluator.evaluate(issue, "ConnectionPool", chunks, scores=[0.1])

    assert evaluation.status == RetrievalEvaluationStatus.INSUFFICIENT.value
    assert evaluation.confidence <= 0.35
    assert len(evaluation.missing_context) > 0


def test_retrieval_evaluator_partial():
    """Verify partial matching yields PARTIAL status with calibrated confidence."""
    chunks = [
        CodeChunk(
            file_path="backend/database/connection.py",
            chunk_type="function",
            symbol_name="connect",
            content="def connect(): pass",
            start_line=1,
            end_line=2,
        )
    ]
    # Issue references connection.py AND ConnectionPool class
    issue = "Update backend/database/connection.py to implement ConnectionPool manager"
    evaluation = RetrievalEvaluator.evaluate(issue, "database connection", chunks, scores=[0.7])

    assert evaluation.status == RetrievalEvaluationStatus.PARTIAL.value
    assert 0.5 <= evaluation.confidence <= 0.85


def test_insufficient_context_empty_results():
    """Verify empty candidate list yields INSUFFICIENT status with 0.0 confidence."""
    evaluation = RetrievalEvaluator.evaluate("Fix critical bug in payment processing", "payment", [])
    assert evaluation.status == RetrievalEvaluationStatus.INSUFFICIENT.value
    assert evaluation.confidence == 0.0


def test_query_rewriting():
    """Verify QueryRewriter incorporates missing symbols and entity names."""
    issue = "Issue with TokenService.refresh_session raising ExpiredTokenError in auth_controller.py"
    current_query = "refresh session"

    rewritten = QueryRewriter.rewrite(issue, current_query, missing_context=["auth_controller.py"], attempt=1)
    assert rewritten is not None
    assert "TokenService" in rewritten or "auth_controller" in rewritten or "ExpiredTokenError" in rewritten


def test_maximum_retrieval_attempts():
    """Verify QueryRewriter halts after reaching MAX_RETRIEVAL_ATTEMPTS."""
    issue = "Some bug"
    rewritten = QueryRewriter.rewrite(issue, "query", attempt=MAX_RETRIEVAL_ATTEMPTS)
    assert rewritten is None  # Enforces bounded execution, never infinite loop


# ---------------------------------------------------------------------------
# 16-18: Planner, Developer Context & Telemetry Integration
# ---------------------------------------------------------------------------


def test_planner_evidence_integration():
    """Verify ExecutionPlan steps can reference concrete files, symbols, and tests."""
    step = PlanStep(
        step_number=1,
        action="Modify token validation logic",
        agent="developer",
        files=["backend/auth/service.py"],
        symbols=["AuthService.validate_token"],
        tests=["tests/test_auth.py"],
    )
    plan = ExecutionPlan(
        goal="Fix token validation",
        steps=[step],
        success_criteria="All auth tests pass",
    )

    assert len(plan.steps) == 1
    assert plan.steps[0].files == ["backend/auth/service.py"]
    assert plan.steps[0].symbols == ["AuthService.validate_token"]
    assert plan.steps[0].tests == ["tests/test_auth.py"]


def test_developer_structured_context():
    """Verify context categorizer separates implementation, tests, and configs."""
    from backend.graph.nodes import format_categorized_context

    chunks = [
        CodeChunk(
            file_path="src/service.py",
            chunk_type="function",
            symbol_name="serve",
            symbol_type="function",
            content="def serve(): pass",
            start_line=1,
            end_line=2,
            imports=["import os", "import sys"],
        ),
        CodeChunk(
            file_path="tests/test_service.py",
            chunk_type="function",
            symbol_name="test_serve",
            symbol_type="test",
            content="def test_serve(): assert True",
            start_line=1,
            end_line=2,
        ),
        CodeChunk(
            file_path="config.py",
            chunk_type="fallback",
            symbol_name=None,
            symbol_type="config",
            content="PORT = 8080",
            start_line=1,
            end_line=1,
        ),
    ]

    formatted = format_categorized_context(chunks)
    assert "[PRIMARY IMPLEMENTATION]" in formatted
    assert "[TESTS]" in formatted
    assert "[DEPENDENCIES]" in formatted
    assert "[CONFIGURATION]" in formatted
    assert "src/service.py" in formatted
    assert "tests/test_service.py" in formatted


def test_rag_telemetry():
    """Verify RAGTelemetry model validates all required telemetry attributes."""
    telemetry = RAGTelemetry(
        retrieval_query="Find database connection helper",
        retrieval_attempt=2,
        documents_retrieved=15,
        scores=[0.92, 0.88, 0.75],
        selected_documents=["db/conn.py", "db/pool.py"],
        evaluation_status="RELEVANT",
        evaluation_confidence=0.91,
        query_rewrite="database ConnectionPool conn",
        missing_context=[],
        retrieval_duration_seconds=0.145,
    )

    assert telemetry.retrieval_attempt == 2
    assert telemetry.evaluation_status == "RELEVANT"
    assert len(telemetry.scores) == 3
    assert telemetry.retrieval_duration_seconds > 0


# ---------------------------------------------------------------------------
# 19-20: Backward Compatibility & HITL/QA Integrity
# ---------------------------------------------------------------------------


def test_existing_rag_backward_compatibility():
    """Verify existing retrieve_project_context returns standard Documents unchanged."""
    mock_store = MagicMock()
    mock_doc = Document(page_content="def legacy(): pass", metadata={"source": "legacy.py"})
    mock_store.similarity_search.return_value = [mock_doc]

    with patch("backend.rag.retriever.load_project_index", return_value=mock_store):
        docs = retrieve_project_context("sample_proj", "legacy search", k=1)
        assert len(docs) == 1
        assert docs[0].page_content == "def legacy(): pass"
        assert docs[0].metadata["source"] == "legacy.py"


def test_existing_hitl_qa_approval_integrity():
    """Verify Phase 1 QA and Phase 2 approval hashing remain intact and functional."""
    from backend.vcs.git_manager import GitWorkspaceManager
    from backend.qa.judge import StructuredQAJudge
    from backend.schemas.qa import QualityCheck, QualityCheckStatus

    # Diff integrity hash
    patch_code = "def secure_fn():\n    return 42\n"
    h1 = GitWorkspaceManager.compute_patch_hash(patch_code)
    h2 = GitWorkspaceManager.compute_patch_hash(patch_code)
    assert h1 == h2
    assert len(h1) == 64

    # QA Judge objective priority
    checks = [
        QualityCheck(name="ast", status=QualityCheckStatus.PASS.value),
        QualityCheck(name="pytest", status=QualityCheckStatus.FAIL.value, stderr_summary="AssertionError"),
    ]
    qa_res = StructuredQAJudge.evaluate(checks)
    assert qa_res.status == "FAIL"
    assert qa_res.failure_category == "TEST_FAILURE"
