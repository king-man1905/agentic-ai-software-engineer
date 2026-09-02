import pytest
from backend.indexer.models import CodeChunk
from backend.indexer.retriever import (
    SimpleBM25Index,
    HybridRetriever,
    reciprocal_rank_fusion,
)


@pytest.fixture
def sample_chunks():
    """
    Returns a list of sample CodeChunks for indexing and search testing.
    """
    return [
        CodeChunk(
            file_path="math.py",
            chunk_type="class",
            symbol_name="Calculator",
            content="class Calculator:\n    def add(self, a, b):\n        return a + b",
            start_line=1,
            end_line=5,
        ),
        CodeChunk(
            file_path="math.py",
            chunk_type="function",
            symbol_name="Calculator.add",
            content="    def add(self, a, b):\n        return a + b",
            start_line=2,
            end_line=3,
        ),
        CodeChunk(
            file_path="string_utils.py",
            chunk_type="function",
            symbol_name="concatenate_strings",
            content="def concatenate_strings(s1, s2):\n    return s1 + s2",
            start_line=1,
            end_line=2,
        ),
        CodeChunk(
            file_path="helper.py",
            chunk_type="function",
            symbol_name="helper_function",
            content="def helper_function():\n    # performs basic mathematical operations\n    pass",
            start_line=1,
            end_line=3,
        ),
    ]


def test_exact_symbol_name_matching(sample_chunks):
    """
    Verify that query matching exact symbol names ranks them highest.
    """
    # Weight symbol_name heavily
    bm25 = SimpleBM25Index(sample_chunks, symbol_weight=5.0)

    # Search for "Calculator"
    results = bm25.search("Calculator", top_n=2)
    assert len(results) > 0
    # First chunk should be Calculator class because "Calculator" is its exact symbol name
    assert results[0][0].symbol_name == "Calculator"

    # Search for "Calculator.add"
    results = bm25.search("Calculator.add", top_n=2)
    assert len(results) > 0
    assert results[0][0].symbol_name == "Calculator.add"


def test_fallback_content_keyword_search(sample_chunks):
    """
    Verify that queries matching keyword tokens inside the content (but not symbol names)
    are correctly found.
    """
    bm25 = SimpleBM25Index(sample_chunks, symbol_weight=5.0)

    # Search for "concatenate" which is only in the symbol name / content of string_utils.py
    results = bm25.search("concatenate", top_n=2)
    assert len(results) > 0
    assert results[0][0].file_path == "string_utils.py"

    # Search for "mathematical" which is only in helper.py content comment
    results = bm25.search("mathematical", top_n=2)
    assert len(results) > 0
    assert results[0][0].symbol_name == "helper_function"


def test_reciprocal_rank_fusion(sample_chunks):
    """
    Verify that Reciprocal Rank Fusion correctly scores and ranks chunks.
    An item appearing in both list rankings should rise to the top.
    """
    chunk_a = sample_chunks[0]  # Calculator
    chunk_b = sample_chunks[1]  # Calculator.add
    chunk_c = sample_chunks[2]  # concatenate_strings
    chunk_d = sample_chunks[3]  # helper_function

    # Sparse: rank A (1), B (2), C (3)
    # Dense: rank C (1), D (2), A (3)
    sparse_candidates = [chunk_a, chunk_b, chunk_c]
    dense_candidates = [chunk_c, chunk_d, chunk_a]

    # Calculate RRF scores for k=60:
    # A (rank 1 in sparse, 3 in dense):
    #   score = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.016393 + 0.015873 ≈ 0.032266 (hybrid)
    #
    # B (rank 2 in sparse, not in dense):
    #   score = 1/(60+2) = 1/62 ≈ 0.016129 (sparse)
    #
    # C (rank 3 in sparse, 1 in dense):
    #   score = 1/(60+3) + 1/(60+1) = 1/63 + 1/61 ≈ 0.032266 (hybrid)
    #
    # D (not in sparse, rank 2 in dense):
    #   score = 1/(60+2) = 1/62 ≈ 0.016129 (dense)

    retriever = HybridRetriever()
    merged = retriever.retrieve(
        sparse_candidates=sparse_candidates,
        dense_candidates=dense_candidates,
        top_k=4,
        k=60,
    )

    assert len(merged) == 4

    # Top two should be C and A (score ≈ 0.032266)
    top_two = {res.chunk.symbol_name for res in merged[:2]}
    assert "concatenate_strings" in top_two
    assert "Calculator" in top_two
    assert merged[0].retrieval_type == "hybrid"
    assert merged[1].retrieval_type == "hybrid"

    # Bottom two should be B and D (score ≈ 0.016129)
    bottom_two = {res.chunk.symbol_name for res in merged[2:]}
    assert "Calculator.add" in bottom_two
    assert "helper_function" in bottom_two

    # Verify retrieval type for non-hybrid matches
    by_symbol = {res.chunk.symbol_name: res for res in merged}
    assert by_symbol["Calculator.add"].retrieval_type == "sparse"
    assert by_symbol["helper_function"].retrieval_type == "dense"
