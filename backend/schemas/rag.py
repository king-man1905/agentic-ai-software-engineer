from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.indexer.models import CodeChunk


class RAGStatus(str, Enum):
    """
    Standard lifecycle and failure states for RAG operations.
    """
    RAG_INDEX_SUCCESS = "RAG_INDEX_SUCCESS"
    RAG_INDEX_FAILURE = "RAG_INDEX_FAILURE"
    RAG_PARSE_FAILURE = "RAG_PARSE_FAILURE"
    RAG_RETRIEVAL_SUCCESS = "RAG_RETRIEVAL_SUCCESS"
    RAG_RETRIEVAL_FAILURE = "RAG_RETRIEVAL_FAILURE"
    RAG_INSUFFICIENT_CONTEXT = "RAG_INSUFFICIENT_CONTEXT"
    RAG_REFINEMENT_EXHAUSTED = "RAG_REFINEMENT_EXHAUSTED"


class RetrievalEvaluationStatus(str, Enum):
    """
    Deterministic evaluation verdict of retrieved context sufficiency.
    """
    RELEVANT = "RELEVANT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


class RetrievedDocumentView(BaseModel):
    """
    Normalized developer-facing view of a retrieved document.
    """
    file: str
    symbol: Optional[str] = None
    line_start: int = 1
    line_end: int = 1
    content: str
    score: float = 0.0
    symbol_type: Optional[str] = None
    source_hash: Optional[str] = None


class RetrievalEvaluation(BaseModel):
    """
    Structured outcome of the deterministic retrieval evaluation layer.
    Determines whether retrieved context is sufficient for safe patch generation.
    """
    status: str = Field(
        default="RELEVANT",
        description="Sufficiency verdict: RELEVANT, PARTIAL, or INSUFFICIENT.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Calibrated retrieval confidence score.",
    )
    relevant_documents: List[RetrievedDocumentView] = Field(
        default_factory=list,
        description="Documents deemed relevant to the issue.",
    )
    missing_context: List[str] = Field(
        default_factory=list,
        description="Identified entities, files, or tests that could not be retrieved.",
    )
    reason: str = Field(
        default="",
        description="Reasoning explaining the evaluation verdict.",
    )


class RAGTelemetry(BaseModel):
    """
    Execution and evaluation telemetry captured during RAG query and refinement.
    """
    retrieval_query: str = Field(
        description="The initial user or task retrieval query."
    )
    retrieval_attempt: int = Field(
        default=1,
        description="Retrieval iteration number (1-indexed)."
    )
    documents_retrieved: int = Field(
        default=0,
        description="Total candidates retrieved before filtering."
    )
    scores: List[float] = Field(
        default_factory=list,
        description="Candidate retrieval scores."
    )
    selected_documents: List[str] = Field(
        default_factory=list,
        description="List of selected file paths or symbol names."
    )
    evaluation_status: str = Field(
        default="RELEVANT",
        description="Evaluation status: RELEVANT, PARTIAL, or INSUFFICIENT."
    )
    evaluation_confidence: float = Field(
        default=1.0,
        description="Evaluation confidence score (0.0 to 1.0)."
    )
    query_rewrite: Optional[str] = Field(
        default=None,
        description="Rewritten query if initial retrieval was insufficient."
    )
    missing_context: List[str] = Field(
        default_factory=list,
        description="Missing context items."
    )
    retrieval_duration_seconds: float = Field(
        default=0.0,
        description="Duration of retrieval operation in seconds."
    )
