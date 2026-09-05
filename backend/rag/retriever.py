from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.rag.indexer import get_embeddings, get_vector_store_path
from backend.schemas.rag import RetrievedDocumentView

VECTOR_STORE_ROOT = Path("vector_store")


def load_project_index(project_id: str, organization_id: Optional[str] = None):
    index_path = get_vector_store_path(project_id, organization_id)

    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector index not found for project: {project_id} (org: {organization_id})"
        )

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


def retrieve_project_context(
    project_id: str,
    query: str,
    k: int = 4,
    organization_id: Optional[str] = None,
) -> List[Document]:
    """
    Standard vector similarity search returning LangChain Documents.
    Maintains 100% backward compatibility with existing callers.
    """
    vector_store = load_project_index(project_id, organization_id)
    return vector_store.similarity_search(query, k=k)


def retrieve_project_context_with_scores(
    project_id: str,
    query: str,
    k: int = 4,
    organization_id: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieves project documents along with their similarity scores.
    """
    vector_store = load_project_index(project_id, organization_id)
    try:
        # returns List[Tuple[Document, float]]
        return vector_store.similarity_search_with_score(query, k=k)
    except Exception:
        docs = vector_store.similarity_search(query, k=k)
        return [(doc, 1.0) for doc in docs]


def retrieve_structured_context(
    project_id: str,
    query: str,
    k: int = 4,
    organization_id: Optional[str] = None,
) -> List[RetrievedDocumentView]:
    """
    Retrieves normalized, metadata-aware documents exposing:
    exact file, symbol, line range, source content, and score.
    """
    results_with_scores = retrieve_project_context_with_scores(
        project_id, query, k=k, organization_id=organization_id
    )
    views: List[RetrievedDocumentView] = []

    for doc, score in results_with_scores:
        meta = doc.metadata or {}
        views.append(
            RetrievedDocumentView(
                file=meta.get("file") or meta.get("source", "unknown"),
                symbol=meta.get("symbol"),
                line_start=meta.get("line_start") or meta.get("start_line", 1),
                line_end=meta.get("line_end") or meta.get("end_line", 1),
                content=doc.page_content,
                score=round(float(score), 4),
                symbol_type=meta.get("symbol_type"),
                source_hash=meta.get("source_hash"),
            )
        )

    return views