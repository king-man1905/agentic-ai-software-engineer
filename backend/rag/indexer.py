from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from backend.core.config import NVIDIA_API_KEY
from backend.indexer.ast_chunker import chunk_file
from backend.indexer.scanner import scan_repository
from backend.rag.loaders import load_project_files
from backend.rag.splitter import split_documents
from backend.vcs.workspace_paths import safe_path_component

VECTOR_STORE_ROOT = Path("vector_store")


def get_embeddings():
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=NVIDIA_API_KEY,
    )


def get_vector_store_path(project_id: str, organization_id: str) -> Path:
    """
    Resolves the tenant-namespaced vector-store directory:
    vector_store/<organization_id>/<project_id>.

    organization_id is mandatory - every vector store is tenant-scoped, and
    silently falling back to an unnamespaced vector_store/<project_id> would
    let two tenants collide on the same directory merely by choosing the
    same project_id (the same bug class P0-3 fixed for workspace/). Fails
    closed with ValueError rather than guessing a path.
    """
    org = safe_path_component(organization_id)
    proj = safe_path_component(project_id)
    if org is None or proj is None:
        raise ValueError(
            "get_vector_store_path requires a valid organization_id and "
            f"project_id (got organization_id={organization_id!r}, "
            f"project_id={project_id!r})"
        )
    return VECTOR_STORE_ROOT / org / proj


def build_project_index(
    project_path: str,
    project_id: str,
    organization_id: str,
):
    project = Path(project_path)
    if not project.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")

    # Use AST-aware repository indexing
    try:
        scanned_files = scan_repository(project_path)
    except Exception:
        scanned_files = []

    documents: List[Document] = []

    if scanned_files:
        for sf in scanned_files:
            chunks = chunk_file(sf.absolute_path, sf.relative_path)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.content,
                    metadata={
                        "source": chunk.file_path,
                        "file": chunk.file_path,
                        "file_name": Path(chunk.file_path).name,
                        "extension": Path(chunk.file_path).suffix.lower(),
                        "symbol": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type or chunk.chunk_type,
                        "line_start": chunk.start_line,
                        "line_end": chunk.end_line,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "source_hash": chunk.source_hash,
                        "imports": chunk.imports,
                        "chunk_id": len(documents),
                        "organization_id": organization_id,
                    },
                )
                documents.append(doc)

    # Fallback to load_project_files if scan_repository found nothing
    if not documents:
        raw_docs = load_project_files(project_path)
        if not raw_docs:
            raise ValueError("No supported project files were found.")
        chunks = split_documents(raw_docs)
        if not chunks:
            raise ValueError("No chunks were generated from project files.")
        documents = chunks

    # Generate embeddings + build FAISS
    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    save_path = get_vector_store_path(project_id, organization_id)
    save_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(str(save_path))

    return {
        "project_id": project_id,
        "organization_id": organization_id,
        "files": len(scanned_files) if scanned_files else len(documents),
        "chunks": len(documents),
        "index_path": str(save_path),
    }