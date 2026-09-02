from pathlib import Path

from langchain_community.vectorstores import FAISS

from backend.rag.indexer import get_embeddings


VECTOR_STORE_ROOT = Path("vector_store")


def load_project_index(project_id: str):
    index_path = VECTOR_STORE_ROOT / project_id

    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector index not found for project: {project_id}"
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
):
    vector_store = load_project_index(project_id)

    documents = vector_store.similarity_search(
        query,
        k=k,
    )

    return documents