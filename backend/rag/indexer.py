from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from backend.core.config import NVIDIA_API_KEY
from backend.rag.loaders import load_project_files
from backend.rag.splitter import split_documents


VECTOR_STORE_ROOT = Path("vector_store")


def get_embeddings():
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=NVIDIA_API_KEY,
    )


def build_project_index(
    project_path: str,
    project_id: str,
):
    # Load project files
    documents = load_project_files(project_path)

    if not documents:
        raise ValueError(
            "No supported project files were found."
        )

    # Split into chunks
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks were generated from project files."
        )

    # Generate embeddings + build FAISS
    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    # Each project gets its own index
    save_path = VECTOR_STORE_ROOT / project_id
    save_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(
        str(save_path)
    )

    return {
        "project_id": project_id,
        "files": len(documents),
        "chunks": len(chunks),
        "index_path": str(save_path),
    }