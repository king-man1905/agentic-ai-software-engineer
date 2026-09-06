from typing import List, Optional

from langchain_core.documents import Document

from backend.services.llm import get_llm
from backend.observability.telemetry import invoke_structured
from backend.rag.retriever import retrieve_project_context
from backend.schemas.knowledge import KnowledgeAnswer


def answer_from_project(
    project_id: str,
    question: str,
    k: int = 4,
    documents: Optional[List[Document]] = None,
) -> KnowledgeAnswer:
    """
    Answers a question using project context. If `documents` is already
    available - e.g. the caller (knowledge_node) already ran its own dense
    retrieval pass over the same project/query for hybrid search - it is
    reused directly instead of loading the vector index and re-running an
    identical similarity search a second time. Pass None (the default) to
    have this function perform its own retrieval, as before.
    """
    if documents is None:
        documents = retrieve_project_context(
            project_id=project_id,
            query=question,
            k=k,
        )
    else:
        documents = documents[:k]

    if not documents:
        return KnowledgeAnswer(
            answer="No relevant project context was found.",
            sources=[],
            sufficient_context=False,
        )

    context_parts = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "unknown"
        )

        context_parts.append(
            f"FILE: {source}\n"
            f"CONTENT:\n{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    llm = get_llm()

    prompt = f"""
You are the Knowledge Agent of an Agentic AI Software Engineer.

Answer the user's question using ONLY the retrieved project context.

PROJECT CONTEXT:

{context}

USER QUESTION:

{question}

RULES:

1. Do not invent project details.
2. Base the answer only on the provided context.
3. Include only source filenames actually used in the answer.
4. If the context does not contain enough information, set
   sufficient_context to false.
5. If sufficient_context is false, clearly state what information
   is missing.
6. Keep the answer concise and technical.
7. Never claim that a file, function, class, or behavior exists
   unless it appears in the provided context.

Return the structured answer.
"""

    return invoke_structured(llm, KnowledgeAnswer, prompt)