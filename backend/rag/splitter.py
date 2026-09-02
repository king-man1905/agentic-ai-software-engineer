from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\nclass ",
            "\ndef ",
            "\nasync def ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    # Add useful metadata to every chunk
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks