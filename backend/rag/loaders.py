from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".html",
    ".css",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
}


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".idea",
    ".vscode",
    "dist",
    "build",
}


IGNORED_FILES = {
    ".env",
}


def load_project_files(project_path: str) -> list[Document]:

    project = Path(project_path)

    if not project.exists():
        raise FileNotFoundError(
            f"Project path does not exist: {project_path}"
        )

    if not project.is_dir():
        raise ValueError(
            f"Project path must be a directory: {project_path}"
        )

    documents = []

    for file_path in project.rglob("*"):

        if not file_path.is_file():
            continue

        # Ignore unwanted directories
        if any(
            part in IGNORED_DIRECTORIES
            for part in file_path.parts
        ):
            continue

        # Never load secret files
        if file_path.name in IGNORED_FILES:
            continue

        # Ignore unsupported file types
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(
                encoding="utf-8"
            )
        except (UnicodeDecodeError, OSError):
            continue

        if not content.strip():
            continue

        relative_path = file_path.relative_to(project)

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(relative_path),
                    "file_name": file_path.name,
                    "extension": file_path.suffix.lower(),
                },
            )
        )

    return documents