from typing import List, Optional
from pydantic import BaseModel, Field


class ScannedFile(BaseModel):
    """
    Represents metadata of a scanned file.
    """
    relative_path: str
    absolute_path: str
    extension: str
    size_bytes: int


class CodeChunk(BaseModel):
    """
    Represents a code chunk (e.g., class, function, async function, or fallback block).
    """
    file_path: str
    chunk_type: str  # 'class', 'function', 'async_function', or 'fallback'
    symbol_name: Optional[str] = None
    content: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    decorators: List[str] = Field(default_factory=list)

    # Enhanced AST & Repository Metadata
    module: Optional[str] = None
    symbol_type: Optional[str] = None  # 'module', 'class', 'function', 'method', 'test', 'config', 'fallback'
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    parent_symbol: Optional[str] = None
    imports: List[str] = Field(default_factory=list)
    source_hash: Optional[str] = None


class IndexingError(BaseModel):
    """
    Represents an error encountered while indexing a specific file.
    Allows indexing of the rest of the repository to proceed without interruption.
    """
    file_path: str
    error_type: str  # 'syntax_error', 'io_error', 'encoding_error', 'unknown'
    message: str
    line_number: Optional[int] = None


class IndexingResult(BaseModel):
    """
    Aggregated result of repository indexing.
    """
    chunks: List[CodeChunk] = Field(default_factory=list)
    errors: List[IndexingError] = Field(default_factory=list)
    total_files: int = 0
    indexed_files: int = 0
