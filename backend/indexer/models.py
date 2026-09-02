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
