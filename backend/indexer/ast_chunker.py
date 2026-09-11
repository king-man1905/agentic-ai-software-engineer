import ast
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

from backend.indexer.models import CodeChunk, IndexingError, IndexingResult
from backend.indexer.scanner import scan_repository


def compute_sha256(content: str) -> str:
    """Computes SHA-256 hex digest for a string content."""
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


def path_to_module(file_path: str) -> str:
    """Converts a relative file path to a Python module name."""
    clean_path = file_path.replace("\\", "/").rstrip("/")
    if clean_path.endswith(".py"):
        clean_path = clean_path[:-3]
    if clean_path.endswith("/__init__"):
        clean_path = clean_path[:-9]
    return clean_path.replace("/", ".")


def is_test_path(file_path: str) -> bool:
    """Checks if a file path is a test file."""
    p = file_path.replace("\\", "/").lower()
    return "/tests/" in p or p.startswith("tests/") or p.endswith("_test.py") or "/test_" in p or p.startswith("test_")


def is_config_path(file_path: str) -> bool:
    """Checks if a file path is a configuration file."""
    p = file_path.replace("\\", "/").lower()
    name = Path(p).name
    return name in {"settings.py", "config.py", "constants.py", "conftest.py"} or p.endswith(".yaml") or p.endswith(".yml") or p.endswith(".json")


def extract_imports(tree: ast.AST) -> List[str]:
    """Extracts all import statements from an AST."""
    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    imports.append(f"import {alias.name} as {alias.asname}")
                else:
                    imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = "." * node.level + (node.module or "")
            names = ", ".join(alias.name if not alias.asname else f"{alias.name} as {alias.asname}" for alias in node.names)
            imports.append(f"from {mod} import {names}")
    return imports


def fallback_chunk(
    content: str,
    file_path: str,
    chunk_size_lines: int = 100,
    overlap_lines: int = 10,
    module: Optional[str] = None,
) -> List[CodeChunk]:
    """
    Fallback chunker that splits file content into line-based chunks.
    Used for non-Python files, Python files with syntax errors, or
    Python files with no structural AST symbols.
    """
    lines = content.splitlines()
    total_lines = len(lines)
    resolved_module = module or path_to_module(file_path)
    sym_type = "test" if is_test_path(file_path) else ("config" if is_config_path(file_path) else "fallback")

    if total_lines == 0:
        return [
            CodeChunk(
                file_path=file_path,
                chunk_type="fallback",
                symbol_name=None,
                content="",
                start_line=1,
                end_line=1,
                docstring=None,
                decorators=[],
                module=resolved_module,
                symbol_type=sym_type,
                source_hash=compute_sha256(""),
            )
        ]

    chunks = []
    start = 0
    while start < total_lines:
        end = min(start + chunk_size_lines, total_lines)
        chunk_lines = lines[start:end]
        chunk_content = "\n".join(chunk_lines)

        chunks.append(
            CodeChunk(
                file_path=file_path,
                chunk_type="fallback",
                symbol_name=None,
                content=chunk_content,
                start_line=start + 1,
                end_line=end,
                docstring=None,
                decorators=[],
                module=resolved_module,
                symbol_type=sym_type,
                source_hash=compute_sha256(chunk_content),
            )
        )

        if end == total_lines:
            break

        start += chunk_size_lines - overlap_lines
        if start >= end:
            start = end

    return chunks


def chunk_python_code(content: str, file_path: str) -> List[CodeChunk]:
    """
    Parses Python source code and extracts ClassDef, FunctionDef, and
    AsyncFunctionDef symbols into AST-aware CodeChunks with rich metadata.
    """
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError) as e:
        raise e

    lines = content.splitlines()
    chunks = []
    mod_name = path_to_module(file_path)
    file_imports = extract_imports(tree)
    is_test_file = is_test_path(file_path)

    def traverse(node: ast.AST, current_scope: List[str], current_class: Optional[str] = None):
        is_class = isinstance(node, ast.ClassDef)
        is_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

        if is_class or is_func:
            name = getattr(node, "name", "")
            qualified_name = ".".join(current_scope + [name])
            parent_symbol = ".".join(current_scope) if current_scope else None

            if is_class:
                chunk_type = "class"
                symbol_type = "test" if (is_test_file or name.startswith("Test")) else "class"
                class_name = name
                func_name = None
            elif isinstance(node, ast.AsyncFunctionDef):
                chunk_type = "async_function"
                if is_test_file or name.startswith("test_"):
                    symbol_type = "test"
                elif current_class:
                    symbol_type = "method"
                else:
                    symbol_type = "function"
                class_name = current_class
                func_name = name
            else:
                chunk_type = "function"
                if is_test_file or name.startswith("test_"):
                    symbol_type = "test"
                elif current_class:
                    symbol_type = "method"
                else:
                    symbol_type = "function"
                class_name = current_class
                func_name = name

            # Determine start line, adjusting for decorators
            start_line = node.lineno
            if hasattr(node, "decorator_list") and node.decorator_list:
                start_line = min(dec.lineno for dec in node.decorator_list)

            end_line = getattr(node, "end_lineno", node.lineno)
            s_idx = max(0, start_line - 1)
            e_idx = min(len(lines), end_line)
            chunk_content = "\n".join(lines[s_idx:e_idx])

            docstring = ast.get_docstring(node)

            decorators = []
            if hasattr(node, "decorator_list"):
                for dec in node.decorator_list:
                    try:
                        decorators.append(ast.unparse(dec))
                    except Exception:
                        decorators.append(ast.dump(dec))

            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    chunk_type=chunk_type,
                    symbol_name=qualified_name,
                    content=chunk_content,
                    start_line=start_line,
                    end_line=end_line,
                    docstring=docstring,
                    decorators=decorators,
                    module=mod_name,
                    symbol_type=symbol_type,
                    class_name=class_name,
                    function_name=func_name,
                    parent_symbol=parent_symbol,
                    imports=file_imports,
                    source_hash=compute_sha256(chunk_content),
                )
            )

            new_scope = current_scope + [name]
            next_class = name if is_class else current_class
        else:
            new_scope = current_scope
            next_class = current_class

        for child in ast.iter_child_nodes(node):
            traverse(child, new_scope, next_class)

    traverse(tree, [])
    return chunks


def chunk_file_with_error(
    absolute_path: str, relative_path: str
) -> Tuple[List[CodeChunk], Optional[IndexingError]]:
    """
    Reads a file and chunks it. If a syntax or reading error occurs, returns fallback
    chunks along with a structured IndexingError so other files continue indexing safely.
    """
    try:
        with open(absolute_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        err = IndexingError(
            file_path=relative_path,
            error_type="io_error",
            message=f"Error reading file: {e}",
        )
        return [
            CodeChunk(
                file_path=relative_path,
                chunk_type="fallback",
                symbol_name=None,
                content=f"Error reading file: {str(e)}",
                start_line=1,
                end_line=1,
                docstring=None,
                decorators=[],
                module=path_to_module(relative_path),
                symbol_type="fallback",
                source_hash=compute_sha256(str(e)),
            )
        ], err

    # Non-Python files use fallback chunking
    if not absolute_path.lower().endswith(".py"):
        return fallback_chunk(content, relative_path), None

    # Python files: attempt AST parsing
    try:
        chunks = chunk_python_code(content, relative_path)
        if not chunks:
            return fallback_chunk(content, relative_path), None
        return chunks, None
    except (SyntaxError, ValueError) as e:
        err = IndexingError(
            file_path=relative_path,
            error_type="syntax_error",
            message=f"Syntax error during AST parse: {e}",
            line_number=getattr(e, "lineno", None),
        )
        return fallback_chunk(content, relative_path), err


def chunk_file(absolute_path: str, relative_path: str) -> List[CodeChunk]:
    """
    Backward-compatible wrapper around chunk_file_with_error.
    """
    chunks, _ = chunk_file_with_error(absolute_path, relative_path)
    return chunks


# Files at or under this size are small enough that giving the developer's
# exact-original-snippet patch-generation prompt one contiguous, verbatim
# view is safe for the LLM's context budget - roughly a few thousand
# tokens, well inside any current model's window - and removes the
# chunk-boundary/overlap-seam ambiguity fallback_chunk() intentionally
# accepts for its actual job (RAG-retrieval-granularity chunking). This is
# NOT a general "load anything" escape hatch: anything larger still goes
# through the existing bounded fallback_chunk() path unchanged.
WHOLE_FILE_CONTEXT_MAX_CHARS = 20_000


def whole_file_chunk_for_patch_context(
    absolute_path: str, relative_path: str
) -> Optional[CodeChunk]:
    """
    Returns a single CodeChunk holding a non-Python file's complete,
    verbatim, on-disk content - read directly from the workspace file, not
    reconstructed from fragments - for use only when assembling context for
    the developer's exact-original-snippet patch-generation prompt.

    Returns None (caller should fall back to chunk_file()/fallback_chunk())
    when the file is Python, missing, unreadable, or larger than
    WHOLE_FILE_CONTEXT_MAX_CHARS. Never used for RAG/indexing.
    """
    if absolute_path.lower().endswith(".py"):
        return None
    try:
        with open(absolute_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None
    if len(content) > WHOLE_FILE_CONTEXT_MAX_CHARS:
        return None

    total_lines = max(len(content.splitlines()), 1)
    return CodeChunk(
        file_path=relative_path,
        chunk_type="fallback",
        symbol_name=None,
        content=content,
        start_line=1,
        end_line=total_lines,
        docstring=None,
        decorators=[],
        module=path_to_module(relative_path),
        symbol_type="test" if is_test_path(relative_path) else ("config" if is_config_path(relative_path) else "fallback"),
        source_hash=compute_sha256(content),
    )


def build_patch_context_chunks(chunks: List[CodeChunk], project_root: str) -> List[CodeChunk]:
    """
    Prepares context specifically for the developer's exact-snippet
    patch-generation prompt: for each small, non-Python file represented in
    `chunks` (regardless of how many fragments of it are present), replaces
    those fragments with one chunk holding the file's complete content from
    whole_file_chunk_for_patch_context(), so the LLM is given a single
    authoritative view of the file it must quote an exact substring from
    instead of overlapping 100-line fallback chunks.

    Python files, and non-Python files that are missing/unreadable/larger
    than the threshold, pass through with their original chunks unchanged.
    This never touches fallback_chunk()/chunk_file() themselves, so
    RAG/indexing and retrieval keep their existing chunking behavior -
    this function only reshapes the list handed to this one prompt.
    """
    root = Path(project_root)
    whole_cache: dict = {}
    emitted_whole: set = set()
    resolved: List[CodeChunk] = []

    for chunk in chunks:
        fp = chunk.file_path
        if fp.lower().endswith(".py"):
            resolved.append(chunk)
            continue

        if fp not in whole_cache:
            whole_cache[fp] = whole_file_chunk_for_patch_context(str(root / fp), fp)
        whole = whole_cache[fp]

        if whole is None:
            resolved.append(chunk)
            continue

        if fp not in emitted_whole:
            resolved.append(whole)
            emitted_whole.add(fp)
        # else: a later fragment of a file already consolidated above - drop it.

    return resolved


def index_repository(
    project_path: str, max_size_kb: int = 500
) -> IndexingResult:
    """
    Scans and indexes an entire repository directory with AST symbol extraction
    and resilient error logging.
    """
    scanned_files = scan_repository(project_path, max_size_kb=max_size_kb)
    all_chunks: List[CodeChunk] = []
    errors: List[IndexingError] = []

    for sf in scanned_files:
        chunks, err = chunk_file_with_error(sf.absolute_path, sf.relative_path)
        all_chunks.extend(chunks)
        if err:
            errors.append(err)

    return IndexingResult(
        chunks=all_chunks,
        errors=errors,
        total_files=len(scanned_files),
        indexed_files=len(scanned_files) - len([e for e in errors if e.error_type == "io_error"]),
    )
