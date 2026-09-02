import ast
from typing import List
from backend.indexer.models import CodeChunk


def fallback_chunk(
    content: str,
    file_path: str,
    chunk_size_lines: int = 100,
    overlap_lines: int = 10,
) -> List[CodeChunk]:
    """
    Fallback chunker that splits file content into line-based chunks.
    Used for non-Python files, Python files with syntax errors, or
    Python files with no structural AST symbols.
    """
    lines = content.splitlines()
    total_lines = len(lines)

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
            )
        )

        if end == total_lines:
            break

        start += chunk_size_lines - overlap_lines
        if start >= end:  # Prevent infinite loop if overlap >= size
            start = end

    return chunks


def chunk_python_code(content: str, file_path: str) -> List[CodeChunk]:
    """
    Parses Python source code and extracts ClassDef, FunctionDef, and
    AsyncFunctionDef symbols into AST-aware CodeChunks.
    """
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError) as e:
        raise e

    lines = content.splitlines()
    chunks = []

    def traverse(node: ast.AST, current_scope: List[str]):
        is_class = isinstance(node, ast.ClassDef)
        is_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

        if is_class or is_func:
            name = getattr(node, "name", "")
            qualified_name = ".".join(current_scope + [name])

            if is_class:
                chunk_type = "class"
            elif isinstance(node, ast.AsyncFunctionDef):
                chunk_type = "async_function"
            else:
                chunk_type = "function"

            # Determine start line, adjusting for decorators if present
            start_line = node.lineno
            if hasattr(node, "decorator_list") and node.decorator_list:
                start_line = min(dec.lineno for dec in node.decorator_list)

            # Determine end line
            end_line = getattr(node, "end_lineno", node.lineno)

            # Extract content lines
            s_idx = max(0, start_line - 1)
            e_idx = min(len(lines), end_line)
            chunk_content = "\n".join(lines[s_idx:e_idx])

            # Get docstring
            docstring = ast.get_docstring(node)

            # Get decorators
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
                )
            )
            new_scope = current_scope + [name]
        else:
            new_scope = current_scope

        # Traverse children recursively
        for child in ast.iter_child_nodes(node):
            traverse(child, new_scope)

    traverse(tree, [])
    return chunks


def chunk_file(absolute_path: str, relative_path: str) -> List[CodeChunk]:
    """
    Reads a file and chunks it. Routes to the AST chunker for Python files,
    falling back to line-based chunking if there are syntax errors, the file
    is non-Python, or no AST symbols are defined.
    """
    try:
        with open(absolute_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
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
            )
        ]

    # Non-Python files use fallback chunking
    if not absolute_path.lower().endswith(".py"):
        return fallback_chunk(content, relative_path)

    # Python files: attempt AST parsing
    try:
        chunks = chunk_python_code(content, relative_path)
    except (SyntaxError, ValueError):
        # Syntax error fallback
        return fallback_chunk(content, relative_path)

    # Fallback to line-based if no class/function symbols are defined
    if not chunks:
        return fallback_chunk(content, relative_path)

    return chunks
