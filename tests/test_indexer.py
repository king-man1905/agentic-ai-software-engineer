import os
import pytest
from backend.indexer.models import CodeChunk, ScannedFile
from backend.indexer.scanner import scan_repository
from backend.indexer.ast_chunker import (
    build_patch_context_chunks,
    chunk_file,
    chunk_python_code,
    fallback_chunk,
    whole_file_chunk_for_patch_context,
    WHOLE_FILE_CONTEXT_MAX_CHARS,
)
from backend.developer.models import FilePatch
from backend.developer.patcher import SafePatcher


def test_scan_repository(tmp_path):
    """
    Test that repository scanning recursively finds files,
    honors max size limits, and properly excludes ignored directories and secrets.
    """
    # Create valid files
    main_file = tmp_path / "main.py"
    main_file.write_text("def hello(): pass", encoding="utf-8")

    readme_file = tmp_path / "README.md"
    readme_file.write_text("# Readme", encoding="utf-8")

    nested_dir = tmp_path / "src"
    nested_dir.mkdir()
    nested_file = nested_dir / "utils.py"
    nested_file.write_text("def utility(): pass", encoding="utf-8")

    # Create ignored directories and files inside them
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("some git config", encoding="utf-8")

    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "activate").write_text("some activation script", encoding="utf-8")

    pycache_dir = tmp_path / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "main.cpython-39.pyc").write_bytes(b"\x00\x01")

    # Create secrets
    (tmp_path / ".env").write_text("API_KEY=123", encoding="utf-8")
    (tmp_path / ".env.dev").write_text("DB_PASS=xyz", encoding="utf-8")
    (tmp_path / "cert.pem").write_text("PEM_CONTENT", encoding="utf-8")
    (tmp_path / "id_rsa.key").write_text("KEY_CONTENT", encoding="utf-8")
    (tmp_path / "secrets.json").write_text('{"secret": "val"}', encoding="utf-8")

    # Create a large file (> 500 KB)
    large_file = tmp_path / "large.py"
    large_file.write_text("a" * (501 * 1024), encoding="utf-8")  # 501 KB

    # Run scanner
    scanned = scan_repository(str(tmp_path), max_size_kb=500)

    # Convert scanned files list to dictionary by relative path
    scanned_dict = {f.relative_path: f for f in scanned}

    # Verify expected files exist
    assert "main.py" in scanned_dict
    assert "README.md" in scanned_dict
    assert "src/utils.py" in scanned_dict

    # Check extension and size_bytes
    assert scanned_dict["main.py"].extension == ".py"
    assert scanned_dict["main.py"].size_bytes == len("def hello(): pass")
    assert scanned_dict["src/utils.py"].absolute_path == str(nested_file).replace("\\", "/")

    # Verify ignored directories are excluded
    for k in scanned_dict.keys():
        assert not k.startswith(".git")
        assert not k.startswith(".venv")
        assert not k.startswith("__pycache__")

    # Verify secrets are excluded
    assert ".env" not in scanned_dict
    assert ".env.dev" not in scanned_dict
    assert "cert.pem" not in scanned_dict
    assert "id_rsa.key" not in scanned_dict
    assert "secrets.json" not in scanned_dict

    # Verify large file is excluded
    assert "large.py" not in scanned_dict


def test_ast_symbol_extraction():
    """
    Test extraction of functions, async functions, classes, and methods.
    """
    code = """
def normal_func(x, y):
    return x + y

async def async_func():
    await sleep(1)

class MyClass:
    def method_one(self):
        pass

    async def method_two(self):
        pass
"""
    chunks = chunk_python_code(code, "test_file.py")

    # We expect chunks for:
    # 1. normal_func
    # 2. async_func
    # 3. MyClass
    # 4. MyClass.method_one
    # 5. MyClass.method_two
    
    assert len(chunks) == 5

    # Index by symbol name
    chunk_map = {c.symbol_name: c for c in chunks}

    assert "normal_func" in chunk_map
    assert chunk_map["normal_func"].chunk_type == "function"
    assert "return x + y" in chunk_map["normal_func"].content

    assert "async_func" in chunk_map
    assert chunk_map["async_func"].chunk_type == "async_function"
    assert "await sleep(1)" in chunk_map["async_func"].content

    assert "MyClass" in chunk_map
    assert chunk_map["MyClass"].chunk_type == "class"
    assert "class MyClass:" in chunk_map["MyClass"].content
    assert "def method_one(self):" in chunk_map["MyClass"].content

    assert "MyClass.method_one" in chunk_map
    assert chunk_map["MyClass.method_one"].chunk_type == "function"
    assert "def method_one(self):" in chunk_map["MyClass.method_one"].content

    assert "MyClass.method_two" in chunk_map
    assert chunk_map["MyClass.method_two"].chunk_type == "async_function"
    assert "async def method_two(self):" in chunk_map["MyClass.method_two"].content


def test_decorator_and_docstring_retention():
    """
    Test that decorators and docstrings are correctly retained,
    and start_line adjusts for decorators.
    """
    code = """@fixture(scope="module")
@other_dec
def my_func():
    \"\"\"
    This is my docstring.
    \"\"\"
    return 42
"""
    chunks = chunk_python_code(code, "test_file.py")
    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk.symbol_name == "my_func"
    assert chunk.docstring == "This is my docstring."
    assert chunk.decorators == ["fixture(scope='module')", "other_dec"]
    # start_line should be 1 (first decorator), end_line should be 7 (last line)
    assert chunk.start_line == 1
    assert chunk.end_line == 7
    assert "@fixture(scope=\"module\")" in chunk.content
    assert "return 42" in chunk.content


def test_fallback_chunker():
    """
    Test line-based fallback chunker for non-python files or syntax errors.
    """
    # Test line splitting with overlap
    content = "\n".join(f"Line {i}" for i in range(1, 15))
    chunks = fallback_chunk(content, "test.txt", chunk_size_lines=5, overlap_lines=2)

    # 14 lines total.
    # Chunk 1: Lines 1-5 (indices 0-4) -> start_line=1, end_line=5
    # Next start: 5 - 2 = 3 (indices 3-7) -> Lines 4-8 -> start_line=4, end_line=8
    # Next start: 8 - 2 = 6 (indices 6-10) -> Lines 7-11 -> start_line=7, end_line=11
    # Next start: 11 - 2 = 9 (indices 9-13) -> Lines 10-14 -> start_line=10, end_line=14
    
    assert len(chunks) == 4
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 5
    assert chunks[0].content == "\n".join(f"Line {i}" for i in range(1, 6))

    assert chunks[1].start_line == 4
    assert chunks[1].end_line == 8
    assert chunks[1].content == "\n".join(f"Line {i}" for i in range(4, 9))

    assert chunks[2].start_line == 7
    assert chunks[2].end_line == 11

    assert chunks[3].start_line == 10
    assert chunks[3].end_line == 14


def test_chunk_file_routing(tmp_path):
    """
    Test chunk_file high-level routing logic.
    """
    # 1. Non-python file
    txt_file = tmp_path / "hello.txt"
    txt_file.write_text("Hello World\nLine 2", encoding="utf-8")
    chunks = chunk_file(str(txt_file), "hello.txt")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "fallback"
    assert chunks[0].symbol_name is None
    assert chunks[0].content == "Hello World\nLine 2"

    # 2. Python file with syntax error
    err_file = tmp_path / "error.py"
    err_file.write_text("class MyClass:\n  def method(\n    pass", encoding="utf-8")
    chunks = chunk_file(str(err_file), "error.py")
    assert len(chunks) > 0
    assert chunks[0].chunk_type == "fallback"
    assert chunks[0].symbol_name is None

    # 3. Python file with no class or function definitions
    script_file = tmp_path / "script.py"
    script_file.write_text("import os\nprint('Hello')\nx = 1\n", encoding="utf-8")
    chunks = chunk_file(str(script_file), "script.py")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "fallback"
    assert chunks[0].symbol_name is None
    assert "print('Hello')" in chunks[0].content


# ---------------------------------------------------------------------------
# Patch-generation context consolidation (developer's exact-snippet prompt)
# ---------------------------------------------------------------------------


def _readme_content(n_lines: int) -> str:
    return "\n".join(f"## Section {i}\nSome body text for section {i}.\n" for i in range(1, n_lines + 1))


def test_build_patch_context_chunks_readme_becomes_one_contiguous_block(tmp_path):
    """A. A small non-Python file (README.md) under the size threshold is
    consolidated into ONE contiguous chunk instead of fallback_chunk()'s
    overlapping fragments."""
    readme = tmp_path / "README.md"
    content = _readme_content(60)  # well over 100 lines, under the char threshold
    readme.write_text(content, encoding="utf-8")
    assert len(content) < WHOLE_FILE_CONTEXT_MAX_CHARS

    fragmented = chunk_file(str(readme), "README.md")
    assert len(fragmented) > 1  # confirms this file DOES get fragmented today

    consolidated = build_patch_context_chunks(fragmented, str(tmp_path))
    readme_chunks = [c for c in consolidated if c.file_path == "README.md"]
    assert len(readme_chunks) == 1
    assert readme_chunks[0].content == content
    assert readme_chunks[0].start_line == 1


def test_build_patch_context_chunks_output_supports_exact_snippet_match(tmp_path):
    """B. An original_code_snippet drawn from the consolidated block is a
    real, exact substring of the on-disk file, so SafePatcher.apply_patch
    (unmodified) accepts it - proving the fragmentation bug is closed."""
    readme = tmp_path / "README.md"
    content = "# Project\n\n## Existing Section\n\nSome existing body text.\n"
    readme.write_text(content, encoding="utf-8")

    fragmented = chunk_file(str(readme), "README.md")
    consolidated = build_patch_context_chunks(fragmented, str(tmp_path))
    readme_chunk = next(c for c in consolidated if c.file_path == "README.md")

    anchor = "## Existing Section\n\nSome existing body text.\n"
    assert anchor in readme_chunk.content  # the LLM's context really contains this verbatim

    patch = FilePatch(
        file_path="README.md",
        original_code_snippet=anchor,
        updated_code_snippet="## New Section\n\nUpdated body text.\n",
        explanation="Add a new section",
    )
    result = SafePatcher.apply_patch(content, patch)
    assert result.is_valid is True
    assert "## New Section" in result.applied_content


def test_build_patch_context_chunks_large_non_python_file_unchanged(tmp_path):
    """C. A non-Python file at/above the threshold keeps the existing
    bounded fallback_chunk() fragmentation - no whole-file consolidation."""
    big_file = tmp_path / "CHANGELOG.md"
    content = "x" * (WHOLE_FILE_CONTEXT_MAX_CHARS + 1)
    big_file.write_text(content, encoding="utf-8")

    assert whole_file_chunk_for_patch_context(str(big_file), "CHANGELOG.md") is None

    fragmented = chunk_file(str(big_file), "CHANGELOG.md")
    consolidated = build_patch_context_chunks(fragmented, str(tmp_path))
    assert consolidated == fragmented


def test_build_patch_context_chunks_python_files_unchanged(tmp_path):
    """D. Python files retain their existing AST-chunking behavior - never
    whole-filed, chunks passed through identically."""
    py_file = tmp_path / "sample.py"
    py_file.write_text("def a():\n    pass\n\n\ndef b():\n    pass\n", encoding="utf-8")

    assert whole_file_chunk_for_patch_context(str(py_file), "sample.py") is None

    original_chunks = chunk_file(str(py_file), "sample.py")
    consolidated = build_patch_context_chunks(original_chunks, str(tmp_path))
    assert consolidated == original_chunks
    assert all(c.chunk_type in ("function", "fallback") for c in consolidated)


def test_safe_patcher_still_rejects_incorrect_snippet(tmp_path):
    """E. SafePatcher.apply_patch (unmodified) still rejects a mismatched
    anchor - patch pre-flight/anchor validation remains mandatory."""
    content = "# Project\n\nSome existing content.\n"
    patch = FilePatch(
        file_path="README.md",
        original_code_snippet="## Section That Does Not Exist\n",
        updated_code_snippet="## New Section\n",
        explanation="Add a new section",
    )
    result = SafePatcher.apply_patch(content, patch)
    assert result.is_valid is False
    assert "Target original snippet not found" in result.syntax_errors[0]


def test_fallback_chunk_and_chunk_file_behavior_unchanged():
    """F. fallback_chunk()/chunk_file() themselves - used by RAG/indexing
    and retrieval - are untouched: re-run the pre-existing fixture from
    test_fallback_chunker() and confirm identical results."""
    content = "\n".join(f"Line {i}" for i in range(1, 15))
    chunks = fallback_chunk(content, "test.txt", chunk_size_lines=5, overlap_lines=2)
    assert len(chunks) == 4
    assert chunks[0].start_line == 1 and chunks[0].end_line == 5
    assert chunks[1].start_line == 4 and chunks[1].end_line == 8
    assert chunks[2].start_line == 7 and chunks[2].end_line == 11
    assert chunks[3].start_line == 10 and chunks[3].end_line == 14
