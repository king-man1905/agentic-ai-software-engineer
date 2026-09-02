import pytest
from backend.developer.models import FilePatch
from backend.developer.patcher import SafePatcher, validate_python_syntax


def test_clean_patch_replacement():
    """
    Test that SafePatcher cleanly replaces a target snippet in valid Python source code.
    """
    source = """def add_numbers(a, b):
    # This is a comment
    return a + b
"""
    patch = FilePatch(
        file_path="math.py",
        original_code_snippet="return a + b",
        updated_code_snippet="return float(a) + float(b)",
        explanation="Make additions float-safe.",
    )

    res = SafePatcher.apply_patch(source, patch)
    assert res.is_valid is True
    assert len(res.syntax_errors) == 0
    assert "return float(a) + float(b)" in res.applied_content
    assert "return a + b" not in res.applied_content


def test_ast_syntax_validation():
    """
    Test that validate_python_syntax accurately flags Python syntax errors.
    """
    # 1. Valid syntax
    valid_code = "def foo():\n    pass\n"
    is_valid, err = validate_python_syntax(valid_code)
    assert is_valid is True
    assert err is None

    # 2. SyntaxError: missing colon
    invalid_code_1 = "def foo()\n    pass\n"
    is_valid, err = validate_python_syntax(invalid_code_1)
    assert is_valid is False
    assert "SyntaxError" in err

    # 3. IndentationError: bad indent
    invalid_code_2 = "def foo():\npass\n"
    is_valid, err = validate_python_syntax(invalid_code_2)
    assert is_valid is False
    assert "SyntaxError" in err  # Python AST exception wraps IndentationError under SyntaxError


def test_patch_ast_syntax_preflight_rejection():
    """
    Verify that SafePatcher rejects patches that result in invalid Python syntax.
    """
    source = """def get_data():
    return {}
"""
    # Patch introducing missing colon
    bad_patch = FilePatch(
        file_path="service.py",
        original_code_snippet="def get_data():",
        updated_code_snippet="def get_data()",  # missing colon
        explanation="Remove colon (broken syntax).",
    )

    res = SafePatcher.apply_patch(source, bad_patch)
    assert res.is_valid is False
    assert len(res.syntax_errors) > 0
    assert res.applied_content is None
    assert "SyntaxError" in res.syntax_errors[0]


def test_multiline_snippet_replacement():
    """
    Verify replacement of multi-line spans of code.
    """
    source = """class MathModule:
    def add(self, x, y):
        # original add
        return x + y

    def subtract(self, x, y):
        return x - y
"""
    patch = FilePatch(
        file_path="math_module.py",
        original_code_snippet="    def add(self, x, y):\n        # original add\n        return x + y",
        updated_code_snippet="    def add(self, x, y):\n        # updated float-safe add\n        return float(x) + float(y)",
        explanation="Update add method to be float-safe.",
    )

    res = SafePatcher.apply_patch(source, patch)
    assert res.is_valid is True
    assert "updated float-safe add" in res.applied_content
    assert "subtract(self, x, y)" in res.applied_content


def test_edge_cases_and_non_python_files():
    """
    Test edge cases: original snippet not found, and patching non-Python files (should not run AST validation).
    """
    source = "hello world"

    # 1. Original snippet not found
    patch_missing = FilePatch(
        file_path="readme.txt",
        original_code_snippet="missing",
        updated_code_snippet="found",
        explanation="Replace missing snippet.",
    )
    res = SafePatcher.apply_patch(source, patch_missing)
    assert res.is_valid is False
    assert "not found" in res.syntax_errors[0]

    # 2. Non-python file with syntax error in text (should still pass validation since it's not python)
    source_txt = "Some text block"
    patch_txt = FilePatch(
        file_path="readme.txt",
        original_code_snippet="Some text block",
        updated_code_snippet="def my_bad_python_code()\n  print('hello'",  # syntactically invalid Python
        explanation="Introduce text (not Python).",
    )
    res_txt = SafePatcher.apply_patch(source_txt, patch_txt)
    assert res_txt.is_valid is True
    assert res_txt.applied_content == "def my_bad_python_code()\n  print('hello'"
