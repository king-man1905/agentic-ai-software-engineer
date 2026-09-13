import ast
from typing import Tuple, Optional
from backend.developer.models import FilePatch, PatchValidationResult


def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validates Python source code syntax using Python's AST parser.
    Returns (True, None) if syntax is correct, or (False, error_message)
    if there are syntax or indentation errors.
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        error_msg = f"SyntaxError: {e.msg} (line {e.lineno}, column {e.offset})"
        if e.text:
            error_msg += f"\nCode line: {e.text.strip()}"
        return False, error_msg
    except Exception as e:
        return False, f"Error: {str(e)}"


class SafePatcher:
    """
    Safely applies patches to code strings and validates the output syntax for Python files.
    """

    @staticmethod
    def apply_patch(source_content: str, patch: FilePatch) -> PatchValidationResult:
        orig = patch.original_code_snippet
        upd = patch.updated_code_snippet

        # An empty original_code_snippet is the full-file-write convention
        # (developer_node's no-repo-context fallback path constructs
        # patches this way): updated_code_snippet IS the file's complete
        # intended content, regardless of whether source_content happens
        # to already be empty (a genuinely new file) or already match it
        # (e.g. that same node writes the file directly to disk in the
        # same call that builds this FilePatch, so a later re-validation -
        # qa/pipeline.py's check_ast - sees non-empty source_content for
        # what is still a full-file replacement, not a snippet to find).
        # Exact-match validation for a real snippet patch (non-empty orig)
        # below is completely unchanged.
        if not orig:
            applied = upd
        else:
            # Check if the target snippet is present in the source code
            if orig not in source_content:
                return PatchValidationResult(
                    is_valid=False,
                    syntax_errors=[
                        f"Target original snippet not found in source file: {patch.file_path}"
                    ],
                    applied_content=None,
                )

            # Replace the first occurrence cleanly
            applied = source_content.replace(orig, upd, 1)

        # Validate syntax if the target file is a Python file
        if patch.file_path.lower().endswith(".py"):
            is_valid, error_msg = validate_python_syntax(applied)
            if not is_valid:
                return PatchValidationResult(
                    is_valid=False,
                    syntax_errors=[error_msg] if error_msg else ["Unknown syntax error"],
                    applied_content=None,
                )

        return PatchValidationResult(
            is_valid=True,
            syntax_errors=[],
            applied_content=applied,
        )
