from typing import List, Optional
from pydantic import BaseModel, Field


class FilePatch(BaseModel):
    """
    Represents a precise patch for a single file.
    """
    file_path: str = Field(description="Relative path of the file to modify.")
    original_code_snippet: str = Field(
        description="The exact original code block in the file that needs replacement."
    )
    updated_code_snippet: str = Field(
        description="The new code block that will replace the original snippet."
    )
    explanation: str = Field(description="Explanation of why this patch is required.")


class PatchValidationResult(BaseModel):
    """
    Represents the result of validating/applying a patch.
    """
    is_valid: bool = Field(description="Whether the patch was applied and is syntactically valid.")
    syntax_errors: List[str] = Field(
        default_factory=list, description="List of syntax error messages/traces."
    )
    applied_content: Optional[str] = Field(
        default=None, description="The complete file content after applying the patch, if valid."
    )
