from pydantic import BaseModel, Field


class FileChange(BaseModel):
    file_path: str = Field(
        description="Relative path of the file that should be changed."
    )

    change_type: str = Field(
        default="MODIFY",
        description="Type of change: CREATE, MODIFY, or DELETE."
    )

    content: str = Field(
        default="",
        description=(
            "Complete proposed content for CREATE/MODIFY. "
            "For DELETE, leave this empty."
        )
    )

    reason: str = Field(
        default="Code update to resolve issue.",
        description="Why this file change is required."
    )



class DeveloperResult(BaseModel):
    summary: str = Field(
        description="Short summary of the implementation."
    )

    changes: list[FileChange] = Field(
        description="Proposed file changes."
    )

    requires_testing: bool = Field(
        description="Whether the proposed changes should be tested."
    )

    notes: list[str] = Field(
        default_factory=list,
        description="Important implementation notes or limitations."
    )