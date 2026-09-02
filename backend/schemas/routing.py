from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CODE_GENERATION = "CODE_GENERATION"
    BUG_FIX = "BUG_FIX"
    CODE_REVIEW = "CODE_REVIEW"
    CODE_EXPLANATION = "CODE_EXPLANATION"
    DOCUMENTATION = "DOCUMENTATION"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    KNOWLEDGE_SEARCH = "KNOWLEDGE_SEARCH"
    GENERAL = "GENERAL"


class RoutingDecision(BaseModel):
    task_type: TaskType = Field(
        description="The category of the user's request."
    )

    requires_planning: bool = Field(
        description="Whether the task requires a multi-step plan."
    )

    requires_knowledge: bool = Field(
        description="Whether external or project-specific context is required."
    )

    reasoning: str = Field(
        description="A short explanation of the routing decision."
    )