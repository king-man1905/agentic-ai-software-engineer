from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_number: int = Field(
        description="Execution order of this step."
    )

    action: str = Field(
        description="Specific action that should be performed."
    )

    agent: str = Field(
        description="Agent responsible for performing this step."
    )

    files: list[str] = Field(
        default_factory=list,
        description="Relevant or target files referenced in this step."
    )

    symbols: list[str] = Field(
        default_factory=list,
        description="Relevant or target symbols referenced in this step."
    )

    tests: list[str] = Field(
        default_factory=list,
        description="Relevant or target test files or cases referenced in this step."
    )


class ExecutionPlan(BaseModel):
    goal: str = Field(
        description="The overall goal of the task."
    )

    steps: list[PlanStep] = Field(
        description="Ordered steps required to complete the task."
    )

    success_criteria: str = Field(
        description="Condition that determines whether the task is complete."
    )