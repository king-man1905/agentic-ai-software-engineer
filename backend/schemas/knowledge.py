from pydantic import BaseModel, Field


class KnowledgeAnswer(BaseModel):
    answer: str = Field(
        description="Answer grounded only in the retrieved project context."
    )

    sources: list[str] = Field(
        description="Project files used to produce the answer."
    )

    sufficient_context: bool = Field(
        description="Whether retrieved context is sufficient to answer reliably."
    )