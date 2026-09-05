import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").strip().lower()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")  # provider-specific default applied in services/llm.py

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def validate_config():
    """Validates that necessary provider keys are present prior to LLM invocation."""
    if LLM_PROVIDER == "nvidia" and not NVIDIA_API_KEY:
        raise ValueError(
            "LLM_PROVIDER is 'nvidia' but NVIDIA_API_KEY is missing. Add it to environment variables."
        )
    if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        raise ValueError(
            "LLM_PROVIDER is 'gemini' but GOOGLE_API_KEY is missing. Add it to environment variables."
        )
