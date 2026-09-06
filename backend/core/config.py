import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").strip().lower()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")  # provider-specific default applied in services/llm.py

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60.0"))
WORKSPACE_LOCK_TIMEOUT_SECONDS = float(os.getenv("WORKSPACE_LOCK_TIMEOUT_SECONDS", "30.0"))
SHUTDOWN_DRAIN_TIMEOUT_SECONDS = float(os.getenv("SHUTDOWN_DRAIN_TIMEOUT_SECONDS", "30.0"))


def validate_config(provider: str | None = None):
    """Validates that necessary provider keys are present prior to LLM invocation."""
    eff_provider = (provider or os.getenv("LLM_PROVIDER", LLM_PROVIDER)).strip().lower()
    nvidia_key = os.getenv("NVIDIA_API_KEY", NVIDIA_API_KEY)
    google_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

    if eff_provider == "nvidia" and not nvidia_key:
        raise ValueError(
            "LLM_PROVIDER is 'nvidia' but NVIDIA_API_KEY is missing. Add it to environment variables."
        )
    if eff_provider == "gemini" and not google_key:
        raise ValueError(
            "LLM_PROVIDER is 'gemini' but GOOGLE_API_KEY is missing. Add it to environment variables."
        )
    if eff_provider == "openai" and not openai_key:
        raise ValueError(
            "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is missing. Add it to environment variables."
        )
