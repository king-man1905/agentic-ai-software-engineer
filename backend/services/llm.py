import os
from typing import Optional

from backend.core.config import (
    LLM_PROVIDER,
    LLM_MODEL_NAME,
    LLM_REQUEST_TIMEOUT_SECONDS,
    NVIDIA_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    validate_config,
)

# Fallback model per provider, used only when LLM_MODEL_NAME isn't set.
# openai/gpt-oss-20b is the last NVIDIA-hosted model confirmed working
# (2026-09-03) after a wave of Llama 3.1/3.3 model deprecations on that
# endpoint - if it goes end-of-life too, set LLM_MODEL_NAME in .env rather
# than editing this file.
_DEFAULT_MODELS = {
    "nvidia": "openai/gpt-oss-20b",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o",
}


def get_llm(provider: Optional[str] = None, timeout: Optional[float] = None):
    """
    Initializes and returns the configured LLM client with bounded timeout.
    Supports 'nvidia', 'gemini', and 'openai'.
    """
    eff_provider = (provider or os.getenv("LLM_PROVIDER", LLM_PROVIDER)).strip().lower()
    default_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", str(LLM_REQUEST_TIMEOUT_SECONDS)))
    eff_timeout = float(timeout) if timeout is not None else default_timeout
    validate_config(eff_provider)
    model = os.getenv("LLM_MODEL_NAME") or LLM_MODEL_NAME or _DEFAULT_MODELS.get(eff_provider)

    nvidia_key = os.getenv("NVIDIA_API_KEY", NVIDIA_API_KEY)
    google_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

    if eff_provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        client = ChatNVIDIA(model=model, api_key=nvidia_key, temperature=0, timeout=eff_timeout)
        setattr(client, "_provider", "nvidia")
        setattr(client, "_timeout", eff_timeout)
        return client

    if eff_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        client = ChatGoogleGenerativeAI(model=model, google_api_key=google_key, temperature=0, timeout=eff_timeout)
        setattr(client, "_provider", "gemini")
        setattr(client, "_timeout", eff_timeout)
        return client

    if eff_provider == "openai":
        from langchain_openai import ChatOpenAI
        client = ChatOpenAI(model=model, api_key=openai_key, temperature=0, timeout=eff_timeout)
        setattr(client, "_provider", "openai")
        setattr(client, "_timeout", eff_timeout)
        return client

    raise ValueError(
        f"Unsupported LLM_PROVIDER: '{eff_provider}'. Supported: 'nvidia', 'gemini', 'openai'."
    )


def get_fallback_provider(primary: Optional[str] = None) -> Optional[str]:
    """
    Resolves the healthy, eligible fallback provider for bounded fallback.
    Returns None if fallback is disabled or no alternative provider is configured.
    """
    if os.getenv("LLM_FALLBACK_ENABLED", "true").strip().lower() in ("0", "false", "no"):
        return None

    explicit_fallback = os.getenv("LLM_FALLBACK_PROVIDER")
    if explicit_fallback:
        candidate = explicit_fallback.strip().lower()
        if candidate in ("nvidia", "gemini", "openai") and candidate != primary:
            return candidate

    primary_clean = (primary or os.getenv("LLM_PROVIDER", LLM_PROVIDER)).strip().lower()
    nvidia_key = os.getenv("NVIDIA_API_KEY", NVIDIA_API_KEY)
    google_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

    if primary_clean == "nvidia":
        if google_key:
            return "gemini"
        if openai_key:
            return "openai"
        return "gemini"

    if primary_clean == "gemini":
        if nvidia_key:
            return "nvidia"
        if openai_key:
            return "openai"
        return "nvidia"

    if primary_clean == "openai":
        if nvidia_key:
            return "nvidia"
        if google_key:
            return "gemini"
        return "gemini"

    return "gemini" if primary_clean != "gemini" else "nvidia"
