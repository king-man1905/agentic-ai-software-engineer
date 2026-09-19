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

    # LLM_MODEL_NAME is a single, provider-agnostic override with no
    # provider of its own attached to it - it exists to pin the
    # CONFIGURED PRIMARY provider to a specific model (e.g. NVIDIA to
    # openai/gpt-oss-20b after a wave of upstream model deprecations), not
    # to name a model for whichever provider happens to be requested.
    # Applying it unconditionally meant a safe provider fallback (e.g.
    # NVIDIA -> Gemini, backend/observability/telemetry.py's explicit
    # get_llm(provider=fallback_provider) call) reused the PRIMARY
    # provider's model name against the FALLBACK provider's API, which has
    # no such model - confirmed in production as a Gemini 404 for the
    # NVIDIA-only model name "openai/gpt-oss-20b". Every other call site in
    # this codebase calls get_llm() with no explicit provider at all, so
    # eff_provider is always the configured primary there already - this
    # guard changes nothing for them.
    configured_primary = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    explicit_model = os.getenv("LLM_MODEL_NAME") or LLM_MODEL_NAME
    model = (explicit_model if eff_provider == configured_primary else None) or _DEFAULT_MODELS.get(eff_provider)

    nvidia_key = os.getenv("NVIDIA_API_KEY", NVIDIA_API_KEY)
    google_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

    if eff_provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        extra_kwargs = {}
        # openai/gpt-oss-* are reasoning models: by default they spend a
        # large, variable amount of hidden "thinking" tokens before the
        # final answer, which is what makes plain classification calls slow
        # enough to threaten LLM_REQUEST_TIMEOUT_SECONDS. `reasoning_effort`
        # is the NVIDIA-hosted, OpenAI-compatible request parameter these
        # models accept to bound that thinking budget; ChatNVIDIA has no
        # dedicated field for it, but its pydantic model passes unknown
        # constructor kwargs through to `model_kwargs`, which is merged
        # into the outbound request payload - the smallest mechanism this
        # installed version supports, without inventing a new parameter.
        if "gpt-oss" in model.lower():
            extra_kwargs["reasoning_effort"] = "low"
        client = ChatNVIDIA(model=model, api_key=nvidia_key, temperature=0, timeout=eff_timeout, **extra_kwargs)
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


def _has_provider_credentials(provider: str) -> bool:
    """
    Centralizes the NVIDIA/Gemini/OpenAI credential-presence check that used
    to be duplicated (and, for fallback selection, only partially applied)
    across get_llm(), get_fallback_provider(), and validate_config(). A
    provider with no key configured is never "available", regardless of
    where in the codebase that's being decided.
    """
    provider = (provider or "").strip().lower()
    if provider == "nvidia":
        return bool(os.getenv("NVIDIA_API_KEY", NVIDIA_API_KEY))
    if provider == "gemini":
        return bool(os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY))
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY", OPENAI_API_KEY))
    return False


# Deterministic fallback preference order per primary provider - unchanged
# from the original hard-coded per-branch order, just centralized into one
# table so get_fallback_provider() can walk it uniformly instead of
# hand-rolling three near-duplicate if/else ladders.
_FALLBACK_ORDER = {
    "nvidia": ("gemini", "openai"),
    "gemini": ("nvidia", "openai"),
    "openai": ("nvidia", "gemini"),
}


def get_fallback_provider(primary: Optional[str] = None) -> Optional[str]:
    """
    Resolves the healthy, eligible fallback provider for bounded fallback.
    Returns None if fallback is disabled, or if no *credentialed*
    alternative provider is configured - a provider with no API key is
    never selectable, not even as a last-resort default (previously, the
    final branch of each preference ladder returned a hardcoded provider
    name unconditionally, so a completely uncredentialed provider could be
    selected and would only fail later, deep inside fallback initialization).
    """
    if os.getenv("LLM_FALLBACK_ENABLED", "true").strip().lower() in ("0", "false", "no"):
        return None

    primary_clean = (primary or os.getenv("LLM_PROVIDER", LLM_PROVIDER)).strip().lower()

    explicit_fallback = os.getenv("LLM_FALLBACK_PROVIDER")
    if explicit_fallback:
        candidate = explicit_fallback.strip().lower()
        if (
            candidate in ("nvidia", "gemini", "openai")
            and candidate != primary_clean
            and _has_provider_credentials(candidate)
        ):
            return candidate
        # An explicit override that's uncredentialed (or equal to the
        # primary) is never selected - fall through to the normal,
        # credential-checked preference order below instead of returning
        # it or giving up outright.

    order = _FALLBACK_ORDER.get(primary_clean, ("gemini", "nvidia", "openai"))
    for candidate in order:
        if candidate != primary_clean and _has_provider_credentials(candidate):
            return candidate

    return None
