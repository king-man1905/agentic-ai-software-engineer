from backend.core.config import (
    LLM_PROVIDER,
    LLM_MODEL_NAME,
    NVIDIA_API_KEY,
    GOOGLE_API_KEY,
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
}


def get_llm():
    validate_config()
    model = LLM_MODEL_NAME or _DEFAULT_MODELS.get(LLM_PROVIDER)

    if LLM_PROVIDER == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(model=model, api_key=NVIDIA_API_KEY, temperature=0)

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=GOOGLE_API_KEY, temperature=0)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'. Supported: 'nvidia', 'gemini'."
    )
