"""
Centralized privacy, sanitization, and secret redaction for all telemetry and analytics.
Prevents any credential, token, key, or sensitive header from entering persistent telemetry storage.
"""

from typing import Any, List, Set

SENSITIVE_FIELD_NAMES: Set[str] = {
    "token",
    "github_token",
    "gh_token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "authorization",
    "bearer",
    "auth",
    "cookie",
    "credentials",
    "private_key",
    "openai_api_key",
    "nvidia_api_key",
}

SENSITIVE_SUBSTRINGS: List[str] = [
    "token",
    "secret",
    "password",
    "auth",
    "bearer",
    "privkey",
    "credential",
    "api_key",
]

MAX_STRING_LENGTH = 4000


def sanitize_telemetry_payload(data: Any) -> Any:
    """
    Recursively scans dictionaries, lists, and primitives to redact credentials
    and bound payload sizes before persistence.
    """
    if data is None:
        return None

    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            k_lower = str(k).lower().strip()
            if k_lower in SENSITIVE_FIELD_NAMES or any(sub in k_lower for sub in SENSITIVE_SUBSTRINGS):
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, (dict, list)):
                sanitized[k] = sanitize_telemetry_payload(v)
            elif isinstance(v, str):
                sanitized[k] = _sanitize_string_value(v)
            else:
                sanitized[k] = v
        return sanitized

    if isinstance(data, list):
        return [sanitize_telemetry_payload(item) for item in data]

    if isinstance(data, str):
        return _sanitize_string_value(data)

    return data


def _sanitize_string_value(val: str) -> str:
    """Detects embedded token patterns (e.g. ghp_, sk-, agy_live_) and redacts them."""
    if not val:
        return val

    # Common token prefixes
    prefixes = ["ghp_", "github_pat_", "sk-", "nvapi-", "agy_live_"]
    for p in prefixes:
        if p in val:
            # Redact whole token if embedded
            import re
            val = re.sub(rf"{p}[A-Za-z0-9_\-]+", "[REDACTED_SECRET]", val)

    if len(val) > MAX_STRING_LENGTH:
        val = val[:MAX_STRING_LENGTH] + "... [TRUNCATED]"

    return val
