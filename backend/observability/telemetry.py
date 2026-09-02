"""
LLM token usage and cost telemetry.

Wraps structured-output LLM calls to capture prompt/completion token counts
and an approximate USD cost, without changing the calling agent functions'
signatures or return types (existing callers and test mocks are unaffected).
Usage from calls made inside a `collect_usage()` block is accumulated and
handed back to the caller to merge into the graph state's `metrics` dict.
"""

import contextvars
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

ZERO_USAGE: Dict[str, Any] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
    # False until a call actually reports usage. Some providers/paths (e.g.
    # ChatNVIDIA's with_structured_output, which doesn't support
    # include_raw) never expose token counts at all - that's "unmeasured",
    # not "zero cost", and callers should render it differently.
    "tracked": False,
}

# Approximate USD cost per 1K tokens, keyed by a substring match against the
# model identifier (case-insensitive). First match wins; extend as needed.
_PRICING_PER_1K = {
    "claude": {"prompt": 0.003, "completion": 0.015},
    "sonnet": {"prompt": 0.003, "completion": 0.015},
    "gemini": {"prompt": 0.000075, "completion": 0.0003},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    "llama": {"prompt": 0.0002, "completion": 0.0002},
}
_FALLBACK_PRICING = {"prompt": 0.0005, "completion": 0.0015}

_current_usage: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar(
    "_current_usage", default=None
)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Approximates USD cost for a call from a small per-model pricing table."""
    model_key = (model or "").lower()
    rates = _FALLBACK_PRICING
    for key, value in _PRICING_PER_1K.items():
        if key in model_key:
            rates = value
            break
    return (prompt_tokens / 1000) * rates["prompt"] + (completion_tokens / 1000) * rates["completion"]


def extract_usage(raw_message: Any, model: str) -> Dict[str, Any]:
    """
    Safely extracts token usage from a LangChain AIMessage: tries the
    standard `usage_metadata` field first, then the provider-specific
    `response_metadata['token_usage']` shape. Never raises; defaults to
    zeroed counts if the provider didn't report usage.
    """
    prompt_tokens = 0
    completion_tokens = 0
    tracked = False
    try:
        usage = getattr(raw_message, "usage_metadata", None)
        if usage:
            prompt_tokens = usage.get("input_tokens", 0) or 0
            completion_tokens = usage.get("output_tokens", 0) or 0
            tracked = True
        else:
            token_usage = (getattr(raw_message, "response_metadata", None) or {}).get(
                "token_usage", {}
            )
            if token_usage:
                prompt_tokens = token_usage.get("prompt_tokens", 0) or 0
                completion_tokens = token_usage.get("completion_tokens", 0) or 0
                tracked = True
    except Exception:
        prompt_tokens = 0
        completion_tokens = 0
        tracked = False

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": estimate_cost_usd(model, prompt_tokens, completion_tokens),
        "tracked": tracked,
    }


def merge_usage(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Adds two usage dicts together, returning a new dict. Missing keys default to 0.
    `tracked` is True if either side actually measured real usage."""
    a = a or ZERO_USAGE
    b = b or ZERO_USAGE
    return {
        "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
        "completion_tokens": a.get("completion_tokens", 0) + b.get("completion_tokens", 0),
        "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
        "estimated_cost_usd": a.get("estimated_cost_usd", 0.0) + b.get("estimated_cost_usd", 0.0),
        "tracked": bool(a.get("tracked", False) or b.get("tracked", False)),
    }


@contextmanager
def collect_usage():
    """
    Context manager that accumulates token usage for every LLM call made
    through `invoke_structured` inside its block. Yields the running dict.
    """
    token = _current_usage.set(dict(ZERO_USAGE))
    try:
        yield _current_usage.get()
    finally:
        _current_usage.reset(token)


def invoke_structured(llm, schema, prompt: str) -> Any:
    """
    Invokes `llm` for structured `schema` output, same as
    `llm.with_structured_output(schema).invoke(prompt)`, except it also
    records token usage (into whatever `collect_usage()` block is active,
    if any) before returning just the parsed result. Preserves the original
    fail-fast behavior: a parsing failure still raises.
    """
    import time
    import re

    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")
    schema_name = getattr(schema, "__name__", str(schema))
    print(f"[LLM] Requesting structured output for {schema_name} from {model_name}...", flush=True)
    t0 = time.time()

    for attempt in range(5):
        try:
            result = None
            try:
                structured = llm.with_structured_output(schema, include_raw=True)
                result = structured.invoke(prompt)
            except (NotImplementedError, Exception) as inner_exc:
                if "guided_json" in str(inner_exc) or "[400]" in str(inner_exc):
                    pass
                elif isinstance(inner_exc, NotImplementedError):
                    try:
                        structured = llm.with_structured_output(schema)
                        res = structured.invoke(prompt)
                        if res is not None:
                            print(f"[LLM] Response received in {time.time()-t0:.2f}s", flush=True)
                            return res
                    except Exception:
                        pass
                else:
                    raise inner_exc


            if isinstance(result, dict):
                if result.get("parsing_error") is not None:
                    raise result["parsing_error"]
                if result.get("parsed") is not None:
                    raw = result.get("raw")
                    if raw is not None:
                        model = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")
                        collector = _current_usage.get()
                        if collector is not None:
                            call_usage = extract_usage(raw, model)
                            collector.update(merge_usage(collector, call_usage))
                    print(f"[LLM] Response received in {time.time()-t0:.2f}s", flush=True)
                    return result["parsed"]

            # Direct fallback if with_structured_output produced None or failed
            raw = llm.invoke(prompt)
            model = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")
            collector = _current_usage.get()
            if collector is not None:
                call_usage = extract_usage(raw, model)
                collector.update(merge_usage(collector, call_usage))

            match = re.search(r"\{.*\}", raw.content, re.DOTALL)
            if match and hasattr(schema, "model_validate_json"):
                try:
                    print(f"[LLM] Response received in {time.time()-t0:.2f}s", flush=True)
                    return schema.model_validate_json(match.group(0))
                except Exception:
                    pass

            # Re-prompt model to extract strict JSON if output was free-form markdown
            schema_doc = schema.model_json_schema() if hasattr(schema, "model_json_schema") else str(schema)
            conv_prompt = (
                f"Convert this text into a single valid JSON object matching this schema:\n{schema_doc}\n\n"
                f"Input text:\n{raw.content}\n\nReturn ONLY the JSON object, no markdown, no explanation."
            )
            conv_res = llm.invoke(conv_prompt)
            conv_match = re.search(r"\{.*\}", conv_res.content, re.DOTALL)
            print(f"[LLM] Response received in {time.time()-t0:.2f}s", flush=True)
            if conv_match and hasattr(schema, "model_validate_json"):
                return schema.model_validate_json(conv_match.group(0))
            if hasattr(schema, "model_validate_json"):
                return schema.model_validate_json(conv_res.content)
            return conv_res.content



        except Exception as e:
            if ("429" in str(e) or "Too Many Requests" in str(e)) and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            if ("guided_json" in str(e) or "[400]" in str(e)) and attempt < 4:
                try:
                    raw = llm.invoke(prompt)
                    match = re.search(r"\{.*\}", raw.content, re.DOTALL)
                    if match and hasattr(schema, "model_validate_json"):
                        return schema.model_validate_json(match.group(0))
                except Exception:
                    pass
            raise

