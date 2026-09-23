"""
Regression tests for the NVIDIA openai/gpt-oss-20b router timeout fix.

Round 1 root cause (superseded, see Round 3 below): gpt-oss-20b is a
reasoning model that, by default, spends a large and variable amount of
hidden "thinking" tokens before answering. The fix passed NVIDIA's
`reasoning_effort="low"` request parameter (via ChatNVIDIA's model_kwargs
passthrough) for gpt-oss models, and trimmed the router prompt.

Round 2 root cause (this file's main coverage): even at reasoning_effort=
"low", the model occasionally still returns an empty completion for a given
request. ChatNVIDIA.with_structured_output(), for hosted endpoints, reacts
to that by silently retrying up to 3 different request formats in sequence
(OpenAI response_format, then guided_json direct, then guided_json via
nvext) - confirmed from the installed langchain-nvidia-ai-endpoints==1.4.3
source - and since the *model* produced nothing, all 3 fail identically,
costing 3 full round-trips (measured up to ~130s on real hardware) before
the existing raw-invoke fallback in `_invoke_single_provider` even runs.
The fix adds `_invoke_nvidia_single_format_structured`, which binds the
same primary response_format directly (via the public `.bind()` mechanism
ChatNVIDIA's own with_structured_output uses internally) for a single
round-trip, then reuses the existing empty/malformed classification
unchanged - gated strictly to NVIDIA gpt-oss models. Unaffected by Round 3.

Round 3 root cause (2026-09-27, corrects Round 1): the router still timed
out at 75s even with reasoning_effort="low" set. Investigation proved
Round 1's premise wrong: the installed SDK's own static model registry
marks "openai/gpt-oss-20b" as `supports_thinking=False` (not a recognized
thinking-controllable model via ChatNVIDIA's first-class mechanism), so
`reasoning_effort` was only ever an unmanaged pydantic "unknown field"
passthrough into model_kwargs. Three direct HTTPS requests to NVIDIA's
endpoint for this model - no reasoning_effort, reasoning_effort at the top
level, and reasoning_effort nested under chat_template_kwargs - all timed
out identically at ~90-92s with zero measurable difference: the parameter
has no effect on this endpoint's latency in any shape NVIDIA's API
recognizes. The fix removes the ineffective passthrough entirely rather
than keep sending a parameter proven to do nothing. The actual bottleneck
is NVIDIA's own hosted response time for this model, which is external and
handled correctly already by the existing LLM_TIMEOUT classification and
provider fallback - unchanged by this round.
"""

from types import SimpleNamespace

import pytest

from backend.observability.telemetry import (
    _invoke_nvidia_single_format_structured,
    invoke_structured,
)
from backend.services.errors import (
    LLMMalformedResponseError,
    LLMTimeoutError,
    classify_llm_exception,
    is_fallback_eligible,
)
from backend.schemas.routing import RoutingDecision, TaskType


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeNvidiaStructuredRunnable:
    """Stands in for the Runnable ChatNVIDIA.with_structured_output(schema)
    (no include_raw) returns - the multi-format cascade path, used only for
    non-gpt-oss NVIDIA models after this fix."""

    def __init__(self, parsed):
        self._parsed = parsed

    def invoke(self, prompt):
        return self._parsed


class FakeNvidiaBoundRunnable:
    """Stands in for the Runnable ChatNVIDIA.bind(response_format=...)
    returns - the single-round-trip fast path this fix adds for gpt-oss."""

    def __init__(self, content):
        self._content = content

    def invoke(self, prompt):
        return SimpleNamespace(content=self._content)


class FakeNvidiaLLM:
    """Mimics the real ChatNVIDIA contract relevant to this fix:
    - include_raw=True raises NotImplementedError unconditionally, before
      any network call.
    - with_structured_output(schema) (no include_raw) is the "expensive"
      multi-format cascade - used for non-gpt-oss models only now.
    - bind(response_format=...) is the new single-call fast path used for
      gpt-oss models.
    - invoke(prompt) is the plain direct-fallback call.
    """

    def __init__(
        self,
        model="openai/gpt-oss-20b",
        plain_result=None,
        bind_content=None,
        direct_content=None,
    ):
        self._provider = "nvidia"
        self.model = model
        self._plain_result = plain_result
        self._bind_content = bind_content
        self._direct_content = direct_content
        self.include_raw_calls = 0
        self.plain_calls = 0
        self.bind_calls = 0
        self.direct_invoke_calls = 0

    def with_structured_output(self, schema, include_raw=False):
        if include_raw:
            self.include_raw_calls += 1
            raise NotImplementedError("include_raw=True is not implemented")
        self.plain_calls += 1
        return FakeNvidiaStructuredRunnable(self._plain_result)

    def bind(self, **kwargs):
        assert "response_format" in kwargs
        assert kwargs["response_format"]["json_schema"]["name"] == "RoutingDecision"
        self.bind_calls += 1
        return FakeNvidiaBoundRunnable(self._bind_content)

    def invoke(self, prompt):
        self.direct_invoke_calls += 1
        return SimpleNamespace(content=self._direct_content, usage_metadata=None)


# ---------------------------------------------------------------------------
# 1. gpt-oss reasoning_effort passthrough (llm.py) - REMOVED in Round 3,
#    empirically proven to have no effect on NVIDIA's hosted endpoint for
#    this model. These tests now lock in its absence.
# ---------------------------------------------------------------------------

class TestNvidiaReasoningEffortConfiguration:
    def test_gpt_oss_default_model_does_not_get_reasoning_effort(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
        monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
        from backend.services import llm as llm_module

        client = llm_module.get_llm(provider="nvidia")
        assert "reasoning_effort" not in client.model_kwargs

    def test_gpt_oss_client_construction_emits_no_unsupported_parameter_warning(self, monkeypatch):
        """Passing an unrecognized constructor kwarg (the old
        reasoning_effort passthrough) triggers a pydantic UserWarning on
        every NVIDIA client construction - confirmed gone now that the
        parameter is no longer sent at all."""
        import warnings

        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
        monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
        from backend.services import llm as llm_module

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            llm_module.get_llm(provider="nvidia")
        assert not any("reasoning_effort" in str(w.message) for w in caught)

    def test_non_gpt_oss_nvidia_model_is_not_given_reasoning_effort(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
        monkeypatch.setenv("LLM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
        from backend.services import llm as llm_module

        client = llm_module.get_llm(provider="nvidia")
        assert "reasoning_effort" not in client.model_kwargs

    def test_other_providers_are_unaffected(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key")
        from backend.services import llm as llm_module

        gemini_client = llm_module.get_llm(provider="gemini")
        assert not hasattr(gemini_client, "model_kwargs") or "reasoning_effort" not in (
            gemini_client.model_kwargs or {}
        )
        openai_client = llm_module.get_llm(provider="openai")
        assert getattr(openai_client, "model_kwargs", None) in (None, {}) or (
            "reasoning_effort" not in openai_client.model_kwargs
        )


# ---------------------------------------------------------------------------
# Direct unit tests of the new single-format helper
# ---------------------------------------------------------------------------

class TestNvidiaSingleFormatStructuredHelper:
    def test_returns_parsed_decision_on_valid_json(self):
        decision = RoutingDecision(
            task_type=TaskType.DOCUMENTATION, requires_planning=False,
            requires_knowledge=False, reasoning="README is documentation.",
        )
        llm = FakeNvidiaLLM(bind_content=decision.model_dump_json())

        result = _invoke_nvidia_single_format_structured(llm, RoutingDecision, "prompt")

        assert result == decision
        assert llm.bind_calls == 1

    def test_returns_none_on_empty_content(self):
        """The exact failure mode reported from EC2: the model returns an
        empty completion. Must return None (not raise) so the caller's
        existing empty-completion handling decides the outcome."""
        llm = FakeNvidiaLLM(bind_content="")
        assert _invoke_nvidia_single_format_structured(llm, RoutingDecision, "prompt") is None

    def test_returns_none_on_malformed_json(self):
        """Content that isn't valid JSON (or doesn't satisfy the schema)
        degrades to None instead of raising - the caller still gets a
        chance via its own raw-invoke fallback."""
        llm = FakeNvidiaLLM(bind_content='{"task_type": "NOT_A_REAL_TYPE"}')
        assert _invoke_nvidia_single_format_structured(llm, RoutingDecision, "prompt") is None

    def test_returns_none_when_no_json_braces_present(self):
        llm = FakeNvidiaLLM(bind_content="I cannot classify this request.")
        assert _invoke_nvidia_single_format_structured(llm, RoutingDecision, "prompt") is None


# ---------------------------------------------------------------------------
# 5. include_raw=True NotImplementedError -> fast single-call fallback
# ---------------------------------------------------------------------------

class TestNvidiaIncludeRawUnsupportedFallback:
    def test_include_raw_true_raises_not_implemented(self):
        llm = FakeNvidiaLLM()
        with pytest.raises(NotImplementedError):
            llm.with_structured_output(RoutingDecision, include_raw=True)

    def test_gpt_oss_falls_back_to_single_bind_call_not_the_cascade(self):
        """The core fix: for gpt-oss, the expensive with_structured_output(
        schema) multi-format cascade must never be invoked - only the
        single bind() call."""
        decision = RoutingDecision(
            task_type=TaskType.BUG_FIX, requires_planning=True,
            requires_knowledge=True, reasoning="Login bug after password reset.",
        )
        llm = FakeNvidiaLLM(bind_content=decision.model_dump_json())

        result = invoke_structured(llm, RoutingDecision, "prompt")

        assert result == decision
        assert llm.include_raw_calls == 1
        assert llm.bind_calls == 1
        assert llm.plain_calls == 0
        assert llm.direct_invoke_calls == 0


# ---------------------------------------------------------------------------
# 6. No regression for non-gpt-oss NVIDIA models
# ---------------------------------------------------------------------------

class TestNoRegressionForNonGptOssNvidia:
    def test_non_gpt_oss_model_still_uses_the_original_cascade_path(self):
        decision = RoutingDecision(
            task_type=TaskType.CODE_REVIEW, requires_planning=False,
            requires_knowledge=True, reasoning="Reviewing existing code.",
        )
        llm = FakeNvidiaLLM(model="meta/llama-3.1-8b-instruct", plain_result=decision)

        result = invoke_structured(llm, RoutingDecision, "prompt")

        assert result == decision
        assert llm.include_raw_calls == 1
        assert llm.plain_calls == 1
        assert llm.bind_calls == 0


# ---------------------------------------------------------------------------
# 2. Successful structured RoutingDecision via the router
# ---------------------------------------------------------------------------

class TestRouterStructuredOutput:
    def test_route_task_returns_structured_routing_decision(self, monkeypatch):
        decision = RoutingDecision(
            task_type=TaskType.DOCUMENTATION,
            requires_planning=False,
            requires_knowledge=True,
            reasoning="User wants README documentation.",
        )
        fake_llm = FakeNvidiaLLM(bind_content=decision.model_dump_json())

        import backend.agents.router as router_module
        monkeypatch.setattr(router_module, "get_llm", lambda: fake_llm)

        result = router_module.route_task("Write a README for the payments module.")

        assert result == decision
        assert fake_llm.include_raw_calls == 1
        assert fake_llm.bind_calls == 1

    def test_router_prompt_instructs_brief_classification_only(self, monkeypatch):
        captured = {}
        fake_llm = FakeNvidiaLLM(bind_content=RoutingDecision(
            task_type=TaskType.GENERAL, requires_planning=False,
            requires_knowledge=False, reasoning="n/a",
        ).model_dump_json())

        import backend.agents.router as router_module
        monkeypatch.setattr(router_module, "get_llm", lambda: fake_llm)

        orig_invoke_structured = router_module.invoke_structured

        def capturing_invoke_structured(llm, schema, prompt):
            captured["prompt"] = prompt
            return orig_invoke_structured(llm, schema, prompt)

        monkeypatch.setattr(router_module, "invoke_structured", capturing_invoke_structured)
        router_module.route_task("hello")

        prompt = captured["prompt"]
        assert "do not reason at length" in prompt.lower()
        assert "one short sentence" in prompt.lower()
        assert len(prompt) < 1500  # old prompt was ~2000+ chars of category examples


# ---------------------------------------------------------------------------
# Timeout classification (unaffected by this fix, re-asserted here)
# ---------------------------------------------------------------------------

class TestTimeoutClassification:
    def test_nvidia_read_timeout_is_classified_and_fallback_eligible(self):
        exc = TimeoutError("Read timed out. (read timeout=75.0)")
        classified = classify_llm_exception(exc, provider="nvidia")
        assert isinstance(classified, LLMTimeoutError)
        assert is_fallback_eligible(classified) is True
        assert classified.provider == "nvidia"


# ---------------------------------------------------------------------------
# 3. Empty completion classification (end-to-end, the exact EC2 scenario)
# ---------------------------------------------------------------------------

class TestMalformedResponseHandling:
    def test_empty_completion_from_bind_and_direct_fallback_is_malformed_fast(self):
        """Reproduces the EC2 "Write a README..." failure: the schema-guided
        single call comes back empty, and so does the plain direct-invoke
        fallback. Must still raise LLMMalformedResponseError (unchanged
        classification), but after only the bind call + one direct
        fallback call - never the old 3-format cascade."""
        llm = FakeNvidiaLLM(bind_content="", direct_content="")

        with pytest.raises(LLMMalformedResponseError):
            invoke_structured(llm, RoutingDecision, "prompt")

        assert llm.include_raw_calls == 1
        assert llm.bind_calls == 1
        assert llm.plain_calls == 0  # the expensive cascade must never run
        assert llm.direct_invoke_calls == 1

    def test_malformed_json_from_bind_recovers_via_direct_fallback(self):
        """If the bind() call returns unparseable content but the plain
        direct-invoke fallback returns valid JSON text, the existing
        regex-extraction path still recovers it."""
        decision = RoutingDecision(
            task_type=TaskType.GENERAL, requires_planning=False,
            requires_knowledge=False, reasoning="Fallback recovered this.",
        )
        llm = FakeNvidiaLLM(
            bind_content="not json at all",
            direct_content=decision.model_dump_json(),
        )

        result = invoke_structured(llm, RoutingDecision, "prompt")

        assert result == decision
        assert llm.bind_calls == 1
        assert llm.direct_invoke_calls == 1


# ---------------------------------------------------------------------------
# 8. Existing bounded primary -> fallback provider behavior preserved
# ---------------------------------------------------------------------------

class TestExistingFallbackBehaviorPreserved:
    def test_primary_timeout_falls_back_to_secondary_provider(self, monkeypatch):
        decision = RoutingDecision(
            task_type=TaskType.BUG_FIX, requires_planning=False,
            requires_knowledge=False, reasoning="short",
        )

        # Primary: bind() path comes back empty, then the direct fallback
        # call times out - exercised via a real TimeoutError from invoke().
        primary = FakeNvidiaLLM(bind_content="")

        def timing_out_invoke(prompt):
            primary.direct_invoke_calls += 1
            raise TimeoutError("Read timed out. (read timeout=75.0)")

        primary.invoke = timing_out_invoke

        monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "gemini")

        from backend.services import llm as llm_module

        # This fake becomes non-gpt-oss (provider "gemini"), so it takes
        # the ORIGINAL with_structured_output(schema) cascade path, not the
        # bind() fast path - set plain_result accordingly.
        fallback_llm = FakeNvidiaLLM(plain_result=decision)
        fallback_llm._provider = "gemini"
        monkeypatch.setattr(llm_module, "get_llm", lambda provider=None: fallback_llm)

        result = invoke_structured(primary, RoutingDecision, "prompt")

        assert result == decision
        assert primary.bind_calls == 1
        assert primary.direct_invoke_calls == 1
