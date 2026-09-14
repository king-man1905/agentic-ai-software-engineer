"""
Structured LLM / Provider Exception Hierarchy & Error Classification.
Ensures unambiguous error classification distinguishing transient provider failures,
timeouts, rate limits, authentication issues, and invalid requests.
"""

from typing import Optional


class LLMError(Exception):
    """Base exception for all structured LLM provider errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_exception: Optional[Exception] = None,
        elapsed_ms: Optional[float] = None,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.original_exception = original_exception
        self.elapsed_ms = elapsed_ms

    def __str__(self) -> str:
        prov_prefix = f"[{self.provider.upper()}] " if self.provider else ""
        return f"{prov_prefix}{self.message}"


class LLMTimeoutError(LLMError, TimeoutError):
    """
    Outbound LLM request timed out (socket timeout, connect timeout, read timeout,
    or provider deadline exceeded). Eligible for safe provider fallback.
    """
    pass


class LLMTransientError(LLMError):
    """
    Transient provider failure (500, 502, 503, 504, connection reset, connection refused,
    temporary service unavailable). Eligible for safe provider fallback.
    """
    pass


class LLMMalformedResponseError(LLMTransientError):
    """
    Provider returned an empty, truncated, or unparseable structured
    response (e.g. an empty completion body, invalid/truncated JSON, or a
    schema parser failing to decode the raw text at all - not a
    well-formed response that simply fails semantic/schema validation).
    This is a response-quality problem, not a request-validity problem:
    retrying the same request (or trying a different provider) is a
    reasonable recovery, so this is retryable and eligible for provider
    fallback like any other transient failure.
    """
    pass


class LLMRateLimitError(LLMError):
    """
    Provider rate limit or quota exhaustion (HTTP 429).
    Eligible for safe provider fallback when bounded.
    """
    pass


class LLMAuthenticationError(LLMError, PermissionError):
    """
    Authentication or authorization failure (HTTP 401, 403, invalid API key, permission denied).
    Deterministic failure: MUST NOT trigger automatic provider fallback.
    """
    pass


class LLMInvalidRequestError(LLMError, ValueError):
    """
    Malformed request, bad parameters, context length exceeded, or schema error (HTTP 400, 422).
    Deterministic failure: MUST NOT trigger automatic provider fallback.
    """
    pass


class LLMPermanentError(LLMError):
    """
    Permanent, unrecoverable, or programming error.
    Deterministic failure: MUST NOT trigger automatic provider fallback.
    """
    pass


def is_fallback_eligible(error: Exception) -> bool:
    """
    Returns True ONLY for explicitly classified transient failures or timeouts.
    Never returns True for authentication, authorization, or invalid request errors.
    """
    return isinstance(error, (LLMTimeoutError, LLMTransientError, LLMRateLimitError))


def classify_llm_exception(exc: Exception, provider: Optional[str] = None) -> LLMError:
    """
    Inspects an arbitrary caught exception from an LLM call or SDK and maps it
    deterministically into the structured LLMError hierarchy.
    """
    if isinstance(exc, LLMError):
        if provider and not exc.provider:
            exc.provider = provider
        return exc

    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)

    # 1. Timeout Checks
    # Common exception class names: Timeout, TimeoutError, ConnectTimeout, ReadTimeout, APITimeoutError
    if (
        "timeout" in exc_type
        or "timeoutexception" in exc_type
        or "readtimedout" in exc_str
        or "connecttimedout" in exc_str
        or "timed out" in exc_str
        or "deadline exceeded" in exc_str
        or "socket timeout" in exc_str
        or "request timed out" in exc_str
    ):
        return LLMTimeoutError(
            f"LLM request timed out: {exc}",
            provider=provider,
            original_exception=exc,
        )

    # 2. Authentication & Authorization Checks (MUST NOT FALLBACK)
    if (
        status_code in (401, 403)
        or "authentication" in exc_type
        or "permissiondenied" in exc_type
        or "unauthorized" in exc_str
        or "invalid api key" in exc_str
        or "api_key" in exc_str and ("invalid" in exc_str or "missing" in exc_str or "expired" in exc_str)
        or "forbidden" in exc_str
        or "401" in exc_str
        or "permission denied" in exc_str
    ):
        return LLMAuthenticationError(
            f"LLM authentication failed: {exc}",
            provider=provider,
            original_exception=exc,
        )

    # 3. Rate Limit Checks (Eligible for fallback)
    if (
        status_code == 429
        or "ratelimit" in exc_type
        or "resourceexhausted" in exc_type
        or "429" in exc_str
        or "too many requests" in exc_str
        or "rate limit" in exc_str
        or "quota exceeded" in exc_str
    ):
        return LLMRateLimitError(
            f"LLM rate limit reached: {exc}",
            provider=provider,
            original_exception=exc,
        )

    # 4. Malformed / Empty Structured Response Checks (Eligible for fallback)
    # Checked BEFORE the Invalid Request rule below: a provider that
    # returned an empty, truncated, or unparseable structured completion is
    # a response-quality problem worth a retry/fallback attempt, not a
    # deterministic "the request itself was invalid" failure - even though
    # the parsing library (Pydantic's ValidationError, LangChain's
    # OutputParserException, or Python's json.JSONDecodeError) raises
    # exactly the same exception *types* a genuine schema/argument error
    # would. Distinguished by message content, not exception type alone: a
    # Pydantic ValidationError only lands here when its own `type=
    # json_invalid` tag (or an equivalent JSON-decoding message) shows the
    # failure was in parsing the raw response text as JSON at all - never
    # for a well-formed JSON response that simply fails semantic/schema
    # validation (e.g. "field required"), which still falls through to the
    # Invalid Request rule below, unchanged, via the generic
    # "validationerror" type check there. A message that describes the
    # *response* as malformed (e.g. "Malformed response body received from
    # provider") belongs here too, not on the Invalid Request rule - guarded
    # against "malformed request" specifically so that phrase (a genuine
    # outbound-request signal) still falls through to the Invalid Request
    # rule below unaffected.
    if (
        "jsondecodeerror" in exc_type
        or "outputparserexception" in exc_type
        or "outputparser" in exc_type
        or "json_invalid" in exc_str
        or "expecting value" in exc_str
        or "eof while parsing" in exc_str
        or "unterminated string" in exc_str
        or "unterminated array" in exc_str
        or "unterminated object" in exc_str
        or ("extra data" in exc_str and "json" in exc_str)
        or ("malformed" in exc_str and "malformed request" not in exc_str)
    ):
        return LLMMalformedResponseError(
            f"LLM returned an empty, truncated, or unparseable structured response: {exc}",
            provider=provider,
            original_exception=exc,
        )

    # 5. Invalid Request / Schema Validation Checks (MUST NOT FALLBACK)
    # Note: deliberately does NOT use `isinstance(exc, (ValueError, TypeError))`
    # as a catch-all - that swept every response-parsing failure (which is
    # also typically a ValueError/TypeError under the hood) into this
    # non-retryable bucket, including the empty/malformed-JSON case rule 4
    # above now correctly classifies as retryable instead. Genuine invalid-
    # request signals (status code, explicit type names, or message
    # content) are matched explicitly. Also deliberately does NOT match a
    # bare "malformed" substring - a provider response that merely
    # describes itself as "malformed" (e.g. "Malformed response body
    # received from provider") is exactly the response-quality problem
    # rule 4 above exists to catch, and is retryable; only a message that
    # specifically calls out the *request* as malformed belongs here.
    if (
        status_code in (400, 422)
        or "badrequest" in exc_type
        or "validationerror" in exc_type
        or "invalidargument" in exc_type
        or "400" in exc_str and "bad request" in exc_str
        or "context window" in exc_str
        or "maximum context length" in exc_str
        or "schema" in exc_str
        or "malformed request" in exc_str
    ):
        return LLMInvalidRequestError(
            f"LLM invalid request: {exc}",
            provider=provider,
            original_exception=exc,
        )

    # 6. Transient Network / Server 5xx Checks (Eligible for fallback)
    if (
        (isinstance(status_code, int) and 500 <= status_code <= 599)
        or "connectionerror" in exc_type
        or "internalservererror" in exc_type
        or "badgateway" in exc_type
        or "serviceunavailable" in exc_type
        or "gatewaytimeout" in exc_type
        or "networkerror" in exc_type
        or "500" in exc_str
        or "502" in exc_str
        or "503" in exc_str
        or "504" in exc_str
        or "connection reset" in exc_str
        or "connection refused" in exc_str
        or "server error" in exc_str
        or "service unavailable" in exc_str
    ):
        return LLMTransientError(
            f"LLM transient provider failure: {exc}",
            provider=provider,
            original_exception=exc,
        )

    # 7. Fallback to Permanent / General LLM Error
    return LLMPermanentError(
        f"LLM provider error: {exc}",
        provider=provider,
        original_exception=exc,
    )
