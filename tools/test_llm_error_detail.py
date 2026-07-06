"""Unit tests for the enriched LLM request-failure diagnostics.

Verifies that a failed provider request is reported with the exception type, the
elapsed time versus the configured timeout, any HTTP status / request-id the SDK
exposes, and the underlying cause of the exception chain. Also covers the retry
runner (transient failures retried and reported) and settings resolution from
env / ~/.ludvart/llm.conf.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tools/test_llm_error_detail.py
"""

import os

from ludvart.llm import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    LLMClient,
    LLMError,
    ProviderConfig,
    _describe_request_error,
    _is_rate_limit,
    _is_retryable,
    _resolve_settings,
    _retry_after_seconds,
    _root_cause,
)


class _FakeTimeout(Exception):
    """Stands in for an SDK timeout error (retryable by class name)."""


_FakeTimeout.__name__ = "APITimeoutError"


def _client(max_retries=2):
    cfg = ProviderConfig(name="anthropic", api_url="x", api_key="k", model="m")
    c = LLMClient(cfg, timeout=1.0, max_retries=max_retries)
    return c


def test_root_cause_walks_chain():
    root = ValueError("boom")
    try:
        try:
            raise root
        except ValueError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as exc:
        assert _root_cause(exc) is root
    assert _root_cause(ValueError("standalone")) is None


def test_describe_includes_timing_and_type():
    exc = TimeoutError("Request timed out or interrupted")
    msg = _describe_request_error("anthropic", exc, 30.02, 30.0)
    assert msg.startswith("anthropic request failed after 30.0s (timeout 30s)")
    assert "TimeoutError: Request timed out or interrupted" in msg


def test_describe_qualifies_module_and_status_and_request_id():
    class APITimeoutError(Exception):
        status_code = 408
        request_id = "req_abc123"

    APITimeoutError.__module__ = "anthropic._exceptions"
    exc = APITimeoutError("the request timed out")
    msg = _describe_request_error("anthropic", exc, 12.5, 30.0)
    assert "anthropic.APITimeoutError" in msg
    assert "the request timed out" in msg
    assert "status=408" in msg
    assert "request_id=req_abc123" in msg


def test_describe_surfaces_underlying_cause():
    try:
        try:
            raise OSError("Connection reset by peer")
        except OSError as inner:
            raise RuntimeError("Request timed out or interrupted") from inner
    except RuntimeError as exc:
        msg = _describe_request_error("openai", exc, 5.3, 30.0)
    assert "RuntimeError: Request timed out or interrupted" in msg
    assert "(cause: OSError: Connection reset by peer)" in msg


def test_is_retryable():
    assert _is_retryable(_FakeTimeout("timed out"))

    class _Status:
        status_code = 429

    assert _is_retryable(_Status())

    class _Bad:
        status_code = 400

    assert not _is_retryable(_Bad())
    assert not _is_retryable(ValueError("nope"))


def test_request_retries_then_succeeds():
    c = _client(max_retries=2)
    notes = []
    c.on_retry = notes.append
    c.timeout = 1.0

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FakeTimeout("Request timed out")
        return "ok"

    # Avoid real backoff sleeps.
    import ludvart.llm as llm

    saved_sleep = llm.time.sleep
    llm.time.sleep = lambda _s: None
    try:
        assert c._request(flaky) == "ok"
    finally:
        llm.time.sleep = saved_sleep
    assert calls["n"] == 3
    assert len(notes) == 2  # two retries reported
    assert "retrying 1/2" in notes[0]
    assert "retrying 2/2" in notes[1]


def test_request_gives_up_after_retries():
    c = _client(max_retries=1)

    import ludvart.llm as llm

    saved_sleep = llm.time.sleep
    llm.time.sleep = lambda _s: None
    try:
        raised = None
        try:
            c._request(lambda: (_ for _ in ()).throw(_FakeTimeout("timed out")))
        except LLMError as exc:
            raised = exc
    finally:
        llm.time.sleep = saved_sleep
    assert raised is not None
    assert "anthropic request failed" in str(raised)


def test_request_non_retryable_raises_immediately():
    c = _client(max_retries=3)
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("bad request")

    try:
        c._request(boom)
    except LLMError:
        pass
    assert calls["n"] == 1  # not retried


def test_resolve_settings():
    for var in ("LUDVART_LLM_TIMEOUT", "LUDVART_LLM_MAX_RETRIES"):
        os.environ.pop(var, None)
    assert _resolve_settings({}) == (DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES)
    # File values used as fallback.
    assert _resolve_settings(
        {"LUDVART_LLM_TIMEOUT": "120", "LUDVART_LLM_MAX_RETRIES": "5"}
    ) == (120.0, 5)
    # Env overrides file; junk falls back to defaults.
    os.environ["LUDVART_LLM_TIMEOUT"] = "90"
    try:
        timeout, retries = _resolve_settings(
            {"LUDVART_LLM_TIMEOUT": "120", "LUDVART_LLM_MAX_RETRIES": "junk"}
        )
    finally:
        del os.environ["LUDVART_LLM_TIMEOUT"]
    assert timeout == 90.0
    assert retries == DEFAULT_MAX_RETRIES


def test_is_rate_limit():
    class _RL(Exception):
        pass

    _RL.__name__ = "RateLimitError"
    assert _is_rate_limit(_RL("slow down")) is True

    class _Status(Exception):
        status_code = 429

    assert _is_rate_limit(_Status()) is True

    class _Other(Exception):
        status_code = 500

    assert _is_rate_limit(_Other()) is False
    assert _is_rate_limit(ValueError("nope")) is False


def test_retry_after_numeric_and_attribute_and_none():
    # Plain retry_after attribute (seconds).
    class _Attr(Exception):
        retry_after = 12

    assert _retry_after_seconds(_Attr()) == 12.0

    # Header on the response object (case-insensitive), clamped to <= 300.
    class _Resp:
        def __init__(self, headers):
            self.headers = headers

    class _WithHeaders(Exception):
        def __init__(self, headers):
            self.response = _Resp(headers)

    assert _retry_after_seconds(_WithHeaders({"retry-after": "20"})) == 20.0
    assert _retry_after_seconds(_WithHeaders({"Retry-After": "45"})) == 45.0
    assert _retry_after_seconds(_WithHeaders({"retry-after": "9999"})) == 300.0

    # No header / no attribute -> None.
    assert _retry_after_seconds(ValueError("x")) is None
    assert _retry_after_seconds(_WithHeaders({})) is None


def test_retry_after_http_date():
    import datetime
    from email.utils import format_datetime

    class _Resp:
        def __init__(self, headers):
            self.headers = headers

    class _WithHeaders(Exception):
        def __init__(self, headers):
            self.response = _Resp(headers)

    future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=30
    )
    secs = _retry_after_seconds(
        _WithHeaders({"retry-after": format_datetime(future)})
    )
    assert secs is not None
    # Allow scheduling slack, but it should be near 30s and non-negative.
    assert 20.0 <= secs <= 31.0

    past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=60
    )
    assert _retry_after_seconds(_WithHeaders({"retry-after": format_datetime(past)})) == 0.0


def test_request_honors_retry_after_and_reports_rate_limit():
    c = _client(max_retries=2)
    notes = []
    c.on_retry = notes.append

    class _Resp:
        headers = {"retry-after": "7"}

    class _RateLimit(Exception):
        status_code = 429
        response = _Resp()

    _RateLimit.__name__ = "RateLimitError"

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _RateLimit("too many requests")
        return "ok"

    import ludvart.llm as llm

    slept = []
    saved_sleep = llm.time.sleep
    llm.time.sleep = slept.append
    try:
        assert c._request(flaky) == "ok"
    finally:
        llm.time.sleep = saved_sleep

    # We honored Retry-After exactly (7s), not the exponential backoff.
    assert slept == [7.0]
    assert len(notes) == 1
    assert "rate limited" in notes[0]
    assert "HTTP 429" in notes[0]
    assert "Retry-After 7s" in notes[0]


def test_google_rate_limit_recognized_with_retry_info():
    # google-genai exposes the HTTP status as ``code`` (not ``status_code``)
    # and puts the wait in a JSON ``RetryInfo`` detail, not a header.
    class _GoogleError(Exception):
        def __init__(self, code, details):
            self.code = code
            self.details = details

    body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "quota exceeded",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "57.238747106s",
                },
            ],
        }
    }
    exc = _GoogleError(429, body)
    assert _is_retryable(exc) is True
    assert _is_rate_limit(exc) is True
    assert _retry_after_seconds(exc) == 57.238747106

    # A non-retryable google error (400) is not retried and has no wait.
    bad = _GoogleError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
    assert _is_retryable(bad) is False
    assert _retry_after_seconds(bad) is None


def main():
    test_root_cause_walks_chain()
    test_describe_includes_timing_and_type()
    test_describe_qualifies_module_and_status_and_request_id()
    test_describe_surfaces_underlying_cause()
    test_is_retryable()
    test_is_rate_limit()
    test_retry_after_numeric_and_attribute_and_none()
    test_retry_after_http_date()
    test_request_honors_retry_after_and_reports_rate_limit()
    test_google_rate_limit_recognized_with_retry_info()
    test_request_retries_then_succeeds()
    test_request_gives_up_after_retries()
    test_request_non_retryable_raises_immediately()
    test_resolve_settings()
    print("ok")


if __name__ == "__main__":
    main()
