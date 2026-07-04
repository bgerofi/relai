"""Unit tests for the enriched LLM request-failure diagnostics.

Verifies that a failed provider request is reported with the exception type, the
elapsed time versus the configured timeout, any HTTP status / request-id the SDK
exposes, and the underlying cause of the exception chain. Also covers the retry
runner (transient failures retried and reported) and settings resolution from
env / ~/.relai/llm.conf.

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_llm_error_detail.py
"""

import os

from relai.llm import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    LLMClient,
    LLMError,
    ProviderConfig,
    _describe_request_error,
    _is_retryable,
    _resolve_settings,
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
    import relai.llm as llm

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

    import relai.llm as llm

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
    for var in ("RELAI_LLM_TIMEOUT", "RELAI_LLM_MAX_RETRIES"):
        os.environ.pop(var, None)
    assert _resolve_settings({}) == (DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES)
    # File values used as fallback.
    assert _resolve_settings(
        {"RELAI_LLM_TIMEOUT": "120", "RELAI_LLM_MAX_RETRIES": "5"}
    ) == (120.0, 5)
    # Env overrides file; junk falls back to defaults.
    os.environ["RELAI_LLM_TIMEOUT"] = "90"
    try:
        timeout, retries = _resolve_settings(
            {"RELAI_LLM_TIMEOUT": "120", "RELAI_LLM_MAX_RETRIES": "junk"}
        )
    finally:
        del os.environ["RELAI_LLM_TIMEOUT"]
    assert timeout == 90.0
    assert retries == DEFAULT_MAX_RETRIES


def main():
    test_root_cause_walks_chain()
    test_describe_includes_timing_and_type()
    test_describe_qualifies_module_and_status_and_request_id()
    test_describe_surfaces_underlying_cause()
    test_is_retryable()
    test_request_retries_then_succeeds()
    test_request_gives_up_after_retries()
    test_request_non_retryable_raises_immediately()
    test_resolve_settings()
    print("ok")


if __name__ == "__main__":
    main()
