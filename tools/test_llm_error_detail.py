"""Unit tests for the enriched LLM request-failure diagnostics.

Verifies that a failed provider request is reported with the exception type, the
elapsed time versus the configured timeout, any HTTP status / request-id the SDK
exposes, and the underlying cause of the exception chain.

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_llm_error_detail.py
"""

from relai.llm import _describe_request_error, _root_cause


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


def main():
    test_root_cause_walks_chain()
    test_describe_includes_timing_and_type()
    test_describe_qualifies_module_and_status_and_request_id()
    test_describe_surfaces_underlying_cause()
    print("ok")


if __name__ == "__main__":
    main()
