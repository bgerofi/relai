"""Unit tests for context-window auto-detection (llm.py).

Builds real provider clients (offline, dummy key) and swaps in a fake SDK
client so the models endpoint can be simulated without network access.

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_context_window_detect.py
"""

from relai.llm import (
    DEFAULT_TIMEOUT,
    ProviderConfig,
    _client_for,
    _first_positive_int,
    _known_context_window,
)


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeModels:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def retrieve(self, model):
        if self._exc:
            raise self._exc
        return self._result

    def get(self, model=None):
        if self._exc:
            raise self._exc
        return self._result


class FakeSDK:
    def __init__(self, models):
        self.models = models


def build(name, model, context_window=0):
    cfg = ProviderConfig(
        name=name,
        api_url="http://localhost:1",
        api_key="x",
        model=model,
        context_window=context_window,
    )
    return _client_for(cfg, DEFAULT_TIMEOUT, 0)


def test_known_table():
    assert _known_context_window("claude-opus-4-6") == 200_000
    assert _known_context_window("gpt-4o-mini") == 128_000
    assert _known_context_window("gpt-4-turbo-2024") == 128_000
    assert _known_context_window("gpt-4") == 8_192
    assert _known_context_window("gpt-3.5-turbo") == 16_385
    assert _known_context_window("gemini-1.5-pro-latest") == 2_097_152
    assert _known_context_window("gemini-2.0-flash") == 1_048_576
    assert _known_context_window("o3-mini") == 200_000
    assert _known_context_window("some-unknown-model") == 0
    assert _known_context_window("") == 0
    print("known-model table: OK")


def test_first_positive_int():
    assert _first_positive_int(Obj(max_model_len=4096), "max_model_len") == 4096
    # falls back to model_extra dict
    assert _first_positive_int(
        Obj(model_extra={"context_length": 8192}), "context_length"
    ) == 8192
    # order: first present positive wins
    assert _first_positive_int(Obj(a=0, b=32000), "a", "b") == 32000
    # booleans and non-positives are ignored
    assert _first_positive_int(Obj(a=True, b=-5, c=0), "a", "b", "c") == 0
    assert _first_positive_int(Obj(), "missing") == 0
    print("_first_positive_int: OK")


def test_property_precedence():
    # 1. explicit env override always wins
    c = build("openai", "gpt-4o", context_window=5000)
    c._client = FakeSDK(FakeModels(result=Obj(max_model_len=999999)))
    c._detected_context_window = 123456
    assert c.context_window == 5000

    # 2. detected value beats the fallback table
    c = build("openai", "gpt-4o")
    c._detected_context_window = 7000
    assert c.context_window == 7000

    # 3. fallback table when nothing else is known
    c = build("anthropic", "claude-opus-4-6")
    assert c._detected_context_window == 0
    assert c.context_window == 200_000

    # 4. truly unknown -> 0 (badge hidden)
    c = build("openai", "mystery-x")
    assert c.context_window == 0
    print("context_window precedence: OK")


def test_openai_detect():
    c = build("openai", "local-model")
    c._client = FakeSDK(FakeModels(result=Obj(max_model_len=40960)))
    assert c.detect_context_window() == 40960
    # no recognizable field -> 0
    c._client = FakeSDK(FakeModels(result=Obj(id="local-model")))
    assert c.detect_context_window() == 0
    # API error -> 0 (never raises)
    c._client = FakeSDK(FakeModels(exc=RuntimeError("boom")))
    assert c.detect_context_window() == 0
    print("openai detect: OK")


def test_anthropic_detect():
    c = build("anthropic", "claude-opus-4-6")
    c._client = FakeSDK(FakeModels(result=Obj(max_input_tokens=200_000, max_tokens=64000)))
    assert c.detect_context_window() == 200_000
    # placeholder 0 from API -> 0 (property will then use the table)
    c._client = FakeSDK(FakeModels(result=Obj(max_input_tokens=0)))
    assert c.detect_context_window() == 0
    assert c.context_window == 200_000  # table fallback
    print("anthropic detect: OK")


def test_google_detect():
    c = build("google", "gemini-2.0-flash")
    c._client = FakeSDK(FakeModels(result=Obj(input_token_limit=1_048_576, output_token_limit=8192)))
    assert c.detect_context_window() == 1_048_576
    c._client = FakeSDK(FakeModels(exc=ValueError("nope")))
    assert c.detect_context_window() == 0
    print("google detect: OK")


def test_verify_triggers_detection():
    c = build("anthropic", "claude-opus-4-6")
    c.complete = lambda *a, **k: "ok"  # no network
    c.detect_context_window = lambda: 314159
    c.verify()
    assert c._detected_context_window == 314159
    assert c.context_window == 314159

    # When pinned via env, verify must NOT auto-detect/override.
    c2 = build("anthropic", "claude-opus-4-6", context_window=111)
    c2.complete = lambda *a, **k: "ok"
    c2.detect_context_window = lambda: 314159
    c2.verify()
    assert c2._detected_context_window == 0
    assert c2.context_window == 111

    # Detection failure during verify is swallowed.
    c3 = build("openai", "mystery-x")
    c3.complete = lambda *a, **k: "ok"
    def boom():
        raise RuntimeError("x")
    c3.detect_context_window = boom
    c3.verify()  # must not raise
    assert c3._detected_context_window == 0
    print("verify triggers detection: OK")


if __name__ == "__main__":
    test_known_table()
    test_first_positive_int()
    test_property_precedence()
    test_openai_detect()
    test_anthropic_detect()
    test_google_detect()
    test_verify_triggers_detection()
    print("\nALL context-window detection tests passed.")
