"""Tests for streamed interim narration.

While the model produces a turn, relai streams its text into a transient,
dim "interim" line above the spinner (an indication of what it is doing), then
removes it once the final reply arrives. These tests cover:

  * the LLM ``converse(..., on_text=...)`` streaming hook (base + Anthropic),
  * the panel rendering the interim line while thinking, and
  * relai wiring: interim is refreshed per turn and cleared on completion.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relai.llm import (  # noqa: E402
    AnthropicClient,
    LLMClient,
    ProviderConfig,
    ToolCall,
    ToolSpec,
    Turn,
    usage_from_response,
)
from relai.panel import AiPanel  # noqa: E402
from relai.relai import Relai  # noqa: E402
from relai.session import SessionStore  # noqa: E402


# -- base converse feeds on_text -------------------------------------------


class _TextClient(LLMClient):
    def __init__(self):
        super().__init__(
            ProviderConfig(name="custom", api_url="u", api_key="k", model="m")
        )

    def complete(self, messages, max_tokens=1024):
        return "hello world"


def test_base_converse_calls_on_text():
    seen = []
    turn = _TextClient().converse(
        [{"role": "user", "content": "hi"}], on_text=seen.append
    )
    assert seen == ["hello world"], seen
    assert turn.text == "hello world", turn.text
    print("base converse calls on_text: OK")


# -- Anthropic streaming path ----------------------------------------------


class _FakeStream:
    """Mimics the anthropic SDK streaming context manager."""

    def __init__(self, deltas, final):
        self.text_stream = deltas
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


def test_anthropic_converse_streams_and_assembles():
    text_block = SimpleNamespace(type="text", text="Let me check")
    tool_block = SimpleNamespace(
        type="tool_use", id="t1", name="capture_screen_history", input={}
    )
    final = SimpleNamespace(
        content=[text_block, tool_block],
        usage=SimpleNamespace(input_tokens=10, output_tokens=3),
    )

    captured = {}

    def _stream(**kwargs):
        captured["kwargs"] = kwargs
        return _FakeStream(["Let me ", "check"], final)

    client = AnthropicClient(
        ProviderConfig(name="anthropic", api_url="http://x", api_key="k", model="m")
    )
    client._client = SimpleNamespace(messages=SimpleNamespace(stream=_stream))

    seen = []
    turn = client.converse(
        [{"role": "user", "content": "hi"}],
        tools=[ToolSpec(name="capture_screen_history", description="d", input_schema={})],
        on_text=seen.append,
    )

    # on_text receives the accumulated text as each delta arrives.
    assert seen == ["Let me ", "Let me check"], seen
    # The final Turn is assembled from the streamed final message.
    assert turn.text == "Let me check", turn.text
    assert len(turn.tool_calls) == 1, turn.tool_calls
    assert turn.tool_calls[0].name == "capture_screen_history"
    assert turn.usage is not None and turn.usage.input_tokens == 10
    # The tools were forwarded to the streaming request.
    assert "tools" in captured["kwargs"], captured["kwargs"]
    print("anthropic converse streams and assembles: OK")


def test_anthropic_non_stream_unchanged():
    """Without on_text, converse still uses the plain create() path."""
    text_block = SimpleNamespace(type="text", text="plain")
    resp = SimpleNamespace(
        content=[text_block],
        usage=SimpleNamespace(input_tokens=5, output_tokens=1),
    )
    client = AnthropicClient(
        ProviderConfig(name="anthropic", api_url="http://x", api_key="k", model="m")
    )
    used = {}

    def _create(**kwargs):
        used["called"] = True
        return resp

    client._client = SimpleNamespace(messages=SimpleNamespace(create=_create))
    turn = client.converse([{"role": "user", "content": "hi"}])
    assert used.get("called") is True
    assert turn.text == "plain", turn.text
    print("anthropic non-stream path unchanged: OK")


# -- panel renders the interim line ----------------------------------------


def test_panel_renders_interim_line():
    panel = AiPanel(cols=40, height=8, provider="fake")
    panel.thinking = True
    panel.interim = "Reading the screen now"
    blob = b"".join(panel.render(8, 40))
    assert b"Reading the screen now" in blob, blob
    # The spinner label is still shown (below the interim line).
    assert b"Thinking" in blob, blob
    # When cleared, the narration is gone.
    panel.interim = ""
    blob = b"".join(panel.render(8, 40))
    assert b"Reading the screen now" not in blob
    print("panel renders interim line: OK")


# -- relai wiring: interim refreshed per turn, cleared on finish -----------


class _StreamingLLM:
    name = "fake"
    model = "m"
    context_window = 1000

    def __init__(self, relai):
        self.on_retry = None
        self._relai = relai
        self.turn = 0
        self.interim_at_entry = []

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        self.turn += 1
        # Record the interim value the harness left when this turn begins; the
        # agent loop must reset it to "" before every turn.
        self.interim_at_entry.append(self._relai._panel.interim)
        if self.turn == 1:
            if on_text:
                on_text("step one")
                on_text("step one narration")
            return Turn(
                text="step one narration",
                tool_calls=[ToolCall(id="t1", name="b64_encode", input={"text": "hi"})],
                assistant_message={"role": "assistant", "content": "..."},
                usage=None,
            )
        if on_text:
            on_text("final")
            on_text("final answer is 42")
        return Turn(
            text="final answer is 42",
            assistant_message={"role": "assistant", "content": "final answer is 42"},
            usage=None,
        )

    def tool_result_message(self, tool_call_id, content):
        return {"role": "user", "content": content}


def _make_relai(root: Path) -> Relai:
    os.environ["RELAI_SESSIONS_DIR"] = str(root)
    r = Relai(["true"])
    r._panel = AiPanel(cols=80, height=8, provider="fake")
    r._phys_rows, r._phys_cols = 24, 80
    r._render_split = lambda: None
    r._session = SessionStore()
    r.snapshot_text = lambda: "SCREEN"
    r.llm = _StreamingLLM(r)
    return r


def test_relai_streams_and_clears_interim(tmp_path: Path):
    r = _make_relai(tmp_path)
    panel = r._panel

    result = r._ask_llm("what is on screen?")
    assert result == "final answer is 42", result

    # The first turn begins with a cleared interim; the second begins with the
    # running narration from the first turn -- its streamed reasoning AND its
    # tool-call note -- so nothing vanishes between tool round-trips.
    assert r.llm.interim_at_entry == [
        "",
        "step one narration\n\u2192 b64_encode(text='hi')",
    ], r.llm.interim_at_entry
    # After the last streamed turn, interim shows the full history above the
    # final streamed narration...
    assert panel.interim == (
        "step one narration\n\u2192 b64_encode(text='hi')\nfinal answer is 42"
    ), panel.interim

    # ...until the turn is delivered, which removes the transient narration.
    r._ask_result = result
    r._deliver = r._deliver_reply
    r._ask_thread = None
    r._finish_ask()
    assert panel.interim == "", panel.interim
    assert panel.messages[-1] == ("relai", "final answer is 42"), panel.messages[-1]
    print("relai streams and clears interim: OK")


def main():
    import tempfile

    test_base_converse_calls_on_text()
    test_anthropic_converse_streams_and_assembles()
    test_anthropic_non_stream_unchanged()
    test_panel_renders_interim_line()
    with tempfile.TemporaryDirectory() as d:
        test_relai_streams_and_clears_interim(Path(d))
    print("ALL streaming/interim tests passed.")


if __name__ == "__main__":
    main()
