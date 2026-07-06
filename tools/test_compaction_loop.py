"""Regression tests for context auto-compaction inside the agent loop.

Two bugs are covered:

  * Compaction used to run only once per user ask (at the top of ``_ask_llm``),
    so a single agentic turn that made many tool round-trips could grow the
    context past the window without ever compacting again. It must now compact
    before every request inside the loop.
  * ``Usage.context_percent`` used to clamp at 100%, hiding a real overshoot.
    It must now report the true value so the badge and the compaction trigger
    see how far over budget the context is.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.llm import ToolCall, Turn, Usage  # noqa: E402
from ludvart.panel import AiPanel  # noqa: E402
from ludvart.ludvart import Ludvart  # noqa: E402
from ludvart.session import SessionStore  # noqa: E402

SUMMARY_TEXT = "COMPACTED-BRIEF: continue the task."


def test_context_percent_reports_overshoot():
    u = Usage(input_tokens=12000, output_tokens=0, total_tokens=12000,
              context_window=8000)
    assert u.context_percent() == 150.0, u.context_percent()
    print("context_percent reports overshoot: OK")


class _ToolLoopLLM:
    """Drives a multi-tool agentic ask whose context keeps growing.

    Each non-summary turn reports a rising context percentage and requests a
    tool call, until ``turns_before_answer`` is reached; then it answers. A
    summarization request (the compaction instruction) returns a fixed brief.
    """

    name = "fake"
    model = "m"
    context_window = 1000

    def __init__(self, pcts):
        self.on_retry = None
        self._pcts = list(pcts)
        self.calls = 0
        self.summarize_calls = 0
        self.history_lens_at_call = []

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        last = messages[-1]["content"] if messages else ""
        if isinstance(last, str) and "Summarize the ENTIRE" in last:
            self.summarize_calls += 1
            return Turn(
                text=SUMMARY_TEXT,
                assistant_message={"role": "assistant", "content": SUMMARY_TEXT},
                usage=None,
            )
        self.history_lens_at_call.append(len(messages))
        idx = self.calls
        self.calls += 1
        pct = self._pcts[min(idx, len(self._pcts) - 1)]
        usage = Usage(
            input_tokens=int(self.context_window * pct / 100.0),
            output_tokens=1,
            total_tokens=1,
            context_window=self.context_window,
        )
        if idx < len(self._pcts) - 1:
            return Turn(
                text="working",
                tool_calls=[ToolCall(id=f"t{idx}", name="b64_encode",
                                     input={"text": "x" * 50})],
                assistant_message={"role": "assistant", "content": "working"},
                usage=usage,
            )
        return Turn(
            text="final answer",
            assistant_message={"role": "assistant", "content": "final answer"},
            usage=usage,
        )

    def tool_result_message(self, tool_call_id, content):
        return {"role": "user", "content": content}


def _make_ludvart(root: Path) -> Ludvart:
    os.environ["LUDVART_SESSIONS_DIR"] = str(root)
    r = Ludvart(["true"])
    r._panel = AiPanel(cols=80, height=8, provider="fake")
    r._phys_rows, r._phys_cols = 24, 80
    r._render_split = lambda: None
    r._session = SessionStore()
    r.snapshot_text = lambda: "SCREEN-SNAPSHOT"
    return r


def test_compacts_inside_agent_loop(tmp_path: Path):
    r = _make_ludvart(tmp_path)
    # A long agentic ask: several tool round-trips whose reported context usage
    # climbs above the 80% threshold mid-loop, then a final answer.
    r.llm = _ToolLoopLLM([30.0, 60.0, 92.0, 40.0])

    result = r._ask_llm("do a multi-step task")
    assert result == "final answer", result

    # It must have compacted mid-loop (not zero, not stuck) when usage crossed
    # the threshold during the single ask.
    assert r.llm.summarize_calls >= 1, r.llm.summarize_calls
    # A compaction marker is in the visible transcript.
    kinds = [k for k, _ in r._panel.messages]
    assert "summary" in kinds, kinds
    # After compaction the model-facing history was reseeded from the summary,
    # so it is small -- not the full unbounded transcript.
    assert len(r._llm_history) <= 6, len(r._llm_history)
    print("compacts inside agent loop: OK")


def test_no_compaction_when_under_threshold(tmp_path: Path):
    r = _make_ludvart(tmp_path)
    r.llm = _ToolLoopLLM([20.0, 30.0, 25.0])
    result = r._ask_llm("short task")
    assert result == "final answer", result
    assert r.llm.summarize_calls == 0, r.llm.summarize_calls
    kinds = [k for k, _ in r._panel.messages]
    assert "summary" not in kinds, kinds
    print("no compaction when under threshold: OK")


def test_badge_reflects_overshoot(tmp_path: Path):
    # Even with a single turn over budget, the panel badge shows the true >100%.
    r = _make_ludvart(tmp_path)
    r.llm = _ToolLoopLLM([135.0])  # single turn, no tools, 135% usage
    # Only one entry -> it is the final answer immediately; usage set on panel.
    r._ask_llm("q")
    assert r._panel.context_pct is not None
    assert r._panel.context_pct == 135.0, r._panel.context_pct
    assert r._panel._prompt_prefix() == "[135%] ", r._panel._prompt_prefix()
    print("badge reflects overshoot: OK")


def main():
    test_context_percent_reports_overshoot()
    with tempfile.TemporaryDirectory() as d:
        test_compacts_inside_agent_loop(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_no_compaction_when_under_threshold(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_badge_reflects_overshoot(Path(d))
    print("ALL compaction-loop tests passed.")


if __name__ == "__main__":
    main()
