"""Live activity progress: elapsed-time hint on the spinner during waits.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_activity_progress.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.panel import AiPanel  # noqa: E402
from ludvart.ludvart import Ludvart  # noqa: E402


def _make_ludvart():
    runner = Ludvart(["true"])
    runner._panel = AiPanel(cols=80, height=10, provider="openai")
    runner._render_split = lambda: None
    return runner


# -- panel rendering --------------------------------------------------------


def test_spinner_shows_elapsed_when_set():
    panel = AiPanel(cols=60, height=8, provider="openai")
    panel.thinking = True
    panel.activity = "Thinking"
    panel.activity_elapsed = 8.0
    blob = b"".join(panel.render(8, 60))
    assert b"Thinking (openai)" in blob, blob
    assert b"Thinking (openai) - 8s" in blob, blob
    print("spinner shows provider and elapsed seconds: OK")


def test_spinner_hides_elapsed_when_none():
    panel = AiPanel(cols=60, height=8, provider="openai")
    panel.thinking = True
    panel.activity = "Thinking"
    panel.activity_elapsed = None
    blob = b"".join(panel.render(8, 60))
    assert b"Thinking (openai)" in blob, blob
    assert b"Thinking (openai) - " not in blob, blob
    print("spinner hides elapsed when unset: OK")


def test_spinner_elapsed_on_tool_label():
    panel = AiPanel(cols=60, height=8, provider="openai")
    panel.thinking = True
    panel.activity = "Calling inject_input"
    panel.activity_elapsed = 12.0
    blob = b"".join(panel.render(8, 60))
    assert b"Calling inject_input" in blob, blob
    assert b"Calling inject_input - 12s" in blob, blob
    print("spinner shows elapsed for a running tool: OK")


# -- controller wait lifecycle ---------------------------------------------


def test_begin_wait_sets_clock_and_label():
    runner = _make_ludvart()
    runner._begin_wait("Thinking")
    assert runner._panel.activity == "Thinking"
    assert runner._panel.activity_elapsed is None
    assert runner._wait_since is not None
    assert runner._wait_streaming is False
    print("begin_wait sets the label and starts the clock: OK")


def test_refresh_wait_below_threshold_hides_elapsed():
    runner = _make_ludvart()
    runner._panel.thinking = True
    runner._begin_wait("Thinking")
    runner._refresh_wait()
    assert runner._panel.activity_elapsed is None
    print("refresh keeps elapsed hidden below the threshold: OK")


def test_refresh_wait_past_threshold_shows_elapsed():
    runner = _make_ludvart()
    runner._panel.thinking = True
    runner._begin_wait("Thinking")
    runner._wait_since = time.monotonic() - 5.0
    runner._refresh_wait()
    assert runner._panel.activity_elapsed is not None
    assert runner._panel.activity_elapsed >= runner.ACTIVITY_ELAPSED_HINT
    print("refresh reveals elapsed once the threshold passes: OK")


def test_streaming_suppresses_elapsed():
    runner = _make_ludvart()
    runner._panel.thinking = True
    runner._begin_wait("Thinking")
    runner._wait_since = time.monotonic() - 5.0
    runner._mark_wait_streaming()
    runner._refresh_wait()
    assert runner._wait_streaming is True
    assert runner._panel.activity_elapsed is None
    print("streaming output suppresses the elapsed counter: OK")


def test_end_wait_clears_clock_and_elapsed():
    runner = _make_ludvart()
    runner._panel.thinking = True
    runner._begin_wait("Thinking")
    runner._wait_since = time.monotonic() - 5.0
    runner._refresh_wait()
    assert runner._panel.activity_elapsed is not None
    runner._end_wait()
    assert runner._wait_since is None
    assert runner._panel.activity_elapsed is None
    print("end_wait stops and hides the elapsed counter: OK")


def test_cancel_ask_clears_wait():
    runner = _make_ludvart()
    runner._panel.thinking = True
    runner._llm_request_in_flight = True
    runner._begin_wait("Calling inject_input")
    runner._wait_since = time.monotonic() - 5.0
    runner._refresh_wait()
    assert runner._panel.activity_elapsed is not None
    runner._cancel_ask()
    assert runner._wait_since is None
    assert runner._panel.activity_elapsed is None
    assert runner._panel.thinking is False
    print("cancelling a request clears the wait indicator: OK")


def main():
    test_spinner_shows_elapsed_when_set()
    test_spinner_hides_elapsed_when_none()
    test_spinner_elapsed_on_tool_label()
    test_begin_wait_sets_clock_and_label()
    test_refresh_wait_below_threshold_hides_elapsed()
    test_refresh_wait_past_threshold_shows_elapsed()
    test_streaming_suppresses_elapsed()
    test_end_wait_clears_clock_and_elapsed()
    test_cancel_ask_clears_wait()
    print("\nALL activity progress tests passed.")


if __name__ == "__main__":
    main()
