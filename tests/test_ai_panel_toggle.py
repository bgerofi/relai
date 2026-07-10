"""Panel toggle behaviour: draft survival and in-flight request confirmation.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_panel_toggle.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.panel import AiPanel  # noqa: E402
from ludvart.ludvart import Ludvart  # noqa: E402


def _make_ludvart():
    r = Ludvart(["true"])
    r._panel = AiPanel(cols=80, height=10, provider="openai")
    r._render_split = lambda: None
    return r


def _systems(r):
    return [t for kind, t in r._panel.messages if kind == "system"]


def test_draft_survives_toggle():
    r = _make_ludvart()
    r._panel.editor.insert("half typed question")
    r._panel.editor.left()
    r._panel.editor.left()
    saved_cursor = r._panel.editor.cursor
    # Simulate closing (leave_split saves) then re-opening (open restores).
    r._save_panel_draft()
    r._panel = AiPanel(cols=80, height=10, provider="openai")
    r._restore_panel_draft()
    assert r._panel.editor.text == "half typed question"
    assert r._panel.editor.cursor == saved_cursor
    print("draft text + cursor survive a toggle: OK")


def test_restore_clamps_cursor():
    r = _make_ludvart()
    r._panel_draft = "abc"
    r._panel_draft_cursor = 999  # stale/oversized cursor is clamped
    r._restore_panel_draft()
    assert r._panel.editor.text == "abc"
    assert r._panel.editor.cursor == 3
    print("restore clamps an oversized cursor: OK")


def test_toggle_closes_immediately_when_idle():
    r = _make_ludvart()
    r._panel.thinking = False
    r._request_toggle_close()
    assert r._panel_closing is True
    assert r._confirm_close is False
    print("idle panel toggles closed without a prompt: OK")


def test_toggle_prompts_during_llm_request():
    r = _make_ludvart()
    r._panel.thinking = True
    r._deliver = r._deliver_reply  # an LLM ask is in flight
    r._request_toggle_close()
    assert r._panel_closing is False
    assert r._confirm_close is True
    assert r._panel.confirm_prompt == (
        "LLM request in progress, cancel request and toggle panel? (y/n)"
    )
    print("in-flight LLM request prompts before closing: OK")


def test_toggle_no_prompt_for_deterministic_action():
    r = _make_ludvart()
    r._panel.thinking = True
    r._deliver = r._deliver_system  # deterministic action, not an LLM ask
    r._request_toggle_close()
    assert r._panel_closing is True
    assert r._confirm_close is False
    print("deterministic action closes without a prompt: OK")


def test_confirm_yes_cancels_and_closes():
    r = _make_ludvart()
    r._panel.thinking = True
    r._deliver = r._deliver_reply
    r._request_toggle_close()
    # Answer 'y' via the input router (confirm state intercepts the keystroke).
    r._panel_input(b"y")
    assert r._panel_closing is True
    assert r._confirm_close is False
    assert r._panel.confirm_prompt == ""
    assert r._ask_cancel.is_set()
    assert r._panel.thinking is False
    print("'y' cancels the request and closes: OK")


def test_confirm_no_keeps_panel_open():
    r = _make_ludvart()
    r._panel.thinking = True
    r._deliver = r._deliver_reply
    r._request_toggle_close()
    r._panel_input(b"n")
    assert r._panel_closing is False
    assert r._confirm_close is False
    assert r._panel.confirm_prompt == ""
    assert not r._ask_cancel.is_set()
    assert r._panel.thinking is True  # request still running
    print("'n' keeps the panel open, request untouched: OK")


def test_confirm_ignores_unrelated_keys():
    r = _make_ludvart()
    r._panel.thinking = True
    r._deliver = r._deliver_reply
    r._request_toggle_close()
    r._panel_input(b"x")  # not y/n -> stays pending
    assert r._confirm_close is True
    assert r._panel_closing is False
    # 'x' must not leak into the input editor.
    assert r._panel.editor.text == ""
    print("confirm ignores unrelated keys and swallows them: OK")


def test_confirm_prompt_renders_on_bottom_input_line():
    r = _make_ludvart()
    r._panel.editor.insert("draft not yet sent")
    r._panel.thinking = True
    r._deliver = r._deliver_reply
    r._request_toggle_close()
    rows = r._panel.render(height=8, cols=80)
    bottom = rows[-1].decode("utf-8", "replace")
    assert "cancel request and toggle panel? (y/n)" in bottom
    # The normal prompt / typed draft must not appear on the input line.
    assert "ludvart>" not in bottom
    assert "draft not yet sent" not in bottom
    print("confirm question renders on the bottom input line: OK")


def main():
    test_draft_survives_toggle()
    test_restore_clamps_cursor()
    test_toggle_closes_immediately_when_idle()
    test_toggle_prompts_during_llm_request()
    test_toggle_no_prompt_for_deterministic_action()
    test_confirm_yes_cancels_and_closes()
    test_confirm_no_keeps_panel_open()
    test_confirm_ignores_unrelated_keys()
    test_confirm_prompt_renders_on_bottom_input_line()
    print("\nALL panel toggle tests passed.")


if __name__ == "__main__":
    main()
