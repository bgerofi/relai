"""Unit test: the /init_helpers panel command triggers helper initialization.

Also asserts the old first-open auto-initialization has been removed.

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_ai_init_helpers.py
"""

from relai.relai import Relai, HELPERS_INIT_PROMPT
from relai.panel import AiPanel


def make_relai(with_llm: bool = True):
    r = Relai(["true"])
    r.llm = object() if with_llm else None  # truthy stub; only presence matters
    r._panel = AiPanel(cols=80, height=8, provider="test")
    r._phys_rows, r._phys_cols = 24, 80
    r._render_split = lambda: None
    # Capture _start_ask instead of spawning a thread.
    calls: list = []
    r._start_ask = lambda q, **kw: calls.append((q, kw))
    return r, calls


def submit(r, text: str) -> None:
    for ch in text:
        r._panel_key(ch.encode())
    r._panel_key(b"\r")


def test_init_helpers_command_starts_turn():
    r, calls = make_relai(with_llm=True)
    submit(r, "/init_helpers")
    assert len(calls) == 1, calls
    question, kwargs = calls[0]
    assert question == HELPERS_INIT_PROMPT
    assert "info" in kwargs and kwargs["info"]
    # The command echo is an ephemeral system line (not persisted, not a user turn).
    kinds = [k for k, _ in r._panel.messages]
    assert kinds and kinds[0] == "system", kinds
    print("/init_helpers starts an agent turn: OK")


def test_init_helpers_without_llm():
    r, calls = make_relai(with_llm=False)
    submit(r, "/init_helpers")
    assert calls == [], "no agent turn should start without an LLM"
    last = r._panel.messages[-1]
    assert last[0] == "system" and "No LLM provider" in last[1], last
    print("/init_helpers without LLM: OK")


def test_autoinit_removed():
    r, _ = make_relai()
    # The first-open auto-initialization machinery must be gone.
    assert not hasattr(r, "_helpers_init_attempted"), "auto-init flag should be gone"
    assert not hasattr(r, "_maybe_init_helpers"), "_maybe_init_helpers should be gone"
    assert not hasattr(r, "_looks_like_shell"), "_looks_like_shell should be gone"
    print("auto-init removed: OK")


def test_tab_completion():
    r, _ = make_relai()
    for ch in "/init":
        r._panel_key(ch.encode())
    r._panel_key(b"\t")
    assert r._panel.editor.text == "/init_helpers ", repr(r._panel.editor.text)
    print("/init tab completion: OK")


if __name__ == "__main__":
    test_init_helpers_command_starts_turn()
    test_init_helpers_without_llm()
    test_autoinit_removed()
    test_tab_completion()
    print("all init-helpers tests passed")
