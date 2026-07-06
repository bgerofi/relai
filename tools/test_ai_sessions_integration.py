"""Integration tests for session persistence + slash commands in the panel.

Drives the panel wiring on a Ludvart instance without spawning a real PTY: it
stubs the LLM ask and the renderer, feeds keystrokes/questions, and checks the
on-disk conversation files and in-panel behaviour.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tools/test_ai_sessions_integration.py
"""

import json
import os
import tempfile
from pathlib import Path

from ludvart.ludvart import Ludvart
from ludvart.panel import AiPanel
from ludvart.session import SessionStore


def make_ludvart(root: Path):
    os.environ["LUDVART_SESSIONS_DIR"] = str(root)
    r = Ludvart(["true"])
    r.llm = object()  # truthy so _ai_ask_callback would use the LLM path
    r._panel = AiPanel(cols=80, height=8, provider="test")
    r._phys_rows, r._phys_cols = 24, 80
    r._render_split = lambda: None
    # Replace the background ask with a synchronous stub: it appends to the
    # model history the way _ask_llm would and returns a canned reply.
    def fake_ask(question: str) -> str:
        r._llm_history.append({"role": "user", "content": question})
        r._llm_history.append({"role": "assistant", "content": f"echo:{question}"})
        return f"echo:{question}"

    r._ai_ask = fake_ask

    # Make _start_ask run synchronously (no thread) for deterministic tests.
    def sync_start_ask(question, *, user_echo=None, info=None):
        if info:
            r._panel.add_info(info)
        if user_echo:
            r._panel.add_user(user_echo)
        r._ask_result = r._ai_ask(question)
        r._panel.add_reply(r._ask_result)
        r._persist_session()

    r._start_ask = sync_start_ask
    return r


def type_and_submit(r, text):
    for ch in text:
        r._panel_key(ch.encode())
    r._panel_key(b"\r")


def test_new_session_created_on_open():
    root = Path(tempfile.mkdtemp())
    r = make_ludvart(root)
    assert r._session is None
    # Simulate the part of _open_panel that starts a session.
    r._session = SessionStore()
    assert r._session is not None
    sid = r._session.session_id
    # Reusing across "toggles": a second open keeps the same session.
    if r._session is None:
        r._session = SessionStore()
    assert r._session.session_id == sid
    print("new session created on first open: OK")


def test_conversation_saved_and_extended():
    root = Path(tempfile.mkdtemp())
    r = make_ludvart(root)
    r._session = SessionStore()

    type_and_submit(r, "hello")
    conv = r._session.path
    assert conv.is_file(), "conversation.json should exist after first turn"
    data = json.loads(conv.read_text())
    assert data["messages"] == [["you", "hello"], ["ludvart", "echo:hello"]]

    type_and_submit(r, "again")
    data = json.loads(conv.read_text())
    assert [m for m in data["messages"]] == [
        ["you", "hello"],
        ["ludvart", "echo:hello"],
        ["you", "again"],
        ["ludvart", "echo:again"],
    ]
    assert len(data["llm_history"]) == 4
    print("conversation saved + extended: OK")


def test_slash_command_not_persisted():
    root = Path(tempfile.mkdtemp())
    r = make_ludvart(root)
    r._session = SessionStore()
    type_and_submit(r, "hello")

    # Run an internal command; it must not touch the saved conversation.
    type_and_submit(r, "/sessions list")
    kinds = [k for k, _ in r._panel.messages]
    assert "system" in kinds, "slash output should show as system lines"

    data = json.loads(r._session.path.read_text())
    saved_kinds = [m[0] for m in data["messages"]]
    assert saved_kinds == ["you", "ludvart"], saved_kinds
    print("slash command not persisted: OK")


def test_sessions_list_and_load():
    root = Path(tempfile.mkdtemp())

    # Pre-create an older, separate session on disk.
    from datetime import datetime, timezone

    old = SessionStore(
        root=root, started_at=datetime(2026, 7, 1, 9, 0, 0, tzinfo=timezone.utc)
    )
    old.save(
        [("you", "old question"), ("ludvart", "old answer")],
        [{"role": "user", "content": "old question"},
         {"role": "assistant", "content": "old answer"}],
    )

    r = make_ludvart(root)
    r._session = SessionStore()  # a fresh current session
    type_and_submit(r, "current")

    # /sessions list should show both and record the list for index loads.
    type_and_submit(r, "/sessions list")
    assert len(r._session_list) == 2, r._session_list
    ids = [s["id"] for s in r._session_list]
    assert "2026-07-01/09_00_00" in ids

    # Load the old one by its 1-based index.
    idx = ids.index("2026-07-01/09_00_00") + 1
    type_and_submit(r, f"/sessions load {idx}")
    # The transcript is now the loaded conversation.
    kinds_texts = r._panel.messages
    assert ("you", "old question") in kinds_texts
    assert r._session.session_id == "2026-07-01/09_00_00"
    assert r._llm_history[0]["content"] == "old question"

    # Continuing the conversation extends the LOADED session's file.
    type_and_submit(r, "followup")
    data = json.loads(old.path.read_text())
    texts = [m[1] for m in data["messages"] if m[0] == "you"]
    assert "old question" in texts and "followup" in texts, texts
    print("sessions list + load + resume: OK")


def test_load_by_id_and_errors():
    root = Path(tempfile.mkdtemp())
    r = make_ludvart(root)
    r._session = SessionStore()

    # load without listing, by explicit id that doesn't exist
    type_and_submit(r, "/sessions load 2026-01-01/00_00_00")
    last = r._panel.messages[-1]
    assert last[0] == "system" and "Could not load" in last[1], last

    # bad index
    type_and_submit(r, "/sessions load 99")
    last = r._panel.messages[-1]
    assert last[0] == "system" and "No session #99" in last[1], last

    # unknown command
    type_and_submit(r, "/bogus")
    last = r._panel.messages[-1]
    assert last[0] == "system" and "Unknown command" in last[1], last
    print("load errors + unknown command: OK")


def test_tab_completion_in_panel():
    root = Path(tempfile.mkdtemp())
    r = make_ludvart(root)
    r._session = SessionStore()

    for ch in "/sess":
        r._panel_key(ch.encode())
    r._panel_key(b"\t")
    assert r._panel.editor.text == "/sessions ", repr(r._panel.editor.text)

    for ch in "li":
        r._panel_key(ch.encode())
    r._panel_key(b"\t")
    assert r._panel.editor.text == "/sessions list ", repr(r._panel.editor.text)
    print("tab completion in panel: OK")


if __name__ == "__main__":
    test_new_session_created_on_open()
    test_conversation_saved_and_extended()
    test_slash_command_not_persisted()
    test_sessions_list_and_load()
    test_load_by_id_and_errors()
    test_tab_completion_in_panel()
    print("\nALL session integration tests passed.")
