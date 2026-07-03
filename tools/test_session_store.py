"""Unit tests for session persistence and slash-command helpers (session.py).

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_session_store.py
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from relai.session import (
    SessionStore,
    complete_slash,
    list_sessions,
    load_session,
    persisted_messages,
    sessions_root,
)


def test_path_layout_utc():
    root = Path(tempfile.mkdtemp())
    when = datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc)
    store = SessionStore(root=root, started_at=when)
    assert store.session_id == "2026-07-02/08_05_09", store.session_id
    assert store.path == root / "2026-07-02" / "08_05_09" / "conversation.json"
    print("path layout (UTC): OK")


def test_path_layout_converts_to_utc():
    # A non-UTC aware time must be converted to UTC for the folder name.
    root = Path(tempfile.mkdtemp())
    from datetime import timedelta

    tz = timezone(timedelta(hours=2))
    when = datetime(2026, 7, 2, 10, 5, 9, tzinfo=tz)  # == 08:05:09 UTC
    store = SessionStore(root=root, started_at=when.astimezone(timezone.utc))
    assert store.session_id == "2026-07-02/08_05_09", store.session_id
    print("path layout converts to UTC: OK")


def test_save_and_reload_roundtrip():
    root = Path(tempfile.mkdtemp())
    when = datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc)
    store = SessionStore(root=root, started_at=when)
    messages = [("you", "hi"), ("relai", "hello"), ("info", "note")]
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "hello"}]
    store.save(messages, history)

    assert store.path.is_file()
    data = json.loads(store.path.read_text())
    assert data["version"] == 1
    assert data["session_id"] == "2026-07-02/08_05_09"
    assert data["started_at"] == "2026-07-02T08:05:09Z"
    assert data["updated_at"].endswith("Z")
    assert data["messages"] == [["you", "hi"], ["relai", "hello"], ["info", "note"]]
    assert data["llm_history"] == history

    loaded = load_session("2026-07-02/08_05_09", root=root)
    assert loaded["messages"] == data["messages"]
    print("save + reload roundtrip: OK")


def test_save_extends_same_file():
    root = Path(tempfile.mkdtemp())
    when = datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc)
    store = SessionStore(root=root, started_at=when)
    store.save([("you", "q1"), ("relai", "a1")], [{"role": "user", "content": "q1"}])
    store.save(
        [("you", "q1"), ("relai", "a1"), ("you", "q2"), ("relai", "a2")],
        [{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}],
    )
    # Still one file; it holds the extended conversation.
    files = list(store.dir.glob("*.json"))
    assert files == [store.path], files
    data = json.loads(store.path.read_text())
    assert len(data["messages"]) == 4
    assert data["messages"][-1] == ["relai", "a2"]
    print("save extends same file: OK")


def test_system_messages_not_persisted():
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root, started_at=datetime.now(timezone.utc))
    messages = [
        ("you", "hi"),
        ("system", "> /sessions list"),
        ("system", "1. 2026.../.."),
        ("relai", "hello"),
    ]
    store.save(messages, [])
    data = json.loads(store.path.read_text())
    kinds = [m[0] for m in data["messages"]]
    assert kinds == ["you", "relai"], kinds
    # And the pure filter helper agrees.
    assert persisted_messages(messages) == [("you", "hi"), ("relai", "hello")]
    print("system messages not persisted: OK")


def test_list_sessions_sorted_with_preview():
    root = Path(tempfile.mkdtemp())
    a = SessionStore(root=root, started_at=datetime(2026, 7, 1, 9, 0, 0, tzinfo=timezone.utc))
    b = SessionStore(root=root, started_at=datetime(2026, 7, 2, 9, 0, 0, tzinfo=timezone.utc))
    a.save([("you", "first question"), ("relai", "ans")], [])
    b.save([("info", "note"), ("you", "second question")], [])

    listed = list_sessions(root=root)
    ids = [s["id"] for s in listed]
    assert ids == ["2026-07-01/09_00_00", "2026-07-02/09_00_00"], ids
    assert listed[0]["preview"] == "first question"
    assert listed[1]["preview"] == "second question"
    assert listed[0]["count"] == 2
    print("list sessions sorted + preview: OK")


def test_list_sessions_empty_and_missing_root():
    assert list_sessions(root=Path(tempfile.mkdtemp())) == []
    assert list_sessions(root=Path(tempfile.mkdtemp()) / "does-not-exist") == []
    print("list sessions empty/missing: OK")


def test_open_existing_binds_to_same_file():
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root, started_at=datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc))
    store.save([("you", "hi"), ("relai", "hello")], [])

    reopened = SessionStore.open_existing("2026-07-02/08_05_09", root=root)
    assert reopened.path == store.path
    reopened.save(
        [("you", "hi"), ("relai", "hello"), ("you", "more")], []
    )
    data = json.loads(store.path.read_text())
    assert data["messages"][-1] == ["you", "more"]
    print("open_existing binds to same file: OK")


def test_sessions_root_env_override(monkeypatch=None):
    import os

    old = os.environ.get("RELAI_SESSIONS_DIR")
    try:
        os.environ["RELAI_SESSIONS_DIR"] = "/tmp/relai-test-root"
        assert sessions_root() == Path("/tmp/relai-test-root")
    finally:
        if old is None:
            os.environ.pop("RELAI_SESSIONS_DIR", None)
        else:
            os.environ["RELAI_SESSIONS_DIR"] = old
    print("sessions_root env override: OK")


def test_complete_slash():
    # command-name completion (unique -> trailing space)
    assert complete_slash("/sess") == "/sessions "
    assert complete_slash("/s") == "/sessions "
    assert complete_slash("/i") == "/init_helpers "
    assert complete_slash("/init") == "/init_helpers "
    # ambiguous at the root ("init_helpers" vs "sessions" share no prefix) -> None
    assert complete_slash("/") is None
    # already complete command name -> add trailing space
    assert complete_slash("/sessions") == "/sessions "
    # subcommand completion
    assert complete_slash("/sessions li") == "/sessions list "
    assert complete_slash("/sessions lo") == "/sessions load "
    # ambiguous subcommand prefix "l" -> common prefix is "l" (== word) -> None
    assert complete_slash("/sessions l") is None
    # a command with no subcommands does not complete its argument
    assert complete_slash("/init_helpers ") is None
    # no completion possible
    assert complete_slash("/xyz") is None
    assert complete_slash("/sessions bogus") is None
    # arguments are not completed
    assert complete_slash("/sessions load 3") is None
    # non-slash input
    assert complete_slash("hello") is None
    print("complete_slash: OK")


if __name__ == "__main__":
    test_path_layout_utc()
    test_path_layout_converts_to_utc()
    test_save_and_reload_roundtrip()
    test_save_extends_same_file()
    test_system_messages_not_persisted()
    test_list_sessions_sorted_with_preview()
    test_list_sessions_empty_and_missing_root()
    test_open_existing_binds_to_same_file()
    test_sessions_root_env_override()
    test_complete_slash()
    print("\nALL session-store tests passed.")
