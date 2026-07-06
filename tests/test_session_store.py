"""Unit tests for session persistence and slash-command helpers (session.py).

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_session_store.py
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ludvart.session import (
    SessionStore,
    complete_slash,
    list_sessions,
    load_session,
    persisted_messages,
    provider_family,
    sanitize_history,
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
    messages = [("you", "hi"), ("ludvart", "hello"), ("info", "note")]
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "hello"}]
    store.save(messages, history)

    assert store.path.is_file()
    data = json.loads(store.path.read_text())
    assert data["version"] == 2
    assert data["session_id"] == "2026-07-02/08_05_09"
    assert data["started_at"] == "2026-07-02T08:05:09Z"
    assert data["updated_at"].endswith("Z")
    assert data["messages"] == [["you", "hi"], ["ludvart", "hello"], ["info", "note"]]
    assert data["llm_history"] == history

    loaded = load_session("2026-07-02/08_05_09", root=root)
    assert loaded["messages"] == data["messages"]
    print("save + reload roundtrip: OK")


def test_save_extends_same_file():
    root = Path(tempfile.mkdtemp())
    when = datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc)
    store = SessionStore(root=root, started_at=when)
    store.save([("you", "q1"), ("ludvart", "a1")], [{"role": "user", "content": "q1"}])
    store.save(
        [("you", "q1"), ("ludvart", "a1"), ("you", "q2"), ("ludvart", "a2")],
        [{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}],
    )
    # Still one file; it holds the extended conversation.
    files = list(store.dir.glob("*.json"))
    assert files == [store.path], files
    data = json.loads(store.path.read_text())
    assert len(data["messages"]) == 4
    assert data["messages"][-1] == ["ludvart", "a2"]
    print("save extends same file: OK")


def test_system_messages_not_persisted():
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root, started_at=datetime.now(timezone.utc))
    messages = [
        ("you", "hi"),
        ("system", "> /sessions list"),
        ("system", "1. 2026.../.."),
        ("ludvart", "hello"),
    ]
    store.save(messages, [])
    data = json.loads(store.path.read_text())
    kinds = [m[0] for m in data["messages"]]
    assert kinds == ["you", "ludvart"], kinds
    # And the pure filter helper agrees.
    assert persisted_messages(messages) == [("you", "hi"), ("ludvart", "hello")]
    print("system messages not persisted: OK")


def test_list_sessions_sorted_with_preview():
    root = Path(tempfile.mkdtemp())
    a = SessionStore(root=root, started_at=datetime(2026, 7, 1, 9, 0, 0, tzinfo=timezone.utc))
    b = SessionStore(root=root, started_at=datetime(2026, 7, 2, 9, 0, 0, tzinfo=timezone.utc))
    a.save([("you", "first question"), ("ludvart", "ans")], [])
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
    store.save([("you", "hi"), ("ludvart", "hello")], [])

    reopened = SessionStore.open_existing("2026-07-02/08_05_09", root=root)
    assert reopened.path == store.path
    reopened.save(
        [("you", "hi"), ("ludvart", "hello"), ("you", "more")], []
    )
    data = json.loads(store.path.read_text())
    assert data["messages"][-1] == ["you", "more"]
    print("open_existing binds to same file: OK")


def test_sessions_root_env_override(monkeypatch=None):
    import os

    old = os.environ.get("LUDVART_SESSIONS_DIR")
    try:
        os.environ["LUDVART_SESSIONS_DIR"] = "/tmp/ludvart-test-root"
        assert sessions_root() == Path("/tmp/ludvart-test-root")
    finally:
        if old is None:
            os.environ.pop("LUDVART_SESSIONS_DIR", None)
        else:
            os.environ["LUDVART_SESSIONS_DIR"] = old
    print("sessions_root env override: OK")


def test_save_records_provider():
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root)
    store.save([("you", "q")], [{"role": "user", "content": "q"}], provider="openai")
    data = json.loads(store.path.read_text())
    assert data["provider"] == "openai"
    # provider defaults to None when not supplied (older callers).
    store2 = SessionStore(root=root)
    store2.save([("you", "q")], [])
    assert json.loads(store2.path.read_text())["provider"] is None
    print("save records provider: OK")


def test_provider_family_mapping():
    assert provider_family("openai") == "openai"
    assert provider_family("custom") == "openai"  # both use the OpenAI shape
    assert provider_family("anthropic") == "anthropic"
    assert provider_family("google") == "google"
    assert provider_family(None) is None
    assert provider_family("weird") is None
    print("provider family mapping: OK")


def test_sanitize_history_same_family_unchanged():
    hist = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
    ]
    # openai <-> custom share a family: history is preserved verbatim.
    out = sanitize_history(hist, "openai", "openai")
    assert out == hist
    # Unknown on both sides also counts as "same" -> unchanged.
    assert sanitize_history(hist, None, None) == hist


def test_sanitize_openai_history_to_anthropic():
    # An OpenAI-shaped history (with a "tool" role) must not keep that role
    # when adapted for Anthropic, which only accepts user/assistant.
    hist = [
        {"role": "user", "content": "what time is it?"},
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "clock", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "12:00"},
        {"role": "assistant", "content": "it is noon"},
    ]
    out = sanitize_history(hist, "openai", "anthropic")
    # No forbidden role and every content is a plain string.
    assert all(m["role"] in ("user", "assistant") for m in out)
    assert all(isinstance(m["content"], str) for m in out)
    flat = " ".join(m["content"] for m in out)
    assert "what time is it?" in flat
    assert "12:00" in flat  # the tool result survives as text
    assert "it is noon" in flat


def test_sanitize_anthropic_history_to_openai():
    # Anthropic uses block lists and a user-role tool_result; flatten to strings.
    hist = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "t1", "name": "echo", "input": {"x": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "done"}
            ],
        },
    ]
    out = sanitize_history(hist, "anthropic", "openai")
    assert all(isinstance(m["content"], str) for m in out)
    assert all(m["role"] in ("user", "assistant") for m in out)
    flat = " ".join(m["content"] for m in out)
    assert "echo" in flat and "done" in flat


def _google_history():
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "function_call", "name": "search", "args": {"q": "x"}}
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "function_response",
                    "name": "search",
                    "response": {"result": "found it"},
                }
            ],
        },
    ]


def test_sanitize_google_history_to_openai():
    out = sanitize_history(_google_history(), "google", "openai")
    assert all(m["role"] in ("user", "assistant") for m in out)
    assert all(isinstance(m["content"], str) for m in out)
    flat = " ".join(m["content"] for m in out)
    assert "search" in flat and "found it" in flat


def test_sanitize_google_history_to_anthropic():
    out = sanitize_history(_google_history(), "google", "anthropic")
    assert all(m["role"] in ("user", "assistant") for m in out)
    flat = " ".join(m["content"] for m in out)
    assert "search" in flat and "found it" in flat


def test_sanitize_coalesces_adjacent_roles():
    # Two tool results in a row (both become user notes) must not produce two
    # adjacent user messages, which some providers reject.
    hist = [
        {"role": "tool", "tool_call_id": "1", "content": "r1"},
        {"role": "tool", "tool_call_id": "2", "content": "r2"},
    ]
    out = sanitize_history(hist, "openai", "anthropic")
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "r1" in out[0]["content"] and "r2" in out[0]["content"]


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
    test_save_records_provider()
    test_provider_family_mapping()
    test_sanitize_history_same_family_unchanged()
    test_sanitize_openai_history_to_anthropic()
    test_sanitize_anthropic_history_to_openai()
    test_sanitize_google_history_to_openai()
    test_sanitize_google_history_to_anthropic()
    test_sanitize_coalesces_adjacent_roles()
    test_complete_slash()
    print("\nALL session-store tests passed.")
