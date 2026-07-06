"""Tests for automatic context compaction (summary-based history compression).

This exercises the *infrastructure* only -- it never calls a real LLM. A
``FakeLLM`` stands in for the model: the conversation turns and the compacted
summary are hand-written to look like something an LLM would produce, so the
tests can verify how ludvart stores, purges, marks, persists and reloads them.

Covers:
- ``working_history`` slicing to the latest summary marker.
- ``_maybe_compact`` firing only past the threshold and reseeding the context.
- The ``/compact`` command (and its no-op / completion behaviour).
- End-to-end: the on-disk log has the right entries and ``/sessions load``
  resumes from the latest summary while keeping the full visible transcript.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_context_compaction.py
"""

import json
import os
import tempfile
from pathlib import Path

from ludvart.ludvart import Ludvart
from ludvart.panel import AiPanel
from ludvart.session import (
    SUMMARY_MARKER,
    SessionStore,
    list_sessions,
    load_session,
    working_history,
)
from ludvart.llm import Turn


# A made-up multi-turn conversation (user request -> assistant answer), written
# to resemble a real ludvart debugging session. Used to populate the history the
# way the panel would before a compaction.
SAMPLE_CONVERSATION = [
    (
        "What is failing in the test suite?",
        "The suite has one failing test, `test_refund_idempotency` in "
        "tests/test_payments.py. It fails intermittently with a "
        "`DuplicateRefundError`, which points to a race condition rather than a "
        "logic bug.",
    ),
    (
        "Can you reproduce it reliably?",
        "Running `pytest -k refund_idempotency -p no:randomly` reproduces it "
        "about 1 in 3 runs. The race is on the shared `RefundLedger` singleton "
        "in src/payments/ledger.py, whose `record()` reads-then-writes with no "
        "lock.",
    ),
    (
        "Fix it with a lock, keep the change minimal.",
        "I've added `self._lock = threading.Lock()` to `RefundLedger.__init__`. "
        "Next I'll wrap the body of `record()` in `with self._lock:` and re-run "
        "the test 20 times to confirm stability.",
    ),
]

# A made-up compaction summary, as if the model wrote it when asked to condense
# the conversation above into a resumable brief.
SAMPLE_SUMMARY = (
    "CONVERSATION SUMMARY (for continuation)\n\n"
    "Goal:\n"
    "- Fix the intermittently failing `test_refund_idempotency` test in the "
    "payments service with a minimal change (user prefers a lock over a DB "
    "transaction).\n\n"
    "Findings:\n"
    "- Failure is a race on the shared `RefundLedger` singleton in "
    "src/payments/ledger.py; `record()` does a read-then-write with no lock.\n"
    "- Reproduced with: pytest -k refund_idempotency -p no:randomly (~1/3 runs)."
    "\n\n"
    "State:\n"
    "- Added `self._lock = threading.Lock()` to RefundLedger.__init__.\n"
    "- `record()` is NOT yet wrapped in the lock.\n\n"
    "Next steps:\n"
    "1. Wrap the body of RefundLedger.record() in `with self._lock:`.\n"
    "2. Re-run `pytest -k refund_idempotency` ~20x to confirm stability.\n"
    "3. Update CHANGELOG.md with the fix."
)


class FakeLLM:
    """Stands in for the model: records calls, returns canned realistic text.

    A summarization request (identified by the compaction instruction) returns
    :data:`SAMPLE_SUMMARY`; any other request returns a short canned answer.
    """

    name = "fake"
    model = "opus-fake"
    context_window = 1000

    def __init__(self):
        self.on_retry = None
        self.calls = []

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        self.calls.append(list(messages))
        last = messages[-1]["content"] if messages else ""
        if isinstance(last, str) and "Summarize the ENTIRE" in last:
            return Turn(
                text=SAMPLE_SUMMARY,
                assistant_message={"role": "assistant", "content": SAMPLE_SUMMARY},
                usage=None,
            )
        if on_text is not None:
            on_text("(canned answer)")
        return Turn(
            text="(canned answer)",
            assistant_message={"role": "assistant", "content": "(canned answer)"},
            usage=None,
        )

    def tool_result_message(self, tool_call_id, content):
        return {"role": "user", "content": content}


def _make_ludvart(root: Path) -> Ludvart:
    os.environ["LUDVART_SESSIONS_DIR"] = str(root)
    r = Ludvart(["true"])
    r.llm = FakeLLM()
    r._panel = AiPanel(cols=80, height=8, provider="fake")
    r._phys_rows, r._phys_cols = 24, 80
    r._render_split = lambda: None
    r._session = SessionStore()
    return r


def _seed_conversation(r: Ludvart, turns=SAMPLE_CONVERSATION) -> None:
    """Populate history + transcript from ``turns`` the way the panel would.

    User turns are wrapped in the ``<screenContext>/<userRequest>`` envelope
    ludvart actually sends, and each turn is persisted like a live session.
    """
    for question, answer in turns:
        r._panel.add_user(question)
        r._llm_history.append(
            {
                "role": "user",
                "content": (
                    "<screenContext>\n(terminal)\n</screenContext>\n"
                    f"<userRequest>\n{question}\n</userRequest>"
                ),
            }
        )
        r._llm_history.append({"role": "assistant", "content": answer})
        r._panel.add_reply(answer)
        r._persist_session()


def _run_actions_sync(r):
    """Make ``_start_action`` run its worker synchronously for tests."""
    def fake_start_action(worker, *, info=None, activity="Working"):
        if info:
            r._panel.add_system(info)
        r._panel.add_system(worker())

    r._start_action = fake_start_action


def test_working_history_slices_to_latest_summary():
    hist = [
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "old a"},
        {"role": "user", "content": f"{SUMMARY_MARKER}\nfirst\n</conversationSummary>"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "mid q"},
        {"role": "user", "content": f"{SUMMARY_MARKER}\nsecond\n</conversationSummary>"},
        {"role": "assistant", "content": "ok2"},
        {"role": "user", "content": "new q"},
    ]
    got = working_history(hist)
    assert len(got) == 3, got
    assert got[0]["content"].lstrip().startswith(SUMMARY_MARKER)
    assert "second" in got[0]["content"]
    # No marker -> unchanged.
    plain = [{"role": "user", "content": "hi"}]
    assert working_history(plain) == plain
    print("working_history slices to latest summary: OK")


def test_no_compaction_below_threshold():
    root = Path(tempfile.mkdtemp())
    r = _make_ludvart(root)
    _seed_conversation(r)
    before = len(r._llm_history)
    r._panel.context_pct = 50.0
    r._maybe_compact()
    assert r.llm.calls == []  # no summary requested
    assert len(r._llm_history) == before
    print("no compaction below threshold: OK")


def test_compaction_above_threshold_reseeds():
    root = Path(tempfile.mkdtemp())
    r = _make_ludvart(root)
    _seed_conversation(r)
    r._panel.context_pct = 85.0
    r._maybe_compact()
    # One summary request was made.
    assert len(r.llm.calls) == 1
    # History purged to a 2-message summary seed carrying the model's summary.
    assert len(r._llm_history) == 2
    assert r._llm_history[0]["role"] == "user"
    assert r._llm_history[0]["content"].lstrip().startswith(SUMMARY_MARKER)
    assert "test_refund_idempotency" in r._llm_history[0]["content"]
    assert "Next steps:" in r._llm_history[0]["content"]
    assert r._llm_history[1]["role"] == "assistant"
    # The purged detail is gone from the model context...
    assert "DuplicateRefundError" not in json.dumps(r._llm_history)
    # ...but the full transcript is retained, plus a summary marker line.
    kinds = [k for (k, _t) in r._panel.messages]
    assert "summary" in kinds
    assert any("DuplicateRefundError" in t for (_k, t) in r._panel.messages)
    # The % counter dropped from 85.
    assert r._panel.context_pct is not None and r._panel.context_pct < 85.0
    print("compaction above threshold reseeds: OK")


def test_seed_only_history_not_recompacted():
    root = Path(tempfile.mkdtemp())
    r = _make_ludvart(root)
    r._llm_history = [
        {"role": "user", "content": f"{SUMMARY_MARKER}\nbrief\n</conversationSummary>"},
        {"role": "assistant", "content": "ok"},
    ]
    r._panel.context_pct = 99.0
    r._maybe_compact()
    assert r.llm.calls == []  # <= 2 messages: left alone
    print("seed-only history not recompacted: OK")


def test_compact_command_compacts():
    root = Path(tempfile.mkdtemp())
    r = _make_ludvart(root)
    _run_actions_sync(r)
    _seed_conversation(r)
    before = len(r._llm_history)
    r._panel.context_pct = 40.0  # below auto threshold, but manual forces it
    r._handle_slash_command("/compact")
    assert len(r.llm.calls) == 1
    assert len(r._llm_history) == 2
    assert r._llm_history[0]["content"].lstrip().startswith(SUMMARY_MARKER)
    systems = [t for (k, t) in r._panel.messages if k == "system"]
    assert any(f"Compacted {before} messages" in t for t in systems), systems
    print("/compact command compacts: OK")


def test_compact_command_noop_when_already_compact():
    root = Path(tempfile.mkdtemp())
    r = _make_ludvart(root)
    _run_actions_sync(r)
    r._llm_history = [
        {"role": "user", "content": f"{SUMMARY_MARKER}\nbrief\n</conversationSummary>"},
        {"role": "assistant", "content": "ok"},
    ]
    r._handle_slash_command("/compact")
    assert r.llm.calls == []
    systems = [t for (k, t) in r._panel.messages if k == "system"]
    assert any("already compact" in t for t in systems), systems
    print("/compact noop when already compact: OK")


def test_compact_tab_completion():
    from ludvart.session import complete_slash

    assert complete_slash("/comp") == "/compact "
    print("/compact tab completion: OK")


def test_end_to_end_log_and_reload():
    """Full path: seed -> compact -> new turn -> reload in a fresh instance.

    Verifies the on-disk log has the right entries and that a session load
    resumes from the latest summary while keeping the full visible transcript.
    """
    root = Path(tempfile.mkdtemp())
    r = _make_ludvart(root)
    _seed_conversation(r)
    session_id = r._session.session_id

    # Before compaction: the stored context is the full conversation.
    before = load_session(session_id, root=root)
    assert len(before["llm_history"]) == 2 * len(SAMPLE_CONVERSATION)
    assert not any(
        isinstance(m.get("content"), str)
        and m["content"].lstrip().startswith(SUMMARY_MARKER)
        for m in before["llm_history"]
    )

    # Compact, then continue with one more turn (persisted).
    r._panel.context_pct = 90.0
    r._maybe_compact()
    r._panel.add_user("Did the lock fix it?")
    r._llm_history.append(
        {"role": "user", "content": "<userRequest>\nDid the lock fix it?\n</userRequest>"}
    )
    r._llm_history.append(
        {"role": "assistant", "content": "Yes -- 20/20 runs passed after the lock."}
    )
    r._panel.add_reply("Yes -- 20/20 runs passed after the lock.")
    r._persist_session()

    # On-disk log: context is seed + the post-compaction turn; the pre-summary
    # detail is gone from the model context but kept in the visible transcript.
    after = load_session(session_id, root=root)
    assert after["llm_history"][0]["content"].lstrip().startswith(SUMMARY_MARKER)
    assert "DuplicateRefundError" not in json.dumps(after["llm_history"])
    kinds = [m[0] for m in after["messages"]]
    assert "summary" in kinds and "you" in kinds and "ludvart" in kinds
    assert any("DuplicateRefundError" in text for (_k, text) in after["messages"])

    # Reload in a FRESH instance (the /sessions load path).
    r2 = _make_ludvart(root)
    r2._session_list = list_sessions()
    r2._load_session(session_id)

    # Model context resumes from the latest summary + the later turn only.
    assert r2._llm_history[0]["content"].lstrip().startswith(SUMMARY_MARKER)
    assert len(r2._llm_history) == 4  # summary seed (2) + one turn (2)
    assert "DuplicateRefundError" not in json.dumps(r2._llm_history)
    assert "Did the lock fix it?" in json.dumps(r2._llm_history)
    # Visible transcript restored in full, including the summary marker.
    kinds2 = [k for (k, _t) in r2._panel.messages]
    assert "summary" in kinds2
    assert any("DuplicateRefundError" in t for (_k, t) in r2._panel.messages)
    print("end-to-end log entries + reload from latest summary: OK")


def main():
    test_working_history_slices_to_latest_summary()
    test_no_compaction_below_threshold()
    test_compaction_above_threshold_reseeds()
    test_seed_only_history_not_recompacted()
    test_compact_command_compacts()
    test_compact_command_noop_when_already_compact()
    test_compact_tab_completion()
    test_end_to_end_log_and_reload()
    print("\nALL context-compaction tests passed.")


if __name__ == "__main__":
    main()
