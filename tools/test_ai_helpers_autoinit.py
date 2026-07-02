"""Unit test: first-open shell detection triggers helper initialization once.

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_ai_helpers_autoinit.py
"""

from relai.relai import Relai, HELPERS_INIT_PROMPT
from relai.panel import AiPanel


def make_relai(prompt: bytes, with_llm: bool):
    r = Relai(["true"])
    if with_llm:
        r.llm = object()  # truthy; _maybe_init_helpers only checks it's not None
    r._panel = AiPanel(cols=80, height=8, provider="test")
    r._phys_rows, r._phys_cols = 24, 80
    r.stream.feed(prompt)  # paint a prompt so the cursor sits after it

    # Capture _start_ask calls instead of spawning a thread / rendering.
    calls = []
    r._start_ask = lambda q, **kw: calls.append((q, kw))
    r._render_split = lambda: None
    return r, calls


def test_shell_detection():
    for prompt, expected in [
        (b"user@host:~/src$ ", True),
        (b"root@box:/# ", True),
        (b"%zsh here % ", True),
        (b">>> ", False),        # python REPL must NOT count
        (b"just some text ", False),
        (b"", False),
    ]:
        r, _ = make_relai(prompt, with_llm=True)
        got = r._looks_like_shell()
        assert got == expected, f"{prompt!r}: expected {expected}, got {got}"
    print("shell detection: OK")


def test_triggers_once_on_shell():
    r, calls = make_relai(b"user@host:~$ ", with_llm=True)
    r._maybe_init_helpers(looks_shell=True)
    assert len(calls) == 1, calls
    assert calls[0][0] == HELPERS_INIT_PROMPT
    assert "info" in calls[0][1]
    assert r._helpers_init_attempted is True

    # A second open must not trigger again.
    r._maybe_init_helpers(looks_shell=True)
    assert len(calls) == 1, "must only attempt once per session"
    print("triggers once on shell: OK")


def test_no_trigger_without_shell_or_llm():
    # Not a shell -> no trigger, and flag stays unset so a later shell open can.
    r, calls = make_relai(b">>> ", with_llm=True)
    r._maybe_init_helpers(looks_shell=False)
    assert calls == [] and r._helpers_init_attempted is False

    # Shell but no LLM configured -> no trigger.
    r2, calls2 = make_relai(b"user@host:~$ ", with_llm=False)
    r2.llm = None
    r2._maybe_init_helpers(looks_shell=True)
    assert calls2 == [] and r2._helpers_init_attempted is False
    print("no trigger without shell or llm: OK")


if __name__ == "__main__":
    test_shell_detection()
    test_triggers_once_on_shell()
    test_no_trigger_without_shell_or_llm()
    print("all helper-autoinit tests passed")
