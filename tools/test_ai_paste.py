"""Bracketed-paste routing test for the AI panel input.

Exercises Relai._panel_input directly (no PTY): a paste split across reads and
containing a newline must be inserted verbatim (newline folded to a space) into
the editor without submitting, and trailing bytes after the end marker are
processed as normal keys.

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_ai_paste.py
"""

from relai.relai import Relai, _PASTE_START, _PASTE_END
from relai.panel import AiPanel


def make_relai():
    r = Relai(["true"])
    r._panel = AiPanel(cols=40, height=8, provider="test")
    r._panel_pasting = False
    r._panel_pastebuf = bytearray()
    return r


def test_paste_single_read():
    r = make_relai()
    r._panel_input(_PASTE_START + b"hello world" + _PASTE_END)
    assert r._panel.editor.text == "hello world", r._panel.editor.text
    assert not r._panel_pasting
    print("paste single read: OK")


def test_paste_with_newline_no_submit():
    r = make_relai()
    submitted = []
    r._panel_submit = lambda: submitted.append(True)
    r._panel_input(_PASTE_START + b"line1\nline2" + _PASTE_END)
    assert r._panel.editor.text == "line1 line2", r._panel.editor.text
    assert submitted == [], "paste must not submit"
    print("paste newline no submit: OK")


def test_paste_split_across_reads():
    r = make_relai()
    # Start marker + first chunk in one read, remainder + end marker in another,
    # with the END marker itself split across the two reads.
    part1 = _PASTE_START + b"abc"
    mid = _PASTE_END[:2]  # split the end marker
    part2 = _PASTE_END[2:] + b"Z"  # rest of marker, then a normal keystroke
    r._panel_input(part1)
    assert r._panel_pasting
    r._panel_input(b"def" + mid)
    assert r._panel_pasting  # end marker not complete yet
    r._panel_input(part2)
    assert not r._panel_pasting
    assert r._panel.editor.text == "abcdefZ", r._panel.editor.text
    print("paste split across reads: OK")


def test_prefix_before_paste():
    r = make_relai()
    r._panel_input(b"hi " + _PASTE_START + b"there" + _PASTE_END)
    assert r._panel.editor.text == "hi there", r._panel.editor.text
    print("text before paste: OK")


if __name__ == "__main__":
    test_paste_single_read()
    test_paste_with_newline_no_submit()
    test_paste_split_across_reads()
    test_prefix_before_paste()
    print("all paste tests passed")
