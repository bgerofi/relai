"""Unit tests for the panel line editor and bracketed-paste input handling.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tools/test_ai_panel_edit.py
"""

from ludvart.lineedit import LineEditor
from ludvart.panel import AiPanel


def test_line_editor():
    ed = LineEditor()
    ed.insert("hello")
    assert ed.text == "hello" and ed.cursor == 5

    # move left twice and insert in the middle
    ed.left(); ed.left()
    assert ed.cursor == 3
    ed.insert("XY")
    assert ed.text == "helXYlo" and ed.cursor == 5

    # backspace removes before cursor
    ed.backspace()
    assert ed.text == "helXlo" and ed.cursor == 4

    # forward delete removes under cursor
    ed.delete()
    assert ed.text == "helXo" and ed.cursor == 4

    # home / end
    ed.home(); assert ed.cursor == 0
    ed.end(); assert ed.cursor == 5

    # right at end is a no-op; left at start is a no-op
    ed.right(); assert ed.cursor == 5
    ed.home(); ed.left(); assert ed.cursor == 0

    # kill to end / start
    ed2 = LineEditor("one two three")
    ed2.home()
    for _ in range(4):
        ed2.right()  # cursor after "one "
    ed2.kill_to_start()
    assert ed2.text == "two three" and ed2.cursor == 0
    ed2.end(); ed2.left(); ed2.left()  # inside "three"
    ed2.kill_to_end()
    assert ed2.text == "two thr"

    # delete word back
    ed3 = LineEditor("foo bar baz")
    ed3.delete_word_back()
    assert ed3.text == "foo bar " and ed3.cursor == len("foo bar ")
    ed3.delete_word_back()
    assert ed3.text == "foo " and ed3.cursor == 4

    # take strips and clears
    ed4 = LineEditor("  spaced  ")
    assert ed4.take() == "spaced"
    assert ed4.text == "" and ed4.cursor == 0
    print("line editor: OK")


def test_input_view_scroll():
    panel = AiPanel(cols=20, height=8, provider="test")
    # prompt "ludvart> " is 7 chars -> avail = 13
    panel.editor.insert("abcdefghij")  # 10 chars, fits
    visible, col = panel._input_view()
    assert visible == "abcdefghij"
    assert col == 7 + 10 + 1  # after last char

    panel.editor.insert("klmnopqrst")  # now 20 chars, wider than avail=13
    visible, col = panel._input_view()
    assert len(visible) == 13
    # cursor is at end -> window shows the tail, cursor clamps to last column
    assert visible.endswith("t")
    assert col == 20

    # move to home: window should show the head, cursor at first input col
    panel.editor.home()
    visible, col = panel._input_view()
    assert visible.startswith("a")
    assert col == 8  # len(prompt)+1
    print("input view scroll: OK")


if __name__ == "__main__":
    test_line_editor()
    test_input_view_scroll()
    print("all panel-edit tests passed")
