"""A minimal, self-contained single-line text editor buffer.

This deliberately avoids any terminal/UI concerns: it is just a string plus a
cursor position and the edit operations a line editor needs (insert, delete,
cursor movement, word/line kills). The ludvart panel feeds it decoded key events
and renders it; nothing here reads or writes the terminal.

The logic is intentionally simple and index-based (one index == one Unicode
code point, matching Python ``str`` indexing) so it maps directly onto a Rust
``Vec<char>`` if this is ported later. No regex, no external dependencies.
"""

from __future__ import annotations


class LineEditor:
    """A single line of editable text with an insertion cursor.

    ``cursor`` is the code-point index in ``text`` where the next insertion
    happens; it ranges over ``0..=len(text)``.
    """

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.cursor = len(text)

    # -- editing -------------------------------------------------------------

    def insert(self, s: str) -> None:
        """Insert ``s`` at the cursor and advance past it."""
        if not s:
            return
        self.text = self.text[: self.cursor] + s + self.text[self.cursor :]
        self.cursor += len(s)

    def backspace(self) -> None:
        """Delete the character before the cursor (Backspace)."""
        if self.cursor > 0:
            self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
            self.cursor -= 1

    def delete(self) -> None:
        """Delete the character under the cursor (Delete/forward)."""
        if self.cursor < len(self.text):
            self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]

    def delete_word_back(self) -> None:
        """Delete the whitespace-delimited word before the cursor (Ctrl-W)."""
        i = self.cursor
        while i > 0 and self.text[i - 1] == " ":
            i -= 1
        while i > 0 and self.text[i - 1] != " ":
            i -= 1
        self.text = self.text[:i] + self.text[self.cursor :]
        self.cursor = i

    def kill_to_start(self) -> None:
        """Delete from the line start up to the cursor (Ctrl-U)."""
        self.text = self.text[self.cursor :]
        self.cursor = 0

    def kill_to_end(self) -> None:
        """Delete from the cursor to the end of the line (Ctrl-K)."""
        self.text = self.text[: self.cursor]

    # -- cursor movement -----------------------------------------------------

    def left(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def right(self) -> None:
        if self.cursor < len(self.text):
            self.cursor += 1

    def home(self) -> None:
        self.cursor = 0

    def end(self) -> None:
        self.cursor = len(self.text)

    # -- whole-buffer --------------------------------------------------------

    def set_text(self, text: str) -> None:
        """Replace the whole buffer with ``text`` and put the cursor at its end."""
        self.text = text
        self.cursor = len(text)

    def clear(self) -> None:
        self.text = ""
        self.cursor = 0

    def take(self) -> str:
        """Return the trimmed text and reset the buffer to empty."""
        value = self.text.strip()
        self.clear()
        return value
