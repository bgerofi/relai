"""Inline AI chat for line-oriented programs.

The AI exchange is injected *above* the child's prompt without ever touching
the prompt itself. A DEC scroll region (DECSTBM) is set to the rows above the
prompt; printing there scrolls older output up (into the terminal's native
scrollback) and leaves the prompt row untouched. The cursor is restored into
the prompt afterward.

Because relai never erases or reprints the child's prompt, the child's line
editor (readline, zle, ...) keeps full ownership of its prompt and input
buffer. There is no resync and no double-printed prompt, regardless of how the
prompt is styled (colors, git status, multiple lines).

Only the standard library is used; the terminal is assumed to be in raw mode.
"""

from __future__ import annotations

import os
from typing import Callable

from .overlay import _wrap, _write_all

_CLEAR_LINE = b"\x1b[K"
_RESET_SGR = b"\x1b[0m"
_DIM = b"\x1b[2m"
_CYAN = b"\x1b[36m"
_HIDE_CURSOR = b"\x1b[?25l"
_SHOW_CURSOR = b"\x1b[?25h"

#: Prefix shown on the question line the user types into.
_QUESTION_PROMPT = "relai> "


def _move_to(row: int, col: int) -> bytes:
    return f"\x1b[{row};{col}H".encode("ascii")


def _set_region(top: int, bottom: int) -> bytes:
    return f"\x1b[{top};{bottom}r".encode("ascii")


_RESET_REGION = b"\x1b[r"


class InlineChat:
    """Run one AI question/answer exchange inline, above the child's prompt."""

    def __init__(self, out_fd: int, in_fd: int, cols: int) -> None:
        self.out_fd = out_fd
        self.in_fd = in_fd
        self.cols = max(1, cols)
        self._bottom = 1  # scroll-region bottom row (the row just above the prompt)

    def run(
        self,
        prompt_row: int,
        cursor_col: int,
        ask: Callable[[str], str],
        provider: str = "",
    ) -> None:
        """Prompt inline above the prompt, call ``ask``, print the reply.

        Parameters
        ----------
        prompt_row:
            1-based screen row where the child's prompt/input cursor sits.
        cursor_col:
            0-based column of the child's cursor within the prompt row.
        ask:
            ``Callable[[str], str]`` returning the reply for a question.
        provider:
            Short provider label for the "thinking" line.
        """
        if prompt_row < 2:
            # No room above the prompt (it is on the top row). Fall back to a
            # plain scroll: print below and let the child redraw on next input.
            self._run_fallback(ask, provider)
            return

        self._bottom = prompt_row - 1
        _write_all(self.out_fd, _HIDE_CURSOR)
        _write_all(self.out_fd, _set_region(1, self._bottom))
        try:
            question = self._read_question()
            if question is None:
                return

            self._open_line()
            label = f"thinking… ({provider})" if provider else "thinking…"
            _write_all(self.out_fd, _DIM + label.encode("utf-8", "replace") + _RESET_SGR)
            try:
                reply = ask(question)
            except Exception as exc:  # surfaced to the user, never crashes relai
                reply = f"[relai] request failed: {exc}"

            self._print_reply(reply)
        finally:
            _write_all(self.out_fd, _RESET_REGION)
            # Restore the cursor into the untouched prompt.
            _write_all(self.out_fd, _move_to(prompt_row, cursor_col + 1))
            _write_all(self.out_fd, _SHOW_CURSOR)

    # -- scroll-region helpers ----------------------------------------------

    def _open_line(self) -> None:
        """Scroll the region up by one, leaving a blank row at the bottom."""
        _write_all(self.out_fd, _move_to(self._bottom, 1))
        _write_all(self.out_fd, b"\n")
        _write_all(self.out_fd, _move_to(self._bottom, 1))

    def _emit(self, text: str, col: int) -> int:
        """Echo ``text`` at the region bottom, wrapping onto new lines.

        Returns the resulting cursor column (0-based).
        """
        for ch in text:
            if col >= self.cols:
                self._open_line()
                col = 0
            _write_all(self.out_fd, ch.encode("utf-8", "replace"))
            col += 1
        return col

    # -- question input ------------------------------------------------------

    def _read_question(self) -> str | None:
        """Read one line of input above the prompt, echoing it."""
        self._open_line()
        _write_all(self.out_fd, _CYAN + _QUESTION_PROMPT.encode("ascii") + _RESET_SGR)
        col = len(_QUESTION_PROMPT)
        buf = bytearray()
        while True:
            chunk = os.read(self.in_fd, 64)
            if not chunk:
                return None
            if chunk in (b"\r", b"\n"):
                text = buf.decode("utf-8", "replace").strip()
                return text if text else None
            if chunk in (b"\x1b", b"\x03"):  # bare Esc / Ctrl-C cancel
                return None
            if chunk[:1] == b"\x1b":  # ignore other escape sequences
                continue
            if chunk in (b"\x7f", b"\x08"):  # Backspace / Delete
                if buf:
                    del buf[-1]
                    while buf and (buf[-1] & 0xC0) == 0x80:
                        del buf[-1]
                    if col > len(_QUESTION_PROMPT):
                        col -= 1
                        _write_all(self.out_fd, b"\b \b")
                continue
            if len(chunk) == 1 and chunk[0] < 0x20:  # other control bytes
                continue
            buf += chunk
            col = self._emit(chunk.decode("utf-8", "replace"), col)

    # -- reply output --------------------------------------------------------

    def _print_reply(self, reply: str) -> None:
        """Replace the 'thinking' line with the reply, wrapped to the width."""
        lines: list[str] = []
        for paragraph in reply.splitlines() or [""]:
            lines.extend(_wrap(paragraph, self.cols) or [""])

        # First reply line overwrites the 'thinking' line in place.
        _write_all(self.out_fd, _move_to(self._bottom, 1) + _CLEAR_LINE)
        _write_all(self.out_fd, lines[0].encode("utf-8", "replace"))
        for extra in lines[1:]:
            self._open_line()
            _write_all(self.out_fd, extra.encode("utf-8", "replace"))

    # -- fallback (prompt on the top row) -----------------------------------

    def _run_fallback(self, ask: Callable[[str], str], provider: str) -> None:
        """Prompt/print inline via plain scrolling (no room above the prompt)."""
        _write_all(self.out_fd, b"\r\n" + _CYAN + _QUESTION_PROMPT.encode("ascii") + _RESET_SGR)
        buf = bytearray()
        while True:
            chunk = os.read(self.in_fd, 64)
            if not chunk or chunk in (b"\x1b", b"\x03"):
                _write_all(self.out_fd, b"\r\n")
                return
            if chunk in (b"\r", b"\n"):
                break
            if chunk[:1] == b"\x1b":
                continue
            if chunk in (b"\x7f", b"\x08"):
                if buf:
                    del buf[-1]
                    while buf and (buf[-1] & 0xC0) == 0x80:
                        del buf[-1]
                    _write_all(self.out_fd, b"\b \b")
                continue
            if len(chunk) == 1 and chunk[0] < 0x20:
                continue
            buf += chunk
            _write_all(self.out_fd, chunk)
        question = buf.decode("utf-8", "replace").strip()
        if not question:
            _write_all(self.out_fd, b"\r\n")
            return
        label = f"thinking… ({provider})" if provider else "thinking…"
        _write_all(self.out_fd, b"\r\n" + _DIM + label.encode("utf-8", "replace") + _RESET_SGR)
        try:
            reply = ask(question)
        except Exception as exc:
            reply = f"[relai] request failed: {exc}"
        _write_all(self.out_fd, b"\r" + _CLEAR_LINE)
        for paragraph in reply.splitlines() or [""]:
            for segment in _wrap(paragraph, self.cols) or [""]:
                _write_all(self.out_fd, segment.encode("utf-8", "replace") + b"\r\n")
