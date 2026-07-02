"""The bottom AI panel: a resizable, scrollable chat pane.

The panel owns only its own state (conversation transcript, the question being
typed, scroll offset, height). It renders itself to a list of drawable row
payloads; the compositor in :mod:`relai` places those on the physical screen
below the (resized) application region. The panel never touches the child's
screen model.
"""

from __future__ import annotations

from .lineedit import LineEditor
from .overlay import _wrap

_RESET = b"\x1b[0m"
_EOL = b"\x1b[K"
_REVERSE = b"\x1b[7m"
_CYAN = b"\x1b[36m"
_DIM = b"\x1b[2m"

_PROMPT = "relai> "

# Animated ellipsis frames: dots grow then shrink after the "thinking" text.
_THINK_FRAMES = ("", ".", "..", "...", "..", ".")


class AiPanel:
    """State and rendering for the bottom AI interaction panel."""

    def __init__(self, cols: int, height: int, provider: str = "") -> None:
        self.cols = max(1, cols)
        self.height = height
        self.provider = provider
        self.editor = LineEditor()
        self.thinking = False
        # The verb shown by the animated indicator while ``thinking`` is True.
        # Defaults to "Thinking"; set to e.g. "Calling inject_input" during a
        # tool call so the user can see what the agent is doing.
        self.activity = "Thinking"
        self.tick = 0  # advances while thinking, drives the spinner animation
        self.scroll = 0  # rows scrolled up from the bottom of the transcript
        self._messages: list[tuple[str, str]] = []

    # -- state mutation ------------------------------------------------------

    def set_cols(self, cols: int) -> None:
        self.cols = max(1, cols)

    @property
    def messages(self) -> list[tuple[str, str]]:
        return self._messages

    def restore(self, messages: list[tuple[str, str]]) -> None:
        """Repopulate the transcript (e.g. after re-opening the panel)."""
        self._messages = list(messages)
        self.scroll = 0

    def add_user(self, text: str) -> None:
        self._messages.append(("you", text))
        self.scroll = 0

    def add_reply(self, text: str) -> None:
        self._messages.append(("relai", text))
        self.scroll = 0

    def add_info(self, text: str) -> None:
        self._messages.append(("info", text))
        self.scroll = 0

    def add_system(self, text: str) -> None:
        """Add an ephemeral in-panel note (slash-command echo/output).

        System messages are shown like info lines but are never persisted to the
        saved conversation nor sent to the LLM.
        """
        self._messages.append(("system", text))
        self.scroll = 0

    def type_text(self, text: str) -> None:
        self.editor.insert(text)
        self.scroll = 0

    def backspace(self) -> None:
        self.editor.backspace()

    def take_input(self) -> str:
        return self.editor.take()

    def scroll_up(self, n: int) -> None:
        self.scroll += n

    def scroll_down(self, n: int) -> None:
        self.scroll = max(0, self.scroll - n)

    # -- rendering -----------------------------------------------------------

    def _input_view(self) -> tuple[str, int]:
        """Return the visible slice of the input and the 1-based cursor column.

        The input is a single line that scrolls horizontally so the cursor is
        always visible even when the text is wider than the panel.
        """
        avail = max(1, self.cols - len(_PROMPT))
        text = self.editor.text
        cur = self.editor.cursor
        if len(text) <= avail:
            start = 0
        else:
            start = max(0, min(cur - avail + 1, len(text) - avail))
            if cur < start:
                start = cur
        visible = text[start : start + avail]
        col = min(self.cols, len(_PROMPT) + (cur - start) + 1)
        return visible, col

    def cursor_col(self) -> int:
        """1-based column of the input cursor on the panel's input row."""
        return self._input_view()[1]

    def _content_lines(self) -> list[bytes]:
        lines: list[bytes] = []
        for kind, text in self._messages:
            logical = text.split("\n")
            if kind == "you":
                segs: list[str] = []
                for para in logical:
                    segs += _wrap(para, max(1, self.cols - 2))
                for i, seg in enumerate(segs):
                    prefix = b"> " if i == 0 else b"  "
                    lines.append(
                        _CYAN + prefix + seg.encode("utf-8", "replace") + _RESET + _EOL
                    )
            elif kind == "info":
                for para in logical:
                    for seg in _wrap(para, self.cols):
                        lines.append(
                            _DIM + seg.encode("utf-8", "replace") + _RESET + _EOL
                        )
            elif kind == "system":
                for para in logical:
                    for seg in _wrap(para, self.cols):
                        lines.append(
                            _CYAN + _DIM + seg.encode("utf-8", "replace")
                            + _RESET + _EOL
                        )
            else:
                for para in logical:
                    for seg in _wrap(para, self.cols):
                        lines.append(seg.encode("utf-8", "replace") + _RESET + _EOL)
        if self.thinking:
            dots = _THINK_FRAMES[self.tick % len(_THINK_FRAMES)]
            base = self.activity
            if self.provider and base == "Thinking":
                base = f"Thinking ({self.provider})"
            label = f"{base}{dots}"
            for seg in _wrap(label, self.cols):
                lines.append(_DIM + seg.encode("utf-8", "replace") + _RESET + _EOL)
        return lines

    def _header(self, more_above: int) -> bytes:
        label = f" relai · {self.provider} " if self.provider else " relai "
        if self.thinking:
            label += f"· {self.activity} "
        hints = "^O/Esc:close  ^G Up/Dn/PgUp/Dn:resize  PgUp/Dn:scroll "
        if more_above > 0:
            hints = f"\u2191{more_above} more  " + hints
        text = (label + "· " + hints)[: self.cols].ljust(self.cols)
        return _REVERSE + text.encode("utf-8", "replace") + _RESET + _EOL

    def _input_line(self) -> bytes:
        visible = self._input_view()[0]
        return _CYAN + _PROMPT.encode("ascii") + _RESET + visible.encode(
            "utf-8", "replace"
        ) + _EOL

    def render(self, height: int, cols: int) -> list[bytes]:
        """Return exactly ``height`` drawable row payloads for the panel."""
        self.set_cols(cols)
        self.height = height
        content_h = max(1, height - 2)

        lines = self._content_lines()
        max_start = max(0, len(lines) - content_h)
        self.scroll = max(0, min(self.scroll, max_start))
        start = max_start - self.scroll
        window = lines[start:start + content_h]
        while len(window) < content_h:
            window.append(_RESET + _EOL)

        rows = [self._header(start)]
        rows += window
        rows.append(self._input_line())
        if len(rows) > height:
            rows = rows[:height]
        while len(rows) < height:
            rows.append(_RESET + _EOL)
        return rows
