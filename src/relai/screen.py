"""Screen model with normal/alternate buffer tracking.

``RelaiScreen`` extends :class:`pyte.HistoryScreen` so that, in addition to the
live viewport, relai keeps a *scrollback* of logical output that scrolled off
the top of the normal screen buffer.

The key behaviour is distinguishing the two terminal screen buffers:

* **Normal buffer** -- scrolling output such as ``ls``, ``cat``, build logs.
  Lines that scroll off the top are meaningful "history" and are kept so the AI
  can read what recently scrolled by.
* **Alternate buffer** -- a live, repainting UI (``vim``, ``htop``, ``less``).
  Programs enter it via the DEC private modes ``?1049h`` / ``?1047h`` / ``?47h``
  and leave via the corresponding ``...l``. This is *not* logical output, so we
  must not let it pollute the scrollback; only the current viewport is
  meaningful while it is active.

pyte's base screen records these modes in ``screen.mode`` but does not maintain
a separate alternate buffer, so we detect the transitions ourselves and keep the
scrollback clean by discarding anything the full-screen app pushed into history
while the alternate buffer was active.
"""

from __future__ import annotations

import pyte

#: DEC private modes that switch to the alternate screen buffer. pyte flags
#: private modes by OR-ing in 1 << 8, but it passes the raw numbers to
#: set_mode/reset_mode alongside a ``private=True`` kwarg, so we match on the
#: raw values.
_ALT_SCREEN_MODES = frozenset({47, 1047, 1049})


def _row_to_text(row) -> str:
    """Render a pyte history row (a mapping of column -> Char) to text."""
    if not row:
        return ""
    return "".join(row[col].data for col in sorted(row)).rstrip()


class RelaiScreen(pyte.HistoryScreen):
    """A pyte screen that tracks the alternate buffer and keeps clean scrollback.

    Attributes
    ----------
    in_alt_screen:
        ``True`` while the child is drawing to the alternate screen buffer
        (i.e. a full-screen app such as vim/htop/less is active).
    """

    def __init__(self, columns: int, lines: int, history: int = 2000) -> None:
        # ratio=1.0 so a full screen's worth of lines is moved into history at
        # once when we page; we mainly use history.top for scrollback text.
        super().__init__(columns, lines, history=history, ratio=1.0)
        self.in_alt_screen: bool = False
        # History length captured when the alternate buffer was entered, so we
        # can truncate back to it on exit and drop the app's repaint noise.
        self._history_len_on_alt_enter: int | None = None

    # -- resize --------------------------------------------------------------

    def resize(self, lines: int | None = None, columns: int | None = None) -> None:
        """Resize the screen the way a real terminal does.

        pyte's own :meth:`pyte.Screen.resize` shrinks by removing lines from the
        *top* unconditionally, which throws away visible content whenever the
        cursor is not already at the bottom of a full screen. Instead we shrink
        like xterm: rows only scroll off the top (into scrollback) when the
        cursor would otherwise fall outside the smaller screen; the unused space
        below the cursor is simply dropped. Growing rows and any column change
        are delegated to pyte.
        """
        lines = lines if lines is not None else self.lines
        columns = columns if columns is not None else self.columns
        if lines < self.lines:
            self._shrink_lines(lines)
        super().resize(lines, columns)
        if self.cursor.y >= self.lines:
            self.cursor.y = self.lines - 1
        if self.cursor.x >= self.columns:
            self.cursor.x = self.columns - 1

    def _shrink_lines(self, new_lines: int) -> None:
        """Reduce the screen to ``new_lines`` rows, preserving recent content."""
        buffer = self.buffer
        self.cursor.y = min(self.cursor.y, self.lines - 1)
        overflow = max(0, (self.cursor.y + 1) - new_lines)
        if overflow:
            # These top rows genuinely fall off the smaller screen -> scrollback.
            if not self.in_alt_screen:
                for y in range(overflow):
                    self.history.top.append(buffer[y])
            for y in range(self.lines):
                src = y + overflow
                if src in buffer:
                    buffer[y] = buffer[src]
                else:
                    buffer.pop(y, None)
            self.cursor.y -= overflow
        # Drop everything below the new bottom row.
        for y in range(new_lines, self.lines):
            buffer.pop(y, None)
        self.lines = new_lines
        self.set_margins()

    # -- mode tracking -------------------------------------------------------

    def set_mode(self, *modes: int, **kwargs) -> None:
        super().set_mode(*modes, **kwargs)
        if kwargs.get("private") and _ALT_SCREEN_MODES.intersection(modes):
            self._enter_alt_screen()

    def reset_mode(self, *modes: int, **kwargs) -> None:
        super().reset_mode(*modes, **kwargs)
        if kwargs.get("private") and _ALT_SCREEN_MODES.intersection(modes):
            self._leave_alt_screen()

    def _enter_alt_screen(self) -> None:
        if self.in_alt_screen:
            return
        self.in_alt_screen = True
        self._history_len_on_alt_enter = len(self.history.top)

    def _leave_alt_screen(self) -> None:
        if not self.in_alt_screen:
            return
        self.in_alt_screen = False
        # Discard any lines the full-screen app pushed into scrollback so the
        # history reflects only logical, normal-buffer output.
        if self._history_len_on_alt_enter is not None:
            excess = len(self.history.top) - self._history_len_on_alt_enter
            for _ in range(max(0, excess)):
                self.history.top.pop()
        self._history_len_on_alt_enter = None

    # -- scrollback access ---------------------------------------------------

    def scrollback_lines(self) -> list[str]:
        """Return logical lines that scrolled off the top (oldest first).

        Excludes the current viewport. While the alternate buffer is active
        this reflects the normal-buffer history captured before entry.
        """
        return [_row_to_text(row) for row in self.history.top]

    def full_text(self, include_scrollback: bool = True) -> list[str]:
        """Return scrollback (optionally) followed by the visible viewport.

        The viewport lines are right-stripped; trailing blank viewport lines
        are dropped so short screens are not padded with blanks.
        """
        lines: list[str] = []
        if include_scrollback:
            lines.extend(self.scrollback_lines())
        viewport = [line.rstrip() for line in self.display]
        while viewport and not viewport[-1]:
            viewport.pop()
        lines.extend(viewport)
        return lines
