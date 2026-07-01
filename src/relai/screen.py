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
