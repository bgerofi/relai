"""Full-screen overlays drawn on the alternate screen buffer.

Overlays let relai temporarily take over the terminal (to show scrollback, an
AI panel, etc.) without disturbing the child program's screen. We switch to the
alternate screen buffer, draw, and on exit switch back -- so whatever the child
had on screen is restored exactly.

This module writes directly to a terminal fd and reads raw key bytes from an
input fd; the caller is responsible for having the terminal in raw mode.
"""

from __future__ import annotations

import os

# ANSI/DEC control sequences used to build overlays.
_ENTER_ALT = b"\x1b[?1049h"   # save screen + switch to alternate buffer
_LEAVE_ALT = b"\x1b[?1049l"   # switch back + restore saved screen
_HIDE_CURSOR = b"\x1b[?25l"
_SHOW_CURSOR = b"\x1b[?25h"
_CLEAR = b"\x1b[2J\x1b[H"     # clear screen, cursor home
_RESET_SGR = b"\x1b[0m"
_INVERSE = b"\x1b[7m"


def _write_all(fd: int, data: bytes) -> None:
    while data:
        n = os.write(fd, data)
        data = data[n:]


class ScrollbackViewer:
    """A simple pager that displays lines on the alternate screen buffer.

    Keys while open:
        Up / Down  or  k / j              scroll one line
        PageUp / PageDown, Space / b,
            or Ctrl-F / Ctrl-B            scroll one page
        Home / g                          jump to top
        End / G                           jump to bottom
        q / Esc / Ctrl-C                  close
    """

    def __init__(self, out_fd: int, in_fd: int, rows: int, cols: int) -> None:
        self.out_fd = out_fd
        self.in_fd = in_fd
        self.rows = rows
        self.cols = cols

    def show(self, lines: list[str], title: str = "relai scrollback") -> None:
        """Open the pager on ``lines`` and block until the user closes it."""
        body_rows = max(1, self.rows - 1)  # reserve one row for the status bar
        # Start scrolled to the bottom (most recent), like a fresh pager view.
        top = max(0, len(lines) - body_rows)

        _write_all(self.out_fd, _ENTER_ALT + _HIDE_CURSOR)
        try:
            while True:
                self._render(lines, top, body_rows, title)
                key = self._read_key()
                if key in ("q", "esc", "ctrl-c"):
                    break
                top = self._apply_key(key, top, len(lines), body_rows)
        finally:
            _write_all(self.out_fd, _SHOW_CURSOR + _LEAVE_ALT)

    # -- rendering -----------------------------------------------------------

    def _render(self, lines: list[str], top: int, body_rows: int, title: str) -> None:
        out = bytearray(_CLEAR)
        window = lines[top : top + body_rows]
        for i in range(body_rows):
            text = window[i] if i < len(window) else ""
            out += text[: self.cols].encode("utf-8", "replace")
            out += b"\r\n"
        # Status bar (inverse video) on the last row.
        total = len(lines)
        shown_end = min(total, top + body_rows)
        status = (
            f" {title} — lines {top + 1}-{shown_end}/{total} "
            f"  [↑/↓ PgUp/PgDn Home/End  q to close] "
        )
        status = status[: self.cols].ljust(self.cols)
        out += _INVERSE + status.encode("utf-8", "replace") + _RESET_SGR
        _write_all(self.out_fd, bytes(out))

    # -- input ---------------------------------------------------------------

    def _apply_key(self, key: str, top: int, total: int, body_rows: int) -> int:
        max_top = max(0, total - body_rows)
        if key == "up":
            top -= 1
        elif key == "down":
            top += 1
        elif key == "pageup":
            top -= body_rows
        elif key == "pagedown":
            top += body_rows
        elif key == "home":
            top = 0
        elif key == "end":
            top = max_top
        return max(0, min(top, max_top))

    def _read_key(self) -> str:
        """Read one logical keypress and return a normalized name.

        Reads a whole chunk at once (terminals deliver an escape sequence
        atomically) so multi-byte keys like the arrows and PageUp/PageDown are
        matched reliably without being split across reads.
        """
        chunk = os.read(self.in_fd, 64)
        if not chunk:
            return "esc"

        # Multi-byte escape sequences (arrows, PageUp/PageDown, Home/End).
        if chunk[:1] == b"\x1b":
            if len(chunk) == 1:
                return "esc"
            return _ESC_KEYS.get(chunk[1:], "other")

        # Single-byte keys.
        b = chunk[:1]
        return _SINGLE_KEYS.get(b, "other")


# Single-byte keys accepted by the viewer.
_SINGLE_KEYS: dict[bytes, str] = {
    b"\x03": "ctrl-c",
    b"q": "q",
    b"Q": "q",
    b"k": "up",
    b"j": "down",
    b" ": "pagedown",
    b"\x06": "pagedown",  # Ctrl-F
    b"b": "pageup",
    b"\x02": "pageup",    # Ctrl-B
    b"g": "home",
    b"G": "end",
}

# Bytes following an initial ESC, mapped to key names. Covers the CSI ("[")
# forms, the application-cursor ("O") forms some terminals/screen emit, and the
# Ctrl-modified variants (parameter ";5").
_ESC_KEYS: dict[bytes, str] = {
    # Cursor keys, CSI form.
    b"[A": "up",
    b"[B": "down",
    b"[C": "right",
    b"[D": "left",
    # Cursor keys, application (SS3 "O") form.
    b"OA": "up",
    b"OB": "down",
    b"OC": "right",
    b"OD": "left",
    # Home / End.
    b"[H": "home",
    b"[F": "end",
    b"[1~": "home",
    b"[4~": "end",
    b"[7~": "home",
    b"[8~": "end",
    b"OH": "home",
    b"OF": "end",
    # PageUp / PageDown.
    b"[5~": "pageup",
    b"[6~": "pagedown",
    # Ctrl-modified PageUp / PageDown.
    b"[5;5~": "pageup",
    b"[6;5~": "pagedown",
}
