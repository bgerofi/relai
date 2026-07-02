"""Render a pyte screen to escape sequences, with a row-level frame differ.

While the AI panel is open, relai stops passing child bytes through verbatim and
instead becomes a compositor: the child writes to the pyte model (its *virtual*
screen), and relai renders that model onto the top region of the physical
screen, drawing the panel below it. This module turns model cells into bytes and
diffs frames so only changed rows are repainted.
"""

from __future__ import annotations

# pyte stores colours either as the string "default", as an ANSI colour name,
# or as a 6-digit hex string (for 256-colour / true-colour cells).
_ANSI = {
    "black": 0, "red": 1, "green": 2, "brown": 3,
    "blue": 4, "magenta": 5, "cyan": 6, "white": 7,
}

_RESET = b"\x1b[0m"


def _color_params(color: str, base: int) -> list[int]:
    """SGR parameters for a pyte colour. ``base`` is 30 (fg) or 40 (bg)."""
    if not color or color == "default":
        return []
    if color in _ANSI:
        return [base + _ANSI[color]]
    if color.startswith("bright") and color[6:] in _ANSI:
        return [base + 60 + _ANSI[color[6:]]]
    if len(color) == 6:
        try:
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
        except ValueError:
            return []
        return [(38 if base == 30 else 48), 2, r, g, b]
    return []


def _is_styled(cell) -> bool:
    return (
        cell.fg != "default"
        or cell.bg != "default"
        or cell.bold
        or cell.italics
        or cell.underscore
        or cell.strikethrough
        or cell.reverse
        or cell.blink
    )


def sgr_bytes(cell) -> bytes:
    """Return the SGR sequence that reproduces ``cell``'s attributes."""
    params: list[int] = []
    if cell.bold:
        params.append(1)
    if cell.italics:
        params.append(3)
    if cell.underscore:
        params.append(4)
    if cell.blink:
        params.append(5)
    if cell.reverse:
        params.append(7)
    if cell.strikethrough:
        params.append(9)
    params += _color_params(cell.fg, 30)
    params += _color_params(cell.bg, 40)
    if not params:
        return _RESET
    return b"\x1b[" + ";".join(str(p) for p in params).encode("ascii") + b"m"


def render_row(screen, y: int, cols: int) -> bytes:
    """Render row ``y`` of ``screen`` to a drawable payload.

    The payload assumes the cursor is already at the start of the row and resets
    SGR and clears to end of line at the end, so shorter rows overwrite cleanly.
    """
    row = screen.buffer[y]
    last = -1
    for x in range(cols):
        cell = row[x]
        if (cell.data and cell.data != " ") or _is_styled(cell):
            last = x

    out = bytearray(_RESET)
    cur = _RESET
    for x in range(last + 1):
        cell = row[x]
        sgr = sgr_bytes(cell)
        if sgr != cur:
            out += sgr
            cur = sgr
        out += (cell.data or " ").encode("utf-8", "replace")
    out += _RESET + b"\x1b[K"
    return bytes(out)


class Compositor:
    """Tracks what each physical row currently shows and emits row diffs."""

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self._shadow: list[bytes | None] = [None] * rows

    def clear(self) -> bytes:
        """Reset the shadow and return a sequence that clears the screen."""
        self._shadow = [None] * self.rows
        return b"\x1b[2J"

    def row_update(self, y: int, payload: bytes) -> bytes:
        """Return the bytes needed to bring physical row ``y`` to ``payload``."""
        if y < 0 or y >= self.rows or self._shadow[y] == payload:
            return b""
        self._shadow[y] = payload
        return b"\x1b[%d;1H" % (y + 1) + payload
