"""Transparent PTY relay.

Spawns a child command in a pseudo-terminal and shuttles bytes between the real
terminal and the child. Output is passed through verbatim (so any program --
including full-screen ncurses apps and nested ssh/tmux sessions -- behaves
exactly as if relai were not there) while also being fed into a ``pyte`` screen
model that maintains a live 2D view of the terminal. That screen model is the
foundation the AI overlay/agent will later read from.
"""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import select
import shutil
import signal
import struct
import sys
import termios
import tty
from typing import Sequence

import pyte


def _get_winsize(fd: int) -> tuple[int, int]:
    """Return (rows, cols) for the terminal on ``fd``.

    Falls back to a sane default if the size cannot be queried.
    """
    try:
        packed = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
        rows, cols, _, _ = struct.unpack("HHHH", packed)
        if rows and cols:
            return rows, cols
    except OSError:
        pass
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.lines, size.columns


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    """Apply the window size ``(rows, cols)`` to the PTY on ``fd``."""
    packed = struct.pack("HHHH", rows, cols, 0, 0)
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)
    except OSError:
        pass


class Relai:
    """A transparent PTY relay around a single child command.

    Parameters
    ----------
    command:
        The argv of the command to spawn (e.g. ``["bash"]`` or
        ``["ssh", "host"]``).
    """

    #: How many bytes to read from a fd at a time.
    READ_SIZE = 65536

    def __init__(self, command: Sequence[str]) -> None:
        self.command = list(command)
        self._child_pid: int = -1
        self._master_fd: int = -1
        self._stdin_fd = sys.stdin.fileno()
        self._stdout_fd = sys.stdout.fileno()
        self._old_term_attrs: list | None = None
        self._resized = False

        rows, cols = _get_winsize(self._stdout_fd)
        # pyte keeps a live model of what the child has drawn on screen.
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.ByteStream(self.screen)

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> int:
        """Spawn the child and relay until it exits. Returns its exit status."""
        self._child_pid, self._master_fd = pty.fork()
        if self._child_pid == 0:
            # Child process: exec the target command. On success this never
            # returns; the child's stdio is already wired to the PTY slave.
            try:
                os.execvp(self.command[0], self.command)
            except OSError as exc:
                sys.stderr.write(f"relai: cannot run {self.command[0]!r}: {exc}\n")
                os._exit(127)

        # Parent process.
        rows, cols = _get_winsize(self._stdout_fd)
        _set_winsize(self._master_fd, rows, cols)

        self._install_raw_mode()
        self._install_winch_handler()
        try:
            return self._loop()
        finally:
            self._restore_term()

    # -- screen inspection (for the AI layer) --------------------------------

    def snapshot_text(self, trim_trailing_blank_lines: bool = True) -> str:
        """Return the visible screen as plain text.

        This is what the user currently sees, rendered by the ``pyte`` screen
        model -- correct even for full-screen ncurses apps. It is the natural
        input to hand to an LLM.

        Parameters
        ----------
        trim_trailing_blank_lines:
            If true, drop empty rows at the bottom so short screens don't come
            with a block of blank padding.
        """
        lines = list(self.screen.display)
        if trim_trailing_blank_lines:
            while lines and not lines[-1].strip():
                lines.pop()
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Return a structured snapshot of the current screen state.

        Includes the plain-text view, the terminal size, and the cursor
        position -- enough context for an agent to reason about the screen and
        decide where input would go.
        """
        return {
            "rows": self.screen.lines,
            "cols": self.screen.columns,
            "cursor": {"row": self.screen.cursor.y, "col": self.screen.cursor.x},
            "text": self.snapshot_text(),
        }

    # -- terminal setup ------------------------------------------------------

    def _install_raw_mode(self) -> None:
        """Put the real terminal into raw mode so keys pass through untouched."""
        if not os.isatty(self._stdin_fd):
            return
        self._old_term_attrs = termios.tcgetattr(self._stdin_fd)
        tty.setraw(self._stdin_fd)

    def _restore_term(self) -> None:
        """Restore the terminal attributes saved by :meth:`_install_raw_mode`."""
        if self._old_term_attrs is not None:
            termios.tcsetattr(
                self._stdin_fd, termios.TCSAFLUSH, self._old_term_attrs
            )
            self._old_term_attrs = None

    def _install_winch_handler(self) -> None:
        """Propagate real-terminal resizes to the child PTY and pyte screen."""

        def _handler(signum, frame):  # noqa: ANN001 - signal handler signature
            self._resized = True

        signal.signal(signal.SIGWINCH, _handler)

    def _handle_resize(self) -> None:
        rows, cols = _get_winsize(self._stdout_fd)
        _set_winsize(self._master_fd, rows, cols)
        self.screen.resize(rows, cols)
        self._resized = False

    # -- main loop -----------------------------------------------------------

    def _loop(self) -> int:
        """Shuttle bytes between stdin and the PTY master until EOF/child exit."""
        master = self._master_fd
        stdin = self._stdin_fd

        while True:
            if self._resized:
                self._handle_resize()

            try:
                readable, _, _ = select.select([master, stdin], [], [])
            except InterruptedError:
                # Interrupted by SIGWINCH (or similar); loop to handle it.
                continue

            if master in readable:
                data = self._read(master)
                if data is None:  # child closed the PTY -> it has exited
                    break
                if data:
                    # Feed the screen model, then pass through verbatim.
                    self.stream.feed(data)
                    self._write_all(self._stdout_fd, data)

            if stdin in readable:
                data = self._read(stdin)
                if data:
                    # Human input goes straight to the child.
                    self._write_all(master, data)

        return self._reap_child()

    def _read(self, fd: int) -> bytes | None:
        """Read from ``fd``. Return ``None`` on EOF/child-gone, else bytes."""
        try:
            data = os.read(fd, self.READ_SIZE)
        except OSError as exc:
            # On Linux, reading the master after the child exits raises EIO.
            if exc.errno == errno.EIO:
                return None
            if exc.errno == errno.EAGAIN:
                return b""
            raise
        if not data:
            return None
        return data

    def _write_all(self, fd: int, data: bytes) -> None:
        """Write all of ``data`` to ``fd``, handling short writes."""
        while data:
            try:
                n = os.write(fd, data)
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    continue
                raise
            data = data[n:]

    def _reap_child(self) -> int:
        """Wait for the child and translate its status into an exit code."""
        try:
            _, status = os.waitpid(self._child_pid, 0)
        except OSError:
            return 0
        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status)
        if os.WIFSIGNALED(status):
            return 128 + os.WTERMSIG(status)
        return 0
