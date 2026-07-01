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
from typing import TYPE_CHECKING, Sequence

import pyte

from .overlay import ScrollbackViewer
from .screen import RelaiScreen

if TYPE_CHECKING:
    from .llm import LLMClient

# relai commands are entered with a prefix key (like screen/tmux) followed by a
# command letter. A single-byte control character is used as the prefix so no
# terminal emulator remaps it and it survives SSH and nested screen/tmux.
#
# Default prefix: Ctrl-G (0x07). Commands:
#   <prefix> s          open the scrollback viewer
#   <prefix> <prefix>   send a literal prefix byte to the child
DEFAULT_PREFIX = b"\x07"  # Ctrl-G


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
    prefix:
        The single-byte prefix key that introduces a relai command. Defaults to
        Ctrl-G. Pressing it twice sends a literal prefix byte to the child.
    llm:
        An optional, already-verified LLM client. When ``None``, relai runs as a
        plain relay with AI features disabled.
    """

    #: How many bytes to read from a fd at a time.
    READ_SIZE = 65536

    def __init__(
        self,
        command: Sequence[str],
        prefix: bytes = DEFAULT_PREFIX,
        llm: "LLMClient | None" = None,
    ) -> None:
        self.command = list(command)
        self.prefix = prefix
        self.llm = llm
        self._child_pid: int = -1
        self._master_fd: int = -1
        self._stdin_fd = sys.stdin.fileno()
        self._stdout_fd = sys.stdout.fileno()
        self._old_term_attrs: list | None = None
        self._resized = False
        # True after the prefix key was pressed, while waiting for the command
        # letter (the next byte selects the relai command).
        self._awaiting_command = False

        rows, cols = _get_winsize(self._stdout_fd)
        # pyte keeps a live model of what the child has drawn on screen, plus a
        # scrollback of normal-buffer output that scrolled off the top.
        self.screen = RelaiScreen(cols, rows)
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

    def scrollback_text(self) -> str:
        """Return logical output that scrolled off the top (oldest first).

        This is normal-buffer scrollback only; output produced by full-screen
        alternate-buffer apps (vim/htop/less) is intentionally excluded.
        """
        return "\n".join(self.screen.scrollback_lines())

    def snapshot(self, include_scrollback: bool = False) -> dict:
        """Return a structured snapshot of the current screen state.

        Includes the plain-text view, the terminal size, the cursor position,
        and whether a full-screen (alternate-buffer) app is active -- enough
        context for an agent to reason about the screen and decide where input
        would go.

        Parameters
        ----------
        include_scrollback:
            If true, also include the normal-buffer scrollback text under the
            ``"scrollback"`` key.
        """
        snap = {
            "rows": self.screen.lines,
            "cols": self.screen.columns,
            "cursor": {"row": self.screen.cursor.y, "col": self.screen.cursor.x},
            "alt_screen": self.screen.in_alt_screen,
            "text": self.snapshot_text(),
        }
        if include_scrollback:
            snap["scrollback"] = self.scrollback_text()
        return snap

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
                    self._handle_input(data)

        return self._reap_child()

    # -- input handling / prefix commands -----------------------------------

    def _handle_input(self, data: bytes) -> None:
        """Forward human input to the child, intercepting relai prefix commands.

        Input is processed one byte at a time so the prefix key and its
        following command letter are recognized even when they arrive in the
        same read or split across reads. The prefix key itself is never
        forwarded unless pressed twice (``<prefix> <prefix>`` sends a literal).
        """
        for i in range(len(data)):
            byte = data[i : i + 1]
            if self._awaiting_command:
                self._awaiting_command = False
                self._run_prefix_command(byte)
            elif byte == self.prefix:
                # Enter command mode; the next byte selects the command.
                self._awaiting_command = True
            else:
                self._write_all(self._master_fd, byte)

    def _run_prefix_command(self, byte: bytes) -> None:
        """Handle the command byte following the prefix key."""
        if byte == self.prefix:
            # Doubled prefix -> send a literal prefix byte to the child.
            self._write_all(self._master_fd, self.prefix)
        elif byte in (b"s", b"S"):
            self._open_scrollback_viewer()
        # Unknown command letters are ignored (not forwarded), matching the
        # screen/tmux convention of swallowing unrecognized prefix commands.

    def _open_scrollback_viewer(self) -> None:
        """Pause passthrough and show the scrollback overlay."""
        rows, cols = _get_winsize(self._stdout_fd)
        lines = self.screen.full_text(include_scrollback=True)
        viewer = ScrollbackViewer(self._stdout_fd, self._stdin_fd, rows, cols)
        viewer.show(lines)

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
