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
import threading
import tty
from typing import TYPE_CHECKING, Sequence

import pyte

from .overlay import ScrollbackViewer
from .panel import AiPanel
from .render import Compositor, render_row
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

    Falls back to a sane default if the size cannot be queried or is reported as
    zero (e.g. an unsized PTY), so callers never receive a 0 dimension.
    """
    try:
        packed = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
        rows, cols, _, _ = struct.unpack("HHHH", packed)
        if rows and cols:
            return rows, cols
    except OSError:
        pass
    size = shutil.get_terminal_size(fallback=(80, 24))
    rows = size.lines if size.lines > 0 else 24
    cols = size.columns if size.columns > 0 else 80
    return rows, cols


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
        # AI panel state. ``_panel`` is non-None only while the split is open;
        # ``_panel_messages`` keeps the transcript alive across toggles.
        self._panel: AiPanel | None = None
        self._panel_closing = False
        self._panel_messages: list[tuple[str, str]] = []
        self._compositor: Compositor | None = None
        self._panel_height = 10
        self._phys_rows = 0
        self._phys_cols = 0
        self._ai_ask = None
        # Background LLM request while the panel spinner animates.
        self._ask_thread: threading.Thread | None = None
        self._ask_result = ""
        self._ask_done = threading.Event()

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
        elif byte in (b"a", b"A"):
            self._open_panel()
        # Unknown command letters are ignored (not forwarded), matching the
        # screen/tmux convention of swallowing unrecognized prefix commands.

    def _open_scrollback_viewer(self) -> None:
        """Pause passthrough and show the scrollback overlay."""
        rows, cols = _get_winsize(self._stdout_fd)
        lines = self.screen.full_text(include_scrollback=True)
        viewer = ScrollbackViewer(self._stdout_fd, self._stdin_fd, rows, cols)
        viewer.show(lines)

    def _ai_ask_callback(self):
        """Return the ``ask`` callable and a short provider label."""
        if self.llm is None:
            def ask(_question: str) -> str:
                return (
                    "No LLM provider is configured. Set the "
                    "{OPENAI,ANTHROPIC,GOOGLE,CUSTOM}_API_URL/_API_KEY/_MODEL "
                    "environment variables and restart relai."
                )

            return ask, "no LLM"
        return self._ask_llm, f"{self.llm.name}:{self.llm.model}"

    # -- AI panel (bottom split) --------------------------------------------

    def _open_panel(self) -> None:
        """Open the AI panel as a bottom split and run it until it is closed.

        The application is resized to the region above the panel (it just sees a
        smaller terminal, via SIGWINCH) and relai switches from passthrough to
        compositing: the child draws into the pyte model, which relai renders
        onto the top region while owning the panel rows below.
        """
        rows, cols = _get_winsize(self._stdout_fd)
        if rows < 5 or cols < 10:
            return  # too small to usefully split
        self._phys_rows, self._phys_cols = rows, cols
        height = max(3, min(self._panel_height, rows - 2))
        self._panel_height = height

        ask, provider = self._ai_ask_callback()
        self._ai_ask = ask
        self._panel = AiPanel(cols, height, provider)
        self._panel.restore(self._panel_messages)
        self._panel_closing = False

        self._apply_split_size()
        self._compositor = Compositor(rows, cols)
        self._write_all(self._stdout_fd, b"\x1b[?25h" + self._compositor.clear())
        self._render_split()
        try:
            self._split_loop()
        finally:
            self._leave_split()

    def _apply_split_size(self) -> None:
        """Resize the model and child PTY to the region above the panel."""
        app_rows = max(1, self._phys_rows - self._panel_height)
        self.screen.resize(app_rows, self._phys_cols)
        _set_winsize(self._master_fd, app_rows, self._phys_cols)

    def _split_loop(self) -> None:
        master = self._master_fd
        stdin = self._stdin_fd
        while not self._panel_closing:
            if self._resized:
                self._handle_split_resize()
            # While waiting on the LLM, wake up periodically to advance the
            # spinner animation.
            timeout = 0.12 if (self._panel and self._panel.thinking) else None
            try:
                readable, _, _ = select.select([master, stdin], [], [], timeout)
            except InterruptedError:
                continue
            if master in readable:
                data = self._read(master)
                if data is None:  # child exited
                    self._panel_closing = True
                    break
                if data:
                    self.stream.feed(data)
                    self._render_split()
            if stdin in readable:
                data = self._read(stdin)
                if data:
                    self._panel_input(data)
                    if not self._panel_closing:
                        self._render_split()
            if self._panel is not None and self._panel.thinking:
                if self._ask_done.is_set():
                    self._finish_ask()
                else:
                    self._panel.tick += 1
                    self._render_split()

    def _handle_split_resize(self) -> None:
        """Re-lay-out the split after the real terminal changed size."""
        self._resized = False
        rows, cols = _get_winsize(self._stdout_fd)
        self._phys_rows, self._phys_cols = rows, cols
        self._panel_height = max(3, min(self._panel_height, rows - 2))
        self._panel.height = self._panel_height
        self._panel.set_cols(cols)
        self._apply_split_size()
        self._compositor = Compositor(rows, cols)
        self._write_all(self._stdout_fd, self._compositor.clear())
        self._render_split()

    def _resize_panel(self, delta: int) -> None:
        """Grow (delta>0) or shrink (delta<0) the panel by ``delta`` rows."""
        height = max(3, min(self._panel_height + delta, self._phys_rows - 2))
        if height == self._panel_height:
            return
        self._panel_height = height
        self._panel.height = height
        self._apply_split_size()
        self._compositor = Compositor(self._phys_rows, self._phys_cols)
        self._write_all(self._stdout_fd, self._compositor.clear())
        self._render_split()

    def _render_split(self) -> None:
        """Composite the app region (from the model) and the panel to screen."""
        comp = self._compositor
        panel = self._panel
        if comp is None or panel is None:
            return
        cols = self._phys_cols
        app_rows = self.screen.lines
        out = bytearray()
        for y in range(app_rows):
            out += comp.row_update(y, render_row(self.screen, y, cols))
        for i, payload in enumerate(panel.render(panel.height, cols)):
            out += comp.row_update(app_rows + i, payload)
        out += b"\x1b[%d;%dH" % (self._phys_rows, panel.cursor_col())
        self._write_all(self._stdout_fd, out)

    def _leave_split(self) -> None:
        """Tear down the split: resize the app back and restore the screen."""
        rows, cols = self._phys_rows, self._phys_cols
        if self._panel is not None:
            self._panel_messages = self._panel.messages  # keep for next toggle
        self.screen.resize(rows, cols)
        _set_winsize(self._master_fd, rows, cols)
        self._compositor = None
        self._panel = None
        # Repaint the full-size app from the model; the child's own SIGWINCH
        # redraw will then flow in via passthrough and stay consistent.
        out = bytearray(b"\x1b[?25h\x1b[2J")
        for y in range(rows):
            out += b"\x1b[%d;1H" % (y + 1) + render_row(self.screen, y, cols)
        out += b"\x1b[%d;%dH" % (self.screen.cursor.y + 1, self.screen.cursor.x + 1)
        self._write_all(self._stdout_fd, out)

    # -- panel input ---------------------------------------------------------

    def _panel_input(self, data: bytes) -> None:
        """Route a stdin read to the panel (editing, scrolling, commands)."""
        if self._awaiting_command:
            self._awaiting_command = False
            self._panel_command(data)
            return
        if data == self.prefix:
            self._awaiting_command = True
            return
        if data[:1] == self.prefix and len(data) > 1:
            self._panel_command(data[1:])
            return
        self._panel_key(data)

    def _panel_command(self, key: bytes) -> None:
        """Handle a prefix command while the panel is open."""
        if key in (b"a", b"A"):
            self._panel_closing = True  # toggle closed
        elif key in (b"\x1b[A", b"\x1bOA"):  # Up -> grow panel
            self._resize_panel(1)
        elif key in (b"\x1b[B", b"\x1bOB"):  # Down -> shrink panel
            self._resize_panel(-1)

    def _panel_key(self, key: bytes) -> None:
        """Handle a normal keystroke while the panel is focused."""
        panel = self._panel
        if key in (b"\r", b"\n"):
            self._panel_submit()
        elif key in (b"\x7f", b"\x08"):
            panel.backspace()
        elif key == b"\x1b":  # bare Esc closes
            self._panel_closing = True
        elif key in (b"\x1b[A", b"\x1bOA"):
            panel.scroll_up(1)
        elif key in (b"\x1b[B", b"\x1bOB"):
            panel.scroll_down(1)
        elif key == b"\x1b[5~":  # PageUp
            panel.scroll_up(max(1, panel.height - 2))
        elif key == b"\x1b[6~":  # PageDown
            panel.scroll_down(max(1, panel.height - 2))
        elif key[:1] == b"\x1b":
            return  # ignore other escape sequences
        else:
            try:
                text = key.decode("utf-8")
            except UnicodeDecodeError:
                return
            text = "".join(ch for ch in text if ch >= " ")
            if text:
                panel.type_text(text)

    def _panel_submit(self) -> None:
        """Send the typed question to the LLM on a background thread.

        The request runs off the render loop so the spinner keeps animating and
        the application region keeps updating while we wait for the reply.
        """
        panel = self._panel
        if panel.thinking:
            return
        question = panel.take_input()
        if not question:
            return
        panel.add_user(question)
        panel.thinking = True
        panel.tick = 0
        self._render_split()  # show the question and the spinner immediately

        ask = self._ai_ask
        self._ask_done = threading.Event()

        def worker() -> None:
            try:
                result = ask(question)
            except Exception as exc:  # surfaced to the user, never crashes relai
                result = f"[relai] request failed: {exc}"
            self._ask_result = result
            self._ask_done.set()

        self._ask_thread = threading.Thread(target=worker, daemon=True)
        self._ask_thread.start()

    def _finish_ask(self) -> None:
        """Deliver the completed background reply into the panel."""
        if self._ask_thread is not None:
            self._ask_thread.join(timeout=1)
            self._ask_thread = None
        panel = self._panel
        if panel is None:
            return
        panel.thinking = False
        panel.add_reply(self._ask_result)
        self._render_split()

    def _ask_llm(self, question: str) -> str:
        """Send the current screen plus the user's question to the LLM."""
        screen_text = self.snapshot_text()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are relai, an assistant embedded in a terminal. The user "
                    "is looking at the terminal screen shown below. Answer their "
                    "question about it concisely and helpfully.\n\n"
                    "--- BEGIN TERMINAL SCREEN ---\n"
                    f"{screen_text}\n"
                    "--- END TERMINAL SCREEN ---"
                ),
            },
            {"role": "user", "content": question},
        ]
        return self.llm.complete(messages)

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
