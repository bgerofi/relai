"""End-to-end: verify the AI panel opens at half the screen height by default,
Ctrl-G PageUp resizes it to half the screen, and Ctrl-G PageDown restores the
previous height.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tools/test_ai_half_resize_e2e.py
"""

import errno, fcntl, os, pty, select, struct, termios, time
import pyte

ROWS, COLS = 24, 90
PREFIX = b"\x07"      # Ctrl-G
PGUP = b"\x1b[5~"
PGDN = b"\x1b[6~"
DOWN = b"\x1b[B"      # Ctrl-G Down -> shrink panel by one row


def pump(fd, stream, seconds):
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if fd in r:
            try:
                d = os.read(fd, 65536)
            except OSError as e:
                if e.errno == errno.EIO:
                    break
                raise
            if not d:
                break
            stream.feed(d)


def header_row(screen):
    """Index of the panel header row (the reverse-video 'ludvart ·' bar)."""
    for i, row in enumerate(screen.display):
        if "ludvart" in row and ("close" in row or "resize" in row):
            return i
    return -1


def panel_height(screen):
    """Panel height = rows from the header down to the bottom of the screen."""
    h = header_row(screen)
    return -1 if h < 0 else ROWS - h


def main():
    pid, m = pty.fork()
    if pid == 0:
        os.environ["PS1"] = "$ "
        os.environ["TERM"] = "xterm"
        os.execvp("ludvart", ["ludvart", "--", "bash", "--norc", "-i"])
    fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)

    pump(m, stream, 6)
    os.write(m, b"\x0f")  # open panel (defaults to half the screen height)
    pump(m, stream, 2)

    default = panel_height(screen)  # new default: half the screen

    # Shrink several rows so the half-resize and restore are observable (the
    # panel already opens at half, so PageUp from there would be a no-op).
    for _ in range(4):
        os.write(m, PREFIX + DOWN)  # Ctrl-G Down -> shrink by 1
        pump(m, stream, 0.4)
    original = panel_height(screen)

    os.write(m, PREFIX + PGUP)  # -> half screen
    pump(m, stream, 1.5)
    half = panel_height(screen)

    os.write(m, PREFIX + PGDN)  # -> restore
    pump(m, stream, 1.5)
    restored = panel_height(screen)

    print(
        f"default={default} original={original} half={half} "
        f"restored={restored}  (rows={ROWS})"
    )
    expect_half = ROWS // 2
    ok = (
        default == expect_half
        and original == expect_half - 4
        and half == expect_half
        and restored == original
    )
    print("RESULT:", "PASS" if ok else "FAIL")

    os.write(m, b"\x0f")
    time.sleep(0.2)
    os.write(m, b"\x03")
    time.sleep(0.3)


if __name__ == "__main__":
    main()
