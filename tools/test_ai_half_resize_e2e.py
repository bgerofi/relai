"""End-to-end: verify Ctrl-G PageUp resizes the AI panel to half the screen and
Ctrl-G PageDown restores the original height.

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_ai_half_resize_e2e.py
"""

import errno, fcntl, os, pty, select, struct, termios, time
import pyte

ROWS, COLS = 24, 90
PREFIX = b"\x07"      # Ctrl-G
PGUP = b"\x1b[5~"
PGDN = b"\x1b[6~"


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
    """Index of the panel header row (the reverse-video 'relai ·' bar)."""
    for i, row in enumerate(screen.display):
        if "relai" in row and ("close" in row or "resize" in row):
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
        os.execvp("relai", ["relai", "--", "bash", "--norc", "-i"])
    fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)

    pump(m, stream, 6)
    os.write(m, b"\x0f")  # open panel
    pump(m, stream, 2)

    original = panel_height(screen)

    os.write(m, PREFIX + PGUP)  # -> half screen
    pump(m, stream, 1.5)
    half = panel_height(screen)

    os.write(m, PREFIX + PGDN)  # -> restore
    pump(m, stream, 1.5)
    restored = panel_height(screen)

    print(f"original={original} half={half} restored={restored}  (rows={ROWS})")
    expect_half = ROWS // 2
    ok = (
        original == 10
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
