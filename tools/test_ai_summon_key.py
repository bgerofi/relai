"""A single Ctrl-O opens the AI panel; a second Ctrl-O closes it."""

import errno, fcntl, os, pty, select, struct, termios, time
import pyte

ROWS, COLS = 24, 90


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


def panel_open(screen):
    text = "\n".join(screen.display)
    return "relai>" in text or "^O/Esc:close" in text


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
    print("panel open before Ctrl-O:", panel_open(screen))

    os.write(m, b"\x0f")  # Ctrl-O -> summon
    pump(m, stream, 3)
    opened = panel_open(screen)
    print("panel open after 1st Ctrl-O:", opened)

    os.write(m, b"\x0f")  # Ctrl-O -> close
    pump(m, stream, 3)
    closed = not panel_open(screen)
    print("panel closed after 2nd Ctrl-O:", closed)

    os.write(m, b"echo READY\r")
    pump(m, stream, 2)
    usable = "READY" in "\n".join(screen.display)
    print("shell usable after close:", usable)

    os.write(m, b"\x03")
    time.sleep(0.3)
    print("RESULT:", "PASS" if (opened and closed and usable) else "FAIL")


if __name__ == "__main__":
    main()
