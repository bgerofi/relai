"""Verify panel scrollback with a reply longer than the panel."""

import errno, fcntl, os, pty, select, struct, termios, time
import pyte

ROWS, COLS = 24, 80


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


def panel_rows(screen):
    return [l.rstrip() for l in screen.display if l.strip()]


def main():
    pid, m = pty.fork()
    if pid == 0:
        os.environ["PS1"] = "$ "
        os.environ["TERM"] = "xterm"
        os.execvp("relai", ["relai", "--", "bash", "--norc", "-i"])
    fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)

    def send(data, wait):
        os.write(m, data)
        pump(m, stream, wait)

    pump(m, stream, 8)
    send(b"\x07", 0.3); send(b"a", 0.5)  # open panel
    send(b"list the numbers one through forty, each on its own line", 0.4)
    send(b"\r", 45)

    def content(label):
        print(f"\n== {label} ==")
        for l in screen.display:
            r = l.rstrip()
            if r:
                print(r)

    content("bottom of transcript (newest)")
    send(b"\x1b[5~", 0.4)  # PageUp
    content("after PageUp")
    send(b"\x1b[5~", 0.4)  # PageUp again
    content("after PageUp x2")
    send(b"\x1b[6~", 0.4); send(b"\x1b[6~", 0.4)  # PageDown back
    content("after PageDown x2 (back to bottom)")

    os.write(m, b"\x07a")
    time.sleep(0.3)


if __name__ == "__main__":
    main()
