"""Reproduce: last number disappears when toggling the panel during output."""

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


def dump(screen, label):
    print(f"\n== {label} ==")
    for i, line in enumerate(screen.display):
        r = line.rstrip()
        if r:
            print(f"{i:2d}|{r}")


def main():
    pid, m = pty.fork()
    if pid == 0:
        os.environ["PS1"] = "$ "
        os.environ["TERM"] = "xterm"
        os.execvp("relai", ["relai", "--no-llm", "--", "bash", "--norc", "-i"])
    fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)

    def send(data, wait):
        os.write(m, data)
        pump(m, stream, wait)

    pump(m, stream, 6)
    send(b'for i in $(seq 1 100); do echo "$i"; sleep 0.4; done\n', 5.0)
    dump(screen, "running loop (before toggle)")

    send(b"\x07", 0.2); send(b"a", 0.6)  # open panel mid-output
    dump(screen, "just after opening panel")

    pump(m, stream, 2.0)
    dump(screen, "a couple numbers later (panel open)")

    os.write(m, b"\x07a\x03")
    time.sleep(0.3)


if __name__ == "__main__":
    main()
