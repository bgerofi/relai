"""The model should cat README, get the settled screen back, and summarize it."""

import errno, fcntl, os, pty, select, struct, termios, time
import pyte

ROWS, COLS = 30, 90


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


def show(screen, label):
    print(f"\n== {label} ==")
    for l in screen.display:
        r = l.rstrip()
        if r:
            print(r)


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
    send(b"\x07", 0.3); send(b"a", 0.5)
    send(b"What is this project about based on the README? "
         b"Read it in the terminal first.", 0.4)
    send(b"\r", 60)  # allow inject + settle polls + final answer
    show(screen, "reply (should summarize README, not ask user)")
    os.write(m, b"\x07a\x03")
    time.sleep(0.3)


if __name__ == "__main__":
    main()
