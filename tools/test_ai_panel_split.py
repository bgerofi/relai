"""Drive the bottom AI panel through a PTY, emulating a real terminal via pyte.

The compositor emits absolute-positioned row diffs, so the harness must render
them like a terminal. We feed all of relai's output into a pyte screen and print
snapshots at each step.
"""

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


def show(screen, label):
    print(f"\n===== {label} =====")
    for i, line in enumerate(screen.display):
        r = line.rstrip()
        if r:
            print(f"row{i:2d}|{r}")


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
    send(b"echo HELLO_42\n", 1.0)
    show(screen, "before panel")

    send(b"\x07", 0.3)   # Ctrl-G
    send(b"a", 0.6)      # open panel
    show(screen, "panel open (height 10)")

    send(b"say hi in three words", 0.4)
    send(b"\r", 40)      # submit question, await reply
    show(screen, "after reply")

    send(b"\x07", 0.2); send(b"\x1b[A", 0.2)  # Ctrl-G Up -> grow
    send(b"\x07", 0.2); send(b"\x1b[A", 0.3)  # Ctrl-G Up -> grow
    show(screen, "after growing panel x2")

    send(b"\x07", 0.2); send(b"\x1b[B", 0.3)  # Ctrl-G Down -> shrink
    show(screen, "after shrinking panel x1")

    send(b"\x1b[5~", 0.3)  # PageUp scroll
    show(screen, "after PageUp scroll")

    send(b"\x07", 0.2); send(b"a", 0.6)  # close panel
    show(screen, "panel closed (app restored)")

    send(b"echo BACK_TO_SHELL\n", 1.0)
    show(screen, "shell usable after close")

    send(b"\x07", 0.2); send(b"a", 0.6)  # re-open panel
    show(screen, "panel re-opened (transcript should persist)")

    os.write(m, b"\x07a\x03exit\n")
    time.sleep(0.3)
    try:
        os.waitpid(pid, os.WNOHANG)
    except OSError:
        pass


if __name__ == "__main__":
    main()
