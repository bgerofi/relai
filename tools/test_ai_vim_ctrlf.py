"""In vim, the model should page down (Ctrl-F) without modifying the buffer."""

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


def show(screen, label):
    print(f"\n== {label} ==")
    for l in screen.display:
        r = l.rstrip()
        if r:
            print(r)


def main():
    # Build a long numbered file to make paging visible.
    with open("/tmp/ludvart_vim_test.txt", "w") as f:
        for i in range(1, 201):
            f.write(f"line_{i:03d}\n")

    pid, m = pty.fork()
    if pid == 0:
        os.environ["PS1"] = "$ "
        os.environ["TERM"] = "xterm"
        os.execvp("ludvart", ["ludvart", "--", "bash", "--norc", "-i"])
    fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)

    def send(data, wait):
        os.write(m, data)
        pump(m, stream, wait)

    pump(m, stream, 8)
    send(b"vim -u NONE /tmp/ludvart_vim_test.txt\r", 3)
    show(screen, "vim opened (top of file)")
    send(b"\x07", 0.3); send(b"a", 0.5)
    send(b"Page down one screen in vim (this is a read-only view, do NOT "
         b"modify the buffer).", 0.4)
    send(b"\r", 60)
    show(screen, "after model pages down (expect later lines, no [+] modified)")
    # Quit vim without saving.
    os.write(m, b"\x07a")
    pump(m, stream, 1)
    os.write(m, b"\x1b:q!\r")
    pump(m, stream, 2)
    os.write(m, b"\x03")
    time.sleep(0.3)


if __name__ == "__main__":
    main()
