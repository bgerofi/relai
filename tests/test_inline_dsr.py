"""Inline AI test harness that emulates a real terminal (answers DSR ESC[6n).

ludvart now asks the terminal for the true cursor position; a dumb pipe never
answers, so this harness feeds all child output through pyte and replies to
ESC[6n with the pyte cursor position, exactly like a real terminal.
"""

import errno, fcntl, os, pty, re, select, struct, termios, time
import pyte

_DSR = re.compile(rb"\x1b\[6n")


def run(fd, screen, stream, seconds):
    """Pump the PTY for a while, feeding pyte and answering DSR queries."""
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
            if _DSR.search(d):
                row, col = screen.cursor.y + 1, screen.cursor.x + 1
                os.write(fd, f"\x1b[{row};{col}R".encode("ascii"))


def scenario(ps1, label, partial=b""):
    pid, m = pty.fork()
    if pid == 0:
        os.environ["PS1"] = ps1
        os.environ["TERM"] = "xterm"
        os.execvp("ludvart", ["ludvart", "--", "bash", "--norc", "-i"])
    fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    screen = pyte.Screen(80, 24)
    stream = pyte.ByteStream(screen)

    run(m, screen, stream, 8)
    os.write(m, b"echo HELLO_42\n"); run(m, screen, stream, 1)
    if partial:
        os.write(m, partial); run(m, screen, stream, 0.4)
    os.write(m, b"\x07"); run(m, screen, stream, 0.3)
    os.write(m, b"a"); run(m, screen, stream, 0.5)
    os.write(m, b"What number is shown?"); run(m, screen, stream, 0.4)
    os.write(m, b"\r"); run(m, screen, stream, 45)
    os.write(m, b"Z"); run(m, screen, stream, 0.6)  # prove line editor intact

    print(f"\n===== {label} =====")
    for i, line in enumerate(screen.display):
        r = line.rstrip()
        if r:
            print(f"row{i:2d}|{r}")
    os.write(m, b"\x03exit\n"); time.sleep(0.3)
    try:
        os.waitpid(pid, os.WNOHANG)
    except OSError:
        pass


def main():
    scenario("$ ", "SINGLE-LINE PROMPT")
    scenario("[demo]\\n$ ", "TWO-LINE PROMPT")
    scenario("$ ", "SINGLE-LINE + PARTIAL BUFFER 'ls -la'", partial=b"ls -la")


if __name__ == "__main__":
    main()
