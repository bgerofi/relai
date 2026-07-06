"""End-to-end: fill the screen, toggle the AI panel on and off, and confirm the
user screen returns to its exact prior state (including trailing blank lines)."""

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


def disp(screen):
    return [row.rstrip() for row in screen.display]


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
    # Fill the screen with numbered lines.
    os.write(m, b"for i in $(seq 1 40); do echo filler_line_$i; done\r")
    pump(m, stream, 3)
    before = disp(screen)
    print("== before toggle ==")
    for r in before:
        if r:
            print(r)

    os.write(m, b"\x0f")   # Ctrl-O open
    pump(m, stream, 2)
    os.write(m, b"\x0f")   # Ctrl-O close
    pump(m, stream, 2)
    after = disp(screen)

    print("\n== after toggle ==")
    for r in after:
        if r:
            print(r)

    match = before == after
    print("\nRESULT:", "PASS" if match else "FAIL")
    if not match:
        for i, (b, a) in enumerate(zip(before, after)):
            if b != a:
                print(f"  row {i}: before={b!r} after={a!r}")

    os.write(m, b"\x03")
    time.sleep(0.3)


if __name__ == "__main__":
    main()
