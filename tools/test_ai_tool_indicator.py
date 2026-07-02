"""Capture the panel indicator; it should show 'Calling inject_input' mid-tool."""

import errno, fcntl, os, pty, select, struct, termios, time
import pyte

ROWS, COLS = 24, 90


def pump_capture(fd, stream, seconds, needle, hits):
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.05)
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
        # Scan the live screen for the indicator text.
        joined = "\n".join(l.rstrip() for l in stream.listener.display)
        if needle in joined:
            hits.append(True)


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
        pump_capture(m, stream, wait, "Calling", hits)

    hits = []
    pump_capture(m, stream, 8, "Calling", hits)
    send(b"\x07", 0.3); send(b"a", 0.5)
    send(b"Run 'ls' in the terminal for me.", 0.4)
    os.write(m, b"\r")
    pump_capture(m, stream, 40, "Calling", hits)

    print("saw 'Calling ...' indicator:", bool(hits))
    print("\n== final screen ==")
    for l in screen.display:
        r = l.rstrip()
        if r:
            print(r)
    os.write(m, b"\x07a\x03")
    time.sleep(0.3)


if __name__ == "__main__":
    main()
