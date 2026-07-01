"""Drive relai through a PTY to test the Ctrl-G a AI panel end-to-end.

Uses select() with timeouts and an overall wall-clock cap so it can never hang.
Prints a transcript tail of what relai rendered, then a PASS/FAIL summary.
"""

import errno
import fcntl
import os
import pty
import select
import struct
import sys
import termios
import time

OVERALL_TIMEOUT = 90.0  # hard cap for the whole test
IDLE_READ = 0.5


def drain(fd, seconds):
    """Read whatever is available from fd for up to `seconds`, return bytes."""
    out = bytearray()
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if fd in r:
            try:
                data = os.read(fd, 65536)
            except OSError as e:
                if e.errno == errno.EIO:
                    break
                raise
            if not data:
                break
            out += data
    return bytes(out)


def main():
    argv = ["relai", "--", "bash", "--norc", "-i"]
    pid, master = pty.fork()
    if pid == 0:
        os.environ["PS1"] = "$ "
        os.environ["TERM"] = "xterm"
        os.execvp(argv[0], argv)

    # Give the PTY a realistic window size (24x80).
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))

    transcript = bytearray()
    start = time.time()

    def send(data, wait=IDLE_READ):
        os.write(master, data)
        transcript.extend(drain(master, wait))

    try:
        transcript.extend(drain(master, 8.0))          # relai + bash start
        send(b"echo HELLO_FROM_SCREEN_42\n", 1.0)      # screen content
        send(b"\x07", 0.3)                             # Ctrl-G
        send(b"a", 0.8)                                # AI panel
        send(b"What number appears on the screen?", 0.5)
        send(b"\r", 0.3)                              # submit
        remaining = max(1.0, min(60.0, OVERALL_TIMEOUT - (time.time() - start)))
        transcript.extend(drain(master, remaining))   # await reply
        send(b"q", 0.5)                               # close reply
        send(b"exit\n", 1.0)                          # exit bash
    finally:
        try:
            os.write(master, b"\x03exit\n")
        except OSError:
            pass
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass

    text = transcript.decode("utf-8", "replace")
    sys.stdout.write("===== RAW TRANSCRIPT (tail) =====\n")
    sys.stdout.write(text[-3000:])
    sys.stdout.write("\n===== SUMMARY =====\n")
    after_answer = text.split("Q: What number")[-1] if "Q: What number" in text else ""
    checks = {
        "panel prompt shown": "Ask relai:" in text,
        "thinking indicator": "Thinking" in text,
        "answer view opened": "relai answer" in text,
        "answer mentions 42": "42" in after_answer,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
