"""Drive relai through a PTY to test the inline Ctrl-G a AI chat end-to-end.

Uses select() with timeouts and an overall wall-clock cap so it can never hang.
Verifies the inline exchange appears in the scroll flow and that relai did NOT
switch to the alternate screen for a line-oriented (shell) session.
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

OVERALL_TIMEOUT = 90.0
IDLE_READ = 0.5


def drain(fd, seconds):
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
        send(b"a", 0.5)                                # inline AI
        send(b"What number appears on the screen?", 0.5)
        send(b"\r", 0.3)                              # submit
        remaining = max(1.0, min(60.0, OVERALL_TIMEOUT - (time.time() - start)))
        transcript.extend(drain(master, remaining))   # await reply
        send(b"echo AFTER_AI_OK\n", 1.0)             # shell still usable
        send(b"exit\n", 1.0)
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
    lower = text.lower()
    sys.stdout.write("===== RAW TRANSCRIPT (tail) =====\n")
    sys.stdout.write(repr(text[-2500:]))
    sys.stdout.write("\n===== SUMMARY =====\n")
    checks = {
        "inline prompt shown": "relai> " in text,
        "question echoed": "What number appears" in text,
        "thinking indicator": "thinking" in lower,
        "answer mentions 42": "42" in lower.split("thinking")[-1],
        "no alt-screen switch": "\x1b[?1049h" not in text,
        "shell usable after": "AFTER_AI_OK" in text,
    }
    ok = True
    for k, v in checks.items():
        ok = ok and v
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  elapsed: {time.time() - start:.1f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
