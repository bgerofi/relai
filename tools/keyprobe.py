"""Print the raw byte sequences the terminal sends for keypresses.

Run directly in a real terminal (not via a pipe):

    python tools/keyprobe.py

Press keys (PageUp, PageDown, Ctrl-PageUp, arrows, ...); press 'q' to quit.
"""

import os
import sys
import termios
import tty


def main() -> None:
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        print("keyprobe must be run in an interactive terminal.")
        return
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    sys.stdout.write("Press keys; 'q' quits.\r\n")
    sys.stdout.flush()
    try:
        while True:
            b = os.read(fd, 64)
            if b == b"q":
                break
            sys.stdout.write(repr(b) + "\r\n")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print("done")


if __name__ == "__main__":
    main()
