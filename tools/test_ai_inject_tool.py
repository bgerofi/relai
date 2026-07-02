"""Verify the LLM can call the inject_input tool to run a command in the shell."""

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
    # Ask the model to run a command for us via the inject tool.
    send(b"\x07", 0.3); send(b"a", 0.5)  # open panel
    send(b"Please create a file called relai_tool_ok.txt in the current "
         b"directory using the touch command. Run it for me.", 0.4)
    send(b"\r", 45)  # allow tool round-trips
    show(screen, "after tool-driven request")
    # Close panel and verify the file was actually created in the shell.
    os.write(m, b"\x07a")
    pump(m, stream, 1)
    send(b"ls relai_tool_ok.txt\r", 3)
    show(screen, "verify file exists in shell")
    os.write(m, b"rm -f relai_tool_ok.txt\r")
    time.sleep(0.3)
    os.write(m, b"\x03")
    time.sleep(0.3)


if __name__ == "__main__":
    main()
