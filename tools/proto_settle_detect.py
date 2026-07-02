"""Prototype + benchmark: faster command-completion detection.

Compares the CURRENT approach (poll every 0.25s, ask the LLM DONE/RUNNING on
every poll) against a HEURISTIC detector that:

  1. Learns the prompt from the cursor line captured *before* injection
     (general: works for bash/zsh/fish, a Python REPL, etc. -- not hardcoded
     to "$"). When that exact prompt reappears with nothing typed after it, the
     command is done -> instant, zero LLM calls.
  2. Falls back to output *quiescence* (screen unchanged for a short window).
  3. Only when a quiet screen is ambiguous does it ask the LLM once to confirm.

Run:
  source .venv/bin/activate && set -a; source .env; set +a
  python tools/proto_settle_detect.py
"""

import os
import pty
import select
import struct
import termios
import fcntl
import time

import pyte

from relai.llm import create_client, LLMNotConfigured

ROWS, COLS = 24, 100


# --------------------------------------------------------------------------- #
# PTY / screen helpers
# --------------------------------------------------------------------------- #
def spawn_bash():
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["PS1"] = "$ "
        os.environ["PROMPT_COMMAND"] = ""
        os.environ["TERM"] = "xterm"
        os.execvp("bash", ["bash", "--norc", "--noprofile", "-i"])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    return fd


def pump_for(fd, stream, budget):
    """Feed whatever the PTY produces for up to ``budget`` seconds."""
    end = time.time() + budget
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            break
        r, _, _ = select.select([fd], [], [], remaining)
        if fd not in r:
            break
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        stream.feed(data)


def snapshot(screen):
    lines = [row.rstrip() for row in screen.display]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def settle_to_prompt(fd, stream, screen):
    """Drain until the screen is quiet, then return the prompt prefix."""
    last = None
    quiet_start = time.time()
    while time.time() - quiet_start < 5:
        pump_for(fd, stream, 0.05)
        text = snapshot(screen)
        if text != last:
            last = text
            quiet_start = time.time()
        elif time.time() - quiet_start >= 0.3:
            break
    return screen.display[screen.cursor.y][: screen.cursor.x]


# --------------------------------------------------------------------------- #
# LLM status check (identical wording to relai._injection_finished)
# --------------------------------------------------------------------------- #
def llm_done(llm, injected, text):
    system = {
        "role": "system",
        "content": (
            "You monitor a terminal. Some keystrokes/command were just injected "
            "into it. Given the injected input and the current screen, decide "
            "whether that input has FINISHED taking effect (output is complete "
            "and the terminal is idle / a shell prompt is waiting) or is STILL "
            "RUNNING. Reply with exactly one word: DONE or RUNNING."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Injected input (repr): {injected!r}\n\n"
            "Current terminal screen:\n--- BEGIN SCREEN ---\n"
            f"{text}\n--- END SCREEN ---\n\n"
            "Has the injected input finished? Answer DONE or RUNNING."
        ),
    }
    try:
        reply = llm.complete([system, user], max_tokens=8)
    except Exception:
        return True
    return "RUNNING" not in reply.strip().upper()


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def detect_llm(fd, stream, screen, prompt_prefix, injected, llm,
               interval=0.25, polls=20):
    """Current approach: sleep, snapshot, ask the LLM -- every poll."""
    t0 = time.time()
    calls = 0
    for _ in range(polls):
        pump_for(fd, stream, interval)
        text = snapshot(screen)
        calls += 1
        if llm_done(llm, injected, text):
            break
    return time.time() - t0, "llm", calls


def detect_heuristic(fd, stream, screen, prompt_prefix, injected, llm,
                     poll=0.05, quiet_window=1.30, max_wait=30):
    """Prompt-return + quiescence, with a single LLM confirmation if unsure.

    Prompt-return is the authoritative fast path for any shell/REPL, so the
    quiescence fallback is deliberately *patient* (only for contexts with no
    recognizable prompt): a normal command's brief silence must not trigger an
    LLM call. Once we do consult the LLM and it says RUNNING, the window widens
    so we back off instead of polling the model repeatedly.
    """
    t0 = time.time()
    last_text = snapshot(screen)
    last_change = t0
    changed_once = False
    calls = 0
    plen = len(prompt_prefix)
    while time.time() - t0 < max_wait:
        pump_for(fd, stream, poll)
        now = time.time()
        text = snapshot(screen)
        if text != last_text:
            last_text = text
            last_change = now
            changed_once = True

        # (1) Prompt-return: the learned prompt is back with nothing typed.
        line = screen.display[screen.cursor.y]
        if (changed_once and screen.cursor.x == plen
                and line[:plen] == prompt_prefix):
            return time.time() - t0, "prompt", calls

        # (2) Quiescence: no change for a window after output started.
        if changed_once and (now - last_change) >= quiet_window:
            if llm is None:
                return time.time() - t0, "quiet", calls
            calls += 1
            if llm_done(llm, injected, text):
                return time.time() - t0, "quiet+llm", calls
            # Really still running; keep waiting, widen the window.
            last_change = now
            quiet_window = min(quiet_window * 2, 2.0)
    return time.time() - t0, "timeout", calls


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
COMMANDS = [
    ("echo (instant)", "echo hello"),
    ("pwd", "pwd"),
    ("ls /", "ls /"),
    ("sleep 1s", "sleep 1; echo done"),
    ("stream ~1s", "for i in 1 2 3; do echo line$i; sleep 0.3; done"),
]


def run_trial(detector, fd, stream, screen, command, llm):
    prompt = settle_to_prompt(fd, stream, screen)
    t0 = time.time()
    os.write(fd, command.encode() + b"\r")
    elapsed, method, calls = detector(
        fd, stream, screen, prompt, command, llm
    )
    return elapsed, method, calls


def main():
    try:
        llm = create_client()
        llm.verify()
        print(f"LLM: {llm.name}:{llm.model}\n")
    except (LLMNotConfigured, Exception) as exc:  # noqa: BLE001
        print(f"No usable LLM ({exc}); heuristic will run without fallback.\n")
        llm = None

    # Measure one bare LLM round-trip for reference.
    if llm is not None:
        t = time.time()
        llm_done(llm, "echo hi", "$ echo hi\nhi\n$ ")
        print(f"single LLM status-check latency: {time.time()-t:0.2f}s\n")

    fd = spawn_bash()
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)
    pump_for(fd, stream, 1.0)

    header = f"{'command':<16}{'HEURISTIC':>28}{'CURRENT (LLM)':>26}"
    print(header)
    print("-" * len(header))
    totals = {"h": 0.0, "l": 0.0, "hc": 0, "lc": 0}
    for name, cmd in COMMANDS:
        he, hm, hc = run_trial(detect_heuristic, fd, stream, screen, cmd, llm)
        le, lm, lc = run_trial(detect_llm, fd, stream, screen, cmd, llm)
        totals["h"] += he
        totals["l"] += le
        totals["hc"] += hc
        totals["lc"] += lc
        speedup = f"{le / he:0.1f}x" if he > 0 else "-"
        print(f"{name:<16}"
              f"{f'{he:0.2f}s [{hm}] {hc} llm':>28}"
              f"{f'{le:0.2f}s {lc} llm':>20}"
              f"{speedup:>6}")

    print("-" * len(header))
    print(f"{'TOTAL':<16}"
          f"{f'{totals[chr(104)]:0.2f}s {totals[chr(104)+chr(99)]} llm':>28}"
          f"{f'{totals[chr(108)]:0.2f}s {totals[chr(108)+chr(99)]} llm':>20}"
          f"{f'{totals[chr(108)]/max(totals[chr(104)],0.01):0.1f}x':>6}")

    os.write(fd, b"exit\r")
    time.sleep(0.3)


if __name__ == "__main__":
    main()
