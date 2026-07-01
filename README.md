# relai

**relai** is a PTY-level relay for your terminal. It spawns a command (your shell
by default), sits transparently on the character stream between you and that
program, and — eventually — lets you pull an AI agent into the loop on demand.

The human is in control by default. relai just relays.

Because it operates at the pseudo-terminal (PTY) layer, it is program- and
session-independent: it works with plain shells, full-screen ncurses apps
(`htop`, `vim`), and even nested remote sessions (`ssh` → `tmux`/`screen` → any
program), since everything is just a byte stream flowing through.

## Status

Early prototype. Current milestone: **transparent passthrough** — spawn a
command, render it through a `pyte` screen model, and let you interact with it
exactly as if relai weren't there.

Planned next: an AI overlay hotkey and the agentic loop.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
./setup.sh
source .venv/bin/activate
```

## Usage

```bash
relai            # spawns your $SHELL
relai -- htop    # spawns any command (everything after -- is the command)
relai -- ssh user@host
```

Exit by exiting the spawned program (e.g. `exit` in the shell).
