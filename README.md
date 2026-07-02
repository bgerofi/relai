# RelAI

**relai** is an AI agent that lives in your terminal. It sits transparently on the
character stream between you and whatever program you run. The agent is always
one keystroke away, no matter where and what you are running.

Because relai operates at the pseudo-terminal (PTY) layer rather than inside any
particular application, it integrates **seamlessly with any terminal and any
program**: plain shells, full-screen ncurses apps (`htop`, `vim`, `less`), and
REPLs all work unchanged. There is nothing to configure per-app; if it runs in a
terminal, relai can drive it.

relai is also **host transparent**. It processes the PTY byte stream, so it
travels with you across `ssh` hops and nested `tmux`/`screen` sessions — the
agent keeps working on the far side without any agent or API key installed on
the remote host. Your session, wherever it goes, carries the agent along.

The human is in control by default. relai just relays, until you summon the
agent. Once you do, it can:

- **Run commands** on your behalf (and read back their output).
- **Control interactive applications** by sending real keystrokes, edit in
  `vim`, page through `less`, drive a Python REPL, and so on.
- **Focus on specific parts of the screen, including scrollback history**, so it
  can reason about exactly what you are looking at, not just the last line.

## Status

Working prototype. Transparent passthrough, an on-demand AI panel (a resizable
bottom split), the agentic tool-calling loop, and screen/scrollback inspection
are all implemented.

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

### Summoning the agent

Press **Ctrl-O** to open (or close) the AI panel — a bottom split where you type
to the agent while your program keeps running above. Ctrl-O is used because
`screen` (Ctrl-A) and `tmux` (Ctrl-B) leave it alone, so it works even inside
nested sessions.

Inside the panel:

- Type your request and press **Enter** to send it; **Esc** or **Ctrl-O** closes
  the panel. The input line is a full editor — arrow keys, Home/End, Ctrl-A/E,
  Ctrl-U/K/W, and mouse (bracketed) paste all work.
- **Up / Down** scroll the conversation; **PageUp / PageDown** scroll by a page.
- **Ctrl-G Up / Down** grow or shrink the panel one row; **Ctrl-G PageUp** snaps
  it to half the screen and **Ctrl-G PageDown** restores the previous height.

### Prefix commands

Press the prefix key (default **Ctrl-G**), then a command letter:

- `Ctrl-G` `a` — open the AI panel (same as Ctrl-O)
- `Ctrl-G` `s` — open the scrollback viewer
- `Ctrl-G` `o` — send a literal Ctrl-O byte to the program underneath
- `Ctrl-G` `Ctrl-G` — send a literal prefix byte to the program underneath

Change the prefix with `--prefix` (e.g. `relai --prefix ctrl-o`).

## LLM configuration

relai selects an LLM provider entirely from environment variables. Set the
three variables for one provider:

| Provider  | URL                 | Key                 | Model             |
|-----------|---------------------|---------------------|-------------------|
| OpenAI    | `OPENAI_API_URL`    | `OPENAI_API_KEY`    | `OPENAI_MODEL`    |
| Anthropic | `ANTHROPIC_API_URL` | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` |
| Google    | `GOOGLE_API_URL`    | `GOOGLE_API_KEY`    | `GOOGLE_MODEL`    |
| Custom    | `CUSTOM_API_URL`    | `CUSTOM_API_KEY`    | `CUSTOM_MODEL`    |

The **custom** provider speaks the OpenAI-compatible API, so it works with local
servers (LM Studio, llama.cpp, vLLM, Ollama's OpenAI shim) and gateways. Google
uses the Gemini (`google-genai`) SDK. If more than one provider is fully
configured, the precedence is custom > google > anthropic > openai.

At startup relai makes a minimal request to verify the provider is reachable. If
no provider is configured, relai runs as a plain relay. Use `--no-llm` to skip
LLM setup entirely.

```bash
# Example: OpenAI
export OPENAI_API_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o
relai
```

## Assistant tools

Once an LLM provider is configured, the AI agent can act inside your terminal
using a set of tools. The primary one is **`inject_input`**.

### `inject_input`

`inject_input` types characters into your terminal exactly as if you had pressed
the keys yourself. Whatever program is currently in the foreground receives the
input, which makes the tool useful in two ways:

- **Run shell commands** on your behalf — e.g. `ls`, `cat`, checking status, or
  installing packages. Submitting the input (pressing Enter) executes them.
- **Send keystrokes to interactive programs** — control characters and TUI
  navigation for editors, pagers, and REPLs (`vim`, `less`, a Python shell, etc.).

Because relai operates at the PTY layer, injected input flows through the same
byte stream as your own keystrokes, so it works with plain shells, full-screen
ncurses apps, and nested remote sessions alike.

| Field    | Description                                                        |
|----------|--------------------------------------------------------------------|
| `text`   | The exact characters to type (may include control characters).     |
| `submit` | If `true`, press Enter after the text to execute it. Default `false`. |

The result of the injected input appears on the terminal screen, which the agent
can then read back and act on.

### `capture_screen_history`

`capture_screen_history` lets the agent focus on a specific part of the screen,
including content that has scrolled off the top. It reads the full logical buffer
(scrollback plus the current viewport) and returns a slice of it, so the agent
can go back and inspect earlier output rather than only the visible rows.

| Field    | Description                                                        |
|----------|--------------------------------------------------------------------|
| `offset` | Where to start; a negative value counts lines above the current position. |
| `length` | How many lines to return.                                          |

Because the injected-input result and the screen history both come from relai's
own `pyte` model of the terminal, the agent always sees exactly what you see —
across plain shells, full-screen apps, and remote sessions alike.
