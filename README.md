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

### In-session commands

Press the prefix key (default **Ctrl-G**), then a command letter:

- `Ctrl-G` `s` — open the scrollback viewer
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
