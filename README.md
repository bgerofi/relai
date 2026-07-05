# RelAI

**relai** is an AI agent that sits transparently on the character stream between
you and whatever terminal-based program you're running, wherever you happen to
be running it. It sees exactly what you see, and it's ready to help with whatever
you need, whenever you need it.

**relai** operates at the pseudo-terminal (PTY) layer rather than inside any
particular application, thus it integrates **seamlessly with any terminal and any
program**: arbitrary shells, full-screen TUI apps (`htop`, `vim`, `claude`), and
REPLs all work unchanged. There is nothing to configure per-app; if it runs in a
terminal, relai can drive it.

relai is **host transparent**, it processes the PTY byte stream and it
travels with you across `ssh` hops and nested `tmux`/`screen` sessions. 
It keeps working on the far side without any agent or API key installed
on the remote host. Your session, wherever it goes, carries the agent along.

You are in control by default, but once you summon the agent, it can:

- **Run commands** on your behalf (and read back their output).
- **Control interactive applications** by sending real keystrokes, edit in
  `vim`, page through `less`, drive a Python REPL, and so on.
- **Focus on specific parts of the screen, including scrollback history**, so it
  can reason about exactly what you have been looking at.
- Through enhanced **helpers** you can also acomplish more complex tasks, such as
  coding or debugging, issue resolution and triaging, etc.

## What RelAI is not?

- RelAI is **not an MCP service or a plugin** for extending other harnesses. 
  It does not exist to hand tools or context to a separate AI harness. It *is* the
  agent, and it drives any harness by itself, at the PTY layer, sending real
  keystrokes and reading the real screen. There is no host application it needs
  to be embedded in and nothing to register on the far side.
- RelAI is **not a terminal emulator with AI bolted on**. It does not implement a
  terminal, swap you onto a different shell, or ask you to adopt a new one. It
  launches your own `$SHELL` (or the command you give it) and runs *inside*
  whatever terminal you already use (xterm, iTerm, Alacritty, ghostty, Windows Terminal,
  a `tmux`/`screen` pane, etc.), relaying the byte stream transparently.
  Your terminal, keybindings, and workflow stay exactly as they were.

## Status

Working prototype. Transparent passthrough, an on-demand AI panel (a resizable
bottom split), the agentic tool-calling loop, screen/scrollback inspection, helpers,
and conversation sessions are implemented.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a local `.venv`, installs relai, and also installs the
optional **LiteLLM gateway** (`litellm[proxy]`) that enables the **GitHub
Copilot** backend. relai works with the other providers even if that step is
skipped.

The first time you run `relai` without a configured provider, it walks you
through a short **interactive setup** that saves your choice to
`~/.relai/llm.conf` — see [LLM configuration](#llm-configuration) below. Use
`--no-llm` to skip it and run as a plain relay.

## LLM configuration

Once you are done with the installation you invoke RelAI simply by typing:

```bash
relai
```

### First-run setup wizard

The first time you start relai without a provider configured, it runs a short
interactive wizard (when stdin/stdout are a TTY) and saves your answers to
`~/.relai/llm.conf`, so you only do this once:

```
Select the API endpoint type:
  1) OpenAI
  2) Anthropic
  3) Google
  4) Custom (OpenAI-compatible)
  5) GitHub Copilot (via local LiteLLM gateway)
```

For providers 1–4 it asks for the endpoint URL, API key (hidden), and model
name. Option 5 (GitHub Copilot) runs GitHub's device-flow authorization and then
lets you pick from the models your account can use — see
[GitHub Copilot](#github-copilot-via-litellm) below. Press Ctrl-C to skip the
wizard and run as a plain relay; you can re-run it later, or configure things
manually as described next.

### Providers

If you'd rather not deal with the interactive
step — or you want to override what it saved — you can configure a provider
directly with a triplet of variables (URL, key, model), either as environment
variables or in `~/.relai/llm.conf`. Set the three variables for one provider:

| Provider  | URL                 | Key                 | Model             |
|-----------|---------------------|---------------------|-------------------|
| OpenAI    | `OPENAI_API_URL`    | `OPENAI_API_KEY`    | `OPENAI_MODEL`    |
| Anthropic | `ANTHROPIC_API_URL` | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` |
| Google    | `GOOGLE_API_URL`    | `GOOGLE_API_KEY`    | `GOOGLE_MODEL`    |
| Custom    | `CUSTOM_API_URL`    | `CUSTOM_API_KEY`    | `CUSTOM_MODEL`    |
| Copilot   | *(set by LiteLLM)*  | *(set by LiteLLM)*  | `COPILOT_MODEL`   |

Environment variables take precedence over the config file, so exporting a value
overrides the wizard's saved settings for that invocation (handy for switching
model per run). Configuring any provider this way also means the wizard won't run
on startup.

The **custom** provider speaks the OpenAI-compatible API, so it works with local
servers (LM Studio, llama.cpp, vLLM, Ollama's OpenAI shim) and gateways. Google
uses the Gemini (`google-genai`) SDK. For **GitHub Copilot** you only set
`COPILOT_MODEL` — the URL and key point at the local LiteLLM gateway, which relai
starts and configures for you (see [below](#github-copilot-via-litellm)).

At startup relai makes a minimal request to verify the provider is reachable. If
no provider is configured (and the wizard is skipped or non-interactive), relai
runs as a plain relay.

### More on GitHub Copilot (via LiteLLM)

relai can use **GitHub Copilot** as an OpenAI-compatible backend by spawning a
local [LiteLLM](https://github.com/BerriAI/litellm) proxy that fronts LiteLLM's
`github_copilot/` provider. relai then talks to that proxy with its normal
OpenAI-compatible client — no endpoint URL or API key to manage.

Requirements and behavior:

- An **active paid GitHub Copilot subscription**.
- The gateway (`litellm[proxy]`) must be installed — `setup.sh` does this, or
  run `uv pip install 'litellm[proxy]'`.
- Authorization uses **GitHub's OAuth device flow**: relai prints a URL and a
  one-time code; you open the URL, enter the code, and approve access. The
  credentials are cached under `~/.config/litellm/github_copilot` and reused on
  later runs, so the gateway starts non-interactively afterwards. (This uses
  GitHub's own OAuth, not `~/.netrc`.)

The easiest way to set this up is the first-run wizard (option 5), which runs the
device flow and lists the models your account can use (Copilot uses its own model
ids, e.g. `gpt-4o`, `claude-opus-4.8`). Only the chosen model is stored:

```ini
# ~/.relai/llm.conf
COPILOT_MODEL=gpt-4o
```

At startup, when no direct provider is configured but `COPILOT_MODEL` is set,
relai authorizes (if needed), spawns the local LiteLLM gateway on loopback, and
points its OpenAI-compatible client at it. The gateway is shut down when relai
exits.

### Timeouts and retries

Each request waits up to **`RELAI_LLM_TIMEOUT`** seconds (default `30`). On a
transient failure — a timeout, a dropped connection, a rate limit, or a `5xx`
response — relai retries up to **`RELAI_LLM_MAX_RETRIES`** times (default `2`,
with exponential backoff).

Both settings are read from the environment or `~/.relai/llm.conf`. If you see
frequent timeouts on slow models or links, raise the timeout (and optionally the
retry count).

### Context window

relai tracks how much of the model's context window a conversation uses (shown
as a `[NN%]` badge in the panel, and used to trigger automatic summarization).
It learns the window size from the provider's API when possible; otherwise it
falls back to a small table of known models.

On first run that table is written to **`~/.relai/context_windows.json`** — a
self-documented JSON file you can edit. Each key is matched as a
case-insensitive substring of the model id and the first match wins, so keep the
most specific ids first:

```jsonc
{
  "claude-opus-4": 1000000,
  "gpt-5": 400000,
  "gpt-4o": 128000,
  "my-local-model": 32768
}
```

relai re-reads the file whenever it changes, so edits take effect on the next
request. Delete the file to regenerate the defaults. For a one-off override you
can instead set `<PROVIDER>_CONTEXT_WINDOW` (e.g. `CUSTOM_CONTEXT_WINDOW`), which
always wins.

## Summoning the agent

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

## Assistant helpers

Tools are the agent's built-in, in-process primitives — `inject_input` and
`capture_screen_history` — and they are the *only* channel through which the
agent touches your machine. Everything the agent does ultimately flows through
them. **Helpers** are a complementary mechanism layered *on top of* those
tools.

Because relai works purely at the PTY layer, the harness has no direct
filesystem or exec access to the (possibly remote) box it is driving — it only
sees the terminal. To work reliably at a higher level, the agent uses a small,
dependency-free helper program, `relai_helper`, under `~/.relai/bin/` on that
machine. The canonical helper ships *with* relai as a version-pinned,
checksummed copy.

The agent does not call helpers directly the way it calls a tool; it *runs*
them by typing a shell command through `inject_input`. `relai_helper` exposes
subcommands for the file operations that are awkward to do safely over a raw
terminal — `read`, `write`, `append`, `replace`, `replace-range`, `search`,
`run`, and `info`. Every content payload is passed as base64 and every result is
sentinel-framed with a real exit code, so edits are immune to quoting, newline,
and escape corruption, and success is read from a reliable status rather than
guessed from screen text.

You can install or repair it at any time with the `/init_helpers` panel command.
This is deterministic and does *not* involve the model: relai injects a short,
self-contained shell command that compares the on-disk copy against the bundled
version by checksum and rewrites it only if it is missing, outdated, or
modified.

### Tools vs. helpers

| | Tools | Helpers |
|---|-------|---------|
| **What** | Built-in agent primitives | A version-pinned script bundled with relai |
| **Where they run** | Inside the relai process (Python) | On the target machine, under `~/.relai/bin/` |
| **How invoked** | Called directly by the model | Run via `inject_input` (typed as shell commands) |
| **Availability** | Always present | Optional; installed/repaired via `/init_helpers` |
| **Lifetime** | Live for the process | Persist across sessions |
| **Purpose** | The only way the agent acts at all | Make file read/edit/search reliable and corruption-proof |

## Related projects

Other projects put an AI agent near the terminal, but they fall into two camps
that are each distinct from RelAI.

**Headless drivers** — the *agent* spawns and owns a session and drives it
programmatically; the human is out of the loop and reviews the result:

- [agent-tty](https://github.com/coder/agent-tty) (coder) — hands a real,
  long-lived PTY to an agent and records reviewable proof (text snapshots, PNG
  screenshots, WebM video, asciicast) via a Ghostty renderer.
- [agent-terminal](https://github.com/jasonkneen/agent-terminal) (jasonkneen) —
  a `node-pty` wrapper exposed as an MCP server; an external agent reads the
  ASCII buffer and sends keys.
- [pilotty](https://github.com/msmps/pilotty) (msmps) — daemon-managed headless
  PTY sessions with VT100 emulation, snapshots, and detected UI elements.

**In-band assistants** — a human drives and the AI rides along in the *live*
session:

- [Butterfish](https://github.com/bakks/butterfish) (bakks) — the closest on UX.
  Wraps your shell in a PTY; you prompt inline (capital letter to ask, `!` for
  agent mode, `@` for a one-shot command) and the AI sees your shell history. It
  reasons over shell command history rather than the rendered screen, and is
  tied to the local `bash`/`zsh`.
- [TmuxAI](https://github.com/alvinunreal/tmuxai) (alvinunreal) — the closest in
  philosophy ("a colleague sitting next to you"). Reads all your tmux panes in
  real time via a chat/exec pane split, but requires tmux.
- [AIShell](https://github.com/changjonathanc/aishell) (changjonathanc) — a
  transparent shell wrapper that captures screen content as AI context, but
  offers help commands rather than an agentic loop.

Further out: [Warp](https://www.warp.dev/) and Microsoft's
[Intelligent Terminal](https://devblogs.microsoft.com/commandline/announcing-intelligent-terminal-version-0-1/)
are full terminal-emulator replacements — the thing relai deliberately is *not*
(see [What RelAI is not?](#what-relai-is-not)).

### How RelAI compares

| Dimension | RelAI | Butterfish | TmuxAI | AIShell | Headless drivers |
|---|:---:|:---:|:---:|:---:|:---:|
| In-band, human-driven | ✅ | ✅ | ✅ | ✅ | ✗ |
| Full terminal- & program-indepence (i.e., nothing to modify) | ✅ | ✗ (its shell) | ✗ (tmux) | ✗ (its shell) | ✗ (program reachable only through their API/CLI) |
| Works at the raw PTY layer (no shell/tmux/emulator dependency) | ✅ | shell-wrapper | needs tmux | shell-wrapper | spawns its own PTY |
| Reasons about the rendered screen **and** scrollback (not just shell history) | ✅ | ✗ | ✅ | partial | ✅ |
| Drives arbitrary full-screen TUIs (`vim`, `htop`, `claude`) | ✅ | ✗ | ✅ | ✗ | ✅ |
| Host-transparent across `ssh`, nothing installed remotely | ✅ | local shell only | tmux-side | local only | ✗ |
| Resizable in-terminal agent panel + conversation sessions | ✅ | inline shell | chat pane | ✗ | N/A |
| Agentic tool-calling loop | ✅ | ✅ | ✅ | ✗ | driven externally |