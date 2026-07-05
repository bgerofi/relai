"""Command-line entry point for relai."""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from .llm import (
    LLMClient,
    LLMError,
    LLMNotConfigured,
    create_client,
    write_provider_conf,
)
from .relai import DEFAULT_PREFIX, Relai


def _default_shell() -> str:
    return os.environ.get("SHELL") or "/bin/sh"


def _parse_prefix(spec: str) -> bytes:
    """Parse a prefix spec like 'C-g', 'ctrl-g', '^g', or '\\x07' into a byte."""
    s = spec.strip().lower()
    if s.startswith(("c-", "ctrl-", "^")):
        letter = s.split("-", 1)[-1] if "-" in s else s[1:]
        if len(letter) == 1 and letter.isalpha():
            # Control character: clear the top three bits (A -> 0x01, G -> 0x07).
            return bytes([ord(letter.upper()) & 0x1F])
    if s.startswith("\\x") and len(s) == 4:
        return bytes([int(s[2:], 16)])
    raise argparse.ArgumentTypeError(
        f"invalid prefix {spec!r}; use e.g. 'C-g', 'ctrl-g', '^g', or '\\x07'"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="relai",
        description=(
            "PTY-level relay: spawn a command and interact with it transparently. "
            "With no command, spawns your $SHELL."
        ),
        epilog=(
            "Everything after '--' is the command to run, e.g.  relai -- htop. "
            "Inside a session, press the prefix key (default Ctrl-G) then 's' to "
            "open the scrollback viewer; press the prefix twice to send it literally."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=_parse_prefix,
        default=DEFAULT_PREFIX,
        metavar="KEY",
        help="Prefix key for relai commands, e.g. 'C-g' (default), 'ctrl-o', '^b'.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run as a plain relay without any LLM (skip provider setup/check).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command (and args) to run. Prefix with '--' to pass flags through.",
    )
    args = parser.parse_args(argv)

    command = args.command
    # argparse.REMAINDER keeps a leading '--' if the user wrote 'relai -- cmd'.
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = [_default_shell()]

    llm = None if args.no_llm else _setup_llm()

    try:
        return Relai(command, prefix=args.prefix, llm=llm).run()
    except KeyboardInterrupt:
        return 130


def _setup_llm() -> LLMClient | None:
    """Resolve and verify an LLM provider, running first-time setup if needed.

    Reuses existing configuration whenever it is present -- provider variables
    from the environment or ``~/.relai/llm.conf``. If nothing is configured and
    the session is interactive, it asks a few questions (endpoint type, URL, API
    key, model), saves them to ``~/.relai/llm.conf``, and continues. Returns a
    ready client, or ``None`` to run as a plain relay when no provider is set up.
    """
    client = _try_create()
    while True:
        if client is None:
            # Nothing configured yet: offer the interactive first-time setup.
            if not _run_setup_wizard():
                sys.stderr.write(
                    "relai: no LLM provider configured. Running as a plain relay.\n"
                    "       Set {OPENAI,ANTHROPIC,GOOGLE,CUSTOM}_API_URL/_API_KEY/"
                    "_MODEL,\n       or edit ~/.relai/llm.conf, or pass --no-llm.\n"
                )
                return None
            client = _try_create()
            if client is None:
                sys.stderr.write("relai: configuration still incomplete.\n")
                continue

        sys.stderr.write(
            f"relai: verifying {client.name} model {client.model!r}... "
        )
        sys.stderr.flush()
        try:
            client.verify()
        except LLMError as exc:
            sys.stderr.write("FAILED\n")
            sys.stderr.write(f"relai: LLM check failed: {exc}\n")
            if sys.stdin.isatty() and _ask_yes_no(
                "relai: re-enter the LLM settings now?", default=True
            ):
                client = None
                continue
            sys.stderr.write("relai: fix the configuration or pass --no-llm.\n")
            sys.exit(2)
        sys.stderr.write("ok\n")
        return client


def _try_create() -> LLMClient | None:
    """Return a client from existing env / file config, or ``None`` if unset."""
    try:
        return create_client()
    except LLMNotConfigured:
        return None


#: Endpoint types offered by the setup wizard, in menu order, with a friendly
#: label and the default endpoint URL (empty = required, no sensible default).
_SETUP_PROVIDERS: tuple[tuple[str, str, str], ...] = (
    ("openai", "OpenAI", "https://api.openai.com/v1"),
    ("anthropic", "Anthropic", "https://api.anthropic.com"),
    ("google", "Google", "https://generativelanguage.googleapis.com"),
    ("custom", "Custom (OpenAI-compatible)", ""),
)


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer; Enter takes ``default``."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        ans = input(prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not ans:
        return default
    return ans in ("y", "yes")


def _run_setup_wizard() -> bool:
    """Interactively collect provider settings and save them to llm.conf.

    Returns ``True`` when a configuration was written, ``False`` if the session
    is non-interactive or the user aborted (Ctrl-C / EOF).
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False

    sys.stderr.write(
        "\nrelai: no LLM provider is configured yet -- let's set one up.\n"
        "Answers are saved to ~/.relai/llm.conf (press Ctrl-C to skip).\n\n"
    )
    sys.stderr.flush()

    try:
        # 1) Endpoint type.
        sys.stderr.write("Select the API endpoint type:\n")
        for i, (_name, label, _url) in enumerate(_SETUP_PROVIDERS, 1):
            sys.stderr.write(f"  {i}) {label}\n")
        sys.stderr.flush()
        provider = default_url = ""
        while not provider:
            choice = input(f"Choice [1-{len(_SETUP_PROVIDERS)}]: ").strip().lower()
            picked = None
            if choice.isdigit() and 1 <= int(choice) <= len(_SETUP_PROVIDERS):
                picked = _SETUP_PROVIDERS[int(choice) - 1]
            else:
                picked = next(
                    (p for p in _SETUP_PROVIDERS if p[0] == choice), None
                )
            if picked is None:
                sys.stderr.write(
                    f"Please enter a number 1-{len(_SETUP_PROVIDERS)}.\n"
                )
                continue
            provider, _label, default_url = picked

        # 2) Endpoint URL (Enter accepts the default when there is one).
        url = ""
        while not url:
            prompt = (
                f"Endpoint URL [{default_url}]: " if default_url else "Endpoint URL: "
            )
            url = input(prompt).strip() or default_url
            if not url:
                sys.stderr.write("An endpoint URL is required.\n")

        # 3) API key (hidden input).
        key = ""
        while not key:
            key = getpass.getpass("API key (input hidden): ").strip()
            if not key:
                sys.stderr.write("An API key is required.\n")

        # 4) Model name.
        model = ""
        while not model:
            model = input(
                "Model name (e.g. gpt-4o, claude-..., gemini-...): "
            ).strip()
            if not model:
                sys.stderr.write("A model name is required.\n")
    except (EOFError, KeyboardInterrupt):
        sys.stderr.write("\nrelai: setup skipped.\n")
        return False

    path = write_provider_conf(provider, url, key, model)
    sys.stderr.write(f"relai: saved provider settings to {path}\n")
    return True


if __name__ == "__main__":
    sys.exit(main())
