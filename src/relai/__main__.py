"""Command-line entry point for relai."""

from __future__ import annotations

import argparse
import os
import sys

from .llm import LLMClient, LLMError, LLMNotConfigured, create_client
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
    """Resolve and verify an LLM provider from the environment.

    Returns a ready client, or ``None`` if no provider is configured. Exits the
    process if a provider *is* configured but the connectivity check fails, so
    the user is not surprised later by a dead backend.
    """
    try:
        client = create_client()
    except LLMNotConfigured:
        sys.stderr.write(
            "relai: no LLM provider configured (set {OPENAI,ANTHROPIC,GOOGLE,"
            "CUSTOM}_API_URL/_API_KEY/_MODEL). Running as a plain relay.\n"
        )
        return None

    sys.stderr.write(
        f"relai: verifying {client.name} model {client.model!r}... "
    )
    sys.stderr.flush()
    try:
        client.verify()
    except LLMError as exc:
        sys.stderr.write("FAILED\n")
        sys.stderr.write(f"relai: LLM check failed: {exc}\n")
        sys.stderr.write("relai: fix the configuration or pass --no-llm.\n")
        sys.exit(2)
    sys.stderr.write("ok\n")
    return client


if __name__ == "__main__":
    sys.exit(main())
