"""Command-line entry point for relai."""

from __future__ import annotations

import argparse
import os
import sys

from .relai import Relai


def _default_shell() -> str:
    return os.environ.get("SHELL") or "/bin/sh"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="relai",
        description=(
            "PTY-level relay: spawn a command and interact with it transparently. "
            "With no command, spawns your $SHELL."
        ),
        epilog="Everything after '--' is the command to run, e.g.  relai -- htop",
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

    try:
        return Relai(command).run()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
