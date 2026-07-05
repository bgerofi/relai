"""Command-line entry point for relai."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from typing import TYPE_CHECKING

from .llm import (
    LLMClient,
    LLMError,
    LLMNotConfigured,
    build_client,
    copilot_model,
    copilot_provider_config,
    create_client,
    ensure_context_windows_file,
    write_copilot_conf,
    write_provider_conf,
)
from .relai import DEFAULT_PREFIX, Relai

if TYPE_CHECKING:
    from .gateway import CopilotGateway


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

    llm = None
    gateway = None
    if not args.no_llm:
        llm, gateway = _setup_llm()

    try:
        return Relai(command, prefix=args.prefix, llm=llm).run()
    except KeyboardInterrupt:
        return 130
    finally:
        if gateway is not None:
            gateway.stop()


def _setup_llm(
    force_wizard: bool = False,
) -> tuple[LLMClient | None, "CopilotGateway | None"]:
    """Resolve and verify an LLM backend, running first-time setup if needed.

    Reuses existing configuration whenever it is present -- provider variables
    from the environment or ``~/.relai/llm.conf``, or a saved GitHub Copilot
    gateway. If nothing is configured and the session is interactive, it asks a
    few questions, saves them to ``~/.relai/llm.conf``, and continues.

    Returns ``(client, gateway)``. ``gateway`` is a running
    :class:`~relai.gateway.CopilotGateway` that the caller must ``stop()`` on
    shutdown (``None`` for direct providers). ``client`` is ``None`` when no
    backend is configured, meaning relai runs as a plain relay.
    """
    # Seed the editable context-window table on first run so users can tune it.
    ensure_context_windows_file()

    if force_wizard and not _run_setup_wizard():
        _report_plain_relay()
        return None, None

    client, gateway = _resolve_client()
    while client is None:
        if not _run_setup_wizard():
            _report_plain_relay()
            return None, None
        client, gateway = _resolve_client()
        if client is None:
            sys.stderr.write("relai: configuration still incomplete.\n")

    sys.stderr.write(f"relai: verifying {client.name} model {client.model!r}... ")
    sys.stderr.flush()
    try:
        client.verify()
    except LLMError as exc:
        sys.stderr.write("FAILED\n")
        sys.stderr.write(f"relai: LLM check failed: {exc}\n")
        if gateway is not None:
            gateway.stop()
        if sys.stdin.isatty() and _ask_yes_no(
            "relai: re-enter the LLM settings now?", default=True
        ):
            return _setup_llm(force_wizard=True)
        sys.stderr.write("relai: fix the configuration or pass --no-llm.\n")
        sys.exit(2)
    sys.stderr.write("ok\n")
    return client, gateway


def _report_plain_relay() -> None:
    sys.stderr.write(
        "relai: no LLM provider configured. Running as a plain relay.\n"
        "       Set {OPENAI,ANTHROPIC,GOOGLE,CUSTOM}_API_URL/_API_KEY/_MODEL,\n"
        "       or edit ~/.relai/llm.conf, or pass --no-llm.\n"
    )


def _resolve_client() -> tuple[LLMClient | None, "CopilotGateway | None"]:
    """Build a client from current config: a direct provider, else the gateway.

    Direct providers (env / ``llm.conf``) take precedence. Otherwise, if a
    GitHub Copilot model is configured, the local LiteLLM gateway is started and
    a client pointed at it. Returns ``(None, None)`` when nothing is configured.
    """
    client = _try_create()
    if client is not None:
        return client, None
    model = copilot_model()
    if model:
        return _start_copilot(model)
    return None, None


def _try_create() -> LLMClient | None:
    """Return a client from existing env / file config, or ``None`` if unset."""
    try:
        return create_client()
    except LLMNotConfigured:
        return None


def _start_copilot(
    model: str,
) -> tuple[LLMClient | None, "CopilotGateway | None"]:
    """Authorize (if needed) and spawn the GitHub Copilot LiteLLM gateway.

    Returns ``(client, gateway)`` on success, or ``(None, None)`` if litellm is
    missing, the session can't authorize, or the gateway fails to start.
    """
    from .gateway import (
        GATEWAY_API_KEY,
        CopilotGateway,
        GatewayError,
        authenticate_copilot,
        copilot_authenticated,
        litellm_available,
    )

    if not litellm_available():
        sys.stderr.write(
            "relai: GitHub Copilot is configured but the LiteLLM gateway isn't\n"
            "       installed. Re-run ./setup.sh, or: uv pip install 'litellm[proxy]'\n"
        )
        return None, None

    if not copilot_authenticated():
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            sys.stderr.write(
                "relai: GitHub Copilot needs authorization; run relai in a "
                "terminal once to set it up.\n"
            )
            return None, None
        sys.stderr.write("relai: authorizing GitHub Copilot...\n")
        sys.stderr.flush()
        try:
            authenticate_copilot()
        except GatewayError as exc:
            sys.stderr.write(f"relai: {exc}\n")
            return None, None

    gateway = CopilotGateway(model)
    sys.stderr.write(
        f"relai: starting the GitHub Copilot gateway (model {model!r})... "
    )
    sys.stderr.flush()
    try:
        gateway.start()
    except GatewayError as exc:
        sys.stderr.write("FAILED\n")
        sys.stderr.write(f"relai: {exc}\n")
        return None, None
    sys.stderr.write("ok\n")

    config = copilot_provider_config(
        gateway.base_url, gateway.litellm_model, GATEWAY_API_KEY
    )
    return build_client(config), gateway


#: Endpoint types offered by the setup wizard, in menu order, with a friendly
#: label and the default endpoint URL (empty = required, no sensible default).
#: ``copilot`` is handled specially (device-flow auth + local gateway).
_SETUP_PROVIDERS: tuple[tuple[str, str, str], ...] = (
    ("openai", "OpenAI", "https://api.openai.com/v1"),
    ("anthropic", "Anthropic", "https://api.anthropic.com"),
    ("google", "Google", "https://generativelanguage.googleapis.com"),
    ("custom", "Custom (OpenAI-compatible)", ""),
    ("copilot", "GitHub Copilot (via local LiteLLM gateway)", ""),
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

        # GitHub Copilot has its own flow (device auth + local gateway).
        if provider == "copilot":
            return _setup_copilot()

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


def _setup_copilot() -> bool:
    """Set up the GitHub Copilot backend: device-flow auth + saved model.

    Walks the user through GitHub's OAuth device flow (paid Copilot subscription
    required), caches the credentials via LiteLLM, and stores the chosen model in
    ``~/.relai/llm.conf``. The gateway itself is spawned later at startup.
    Returns ``True`` on success, ``False`` if unavailable or aborted.
    """
    from .gateway import GatewayError, authenticate_copilot, litellm_available

    if not litellm_available():
        sys.stderr.write(
            "relai: the LiteLLM gateway isn't installed, so the GitHub Copilot\n"
            "       option is unavailable. Re-run ./setup.sh, or:\n"
            "       uv pip install 'litellm[proxy]'\n"
        )
        return False

    sys.stderr.write(
        "\nGitHub Copilot requires an active paid GitHub Copilot subscription.\n"
        "You'll authorize relai through GitHub's device flow: a URL and one-time\n"
        "code appear below -- open the URL, enter the code, and approve access.\n"
        "Credentials are cached under ~/.config/litellm and reused afterwards.\n"
        "(This uses GitHub's own OAuth, not ~/.netrc.)\n\n"
    )
    sys.stderr.flush()

    sys.stderr.write("relai: starting GitHub authentication...\n")
    sys.stderr.flush()
    try:
        authenticate_copilot()
    except GatewayError as exc:
        sys.stderr.write(f"\nrelai: {exc}\n")
        return False

    model = _choose_copilot_model()
    if not model:
        sys.stderr.write("\nrelai: setup skipped.\n")
        return False

    path = write_copilot_conf(model)
    sys.stderr.write(
        f"relai: GitHub Copilot authorized; saved model {model!r} to {path}\n"
    )
    return True


def _choose_copilot_model(default: str = "gpt-4o") -> str:
    """Prompt for a Copilot model, listing the account's available slugs.

    Returns the chosen slug (e.g. ``claude-opus-4.8``), or ``""`` if aborted.
    Copilot uses its own model ids, so we list what the account can actually use
    rather than making the user guess.
    """
    from .gateway import list_copilot_models

    models = list_copilot_models()
    try:
        if not models:
            # Listing failed; fall back to a free-text prompt.
            return input(f"Copilot model [{default}]: ").strip() or default

        sys.stderr.write("\nModels available to your GitHub Copilot account:\n")
        for i, name in enumerate(models, 1):
            sys.stderr.write(f"  {i:2}) {name}\n")
        sys.stderr.flush()
        hint = default if default in models else models[0]
        raw = input(f"Choose a model [number or name, default {hint}]: ").strip()
        if not raw:
            return hint
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        # Accept a typed slug even if it's not listed (verify() catches a
        # genuinely unsupported one).
        return raw
    except (EOFError, KeyboardInterrupt):
        return ""


if __name__ == "__main__":
    sys.exit(main())
