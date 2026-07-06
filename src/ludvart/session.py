"""Persistent conversation storage and in-panel slash commands.

Every ludvart run that opens the AI panel starts a *session*: a timestamped
directory under ``~/.ludvart/sessions/`` holding a ``conversation.json`` file that
is (re)written as the conversation grows. The layout is::

    ~/.ludvart/sessions/YYYY-MM-DD/HH_MM_SS/conversation.json

where the date and time are UTC. A new session is created the first time the
panel is opened in a process; subsequent opens in the same process keep
extending the same file (unless a different session is loaded).

This module has no terminal/UI dependencies so it can be unit tested directly.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SESSIONS_VERSION = 1

_CONV_NAME = "conversation.json"
_DAY_FMT = "%Y-%m-%d"
_TIME_FMT = "%H_%M_%S"

# Message kinds that are part of the real conversation and therefore persisted.
# Slash-command echoes/output use the "system" kind and are never saved. The
# "summary" kind marks an automatic context-compaction point (see the compaction
# logic in :mod:`ludvart`).
_PERSISTED_KINDS = ("you", "ludvart", "info", "summary")

# Text marker wrapping a compaction summary in the model-facing ``llm_history``.
# It is ordinary message content (safe to send to any provider) and lets us find
# the latest summary when resuming a conversation.
SUMMARY_MARKER = "<conversationSummary>"
SUMMARY_MARKER_END = "</conversationSummary>"


def sessions_root() -> Path:
    """Return the root directory that holds all session folders.

    Honours ``LUDVART_SESSIONS_DIR`` (used by tests) and otherwise defaults to
    ``~/.ludvart/sessions``.
    """
    override = os.environ.get("LUDVART_SESSIONS_DIR")
    if override:
        return Path(override)
    return Path(os.path.expanduser("~/.ludvart/sessions"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    """Format a UTC datetime as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def persisted_messages(
    messages: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Drop ephemeral (slash-command) messages so only the conversation saves."""
    return [(k, t) for (k, t) in messages if k in _PERSISTED_KINDS]


class SessionStore:
    """Owns the on-disk file for one conversation and rewrites it on save."""

    def __init__(
        self,
        root: Path | str | None = None,
        started_at: datetime | None = None,
        session_id: str | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else sessions_root()
        if session_id is not None:
            # Bind to an existing session (e.g. after ``/sessions load``).
            self.session_id = session_id
            self.started_at = started_at or _parse_session_id(session_id)
        else:
            self.started_at = started_at or _utc_now()
            day = self.started_at.strftime(_DAY_FMT)
            clock = self.started_at.strftime(_TIME_FMT)
            self.session_id = f"{day}/{clock}"
        self.dir = self.root / self.session_id
        self.path = self.dir / _CONV_NAME

    @classmethod
    def open_existing(
        cls, session_id: str, root: Path | str | None = None
    ) -> "SessionStore":
        """Return a store bound to an already-saved session directory."""
        return cls(root=root, session_id=session_id)

    def save(
        self,
        messages: list[tuple[str, str]],
        llm_history: list[dict[str, Any]],
    ) -> None:
        """Atomically (re)write the conversation file with the current state."""
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": SESSIONS_VERSION,
            "session_id": self.session_id,
            "started_at": _iso(self.started_at),
            "updated_at": _iso(_utc_now()),
            "messages": [list(m) for m in persisted_messages(messages)],
            "llm_history": llm_history,
        }
        tmp = self.dir / (_CONV_NAME + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(tmp, self.path)


def _parse_session_id(session_id: str) -> datetime:
    """Best-effort parse of ``YYYY-MM-DD/HH_MM_SS`` into a UTC datetime."""
    try:
        day, clock = session_id.split("/", 1)
        dt = datetime.strptime(f"{day} {clock}", f"{_DAY_FMT} {_TIME_FMT}")
        return dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return _utc_now()


def list_sessions(root: Path | str | None = None) -> list[dict[str, Any]]:
    """Return a summary of every saved session, oldest first.

    Each summary has ``id``, ``started_at``, ``updated_at``, ``count`` (number
    of persisted messages) and ``preview`` (the first user line, if any).
    """
    base = Path(root) if root is not None else sessions_root()
    out: list[dict[str, Any]] = []
    if not base.exists():
        return out
    for day in sorted(p for p in base.iterdir() if p.is_dir()):
        for sess in sorted(p for p in day.iterdir() if p.is_dir()):
            conv = sess / _CONV_NAME
            if not conv.is_file():
                continue
            try:
                data = json.loads(conv.read_text())
            except (OSError, ValueError):
                continue
            messages = data.get("messages", [])
            preview = ""
            for kind, text in messages:
                if kind == "you":
                    preview = text
                    break
            out.append(
                {
                    "id": data.get("session_id", f"{day.name}/{sess.name}"),
                    "started_at": data.get("started_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "count": len(messages),
                    "preview": preview,
                }
            )
    return out


def load_session(
    session_id: str, root: Path | str | None = None
) -> dict[str, Any]:
    """Load and return the full stored data for ``session_id``."""
    base = Path(root) if root is not None else sessions_root()
    conv = base / session_id / _CONV_NAME
    data = json.loads(conv.read_text())
    return data


def working_history(
    llm_history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the resumable slice of a stored ``llm_history``.

    Automatic compaction replaces the model-facing context with a summary seed
    whose first message content starts with :data:`SUMMARY_MARKER`. When a
    conversation is resumed we start from the *last* such summary, so it always
    continues from its latest compression point and never replays the purged
    pre-summary turns. Histories without a marker are returned unchanged.
    """
    start = 0
    for i, msg in enumerate(llm_history):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.lstrip().startswith(SUMMARY_MARKER):
            start = i
    return list(llm_history[start:])


# -- slash commands ---------------------------------------------------------

# Registry of in-panel commands and their subcommands. Used both to dispatch and
# to drive Tab completion. Keep the subcommand lists sorted for stable output.
SLASH_COMMANDS: dict[str, list[str]] = {
    "compact": [],
    "help": [],
    "init_helpers": [],
    "mcp_refresh": [],
    "sessions": ["list", "load"],
}

# One-line usage + description for each command, shown by ``/help``. Ordered the
# way they should be listed. Keep in sync with :data:`SLASH_COMMANDS`.
SLASH_COMMAND_HELP: list[tuple[str, str]] = [
    ("/help", "Show this list of internal panel commands."),
    (
        "/compact",
        "Summarise the conversation so far and replace the working context "
        "with that summary, freeing up the context window.",
    ),
    (
        "/init_helpers",
        "Install or verify ~/.ludvart/bin/ludvart_helper on the foreground host "
        "(for precise file read/edit/search).",
    ),
    (
        "/mcp_refresh",
        "Reconnect to the MCP servers in ~/.ludvart/mcp.json and refresh their "
        "tool definitions.",
    ),
    ("/sessions list", "List saved conversation sessions (current is marked *)."),
    ("/sessions load <n>|<id>", "Load and resume a saved session by number or id."),
]


def _common_completion(word: str, matches: list[str]) -> str | None:
    """Return the completion for ``word`` given candidate ``matches``.

    A single match completes to that match; multiple matches complete to their
    longest common prefix (only if it extends ``word``); no match yields None.
    """
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    prefix = os.path.commonprefix(matches)
    if len(prefix) > len(word):
        return prefix
    return None


def complete_slash(text: str) -> str | None:
    """Tab-complete a slash command line.

    Returns the completed line, or ``None`` if there is nothing to complete.
    A uniquely completed command/subcommand gains a trailing space so the next
    token can be typed straight away. Only command names and their immediate
    subcommands are completed (arguments are left untouched).
    """
    if not text.startswith("/"):
        return None
    body = text[1:]
    parts = body.split(" ")

    if len(parts) == 1:
        word = parts[0]
        matches = [c for c in sorted(SLASH_COMMANDS) if c.startswith(word)]
        completion = _common_completion(word, matches)
        if completion is None:
            return None
        new = "/" + completion
        if len(matches) == 1 and completion == matches[0]:
            new += " "
        return new if new != text else None

    if len(parts) == 2:
        cmd, word = parts
        subs = SLASH_COMMANDS.get(cmd)
        if not subs:
            return None
        matches = [s for s in subs if s.startswith(word)]
        completion = _common_completion(word, matches)
        if completion is None:
            return None
        new = "/" + cmd + " " + completion
        if len(matches) == 1 and completion == matches[0]:
            new += " "
        return new if new != text else None

    return None
