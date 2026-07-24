"""The ludvart backend server: run the agent loop over a framed channel.

``python -m ludvart serve`` starts this on a duplex byte stream (its stdin and
stdout), which the client reaches either by forking it locally or by spawning it
on a remote host over SSH. It reads ``SUBMIT`` frames, runs an
:class:`~ludvart.agent_core.AgentCore` turn against a
:class:`~ludvart.remote_host.RemoteTerminalHost` (so terminal tools execute back
on the client), and returns a ``REPLY``.

State stays under ``~/.ludvart/`` on the backend host. Only protocol frames are
written to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

from .agent_core import DEFAULT_CLIENT_TOOLS, AgentCore
from .llm import LLMClient, ProviderConfig, ToolCall, ToolSpec, Turn
from .protocol import (
    DEFAULT_MAX_FRAME,
    FrameChannel,
    MsgType,
    message,
    msg_type,
)
from .remote_host import RemoteTerminalHost

#: Compact system prompt used by the split backend prototype. The full,
#: helper-aware prompt still lives in the monolithic client path.
_SYSTEM_PROMPT = (
    "You are ludvart, an assistant embedded in a terminal. Each user message "
    "carries a <screenContext> snapshot followed by a <userRequest>. Use the "
    "tools available to you to act in the terminal; keep replies concise and in "
    "plain ASCII."
)


def _prototype_tools() -> list[ToolSpec]:
    """The tool set the split backend advertises (prototype subset)."""
    return [
        ToolSpec(
            name="inject_input",
            description="Type characters into the user's terminal.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "submit": {"type": "boolean"},
                    "interpret_escapes": {"type": "boolean"},
                },
                "required": ["text"],
            },
        ),
        ToolSpec(
            name="capture_screen_history",
            description="Read lines from the terminal scrollback history.",
            input_schema={
                "type": "object",
                "properties": {
                    "offset": {"type": "integer"},
                    "length": {"type": "integer"},
                },
                "required": ["offset", "length"],
            },
        ),
        ToolSpec(
            name="b64_encode",
            description="Base64-encode UTF-8 text.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        ToolSpec(
            name="b64_decode",
            description="Base64-decode to UTF-8 text.",
            input_schema={
                "type": "object",
                "properties": {"b64": {"type": "string"}},
                "required": ["b64"],
            },
        ),
    ]


class _FakeBackendLLM(LLMClient):
    """A deterministic offline LLM for hermetic backend tests.

    First model call of a turn requests one ``inject_input`` tool call; once a
    tool result is present in the replayed history, it returns a final text
    reply that echoes the tool output. No network is used.
    """

    def __init__(self) -> None:
        super().__init__(
            ProviderConfig(name="custom", api_url="x", api_key="k", model="fake")
        )

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        has_tool_result = any(
            isinstance(m, dict) and m.get("role") == "tool" for m in messages
        )
        if on_text:
            on_text("working on it")
        if not has_tool_result and tools:
            call = ToolCall(
                id="c1",
                name="inject_input",
                input={"text": "echo hi", "submit": True},
            )
            return Turn(
                text="working on it",
                tool_calls=[call],
                assistant_message={"role": "assistant", "content": "working on it"},
                usage=None,
            )
        tool_output = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "tool":
                tool_output = str(m.get("content", ""))
        return Turn(
            text=f"done ({tool_output[:40]})",
            assistant_message={"role": "assistant", "content": "done"},
            usage=None,
        )


def _build_llm() -> LLMClient:
    """Build the backend LLM: a fake for tests, else the active registered model."""
    if os.environ.get("LUDVART_BACKEND_FAKE_LLM"):
        return _FakeBackendLLM()
    # Real path: activate the registered model on the backend host.
    from .backend import build_backend
    from .models import active_index, load_registry

    models = load_registry()
    idx = active_index(models) if models else None
    if idx is None:
        raise RuntimeError("no active model registered on the backend")
    backend = build_backend(models[idx])
    return backend.client


def serve(channel: FrameChannel, *, llm: LLMClient | None = None) -> None:
    """Run the backend request loop on ``channel`` until the client disconnects.

    One turn at a time: read a ``SUBMIT``, run it, send a ``REPLY``. A ``BYE`` or
    a clean end-of-stream ends the loop.
    """
    client = llm if llm is not None else _build_llm()
    host = RemoteTerminalHost(channel)
    core = AgentCore(
        client,
        host,
        system_prompt=_SYSTEM_PROMPT,
        tools=_prototype_tools(),
        client_tools=DEFAULT_CLIENT_TOOLS,
    )
    channel.send(message(MsgType.HELLO, app="ludvart", protocol=1))
    while True:
        msg = channel.recv()
        if msg is None:
            return
        kind = msg_type(msg)
        if kind == MsgType.BYE:
            return
        if kind == MsgType.SUBMIT:
            text = msg.get("text", "")
            snapshot = msg.get("snapshot", "")
            try:
                reply = core.run_turn(text, snapshot)
            except ConnectionError:
                return  # client vanished mid-turn
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                reply = f"[ludvart] backend error: {exc}"
            channel.send(message(MsgType.REPLY, text=reply))
        # Other client message kinds are ignored in the prototype.


def serve_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ``ludvart serve``: bind the framed channel to stdio.

    Reads frames from stdin and writes them to stdout; nothing else may touch
    stdout or the protocol stream is corrupted.
    """
    reader = sys.stdin.buffer
    writer = sys.stdout.buffer
    channel = FrameChannel(reader, writer, max_frame=DEFAULT_MAX_FRAME)
    try:
        serve(channel)
    except (BrokenPipeError, ConnectionError):
        return 0
    return 0
