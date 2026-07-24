"""Client-side driver for a remote (or forked) backend agent loop.

:class:`BackendClient` submits a user question over a
:class:`~ludvart.protocol.FrameChannel`, then services the backend's
``REQUEST`` frames (snapshot / terminal tool) using the local
:class:`~ludvart.terminal_host.TerminalHost`, applies ``PANEL_UPDATE``
notifications, and returns the final ``REPLY`` text.

This is the counterpart to :class:`ludvart.remote_host.RemoteTerminalHost`.
"""

from __future__ import annotations

from .protocol import FrameChannel, MsgType, message, msg_type
from .terminal_host import TerminalHost


class BackendClient:
    """Runs one question against a backend, servicing its terminal requests."""

    def __init__(self, channel: FrameChannel) -> None:
        self._channel = channel

    def ask(self, question: str, snapshot: str, host: TerminalHost) -> str:
        """Submit ``question`` (with ``snapshot``) and return the reply text.

        Blocks pumping frames until the backend finishes the turn: it answers
        each ``REQUEST`` with a ``RESPONSE`` (using ``host``) and forwards
        ``PANEL_UPDATE`` notifications to ``host`` for rendering.
        """
        self._channel.send(
            message(MsgType.SUBMIT, text=question, snapshot=snapshot)
        )
        while True:
            msg = self._channel.recv()
            if msg is None:
                raise ConnectionError("backend disconnected during a turn")
            kind = msg_type(msg)
            if kind == MsgType.REPLY:
                return msg.get("text", "")
            self._handle(msg, host)

    def _handle(self, msg: dict, host: TerminalHost) -> None:
        kind = msg_type(msg)
        if kind == MsgType.REQUEST:
            result = self._serve_request(msg, host)
            self._channel.send(
                message(
                    MsgType.RESPONSE, call_id=msg.get("call_id"), result=result
                )
            )
        elif kind == MsgType.PANEL_UPDATE:
            self._apply_panel_update(msg, host)
        elif kind == MsgType.LOG:
            host.add_info(msg.get("text", ""))
        # Unknown message kinds are ignored so the protocol can grow.

    @staticmethod
    def _serve_request(msg: dict, host: TerminalHost) -> str:
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "snapshot":
            return host.snapshot()
        if method == "tool":
            return host.run_terminal_tool(
                params.get("name", ""), params.get("args") or {}
            )
        return f"[ludvart] unknown host request: {method!r}"

    @staticmethod
    def _apply_panel_update(msg: dict, host: TerminalHost) -> None:
        kind = msg.get("kind")
        if kind == "interim":
            host.narrate(msg.get("text", ""))
        elif kind == "activity":
            host.set_activity(msg.get("label", ""))
        elif kind == "info":
            host.add_info(msg.get("text", ""))
