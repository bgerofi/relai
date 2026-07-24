"""Integration tests for the client<->backend RPC over framed channels.

Two levels:

* In-process loopback: run ``serve`` (with a fake LLM) on one thread and a
  ``BackendClient`` on another, connected by a pair of OS pipes. Exercises the
  full RemoteTerminalHost <-> BackendClient request/response + panel path
  without a subprocess.
* Real subprocess: fork ``python -m ludvart serve`` with the fake-LLM env and
  drive it through a ``Transport``, proving the CLI entry and stdio framing.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_backend_rpc.py
"""

import os
import subprocess
import sys
import threading

from ludvart.backend_client import BackendClient
from ludvart.protocol import FrameChannel
from ludvart.server import _FakeBackendLLM, serve
from ludvart.terminal_host import TerminalHost
from ludvart.transport import local_backend


class RecordingHost(TerminalHost):
    def __init__(self):
        self.tool_calls = []
        self.narrations = []
        self.activities = []
        self.infos = []
        self.snapshots = 0

    def snapshot(self):
        self.snapshots += 1
        return "CLIENT-SCREEN"

    def run_terminal_tool(self, name, args):
        self.tool_calls.append((name, args))
        return f"Injected via {name}"

    def narrate(self, text):
        self.narrations.append(text)

    def set_activity(self, label):
        self.activities.append(label)

    def add_info(self, text):
        self.infos.append(text)


def _pipe_pair():
    """Return (client_channel, backend_channel) connected by two OS pipes."""
    a_r, a_w = os.pipe()  # backend -> client
    b_r, b_w = os.pipe()  # client -> backend
    client = FrameChannel(os.fdopen(a_r, "rb"), os.fdopen(b_w, "wb"))
    backend = FrameChannel(os.fdopen(b_r, "rb"), os.fdopen(a_w, "wb"))
    return client, backend


def test_loopback_turn_with_client_tool():
    client_ch, backend_ch = _pipe_pair()

    def run_backend():
        serve(backend_ch, llm=_FakeBackendLLM())

    t = threading.Thread(target=run_backend, daemon=True)
    t.start()

    client = BackendClient(client_ch)
    host = RecordingHost()

    # The backend sends HELLO first; skip it before submitting.
    hello = client_ch.recv()
    assert hello["type"] == "hello", hello

    reply = client.ask("please echo", "SNAPSHOT-AT-ASK", host)

    # The fake LLM requests an inject_input tool call, then answers echoing it.
    assert host.tool_calls == [
        ("inject_input", {"text": "echo hi", "submit": True})
    ], host.tool_calls
    assert reply.startswith("done ("), reply
    assert "Injected via inject_input" in reply, reply
    # Narration + activity flowed to the client host.
    assert "working on it" in host.narrations
    assert any("Calling inject_input" in a for a in host.activities)

    client_ch.close()
    t.join(timeout=2)
    assert not t.is_alive()
    backend_ch.close()
    print("loopback turn drives a client tool and returns the reply: OK")


def test_loopback_plain_turn_without_tools():
    client_ch, backend_ch = _pipe_pair()

    class NoToolLLM(_FakeBackendLLM):
        def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
            if on_text:
                on_text("hi")
            from ludvart.llm import Turn

            return Turn(
                text="just text",
                assistant_message={"role": "assistant", "content": "just text"},
                usage=None,
            )

    t = threading.Thread(
        target=lambda: serve(backend_ch, llm=NoToolLLM()), daemon=True
    )
    t.start()

    client = BackendClient(client_ch)
    host = RecordingHost()
    assert client_ch.recv()["type"] == "hello"
    reply = client.ask("hi", "S", host)
    assert reply == "just text"
    assert host.tool_calls == []

    client_ch.close()
    t.join(timeout=2)
    backend_ch.close()
    print("loopback plain turn returns text with no tool calls: OK")


def test_subprocess_serve_end_to_end():
    # Fork the real `python -m ludvart serve` with the offline fake LLM.
    env = dict(os.environ)
    env["LUDVART_BACKEND_FAKE_LLM"] = "1"
    transport = local_backend(env=env, stderr=subprocess.DEVNULL)
    try:
        client = BackendClient(transport.channel)
        host = RecordingHost()
        # Backend greets with HELLO.
        assert transport.channel.recv()["type"] == "hello"
        reply = client.ask("do it", "SNAP", host)
        assert host.tool_calls == [
            ("inject_input", {"text": "echo hi", "submit": True})
        ], host.tool_calls
        assert reply.startswith("done ("), reply
    finally:
        transport.close()
    assert transport.poll() is not None  # backend reaped, no leak
    print("forked `ludvart serve` runs a full turn over stdio: OK")


def main():
    test_loopback_turn_with_client_tool()
    test_loopback_plain_turn_without_tools()
    test_subprocess_serve_end_to_end()
    print("\nALL backend RPC tests passed.")


if __name__ == "__main__":
    main()
