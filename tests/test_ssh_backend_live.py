"""Live SSH backend round-trip (opt-in, real network).

Skipped unless ``LUDVART_SSH_BACKEND`` is set to a ``host:folder`` reachable by
key-based SSH, whose ``folder`` holds this ludvart checkout with a ``.venv``.
It spawns the real backend over SSH (with the offline fake LLM) and drives one
full agent turn -- including a client-side ``inject_input`` -- proving the whole
protocol works across a real SSH stdio pipe.

Run it against this dev host:

    LUDVART_SSH_BACKEND=hpc-doit-dev01:/local_home/bgerofi1/src/ludvart \
        tests/run.sh -k ssh_backend_live -s
"""

import os
import subprocess

import pytest

from ludvart.backend_client import BackendClient
from ludvart.terminal_host import TerminalHost
from ludvart.transport import Transport, parse_backend_spec, ssh_backend_argv


class _RecordingHost(TerminalHost):
    def __init__(self):
        self.tools = []
        self.narrations = []

    def snapshot(self):
        return "REMOTE-SNAP"

    def run_terminal_tool(self, name, args):
        self.tools.append((name, args))
        return f"injected:{name}"

    def narrate(self, text):
        self.narrations.append(text)

    def set_activity(self, label):
        pass

    def add_info(self, text):
        pass


def test_ssh_backend_live():
    spec = os.environ.get("LUDVART_SSH_BACKEND")
    if not spec:
        pytest.skip("set LUDVART_SSH_BACKEND=host:folder to run the live SSH test")
    host, folder = parse_backend_spec(spec)

    # Real SSH transport, but tell the remote backend to use the offline fake LLM
    # so the test is deterministic and needs no provider credentials.
    argv = ssh_backend_argv(
        host, folder, remote_env={"LUDVART_BACKEND_FAKE_LLM": "1"}
    )
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
    )
    transport = Transport(proc)
    try:
        client = BackendClient(transport.channel)
        rec = _RecordingHost()
        hello = transport.channel.recv()
        assert hello and hello.get("type") == "hello", hello
        reply = client.ask("do it over ssh", "SNAP", rec)
        assert rec.tools == [
            ("inject_input", {"text": "echo hi", "submit": True})
        ], rec.tools
        assert reply.startswith("done ("), reply
        assert "injected:inject_input" in reply, reply
    finally:
        transport.close()
    assert transport.poll() is not None  # backend reaped over SSH
    print("live SSH backend round-trip: OK")


if __name__ == "__main__":
    test_ssh_backend_live()
    print("\nlive SSH backend test passed.")
