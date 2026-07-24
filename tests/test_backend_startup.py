"""Client-side backend startup handshake: stream progress, read HELLO.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_backend_startup.py
"""

import contextlib
import io

from ludvart.__main__ import _read_backend_hello


class _FakeChannel:
    def __init__(self, frames):
        self._frames = list(frames)

    def recv(self):
        return self._frames.pop(0) if self._frames else None


class _FakeTransport:
    def __init__(self, frames):
        self.channel = _FakeChannel(frames)


def test_startup_streams_log_then_returns_label():
    frames = [
        {"type": "log", "text": "verifying copilot:opus (model 'x')..."},
        {"type": "log", "text": "starting the GitHub Copilot gateway (model 'x')..."},
        {"type": "log", "text": "copilot:opus: ok"},
        {"type": "log", "text": "verifying anthropic:claude..."},
        {"type": "log", "text": "anthropic:claude: ok"},
        {
            "type": "hello",
            "active_label": "copilot:opus",
            "verified": True,
            "verify_error": None,
        },
    ]
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        label = _read_backend_hello(_FakeTransport(frames))
    out = err.getvalue()
    assert label == "copilot:opus", label
    assert "starting the GitHub Copilot gateway" in out, out
    assert "verifying anthropic:claude" in out, out
    assert "backend model copilot:opus... ok" in out, out
    print("startup streams gateway + verification logs, returns label: OK")


def test_startup_reports_verification_failure():
    frames = [
        {"type": "log", "text": "verifying copilot:opus (model 'x')..."},
        {"type": "log", "text": "copilot:opus: FAILED (boom)"},
        {
            "type": "hello",
            "active_label": "copilot:opus",
            "verified": False,
            "verify_error": "boom",
        },
    ]
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        label = _read_backend_hello(_FakeTransport(frames))
    out = err.getvalue()
    assert label == "copilot:opus"
    assert "FAILED (boom)" in out, out
    assert "backend model copilot:opus... FAILED (boom)" in out, out
    print("startup surfaces a verification failure: OK")


def test_startup_handles_early_close():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        label = _read_backend_hello(_FakeTransport([]))
    assert label is None
    assert "closed before handshake" in err.getvalue()
    print("startup handles a backend that closes before HELLO: OK")


def main():
    test_startup_streams_log_then_returns_label()
    test_startup_reports_verification_failure()
    test_startup_handles_early_close()
    print("\nALL backend startup tests passed.")


if __name__ == "__main__":
    main()
