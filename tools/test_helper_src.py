"""Unit tests for the embedded golden relai_helper (helper_src.py).

Covers: the asset loads and matches its pinned checksum; the install command is
well-formed, single-quote-safe, and self-contained; and running it against a
throwaway HOME installs, then reports "current", then repairs a tampered copy.

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_helper_src.py
"""

import hashlib
import os
import re
import subprocess
import tempfile

from relai.helper_src import (
    RELAI_HELPER_MD5,
    RELAI_HELPER_MD5_EXPECTED,
    RELAI_HELPER_SOURCE,
    RELAI_HELPER_VERSION,
    helper_install_command,
    helper_install_payload_b64,
)


def test_asset_integrity():
    assert RELAI_HELPER_MD5 == RELAI_HELPER_MD5_EXPECTED
    assert RELAI_HELPER_MD5 == hashlib.md5(RELAI_HELPER_SOURCE).hexdigest()
    # Version derived from the source matches the source's VER line.
    assert re.search(rb'^VER\s*=\s*"%s"' % RELAI_HELPER_VERSION.encode(),
                     RELAI_HELPER_SOURCE, re.MULTILINE)
    assert RELAI_HELPER_SOURCE.startswith(b"#!")
    print("asset integrity + version: OK")


def test_command_is_quote_safe():
    cmd = helper_install_command()
    # Wrapped in single quotes; the inner program must contain no single quote
    # (so the outer shell quoting can never be broken by the payload/program).
    assert cmd.startswith("python3 -c '") and cmd.endswith("'")
    inner = cmd[len("python3 -c '"):-1]
    assert "'" not in inner, "inner program must not contain a single quote"
    # The payload embedded in the command is exactly the golden source b64.
    assert helper_install_payload_b64() in cmd
    print("command is quote-safe + carries golden payload: OK")


def _run(cmd, home):
    env = dict(os.environ, HOME=home)
    return subprocess.run(["bash", "-c", cmd], env=env,
                          capture_output=True, text=True)


def test_install_current_and_repair():
    cmd = helper_install_command()
    with tempfile.TemporaryDirectory() as home:
        dest = os.path.join(home, ".relai", "bin", "relai_helper")

        # 1. Fresh install.
        out = _run(cmd, home).stdout
        assert "status=installed" in out and "ok=1" in out and "reason=missing" in out, out
        assert os.path.isfile(dest)
        assert os.access(dest, os.X_OK), "helper must be executable"
        assert hashlib.md5(open(dest, "rb").read()).hexdigest() == RELAI_HELPER_MD5

        # 2. Already current -> no rewrite.
        out = _run(cmd, home).stdout
        assert "status=current" in out and "reason=match" in out, out

        # 3. Tamper, then repair.
        with open(dest, "ab") as fh:
            fh.write(b"\n# sneaky change\n")
        assert hashlib.md5(open(dest, "rb").read()).hexdigest() != RELAI_HELPER_MD5
        out = _run(cmd, home).stdout
        assert "status=installed" in out and "reason=stale_or_modified" in out, out
        assert hashlib.md5(open(dest, "rb").read()).hexdigest() == RELAI_HELPER_MD5

        # 4. The repaired helper actually runs.
        info = subprocess.run([dest, "info"], capture_output=True, text=True)
        assert info.returncode == 0 and "RELAI:BEGIN op=info" in info.stdout, info.stdout
    print("install / current / repair round-trip: OK")


if __name__ == "__main__":
    test_asset_integrity()
    test_command_is_quote_safe()
    test_install_current_and_repair()
    print("all helper_src tests passed")
