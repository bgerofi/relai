"""Tests for external MCP server support (src/relai/mcp.py).

These spin up a *real* stdio MCP server (a tiny FastMCP script in a temp file)
and drive it through :class:`McpManager`, so discovery, tool namespacing and
tool-call routing are exercised end to end -- no mocking of the protocol. A
second, deliberately broken server checks that a failing/unreachable server is
reported without hanging or taking the others down.

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_mcp.py
"""

import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

from relai.mcp import (
    McpManager,
    McpConfigError,
    _expand,
    _public_name,
    load_config,
)


# A minimal, self-contained stdio MCP server used as a test fixture. It exposes
# two tools so we can check listing and calling.
_SERVER_SRC = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("relai-test")

    @mcp.tool()
    def echo(text: str) -> str:
        "Echo the given text back."
        return "echo:" + text

    @mcp.tool()
    def add(a: int, b: int) -> int:
        "Add two integers."
        return a + b

    if __name__ == "__main__":
        mcp.run()
    """
)


def _write_server(root: Path) -> Path:
    path = root / "echo_server.py"
    path.write_text(_SERVER_SRC)
    return path


def _write_config(root: Path, servers: dict) -> Path:
    path = root / "mcp.json"
    path.write_text(json.dumps({"servers": servers}))
    return path


def test_load_config_variants():
    root = Path(tempfile.mkdtemp())
    # missing file -> empty
    assert load_config(str(root / "nope.json")) == {}
    # mcpServers alias + disabled filtering
    p = root / "c.json"
    p.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "a": {"command": "x"},
                    "b": {"command": "y", "disabled": True},
                }
            }
        )
    )
    cfg = load_config(str(p))
    assert set(cfg) == {"a"}
    # malformed JSON -> error
    bad = root / "bad.json"
    bad.write_text("{ not json")
    try:
        load_config(str(bad))
    except McpConfigError:
        pass
    else:
        raise AssertionError("expected McpConfigError for malformed json")
    print("load_config variants: OK")


def test_expand_and_public_name():
    os.environ["RELAI_TEST_TOKEN"] = "secret123"
    assert _expand("Bearer ${env:RELAI_TEST_TOKEN}") == "Bearer secret123"
    assert _expand("${env:RELAI_DOES_NOT_EXIST}") == ""

    used: set[str] = set()
    n1 = _public_name("git.hub", "search repos", used)
    n2 = _public_name("git.hub", "search repos", used)  # forced collision
    assert n1 == "mcp_git_hub_search_repos"
    assert n2 != n1 and n2 not in ("", None)
    # schema-legal characters only, within length bounds
    import re

    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", n1)
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", n2)
    long = _public_name("s" * 40, "t" * 40, set())
    assert len(long) <= 64
    print("expand + public name: OK")


def test_discovery_and_call():
    root = Path(tempfile.mkdtemp())
    server = _write_server(root)
    cfg = _write_config(
        root,
        {"echo": {"command": sys.executable, "args": [str(server)]}},
    )
    mgr = McpManager(config_file=str(cfg))
    try:
        status = mgr.refresh()
        assert status.servers.get("echo") == (2, None), status.servers
        assert status.total_tools == 2

        specs = {s.name: s for s in mgr.tool_specs()}
        echo_name = _public_name("echo", "echo", set())
        add_name = _public_name("echo", "add", set())
        assert echo_name in specs and add_name in specs
        # description is namespaced and schema is a JSON object
        assert specs[echo_name].description.startswith("[MCP:echo]")
        assert specs[echo_name].input_schema.get("type") == "object"
        assert mgr.is_mcp_tool(echo_name) and not mgr.is_mcp_tool("nope")

        # call routing
        out = mgr.call_tool(echo_name, {"text": "hello"})
        assert out == "echo:hello", repr(out)
        out2 = mgr.call_tool(add_name, {"a": 2, "b": 40})
        assert out2 == "42", repr(out2)
        # unknown tool
        assert "unknown MCP tool" in mgr.call_tool("mcp_missing", {})
        print("discovery + call: OK")
    finally:
        mgr.close()


def test_refresh_reconnects():
    root = Path(tempfile.mkdtemp())
    server = _write_server(root)
    cfg_path = root / "mcp.json"
    cfg_path.write_text(
        json.dumps(
            {"servers": {"echo": {"command": sys.executable, "args": [str(server)]}}}
        )
    )
    mgr = McpManager(config_file=str(cfg_path))
    try:
        assert mgr.refresh().total_tools == 2
        # Edit the config to remove the server, then refresh again.
        cfg_path.write_text(json.dumps({"servers": {}}))
        status = mgr.refresh()
        assert status.total_tools == 0 and status.servers == {}
        assert mgr.tool_specs() == []
        print("refresh reconnects: OK")
    finally:
        mgr.close()


def test_broken_server_reported():
    root = Path(tempfile.mkdtemp())
    server = _write_server(root)
    cfg = _write_config(
        root,
        {
            "echo": {"command": sys.executable, "args": [str(server)]},
            "broken": {"command": "relai_nonexistent_binary_xyz", "args": []},
        },
    )
    # Short connect timeout so the broken server fails fast.
    mgr = McpManager(config_file=str(cfg), connect_timeout=8.0)
    try:
        status = mgr.refresh()
        # The good server still works...
        assert status.servers["echo"] == (2, None), status.servers
        # ...and the broken one is reported with an error, not a crash.
        count, err = status.servers["broken"]
        assert count == 0 and err, status.servers
        # A call against the good server still succeeds.
        echo_name = _public_name("echo", "echo", set())
        assert mgr.call_tool(echo_name, {"text": "x"}) == "echo:x"
        print("broken server reported: OK")
    finally:
        mgr.close()


def main():
    test_load_config_variants()
    test_expand_and_public_name()
    test_discovery_and_call()
    test_refresh_reconnects()
    test_broken_server_reported()
    print("\nALL MCP tests passed.")


if __name__ == "__main__":
    main()
