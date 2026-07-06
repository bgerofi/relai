"""Unit tests for the first-run LLM setup: ``write_provider_conf``.

Verifies that the setup wizard's file writer persists the correct provider
variable names to ``~/.ludvart/llm.conf``, updates them in place on re-run,
preserves unrelated lines, applies 0600 permissions, and round-trips back
through the normal config resolution (``_load_conf`` / ``resolve_config``).

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_llm_setup.py
"""

import os
import stat
import tempfile

from ludvart.llm import _load_conf, _read_provider, write_provider_conf


def test_writes_expected_var_names(tmp):
    path = os.path.join(tmp, "llm.conf")
    write_provider_conf(
        "anthropic",
        "https://api.anthropic.com",
        "sk-secret",
        "claude-x",
        path=path,
    )
    conf = _load_conf(path)
    assert conf["ANTHROPIC_API_URL"] == "https://api.anthropic.com"
    assert conf["ANTHROPIC_API_KEY"] == "sk-secret"
    assert conf["ANTHROPIC_MODEL"] == "claude-x"
    print("writes expected var names: OK")


def test_file_is_owner_only(tmp):
    path = os.path.join(tmp, "llm.conf")
    write_provider_conf("openai", "https://api.openai.com/v1", "sk", "gpt", path=path)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, oct(mode)
    print("file is owner-only (0600): OK")


def test_resolves_after_write(tmp):
    path = os.path.join(tmp, "llm.conf")
    write_provider_conf(
        "custom", "https://proxy.local/v1", "sk-custom", "my-model", path=path
    )
    # Make sure env doesn't interfere, then resolve straight from the file.
    for var in (
        "CUSTOM_API_URL",
        "CUSTOM_API_KEY",
        "CUSTOM_MODEL",
    ):
        os.environ.pop(var, None)
    cfg = _read_provider("custom", _load_conf(path))
    assert cfg is not None
    assert cfg.name == "custom"
    assert cfg.api_url == "https://proxy.local/v1"
    assert cfg.api_key == "sk-custom"
    assert cfg.model == "my-model"
    print("resolves after write: OK")


def test_update_in_place_preserves_other_lines(tmp):
    path = os.path.join(tmp, "llm.conf")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "# my config\n"
            "OPENAI_API_URL=https://old/v1\n"
            "export OPENAI_API_KEY = old-key\n"
            "OPENAI_MODEL=old-model\n"
            "ANTHROPIC_CONTEXT_WINDOW=1000000\n"
        )
    write_provider_conf(
        "openai", "https://new/v1", "new-key", "new-model", path=path
    )
    text = open(path, encoding="utf-8").read()
    # The comment and the unrelated tuning var survive.
    assert "# my config" in text
    assert "ANTHROPIC_CONTEXT_WINDOW=1000000" in text
    # Updated values, and no leftover/duplicate old values.
    conf = _load_conf(path)
    assert conf["OPENAI_API_URL"] == "https://new/v1"
    assert conf["OPENAI_API_KEY"] == "new-key"
    assert conf["OPENAI_MODEL"] == "new-model"
    assert "old-key" not in text
    assert "old-model" not in text
    assert "https://old/v1" not in text
    # Exactly one assignment of each updated key.
    assert text.count("OPENAI_API_URL=") == 1
    assert text.count("OPENAI_API_KEY=") == 1
    assert text.count("OPENAI_MODEL=") == 1
    print("update in place preserves other lines: OK")


def test_rejects_unknown_provider(tmp):
    path = os.path.join(tmp, "llm.conf")
    try:
        write_provider_conf("bogus", "u", "k", "m", path=path)
    except ValueError:
        print("rejects unknown provider: OK")
        return
    raise AssertionError("expected ValueError for unknown provider")


def _run():
    tests = [
        test_writes_expected_var_names,
        test_file_is_owner_only,
        test_resolves_after_write,
        test_update_in_place_preserves_other_lines,
        test_rejects_unknown_provider,
    ]
    for fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            fn(tmp) if fn.__code__.co_argcount else fn()
    print("\nall test_llm_setup tests passed")


if __name__ == "__main__":
    _run()
