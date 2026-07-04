"""Unit tests for LLM config resolution from env and ~/.relai/llm.conf.

Verifies that provider variables are read from the environment, fall back to a
``~/.relai/llm.conf`` file, and that the environment always overrides the file.

Run:
    cd /local_home/bgerofi1/src/relai && source .venv/bin/activate \
        && python tools/test_llm_conf.py
"""

import os

from relai.llm import _getvar, _load_conf, _read_provider, resolve_config


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_load_conf_parsing(tmp):
    path = os.path.join(tmp, "llm.conf")
    _write(
        path,
        "# a comment\n"
        "\n"
        "OPENAI_API_URL=https://api.openai.com/v1\n"
        "export OPENAI_API_KEY = sk-file \n"
        'OPENAI_MODEL="gpt-4o"\n'
        "NOEQUALS\n"
        "SINGLE='q u o t e d'\n",
    )
    conf = _load_conf(path)
    assert conf["OPENAI_API_URL"] == "https://api.openai.com/v1"
    assert conf["OPENAI_API_KEY"] == "sk-file"       # export + spaces stripped
    assert conf["OPENAI_MODEL"] == "gpt-4o"          # double quotes stripped
    assert conf["SINGLE"] == "q u o t e d"           # single quotes stripped
    assert "NOEQUALS" not in conf                      # lines without '=' skipped
    assert _load_conf(os.path.join(tmp, "missing.conf")) == {}
    print("load_conf parsing: OK")


def test_getvar_env_overrides_file():
    conf = {"OPENAI_API_KEY": "from-file"}
    os.environ.pop("OPENAI_API_KEY", None)
    assert _getvar(conf, "OPENAI_API_KEY") == "from-file"
    os.environ["OPENAI_API_KEY"] = "from-env"
    try:
        assert _getvar(conf, "OPENAI_API_KEY") == "from-env"
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
    assert _getvar(conf, "MISSING_VAR") is None
    print("getvar env overrides file: OK")


def test_read_provider_from_file_and_override():
    conf = {
        "OPENAI_API_URL": "https://file/v1/",
        "OPENAI_API_KEY": "sk-file",
        "OPENAI_MODEL": "gpt-4o",
        "OPENAI_CONTEXT_WINDOW": "4242",
    }
    for k in list(os.environ):
        if k.startswith("OPENAI_"):
            os.environ.pop(k, None)
    cfg = _read_provider("openai", conf)
    assert cfg is not None
    assert cfg.api_url == "https://file/v1"   # trailing slash stripped
    assert cfg.api_key == "sk-file"
    assert cfg.model == "gpt-4o"
    assert cfg.context_window == 4242

    # Environment overrides an individual file value.
    os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
    try:
        cfg2 = _read_provider("openai", conf)
        assert cfg2.model == "gpt-4o-mini"
    finally:
        os.environ.pop("OPENAI_MODEL", None)

    # Missing one field -> not configured.
    partial = dict(conf)
    partial.pop("OPENAI_MODEL")
    assert _read_provider("openai", partial) is None
    print("read_provider file + override: OK")


def test_resolve_config_uses_conf_file(tmp):
    # Point HOME at a temp dir so ~/.relai/llm.conf resolves there.
    path = os.path.join(tmp, ".relai", "llm.conf")
    _write(
        path,
        "ANTHROPIC_API_URL=https://api.anthropic.com\n"
        "ANTHROPIC_API_KEY=sk-ant-file\n"
        "ANTHROPIC_MODEL=claude-opus-4-8\n",
    )
    old_home = os.environ.get("HOME")
    # Clear any provider vars so only the file configures a provider.
    saved = {k: os.environ.pop(k) for k in list(os.environ)
             if k.split("_")[0] in {"OPENAI", "ANTHROPIC", "GOOGLE", "CUSTOM"}}
    os.environ["HOME"] = tmp
    try:
        cfg = resolve_config()
        assert cfg is not None
        assert cfg.name == "anthropic"
        assert cfg.api_key == "sk-ant-file"
    finally:
        if old_home is not None:
            os.environ["HOME"] = old_home
        else:
            os.environ.pop("HOME", None)
        os.environ.update(saved)
    print("resolve_config uses conf file: OK")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_load_conf_parsing(d)
        test_getvar_env_overrides_file()
        test_read_provider_from_file_and_override()
        test_resolve_config_uses_conf_file(d)
    print("\nALL llm.conf tests passed.")
