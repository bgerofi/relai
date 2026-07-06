#!/usr/bin/env bash
# Set up the ludvart development environment using uv.
# Creates a local .venv and installs ludvart (editable) plus its dependencies,
# then verifies the install so a half-configured environment fails loudly.
set -euo pipefail

cd "$(dirname "$0")"

# Ensure uv is available. If it isn't on PATH, bootstrap a project-local copy
# under ./.uv using the official installer instead of asking the user to do it.
if ! command -v uv >/dev/null 2>&1; then
    echo "==> 'uv' not found; installing a project-local copy into ./.uv"
    export UV_INSTALL_DIR="$PWD/.uv"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
    else
        echo "error: need 'curl' or 'wget' to download uv." >&2
        exit 1
    fi
    export PATH="$UV_INSTALL_DIR:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is still not available after attempting to install it." >&2
    exit 1
fi

echo "==> Creating virtual environment (.venv)"
# Reuse an existing .venv (uv would otherwise prompt interactively, which hangs
# non-interactive runs). The install step below repairs/updates it regardless.
if [[ -d .venv ]]; then
    echo "    .venv already exists; reusing it."
else
    uv venv
fi

# Point uv (and our verification below) unambiguously at this .venv, regardless
# of any other environment that happens to be active. Without this, an inactive
# venv can cause 'uv pip install' to target the wrong interpreter, leaving the
# ludvart module and launcher missing (the "No module named ludvart" symptom).
export VIRTUAL_ENV="$PWD/.venv"
VENV_PY="$VIRTUAL_ENV/bin/python"

echo "==> Installing ludvart (editable) with dev tools (pytest) and dependencies"
uv pip install --python "$VENV_PY" -e ".[dev]"

echo "==> Installing the LiteLLM gateway (enables the GitHub Copilot endpoint)"
# Optional feature: ludvart works with the other providers without it, so a
# failure here warns but does not abort the whole setup.
if ! uv pip install --python "$VENV_PY" "litellm[proxy]"; then
    echo "warning: could not install litellm[proxy]; the GitHub Copilot gateway" >&2
    echo "         option will be unavailable (other providers still work)." >&2
fi

echo "==> Verifying installation"
if ! "$VENV_PY" -c "import ludvart, ludvart.__main__" >/dev/null 2>&1; then
    echo "error: ludvart did not install correctly (cannot import 'ludvart')." >&2
    echo "Try re-running ./setup.sh; if it persists, check the output of:" >&2
    echo "    uv pip install --python \"$VENV_PY\" -e ." >&2
    exit 1
fi
if [[ ! -x "$VIRTUAL_ENV/bin/ludvart" ]]; then
    echo "error: the 'ludvart' launcher was not created in .venv/bin." >&2
    echo "Confirm pyproject.toml has a [project.scripts] entry for ludvart." >&2
    exit 1
fi

echo
echo "Done. Activate with:"
echo "    source .venv/bin/activate"
echo "Then run:"
echo "    ludvart            # spawns your \$SHELL"
echo "    ludvart -- htop    # spawns any command"
echo "Run the tests with:"
echo "    pytest -m 'not e2e'   # fast unit tests only"
echo "    pytest                # also runs e2e (needs a configured LLM; real tokens)"
