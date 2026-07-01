#!/usr/bin/env bash
# Set up the relai development environment using uv.
# Creates a local .venv and installs relai (editable) plus its dependencies.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' is not installed or not on PATH." >&2
    echo "Install it from https://docs.astral.sh/uv/ and re-run." >&2
    exit 1
fi

echo "==> Creating virtual environment (.venv)"
uv venv

echo "==> Installing relai (editable) and dependencies"
uv pip install -e .

echo
echo "Done. Activate with:"
echo "    source .venv/bin/activate"
echo "Then run:"
echo "    relai            # spawns your \$SHELL"
echo "    relai -- htop    # spawns any command"
