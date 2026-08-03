#!/bin/bash
# Select the shared K3 Python environment for tokenizer/staging helpers.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ARCH=$(uname -m)
PYTHON=${K3_PYTHON:-$SCRIPT_DIR/.venv-$ARCH/bin/python}

if [[ ! -x "$PYTHON" ]]; then
    echo "k3_python: missing executable: $PYTHON" >&2
    echo "create it with: uv venv --python 3.11 $SCRIPT_DIR/.venv-$ARCH" >&2
    exit 127
fi

exec "$PYTHON" "$@"
