#!/bin/sh
# Virtual-human eye demo: http://127.0.0.1:8790/
#   sh server/vhuman/run.sh [--port 8790] [--mock]
# The server itself needs only numpy + Pillow (system python3). Qwen-Image
# iris plates run cuda/qimg21/native_generate.py with --qwen-python
# (default tmp/qimg21-ref-venv/bin/python, which has torch).
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/tmp/vhuman-runtime"
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$root/tmp/vhuman-cache}"
mkdir -p "$XDG_CACHE_HOME"
export LD_LIBRARY_PATH="/opt/rocm/core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python=${PYTHON:-python3}
if [ -x "$root/tmp/vhuman-rocm-venv/bin/python" ]; then python=${PYTHON:-$root/tmp/vhuman-rocm-venv/bin/python}; fi
exec "$python" -m server.vhuman.app "$@"
