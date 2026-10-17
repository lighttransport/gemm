#!/bin/sh
# Virtual-human eye demo: http://127.0.0.1:8790/
#   sh server/vhuman/run.sh [--port 8790] [--mock]
# The server itself needs only numpy + Pillow (system python3). Qwen-Image
# iris plates use the selected native Qwen backend and --qwen-python.
# The installed tmp/vhuman-rocm-venv interpreter is selected automatically for AMD.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/tmp/vhuman-runtime"
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$root/tmp/vhuman-cache}"
mkdir -p "$XDG_CACHE_HOME"
rocm_lib=$(PYTHONPATH="$root" python3 -B -m server.vhuman.rocm_runtime)
export LD_LIBRARY_PATH="$rocm_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python=${PYTHON:-python3}
if [ -x "$root/tmp/vhuman-rocm-venv/bin/python" ]; then python=${PYTHON:-$root/tmp/vhuman-rocm-venv/bin/python}; fi
exec "$python" -m server.vhuman.app "$@"
