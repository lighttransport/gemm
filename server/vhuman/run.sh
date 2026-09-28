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
exec python3 -m server.vhuman.app "$@"
