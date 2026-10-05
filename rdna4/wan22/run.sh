#!/bin/sh
set -eu
wan_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
wan_python="$wan_root/tmp/vhuman-rocm-venv/bin/python"
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$wan_root/tmp/video-rocm/wan22-build"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
exec "$wan_python" "$wan_root/rdna4/wan22/generate.py" "$@"
