#!/bin/sh
set -eu
h3_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
h3_python=$1
shift
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$h3_root/tmp/video-rocm/h3-build"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
exec "$h3_python" "$h3_root/rdna4/minimax_h3/conditioning.py" "$@"
