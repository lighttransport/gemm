#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
export TMPDIR="$root/tmp/qimg21-runtime"
mkdir -p "$TMPDIR"
python="$root/tmp/qimg21-ref-venv/bin/python"
if [ ! -x "$python" ] && [ -x "$root/tmp/vhuman-rocm-venv/bin/python" ]; then
    python="$root/tmp/vhuman-rocm-venv/bin/python"
    export LD_LIBRARY_PATH="/opt/rocm/core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "${PYTHON:-$python}" "$root/server/qwen_image21/app.py" "$@"
