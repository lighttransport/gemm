#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backend=${1:?Usage: run.sh cpu|cuda|rocm SCRIPT [ARGS...]}
shift
case "$backend" in cpu|cuda|rocm) ;; *) echo "Invalid backend: $backend" >&2; exit 2;; esac
export TMPDIR="$project_dir/../../tmp/pixal3d"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$project_dir/../../tmp/vhuman-cache}"
mkdir -p "$XDG_CACHE_HOME"
export UV_CACHE_DIR="$project_dir/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$project_dir/.cache/python"
export UV_PROJECT_ENVIRONMENT="$project_dir/.venv-$backend"
unset VIRTUAL_ENV
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$project_dir/.cache/extensions-$backend"
export TORCHINDUCTOR_CACHE_DIR="$project_dir/.cache/inductor-$backend"
export TRITON_CACHE_DIR="$project_dir/.cache/triton-$backend"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
if [ "$backend" = rocm ] && [ -d /opt/rocm/core/lib ]; then
    export LD_LIBRARY_PATH="/opt/rocm/core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Native input preparation shares vhuman's pinned ROCm environment.
if [ "$backend" = rocm ] && { [ "$(basename "${1:-}")" = prepare_input.py ] || [ "${1:-}" = "$project_dir/../../server/pixal3d/app.py" ]; } && [ -x "$project_dir/../../tmp/vhuman-rocm-venv/bin/python" ]; then
    exec "$project_dir/../../tmp/vhuman-rocm-venv/bin/python" "$@"
fi
exec uv run --project "$project_dir" --frozen --extra "$backend" python "$@"
