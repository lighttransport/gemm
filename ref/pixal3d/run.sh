#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backend=${1:?Usage: run.sh cpu|cuda|rocm SCRIPT [ARGS...]}
shift
case "$backend" in cpu|cuda|rocm) ;; *) echo "Invalid backend: $backend" >&2; exit 2;; esac
export TMPDIR="$project_dir/../../tmp/pixal3d"
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
exec uv run --project "$project_dir" --frozen --extra "$backend" python "$@"
