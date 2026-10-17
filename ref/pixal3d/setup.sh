#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backend=${1:?Usage: setup.sh cpu|cuda|rocm}
case "$backend" in cpu|cuda|rocm) ;; *) echo "Invalid backend: $backend" >&2; exit 2;; esac
export TMPDIR="$project_dir/../../tmp/pixal3d"
export UV_CACHE_DIR="$(python3 "$project_dir/../../server/python_env.py" cache)"
export UV_PYTHON_INSTALL_DIR="$project_dir/.cache/python"
export UV_PROJECT_ENVIRONMENT="$(python3 "$project_dir/../../server/python_env.py" venv "$project_dir/.venv-$backend" --link)"
mkdir -p "$TMPDIR"
if [ "$backend" = rocm ] && [ -d /opt/rocm/core/lib ]; then
    export LD_LIBRARY_PATH="/opt/rocm/core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
uv sync --frozen --project "$project_dir" --extra "$backend" --python 3.12
exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$project_dir/check_device.py" "$backend"
