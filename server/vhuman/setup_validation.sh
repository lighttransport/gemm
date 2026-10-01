#!/bin/sh
# Install local validation dependencies; optionally run the complete suite.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
tools="$root/tmp/vhuman-tools"
run_tests=0
jobs=4
while [ "$#" -gt 0 ]; do
    case "$1" in
        --tools-root) tools=${2:?--tools-root needs a directory}; shift ;;
        --jobs) jobs=${2:?--jobs needs a positive integer}; shift ;;
        --run) run_tests=1 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done
python="$root/tmp/vhuman-rocm-venv/bin/python"
[ -x "$python" ] || { echo 'Run server/vhuman/setup_rocm.sh first' >&2; exit 1; }
export TMPDIR="$root/tmp/vhuman-runtime"
export UV_CACHE_DIR="$root/tmp/uv-cache"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$root/tmp/vhuman-cache}"
export LD_LIBRARY_PATH="/opt/rocm/core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$tools"
tools=$(CDPATH= cd -- "$tools" && pwd)
"$python" -B vulkan/tools/setup_glslc.py --prefix "$tools" --jobs "$jobs"
uv pip install --python "$python" -r server/vhuman/requirements-validation.txt
export PATH="$tools/bin:$PATH"
export GLSLC="$tools/bin/glslc"
export VHUMAN_RIG_PYTHON="$python"
export VHUMAN_DATASET_TEST_ROOT="${VHUMAN_DATASET_TEST_ROOT:-$tools/test-datasets}"
make -C vulkan/vhuman libvhuman_deformer_vk.so
if [ "$run_tests" -eq 1 ]; then
    command -v xvfb-run >/dev/null || { echo 'Install xvfb and xauth to run browser tests' >&2; exit 1; }
    export VHUMAN_BROWSER_MODE=xvfb
    export OPENBLAS_NUM_THREADS=4
    export OMP_NUM_THREADS=4
    xvfb-run -a -s '-screen 0 1280x900x24' "$python" -u -B -m server.vhuman.test_all -v
fi
