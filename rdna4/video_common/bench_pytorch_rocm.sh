#!/bin/sh
# Run the isolated ROCm/ComfyUI benchmark environment prepared in repo tmp/.
set -eu
bench_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
bench_scratch="$bench_root/tmp/video-rocm"
bench_python="$bench_root/tmp/vhuman-rocm-venv/bin/python"
bench_comfy="$bench_scratch/pytorch-bench-comfy"
bench_deps="$bench_scratch/pytorch-bench-deps"
if [ ! -x "$bench_python" ] || [ ! -d "$bench_comfy/comfy" ] || [ ! -d "$bench_deps/comfy_kitchen" ]; then
    echo "Benchmark environment is missing; see rdna4/README.md." >&2
    exit 1
fi
cd "$bench_root"
export TMPDIR="$bench_scratch"
export PYTHONPATH="$bench_comfy:$bench_deps:$bench_scratch/reference-deps${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$bench_scratch/pytorch-bench-inductor"
mkdir -p "$bench_root/tmp/pixal3d/device-locks"
if [ "${1-}" = "--help" ] || [ "${1-}" = "-h" ]; then
    exec "$bench_python" "$bench_root/rdna4/video_common/bench_pytorch.py" "$@"
fi
exec flock "$bench_root/tmp/pixal3d/device-locks/rocm-0.lock" \
    "$bench_python" "$bench_root/rdna4/video_common/bench_pytorch.py" "$@"
