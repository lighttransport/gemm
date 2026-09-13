#!/bin/sh
# Optional CUDA-only mesh oracle. Never required by native inference.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export TMPDIR="$project_dir/../../tmp/pixal3d"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export MAX_JOBS=${MAX_JOBS:-4}
mkdir -p "$TMPDIR"
cd "$project_dir/cumesh-upstream"
git submodule update --init --recursive
# Some toolkit installations omit headers shipped by the local PyTorch wheels.
site="$project_dir/.venv-cuda/lib/python3.12/site-packages/nvidia"
export CPATH="$site/cusparse/include:$site/cublas/include:$site/cusolver/include${CPATH:+:$CPATH}"
exec "$project_dir/.venv-cuda/bin/python" setup.py build_ext --inplace
