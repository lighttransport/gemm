#!/usr/bin/env bash
# Repository-local runtime environment for full Pixal3D inference under WSL2.
set -e

PIXAL3D_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PIXAL3D_OPENCV="$PIXAL3D_ROOT/ref/pixal3d/deps/opencv/usr/lib/x86_64-linux-gnu"
PIXAL3D_OPENBLAS_ROOT="$PIXAL3D_ROOT/tmp/pixal3d/openblas/usr"
PIXAL3D_OPENBLAS="$PIXAL3D_OPENBLAS_ROOT/lib/x86_64-linux-gnu/openblas-pthread"
PIXAL3D_CUDA_ROOT="$PIXAL3D_ROOT/.cuda-wsl/13.3"
PIXAL3D_CUDA_LIBS="$PIXAL3D_CUDA_ROOT/lib:$PIXAL3D_CUDA_ROOT/lib64:$PIXAL3D_CUDA_ROOT/targets/x86_64-linux/lib"

export C_INCLUDE_PATH="$PIXAL3D_OPENBLAS_ROOT/include/x86_64-linux-gnu/openblas-pthread${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$PIXAL3D_OPENBLAS_ROOT/include/x86_64-linux-gnu/openblas-pthread:$PIXAL3D_ROOT/tmp/pixal3d/boost/usr/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$PIXAL3D_OPENBLAS:$PIXAL3D_OPENCV${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$PIXAL3D_CUDA_LIBS:$PIXAL3D_OPENBLAS:$PIXAL3D_OPENCV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
