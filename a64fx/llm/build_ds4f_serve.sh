#!/bin/sh
# build_ds4f_serve.sh -- build the single-node DS4F serving library (x86).
#
# Produces ./libds4f_serve.so (ctypes-loadable by ds4f_serve_runner.py).  The
# model math is the exact/tierb2/mHC forward; the dense bank goes to the ROCm
# when DS4F_SERVE_HIP is defined. Large prompt tiles may stream each layer's
# raw MXFP4 experts through the GPU via --hip-expert-stream.
#
# Run from the repo root:
#   sh a64fx/llm/build_ds4f_serve.sh
set -e
CC="${CC:-gcc}"
ARCH="${ARCH:--march=native -mavx2 -mfma -mf16c -ffp-contract=fast}"
HIP="${HIP:-/opt/rocm/include}"
ROCM_LIB="$(ls -d /opt/rocm/core-*/lib 2>/dev/null | head -1)"

# locate the DS4F source files
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

CFLAGS="-O2 -std=c11 -D_GNU_SOURCE -fPIC -shared $ARCH -I. -I./common -I./hetero/ds4f -I./rdna4 -I./cuda"
if [ -n "$ROCM_LIB" ]; then
    CFLAGS="$CFLAGS -DDS4F_SERVE_HIP -I$HIP -L$ROCM_LIB -Wl,-rpath,$ROCM_LIB"
    HIP_LIBS="-lhiprtc -lamdhip64"
else
    HIP_LIBS=""
fi

$CC $CFLAGS -o libds4f_serve.so \
    a64fx/llm/ds4f_serve_lib.c \
    hetero/ds4f/hip_ds4f_dense.c hetero/ds4f/dual_ds4f_prefill.c \
    hetero/ds4f/cuda_ds4f_mxfp4.c cuda/cuew.c rdna4/rocew.c \
    $HIP_LIBS -ldl -lm

echo "built ./libds4f_serve.so"$([ -n "$ROCM_LIB" ] && echo " (ROCm: $ROCM_LIB)" || echo " (CPU-only)")
