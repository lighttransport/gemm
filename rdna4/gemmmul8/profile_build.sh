#!/bin/sh
# Build an instrumented copy without replacing the normal timing library.
set -eu
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)
PROFILE_ROOT="$PROJECT_ROOT/../../tmp/gemmmul8/profile"
TASK_TMP="$PROJECT_ROOT/../../tmp/gemmmul8/buildtmp"
mkdir -p "$PROFILE_ROOT" "$TASK_TMP"
tar --exclude='./vendor/GEMMul8/build' --exclude='./vendor/GEMMul8/lib' \
    --exclude='./vendor/GEMMul8/compile_info' --exclude='./*.so' \
    --exclude='./bench_ozaki' --exclude='./bench_modular' --exclude='./bench_wmma' --exclude='./test_api' \
    --exclude='./test_modular' --exclude='./tune_ozaki' --exclude='./__pycache__' \
    -cf - . | tar -xf - -C "$PROFILE_ROOT"
ROCM_ROOT=$(/opt/rocm/bin/hipconfig --path)
TMPDIR="$TASK_TMP" make -C "$PROFILE_ROOT/vendor/GEMMul8" -j "${OZAKI_JOBS:-8}" \
    BACKEND=hip HIP_PATH="$ROCM_ROOT" GPU_ARCH=gfx1201 OPS=gemm \
    OZ2_BACKENDS='INT8 FP8' PROFILE=on TEMPDIR="$TASK_TMP"
TMPDIR="$TASK_TMP" /opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx1201 \
    "$PROFILE_ROOT/profile_ozaki.hip" -L"$PROFILE_ROOT/vendor/GEMMul8/lib" \
    -Wl,-rpath,"$PROFILE_ROOT/vendor/GEMMul8/lib" -lgemmul8 \
    -L"$ROCM_ROOT/lib" -Wl,-rpath,"$ROCM_ROOT/lib" -lhipblaslt -lamdhip64 \
    -o "$PROFILE_ROOT/profile_ozaki"
printf 'Instrumented benchmark: %s/profile_ozaki\n' "$PROFILE_ROOT"
