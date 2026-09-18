#!/bin/bash
# Native A64FX llama.cpp/ggml diagnostic build.
# Keep this separate from any workstation build directory: the checkout is
# shared, while the compiler, ISA, and generated objects are node-local.
set -euo pipefail

SRC=${LLAMA_SRC:-$HOME/work/llama.cpp}
BUILD=${LLAMA_BUILD:-/local/glm53f-llama-a64fx}
JOBS=${JOBS:-8}
REPO=${REPO:-$HOME/work/gemm/glm53f}

mkdir -p "$BUILD"
# Some Fugaku images do not provide /tmp to the compute-side shell.  Keep
# compiler/CMake temporaries on the node-local filesystem as required by the
# remote procedure.
mkdir -p "$BUILD/tmp"
export TMPDIR=${TMPDIR:-$BUILD/tmp}
cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=fcc -DCMAKE_CXX_COMPILER=FCC \
    -DGIT_EXECUTABLE=/bin/false \
    -DCMAKE_C_FLAGS="-Nclang -O3 -fopenmp -include $REPO/a64fx/glm5/llama_a64fx_disable_vectorization.h" \
    -DCMAKE_CXX_FLAGS="-Nclang -O3 -fopenmp -include $REPO/a64fx/glm5/llama_a64fx_disable_vectorization.h" \
    -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH='armv8.2-a+sve' \
    -DGGML_CPU=ON -DGGML_OPENMP=ON -DGGML_CCACHE=OFF \
    -DGGML_CPU_HBM=OFF -DGGML_CPU_KLEIDIAI=OFF \
    -DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_BMI2=OFF \
    -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_AVX512=OFF \
    -DGGML_LASX=OFF -DGGML_LSX=OFF -DGGML_RVV=OFF \
    -DGGML_RV_ZFH=OFF -DGGML_RV_ZVFH=OFF -DGGML_RV_ZICBOP=OFF \
    -DGGML_RV_ZIHINTPAUSE=OFF -DGGML_VXE=OFF \
    -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF -DGGML_RPC=OFF -DGGML_SYCL=OFF \
    -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF \
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_MTMD=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_LLGUIDANCE=OFF -DLLAMA_SUBPROCESS=OFF \
    -DLLAMA_ALL_WARNINGS=OFF

# llama-debug in the current shared checkout is not buildable independently:
# its debug.cpp references a removed base_callback_data type.  llama-simple
# exercises the same ggml/llama shared-library link without that stale source.
targets="ggml-base ggml-cpu ggml llama llama-common llama-simple llama-gguf"
ninja -C "$BUILD" -j "$JOBS" $targets

echo "A64FX_LLAMA_BUILD PASS build=$BUILD"
file "$BUILD/bin/llama-simple"
# llama.cpp examples conventionally return 1 after printing usage when no
# model is supplied; accept that usage-path exit while rejecting crashes or
# dynamic-loader failures.
set +e
"$BUILD/bin/llama-simple" --help >/dev/null 2>&1
rc=$?
set -e
test "$rc" -eq 0 -o "$rc" -eq 1
echo "A64FX_LLAMA_EXEC PASS"
file "$BUILD/bin/llama-gguf"
echo "A64FX_LLAMA_GGUF PASS"
GGUF_SMOKE=${GGUF_SMOKE:-$SRC/models/ggml-vocab-gemma-4.gguf}
if [ -r "$GGUF_SMOKE" ]; then
    "$BUILD/bin/llama-gguf" "$GGUF_SMOKE" r n >/dev/null 2>&1
    echo "A64FX_LLAMA_GGUF_READ PASS file=$GGUF_SMOKE"
else
    echo "A64FX_LLAMA_GGUF_READ SKIP file=$GGUF_SMOKE"
fi
