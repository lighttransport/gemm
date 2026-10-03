#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Optional deployment build only. Local compression never depends on this.
export CUDACXX="${V100_CUDACXX:-/usr/local/cuda-12.9/bin/nvcc}"
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build-v100 \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_CURL=OFF
cmake --build vendor/llama.cpp/build-v100 --parallel 8
