#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p vendor artifacts .cache
bash scripts/setup-python.sh
for entry in 'reap CerebrasResearch/reap' 'GSQ IST-DASLab/GSQ' 'RCO IST-DASLab/RCO' 'llama.cpp ggml-org/llama.cpp'; do
    read -r name repo <<< "$entry"
    if [[ ! -d "vendor/$name/.git" ]]; then
        git clone --depth 1 "https://github.com/$repo.git" "vendor/$name"
    fi
    git -C "vendor/$name" rev-parse HEAD > "artifacts/$name.revision"
done
rg -q 'class Glm5NextModel' vendor/llama.cpp/conversion/glm.py || { echo 'llama.cpp checkout lacks GLM5Next support'; exit 1; }
CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
export CUDACXX
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build-compression -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build vendor/llama.cpp/build-compression --parallel 16
# Local quality checks and GGUF inference use the RTX 5060 Ti with CUDA 13.
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build-local -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF
cmake --build vendor/llama.cpp/build-local --parallel 8
.venv/bin/python -m glm_reap.cli preflight --report artifacts/preflight.json
