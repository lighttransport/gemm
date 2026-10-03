#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output="${GLM_OUTPUT:-/mnt/nvme01/models/glm53f/reap}"
exec vendor/llama.cpp/build-v100/bin/llama-server \
    --model "$output/glm53f-reap-gsq-rco.gguf" \
    --mmproj "$output/mmproj-glm53f-f16.gguf" \
    --ctx-size 131072 --parallel 1 --n-gpu-layers 999 \
    --split-mode layer --tensor-split 1,1 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --host 127.0.0.1 --port 8080 "$@"
