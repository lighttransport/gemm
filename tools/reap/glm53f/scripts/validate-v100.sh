#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output="${GLM_OUTPUT:-/mnt/nvme01/models/glm53f/reap}"
# A successful short prompt does not prove long-context capacity. Force KV
# allocation at the requested context and preserve startup memory reports.
mkdir -p artifacts/v100
nvidia-smi > artifacts/v100/hardware.txt
vendor/llama.cpp/build-v100/bin/llama-cli \
    --model "$output/glm53f-reap-gsq-rco.gguf" \
    --ctx-size 131072 --n-gpu-layers 999 --split-mode layer \
    --tensor-split 1,1 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --prompt 'Write a Python function that returns Fibonacci numbers.' \
    --n-predict 128 > artifacts/v100/smoke.txt 2> artifacts/v100/memory.txt
echo 'Inspect memory.txt for CPU-offloaded layers and per-device VRAM.'
echo 'Long-prefill quality, tool calls, and vision still require evaluation.'
