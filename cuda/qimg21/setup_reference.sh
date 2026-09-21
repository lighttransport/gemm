#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QDIR="$ROOT/cuda/qimg21"
ENV_DIR="$ROOT/tmp/qimg21-ref-venv"
CACHE_DIR="$ROOT/tmp/uv-cache"
TMP_DIR="$ROOT/tmp/qimg21-tmp"

mkdir -p "$CACHE_DIR" "$TMP_DIR"
export UV_CACHE_DIR="$CACHE_DIR"
export TMPDIR="$TMP_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv --python 3.12 "$ENV_DIR"
fi

uv pip install --python "$ENV_DIR/bin/python" \
  'torch==2.14.0' 'torchvision==0.29.0' \
  'numpy>=2.0' 'Pillow>=10' 'safetensors>=0.5' 'accelerate>=1.10' \
  'qwen-vl-utils>=0.0.14' \
  'diffusers @ git+https://github.com/huggingface/diffusers@80c7ed262aeffbeb43ef13ae04baeb9b84515a69' \
  'transformers @ git+https://github.com/huggingface/transformers@c587bc884db2c2e31fc2b8102314656b17aa07b1'

"$ENV_DIR/bin/python" - <<'PY'
import torch
import diffusers
import transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

echo "reference environment ready: $ENV_DIR"
