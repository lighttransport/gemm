#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_LINK_MODE=copy
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p artifacts .cache
UV="${UV:-uv}"
if [[ ! -x .venv/bin/python ]]; then
    "$UV" venv --python 3.12 .venv
fi
torch_metadata=(.venv/lib/python*/site-packages/torch-*.dist-info/METADATA)
if [[ -f "${torch_metadata[0]}" ]] && .venv/bin/python -c 'import packaging' 2>/dev/null; then
    .venv/bin/python scripts/repair-uv-cache.py --cache "$UV_CACHE_DIR" \
        --torch-metadata "${torch_metadata[0]}"
fi
"$UV" pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu130
"$UV" pip install --python .venv/bin/python -e . pytest
.venv/bin/python -m glm_reap.cli --help
