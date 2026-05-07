#!/usr/bin/env bash
# One-shot setup for rdna4/hy3d ROCm PyTorch reference env.
#
# Mirrors cuda/trellis2/setup_cuda.sh but pulls torch+torchvision from the
# Radeon ROCm 7.2.2 manylinux wheels instead of the CUDA index. Used to
# generate timed baseline runs of ref/hy3d/run_full_pipeline.py on RX 9070 XT
# (gfx1201) for comparison against the rdna4/hy3d HIP path.
#
# Re-run is idempotent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if ! command -v rocminfo >/dev/null 2>&1 && ! command -v rocm-smi >/dev/null 2>&1; then
    echo "WARNING: no ROCm tooling on PATH — torch wheels will install but" >&2
    echo "         won't be usable without a ROCm runtime." >&2
fi

cd "$SCRIPT_DIR"

# 1. venv (3.12 required for the rocm-rel-7.2.2 cp312 wheels)
if [[ ! -d .venv ]]; then
    echo "[1/2] Creating .venv (python 3.12)"
    uv venv --python 3.12 .venv
else
    echo "[1/2] Reusing existing .venv"
fi

# 2. torch ROCm + ref deps
echo "[2/2] Installing torch-rocm722 wheels + ref deps"
uv pip install -e ".[torch-rocm722]"

echo
echo "Setup complete."
echo "Run reference shape-gen with:"
echo "  bash $SCRIPT_DIR/run_rocm_ref.sh"
