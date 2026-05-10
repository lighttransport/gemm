#!/usr/bin/env bash
# One-shot setup for cuda/trellis2 reference texgen environment.
#
# Creates a Python 3.12 venv, installs pinned PyTorch CUDA wheels +
# the ref-pipeline stack, then builds o-voxel and FlexGEMM from the
# in-tree sources targeting CUDA.
#
# Re-run is idempotent (uv pip install is incremental).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found — no NVIDIA driver detected." >&2
    echo "         The CUDA torch wheels will install but won't be usable." >&2
fi

cd "$SCRIPT_DIR"

# 1. venv
if [[ ! -d .venv ]]; then
    echo "[1/4] Creating .venv (python 3.12)"
    uv venv --python 3.12 .venv
else
    echo "[1/4] Reusing existing .venv"
fi

# 2. base deps + torch + ref-pipeline
echo "[2/4] Installing torch-cu124 + ref-pipeline extras"
uv pip install -e ".[torch-cu124,ref-pipeline]"

# 3. o-voxel (sparse conv kernels). BUILD_TARGET=cuda picks the CUDA backend.
OVOXEL_SRC="$REPO_ROOT/cpu/trellis2/trellis2_repo/o-voxel"
if [[ -d "$OVOXEL_SRC" ]]; then
    echo "[3/4] Building o-voxel (CUDA)"
    BUILD_TARGET=cuda uv pip install -e "$OVOXEL_SRC" --no-build-isolation
else
    echo "[3/4] SKIP o-voxel: $OVOXEL_SRC not found"
fi

# 4. FlexGEMM
FLEXGEMM_SRC="$REPO_ROOT/rdna4/trellis2/deps/FlexGEMM"
if [[ -d "$FLEXGEMM_SRC" ]]; then
    echo "[4/4] Building FlexGEMM (CUDA)"
    BUILD_TARGET=cuda uv pip install -e "$FLEXGEMM_SRC" --no-build-isolation
else
    echo "[4/4] SKIP FlexGEMM: $FLEXGEMM_SRC not found"
fi

echo
echo "Setup complete."
echo "Run reference texgen with:"
echo "  ./run_stage2_ref.sh"
