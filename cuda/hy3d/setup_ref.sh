#!/usr/bin/env bash
# setup_ref.sh - Bootstrap PyTorch reference env + weights for Hunyuan3D-2.1.
#
# Usage:
#   bash cuda/hy3d/setup_ref.sh [--models-dir /mnt/disk01/models/Hunyuan3D-2.1]
#
# This is a thin wrapper that delegates to ref/hy3d/ (mirrors ref/qwen_image/):
#   1. Creates a uv venv under ref/hy3d/.venv with torch + deps.
#   2. Downloads hunyuan3d-dit-v2-1 and hunyuan3d-vae-v2-1 from
#      tencent/Hunyuan3D-2.1 on HuggingFace.
#   3. Exports the combined .ckpt files to per-component .safetensors using
#      ref/hy3d/export_safetensors.py and ref/hy3d/export_vae_safetensors.py.
#
# Prerequisites:
#   - uv  (https://docs.astral.sh/uv/)
#   - hf  (huggingface_hub CLI — old `huggingface-cli` is deprecated)
#   - ~8 GB free disk on --models-dir
#
# SPDX-License-Identifier: MIT
# Copyright 2025 - Present, Light Transport Entertainment Inc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REF_DIR="${REPO_ROOT}/ref/hy3d"
MODELS_DIR="/mnt/disk01/models/Hunyuan3D-2.1"
HF_REPO="tencent/Hunyuan3D-2.1"

for arg in "$@"; do
    case $arg in
        --models-dir=*) MODELS_DIR="${arg#*=}" ;;
        --models-dir)   shift; MODELS_DIR="${1:-$MODELS_DIR}" ;;
    esac
done

echo "=== Hunyuan3D-2.1 Reference Environment Setup ==="
echo "  Ref dir:    ${REF_DIR}"
echo "  Models dir: ${MODELS_DIR}"
echo "  HF repo:    ${HF_REPO}"
echo ""

# ---- Step 0: Check tooling ----
if ! command -v uv &>/dev/null; then
    echo "ERROR: 'uv' not found. Install from https://docs.astral.sh/uv/"
    exit 1
fi
if ! command -v hf &>/dev/null; then
    echo "ERROR: 'hf' not found. Install with: pip install -U huggingface_hub"
    exit 1
fi
echo "[0/4] Tooling OK: uv=$(uv --version), hf=$(hf --version 2>&1 | head -1)"

# ---- Step 1: uv venv ----
echo "[1/4] Syncing Python venv in ${REF_DIR}..."
[ -f "${REF_DIR}/pyproject.toml" ] || {
    echo "ERROR: ${REF_DIR}/pyproject.toml missing. Did you pull ref/hy3d/?"
    exit 1
}
(cd "${REF_DIR}" && uv sync)
echo "  venv ready at ${REF_DIR}/.venv"

# ---- Step 2: Download weights ----
echo "[2/4] Downloading weights to ${MODELS_DIR}..."
mkdir -p "${MODELS_DIR}"

DIT_DIR="${MODELS_DIR}/hunyuan3d-dit-v2-1"
VAE_DIR="${MODELS_DIR}/hunyuan3d-vae-v2-1"

if [ ! -f "${DIT_DIR}/model.fp16.ckpt" ]; then
    echo "  Fetching DiT..."
    hf download "${HF_REPO}" --local-dir "${MODELS_DIR}" --include "hunyuan3d-dit-v2-1/*"
else
    echo "  DiT already present: ${DIT_DIR}/model.fp16.ckpt"
fi

if [ ! -f "${VAE_DIR}/model.fp16.ckpt" ]; then
    echo "  Fetching VAE..."
    hf download "${HF_REPO}" --local-dir "${MODELS_DIR}" --include "hunyuan3d-vae-v2-1/*"
else
    echo "  VAE already present: ${VAE_DIR}/model.fp16.ckpt"
fi

# ---- Step 3: Export .ckpt -> .safetensors ----
echo "[3/4] Exporting safetensors..."
if [ -f "${DIT_DIR}/model.fp16.ckpt" ]; then
    if [ ! -f "${MODELS_DIR}/conditioner.safetensors" ] || \
       [ ! -f "${MODELS_DIR}/model.safetensors" ]; then
        (cd "${REF_DIR}" && uv run python export_safetensors.py \
            --ckpt "${DIT_DIR}/model.fp16.ckpt" \
            --outdir "${MODELS_DIR}")
    else
        echo "  DiT safetensors already exported"
    fi
fi
if [ -f "${VAE_DIR}/model.fp16.ckpt" ]; then
    if [ ! -f "${MODELS_DIR}/vae.safetensors" ]; then
        (cd "${REF_DIR}" && uv run python export_vae_safetensors.py \
            --ckpt "${VAE_DIR}/model.fp16.ckpt" \
            --outdir "${MODELS_DIR}")
    else
        echo "  VAE safetensors already exported"
    fi
fi

# ---- Step 4: Summary ----
echo "[4/4] Weight inventory:"
for f in "${MODELS_DIR}/conditioner.safetensors" \
         "${MODELS_DIR}/model.safetensors" \
         "${MODELS_DIR}/vae.safetensors"; do
    if [ -f "$f" ]; then
        echo "  OK       $f ($(du -sh "$f" | cut -f1))"
    else
        echo "  MISSING  $f"
    fi
done

cat <<EOF

=== Setup complete ===

Generate PyTorch reference outputs (from ${REF_DIR}):
  cd ${REF_DIR}
  uv run python dump_dinov2.py --ckpt ${DIT_DIR}/model.fp16.ckpt
  uv run python dump_vae.py --vae-path ${MODELS_DIR}/vae.safetensors --grid-res 8
  HY3D_REPO=/path/to/Hunyuan3D-2.1-repo \\
      uv run python dump_dit_single_step.py --ckpt ${DIT_DIR}/model.fp16.ckpt

Build & verify CUDA runner:
  cd ${SCRIPT_DIR} && make verify
  ./verify_dinov2 ${MODELS_DIR}/conditioner.safetensors \\
      --ref-dir ${REF_DIR}/output --out-dir cuda_output
  ./verify_vae   ${MODELS_DIR}/vae.safetensors \\
      --ref-dir ${REF_DIR}/output --out-dir cuda_output
  ./verify_dit   ${MODELS_DIR}/model.safetensors \\
      --ref-dir ${REF_DIR}/output --out-dir cuda_output

End-to-end run:
  ./test_cuda_hy3d \\
      ${MODELS_DIR}/conditioner.safetensors \\
      ${MODELS_DIR}/model.safetensors \\
      ${MODELS_DIR}/vae.safetensors \\
      -i input.ppm -o output.obj
EOF
