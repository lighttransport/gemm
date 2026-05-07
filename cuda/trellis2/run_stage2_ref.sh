#!/usr/bin/env bash
# Reference Stage 2 (texgen) on CUDA — produces ground-truth textured GLB
# for comparing against the ROCm path.
#
# Inputs (override via env):
#   MESH    — input mesh OBJ (default: knight from /mnt/disk1/tmp/tex_knight_r512_fresh)
#   IMAGE   — conditioning image PNG (RGBA preferred; RGB+rembg also OK)
#   DINOV3  — local timm DINOv3 ViT-L/16 safetensors
#   STAGE1  — Stage 1 DiT safetensors (only used by run_stage1_ref.sh)
#   OUTDIR  — output directory (default: /tmp/t2_cuda_ref)
#
# Prereqs:
#   1. uv venv created per cuda/trellis2/pyproject.toml:
#        cd cuda/trellis2 && uv venv --python 3.12 .venv
#        uv pip install -e ".[torch-cu124,ref-pipeline]"
#   2. o-voxel + FlexGEMM built with BUILD_TARGET=cuda (see pyproject.toml).
#   3. NVIDIA driver supporting CUDA 12.4+.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: $VENV_PY not found. Create the venv first:" >&2
    echo "  cd $SCRIPT_DIR && uv venv --python 3.12 .venv && uv pip install -e \".[torch-cu124,ref-pipeline]\"" >&2
    exit 1
fi

MESH="${MESH:-/mnt/disk1/tmp/tex_knight_r512_fresh/preprocess_mesh.obj}"
IMAGE="${IMAGE:-/mnt/disk1/tmp/tex_knight_r512_fresh/preprocess_image.png}"
DINOV3="${DINOV3:-/mnt/disk1/models/dinov3-vitl16/model.safetensors}"
OUTDIR="${OUTDIR:-/tmp/t2_cuda_ref}"
RESOLUTION="${RESOLUTION:-512}"
TEXSIZE="${TEXSIZE:-1024}"
SEED="${SEED:-42}"

mkdir -p "$OUTDIR"

# preprocess_image.png is RGB without alpha. The pipeline's preprocess_image()
# falls back to BiRefNet rembg when alpha is uniform. To avoid the gated HF
# fetch, synthesize an alpha from the (already bg-removed) RGB image so the
# bbox-crop branch fires instead.
if file -b "$IMAGE" | grep -qi 'PNG.* 3 channels\|, RGB,\|RGB '; then
    RGBA="$OUTDIR/$(basename "$IMAGE" .png)_rgba.png"
    "$VENV_PY" - <<PY
from PIL import Image
import numpy as np
im = np.array(Image.open("$IMAGE").convert("RGB"))
alpha = ((im.sum(-1) > 15) * 255).astype(np.uint8)
Image.fromarray(np.dstack([im, alpha]), "RGBA").save("$RGBA")
PY
    IMAGE="$RGBA"
fi

cd "$REPO_ROOT/ref/trellis2"
# gen_stage2_ref.py imports its software-rasterizer / cumesh / flash_attn
# shims from $RDNA4_DIR. Reuse the rdna4/trellis2 ones — they're plain Python
# and run on CUDA fine (sw rast is pure torch, flash_attn shim falls back to
# SDPA, cumesh shim wraps xatlas).
DINOV3_WEIGHTS="$DINOV3" RDNA4_DIR="$REPO_ROOT/rdna4/trellis2" "$VENV_PY" gen_stage2_ref.py \
    --mesh "$MESH" \
    --image "$IMAGE" \
    --output-dir "$OUTDIR" \
    --resolution "$RESOLUTION" \
    --texture-size "$TEXSIZE" \
    --seed "$SEED" \
    --dinov3 "$DINOV3"

# Dump the textures from the GLB so visual diff is one-step.
"$VENV_PY" - <<PY
import trimesh, numpy as np
s = trimesh.load("$OUTDIR/textured.glb", process=False)
geom = list(s.geometry.values())[0]
m = geom.visual.material
m.baseColorTexture.save("$OUTDIR/basecolor.png")
if getattr(m, 'metallicRoughnessTexture', None) is not None:
    m.metallicRoughnessTexture.save("$OUTDIR/metalrough.png")
print(f"verts={len(geom.vertices)} faces={len(geom.faces)}")
a = np.array(m.baseColorTexture)
nz = a[..., 3] > 0
print(f"alpha-nz: {nz.mean()*100:.2f}% mean RGB over nz: {a[nz][:,:3].mean(0)}")
PY

echo "Done. Outputs in $OUTDIR/ (textured.glb, basecolor.png, metalrough.png)"
