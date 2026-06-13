#!/usr/bin/env bash
# Run the PyTorch+ROCm Hunyuan3D-2.1 shape-gen reference end-to-end on RX 9070
# XT, producing timed per-stage JSON + .npy traces that the HIP path consumes
# for verification and perf comparison.
#
# Inputs (env-overridable):
#   IMAGE     — input PNG (default: Hunyuan3D-2.1 demo image)
#   STEPS     — DiT inference steps (default: 30)
#   GUIDANCE  — CFG scale (default: 5.0)
#   OCTREE    — marching cubes grid (default: 256)
#   SEED      — RNG seed (default: 42)
#   DTYPE     — fp16 / fp32 (default: fp16)
#   TRACE_DIR — where .npy traces land (default: ../traces/rocm_ref)
#   TIMING_JSON — per-stage timing output JSON
#                 (default: ../traces/timings_pytorch.json)
#   OUT_GLB   — output mesh path (default: ../traces/rocm_ref.glb)
#
# Prereqs:
#   - .venv created via setup_rocm.sh
#   - Hunyuan3D-2.1 weights at $HY3DGEN_MODELS/$HY3D_MODEL_NAME (default
#     /mnt/disk01/models/Hunyuan3D-2.1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDNA4_HY3D="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RDNA4_HY3D/../.." && pwd)"

VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: $VENV_PY not found. Run: bash $SCRIPT_DIR/setup_rocm.sh" >&2
    exit 1
fi

IMAGE="${IMAGE:-/mnt/disk1/models/Hunyuan3D-2.1-repo/hy3dshape/demos/demo.png}"
export HY3DGEN_MODELS="${HY3DGEN_MODELS:-/mnt/disk1/models}"
export HY3D_REPO="${HY3D_REPO:-/mnt/disk1/models/Hunyuan3D-2.1-repo/hy3dshape}"
STEPS="${STEPS:-30}"
GUIDANCE="${GUIDANCE:-5.0}"
OCTREE="${OCTREE:-256}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-fp16}"
TRACE_DIR="${TRACE_DIR:-$RDNA4_HY3D/traces/rocm_ref}"
TIMING_JSON="${TIMING_JSON:-$RDNA4_HY3D/traces/timings_pytorch.json}"
OUT_GLB="${OUT_GLB:-$RDNA4_HY3D/traces/rocm_ref.glb}"

mkdir -p "$TRACE_DIR" "$(dirname "$TIMING_JSON")" "$(dirname "$OUT_GLB")"

echo "=== Hunyuan3D-2.1 PyTorch+ROCm reference run ==="
echo "  IMAGE       $IMAGE"
echo "  STEPS       $STEPS    GUIDANCE $GUIDANCE    OCTREE $OCTREE    SEED $SEED"
echo "  DTYPE       $DTYPE"
echo "  TRACE_DIR   $TRACE_DIR"
echo "  TIMING_JSON $TIMING_JSON"
echo "  OUT_GLB     $OUT_GLB"
echo

# HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES caller-overridable.
"$VENV_PY" "$REPO_ROOT/ref/hy3d/run_full_pipeline.py" \
    --image "$IMAGE" \
    --out "$OUT_GLB" \
    --device cuda \
    --dtype "$DTYPE" \
    --steps "$STEPS" \
    --guidance "$GUIDANCE" \
    --octree "$OCTREE" \
    --seed "$SEED" \
    --trace-dir "$TRACE_DIR" \
    --per-stage-timing "$TIMING_JSON"

echo
echo "Done. Use the traces with:"
echo "  $RDNA4_HY3D/test_hip_hy3d <cond.st> <model.st> <vae.st> \\"
echo "      --init-trace-dir $TRACE_DIR \\"
echo "      --per-stage-timing $RDNA4_HY3D/traces/timings_hip.json"
