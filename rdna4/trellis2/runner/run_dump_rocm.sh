#!/usr/bin/env bash
# Dump ROCm-path intermediates for diffing against cuda/trellis2/verify-dumps/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDNA4_T2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RDNA4_T2_DIR/../.." && pwd)"
VENV_PY="$RDNA4_T2_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: $VENV_PY missing. Set up .venv at $RDNA4_T2_DIR/.venv per rdna4/trellis2/pyproject.toml first." >&2
    exit 1
fi

IMAGE="${IMAGE:-$REPO_ROOT/cpu/trellis2/trellis2_repo/assets/example_image/T.png}"
DINOV3="${DINOV3:-/mnt/disk1/models/dinov3-vitl16/model.safetensors}"
MODEL_ID="${MODEL_ID:-microsoft/TRELLIS.2-4B}"
OUTDIR="${OUTDIR:-$RDNA4_T2_DIR/verify-dumps-rocm}"
SEED="${SEED:-42}"
DUMP_PER_STEP_FLAG=""
if [[ "${DUMP_PER_STEP:-0}" == "1" ]]; then
    DUMP_PER_STEP_FLAG="--dump-per-step"
fi

mkdir -p "$OUTDIR"

# Match the CUDA dumper env: ATTN_BACKEND=sdpa, OPENCV EXR.
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export OPENCV_IO_ENABLE_OPENEXR=1

# Activate vendored hipBLASLt subset if present (run vendor/setup_vendor.sh
# to populate). Falls through to the system /opt/rocm path if unset.
VENDOR_HBL="$RDNA4_T2_DIR/vendor/hipblaslt"
if [[ -d "$VENDOR_HBL/library" ]]; then
    export LD_LIBRARY_PATH="$VENDOR_HBL:${LD_LIBRARY_PATH:-}"
    export HIPBLASLT_TENSILE_LIBPATH="$VENDOR_HBL/library"
    echo "[run] using vendored hipBLASLt: $VENDOR_HBL"
else
    echo "[run] vendor/hipblaslt missing; run vendor/setup_vendor.sh to bundle (using system /opt/rocm)" >&2
fi

DINOV3_WEIGHTS="$DINOV3" RDNA4_DIR="$RDNA4_T2_DIR" REPO_DIR="$REPO_ROOT/cpu/trellis2/trellis2_repo" \
    "$VENV_PY" "$SCRIPT_DIR/dump_rocm.py" \
        --image "$IMAGE" \
        --model-id "$MODEL_ID" \
        --dinov3 "$DINOV3" \
        --output-dir "$OUTDIR" \
        --pipeline-type 512 \
        --seed "$SEED" \
        $DUMP_PER_STEP_FLAG

echo "ROCm dumps -> $OUTDIR"
