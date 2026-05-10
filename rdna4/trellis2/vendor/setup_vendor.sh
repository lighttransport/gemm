#!/usr/bin/env bash
# Populate vendor/hipblaslt/ from /opt/rocm-7.2.2 so the runner can launch
# without the system hipBLASLt path on LD_LIBRARY_PATH.
#
# Modes:
#   --prune (default): keep gfx1201 BB/HH/SS/DD Tensile libs + the master
#       hsaco + extop. Drops FP8 / INT8 / non-gfx1201 architectures.
#       Footprint ~60 MB. Sufficient for TRELLIS-2 stage-2 (BF16 DiT/ViT,
#       F16 in some autocast spots; F32 SparseLinear is handled by the
#       in-tree linear_f32 HIP kernel).
#   --full: every gfx1201 file (~225 MB) for safety.
#   --check: smoke-test that the populated subset loads (runs
#       debug/repro_hipblaslt_tall_skinny.py — exits 0 on success).
#
# Source override: --from-rocm DIR (default /opt/rocm-7.2.2).

set -euo pipefail

MODE=prune
ROCM_ROOT=/opt/rocm-7.2.2
DO_CHECK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prune) MODE=prune; shift ;;
        --full)  MODE=full; shift ;;
        --check) DO_CHECK=1; shift ;;
        --from-rocm) ROCM_ROOT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

VENDOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$VENDOR_DIR/hipblaslt"
SRC_LIB="$ROCM_ROOT/lib"
SRC_TENSILE="$SRC_LIB/hipblaslt/library"

if [[ ! -d "$SRC_TENSILE" ]]; then
    echo "ERROR: $SRC_TENSILE missing. Set --from-rocm DIR." >&2
    exit 1
fi

echo "[vendor] mode=$MODE  src=$ROCM_ROOT  dest=$DEST"
mkdir -p "$DEST/library"

# libhipblaslt.so.1 + symlink chain.
cp -af "$SRC_LIB/libhipblaslt.so.1.2.70202" "$DEST/"
ln -sf libhipblaslt.so.1.2.70202 "$DEST/libhipblaslt.so.1"
ln -sf libhipblaslt.so.1         "$DEST/libhipblaslt.so"

# Master kernel blob + extop tables (always needed).
cp -af "$SRC_TENSILE/Kernels.so-000-gfx1201.hsaco" "$DEST/library/"
cp -af "$SRC_TENSILE/extop_gfx1201.co"             "$DEST/library/"
cp -af "$SRC_TENSILE/hipblasltExtOpLibrary.dat"    "$DEST/library/"
cp -af "$SRC_TENSILE/hipblasltTransform.hsaco"     "$DEST/library/"
cp -af "$SRC_TENSILE/TensileLiteLibrary_lazy_Mapping.dat" "$DEST/library/"

KEEP_PREFIXES=(BB HH SS DD)
if [[ "$MODE" == "full" ]]; then
    KEEP_PREFIXES=(BB HH SS DD B8B8 B8F8 F8B8 F8F8 I8I8 lazy)
else
    KEEP_PREFIXES+=(lazy)
fi

n=0
for prefix in "${KEEP_PREFIXES[@]}"; do
    while IFS= read -r f; do
        cp -af "$f" "$DEST/library/"
        n=$((n + 1))
    done < <(ls "$SRC_TENSILE"/TensileLibrary_${prefix}_*gfx1201* 2>/dev/null || true)
done

echo "[vendor] copied $n Tensile files (+ libhipblaslt + master hsaco + extop)"
du -sh "$DEST"

# Manifest.
{
    echo "# vendor/hipblaslt/ manifest — generated $(date -Iseconds)"
    echo "# source: $ROCM_ROOT  mode: $MODE"
    (cd "$DEST" && find . -type f -o -type l | sort)
} > "$VENDOR_DIR/manifest.txt"
echo "[vendor] wrote $VENDOR_DIR/manifest.txt"

if [[ "$DO_CHECK" -eq 1 ]]; then
    echo "[vendor] smoke test: repro_hipblaslt_tall_skinny.py with vendored libs"
    SCRIPT_DIR="$(cd "$VENDOR_DIR/.." && pwd)"
    LD_LIBRARY_PATH="$DEST:${LD_LIBRARY_PATH:-}" \
    HIPBLASLT_TENSILE_LIBPATH="$DEST/library" \
        "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/debug/repro_hipblaslt_tall_skinny.py"
fi
