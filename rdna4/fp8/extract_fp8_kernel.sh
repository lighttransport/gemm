#!/usr/bin/env bash
# Extract the FP8 GEMM kernel ELF used by bench_fp8_extracted from the local
# ROCm install. The source library is a CCOB+zstd container holding a clang
# offload bundle that wraps the AMDGPU ELF.
#
# Layout:
#   bytes 0..31   : CCOB header ("CCOB" magic + 28 bytes)
#   bytes 32..    : zstd stream
#   after decomp  : __CLANG_OFFLOAD_BUNDLE__ header (4096 bytes)
#   byte 4096     : AMDGPU ELF (the kernel code object)
#
# We strip CCOB + offload-bundle wrappers and write the bare ELF, which is what
# hipModuleLoad in bench_fp8_extracted.cpp expects.
#
# The extracted ELF is ~38 MB (the entire FP8 SAB_SCD Tensile catalog for
# gfx1201; we only call one kernel from it). The compressed source on disk is
# ~1 MB. We do NOT vendor either into git — re-run this script to regenerate.
set -euo pipefail

ROCM_LIB=${ROCM_LIB:-/opt/rocm/lib/hipblaslt/library}
SRC="$ROCM_LIB/TensileLibrary_F8F8_SF8_HA_Bias_SAB_SCD_SAV_UA_Type_F8S_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1201.co"
OUT=${1:-fp8_kernel_gfx1201.co}

if [[ ! -f "$SRC" ]]; then
  echo "error: source not found: $SRC" >&2
  echo "       set ROCM_LIB=/path/to/hipblaslt/library or install ROCm 7.x." >&2
  exit 1
fi

if ! command -v zstd >/dev/null 2>&1; then
  echo "error: zstd CLI not found (apt install zstd)" >&2
  exit 1
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
dd if="$SRC" bs=1 skip=32 status=none | zstd -d -q -o "$tmp" -f
dd if="$tmp" of="$OUT" bs=4096 skip=1 status=none

# Sanity-check: must start with ELF magic and contain our kernel name.
magic=$(head -c 4 "$OUT" | xxd -p)
if [[ "$magic" != "7f454c46" ]]; then
  echo "error: extracted file is not an ELF (magic=$magic)" >&2
  exit 1
fi
if ! strings "$OUT" | grep -q 'Cijk_Alik_Bljk_F8SS_BH_Bias_SHB_HA_S_SAB_SCD_SAV'; then
  echo "warning: expected FP8 kernel symbol not found in $OUT" >&2
fi

echo "wrote $OUT ($(stat -c%s "$OUT") bytes)"
