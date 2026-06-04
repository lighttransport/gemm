#!/bin/bash
# A/B the single-node ds4f_runner on ONE clean (non-origin) node via mpiexec -np 1,
# so the 48-thread measurement is not contaminated by the co-located claude session
# on relative coord (0,0,0). Compares dense-matvec kernels back-to-back:
#   FP8 gather (default) | BF16 predequant (plain) | BF16 predequant (pv layout).
# bf16_pv MUST be byte-identical to plain bf16 (same argmax + ||x||), only faster.
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"
OUT="$LLM_DIR/bf16pv_clean.log"; : > "$OUT"
BIN="$LLM_DIR/build/ds4f_runner"

VC="$LLM_DIR/vc_bf16pv.txt"; echo "(1,1,1)" > "$VC"
COMMON="OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores DS4F_PREFILL=8 DS4F_MAXGEN=16 DS4F_PROF=1"

run() { # label env...
    local lbl="$1"; shift
    echo "===== $lbl =====" >> "$OUT"
    mpiexec -np 1 -vcoordfile "$VC" \
        sh -c "$COMMON $* '$BIN' >> '$OUT' 2>&1" || echo "(run failed)" >> "$OUT"
}
run "A FP8 gather (default)"            ""
run "B BF16 predequant (plain)"        "DS4F_FP8_BF16=1"
run "C BF16 predequant (pv layout)"    "DS4F_FP8_BF16=1 DS4F_BF16_PV=1"
echo "=== DONE ===" >> "$OUT"
cat "$OUT"
