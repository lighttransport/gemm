#!/bin/bash
# PREFILL_K sweep — now that DS4F_PREFILL_GEMM is CORRECT (forward_verify x TP_WOB fix, f9daca59).
#
# The mechanism: batching the prefill through the verify path fires the per-layer EP all-reduce once
# per K tokens instead of once per token, so comm ~ /K. Since comm ~ 3 x expert_time dominates here,
# larger K should keep paying -- but K was NEVER swept on a path that produced correct output, so
# the shape of the curve is unknown. The verify path caps K at 128.
#
# EVERY point is output-gated: a K that is fast but incoherent is worth nothing. That is the whole
# lesson of the "35 tok/s of garbage" episode -- see a64fx/ds4f.md.
#
#   DS4F_DENSE=q8pv DS4F_EXPERTS=q8pv DS4F_STAGE_DIR=/local/ds4fbase_q8 DS4F_TP_ATTN=0 \
#     ./sweep_prefill_k.sh
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"

OUT=${OUT:-sweep_prefill_k.txt}; : > "$OUT"
PF="$LLM_DIR/sweep_k_prompt.txt"
cat > "$PF" <<'EOF'
Below is a complete, working implementation of an in-place quicksort in Python, together with a
short explanation of how the partition step works and why the average time complexity is O(n log n).
The implementation uses the Lomuto partition scheme and recurses on both halves.

def quicksort(arr, lo=0, hi=None):
EOF

for K in 1 8 16 32 64 128; do
    [ "$K" = 1 ] && G=0 || G=1        # K=1 -> token-at-a-time CONTROL (PREFILL_GEMM off)
    # per-K sentinel/log: a STALE sentinel has produced wrong conclusions here before.
    S="$LLM_DIR/sweep_k_sent_$K.txt"; L="$LLM_DIR/sweep_k_$K.log"; rm -f "$S" "$L"
    PROMPT_FILE="$PF" MAX_NEW=${MAX_NEW:-24} \
      DS4F_GEN_SENTINEL="$S" DS4F_GEN_LOG="$L" \
      DS4F_PREFILL_GEMM=$G DS4F_PREFILL_K=$K \
      ./run_ds4fbase_gen_12n.sh > /dev/null 2>&1 || true

    pf=$(grep -hE '^prefill:' "$S" 2>/dev/null | head -1)
    cp=$(sed -n '/<<<COMPLETION>>>/,/<<<END>>>/p' "$S" 2>/dev/null | sed '1d;$d' | tr '\n' ' ' | cut -c1-90)
    printf '%-6s gemm=%s  %s\n' "K=$K" "$G" "${pf:-NO PREFILL LINE (run failed)}" | tee -a "$OUT"
    printf '        gen: %s\n' "${cp:-<none>}"                                      | tee -a "$OUT"
done
echo; echo "=== summary (gen text must stay coherent, else the tok/s is worthless) ==="; cat "$OUT"
