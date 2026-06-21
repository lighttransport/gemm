#!/bin/bash
# int8 index_score SPEED A/B: ctx-warm 10240, DS4F_IDX_INT8=0 (f32 ref) vs 1 (int8/svdot).
# Deterministic ctx-warm fill (lockstep-preserved) => pure speed test of the indexer scan.
# Full per-leg output -> /tmp logs; compact verdict to stdout.
set -u
cd /vol0006/mdt0/data/hp250467/work/gemm/ds4f/a64fx/llm
CTX=${CTX:-10240}
VERD=/tmp/ab_idx_speed_verdict.txt
: > "$VERD"

run_leg () {  # $1=tag $2=int8val $3=sentinel
  local tag=$1 i8=$2 sent=$3
  : > "$sent"
  echo "=== SPEED leg $tag: DS4F_IDX_INT8=$i8 ctx=$CTX ===" | tee -a "$VERD"
  DS4F_IDX_INT8=$i8 DS4F_CTX_WARM=$CTX DS4F_FP8_BF16=1 DS4F_PROF=1 \
    DS4F_LCTX_SENTINEL=$sent DS4F_QUIET=1 \
    ./run_ds4f_longctx_11n.sh > /tmp/ab_idx_speed_$tag.log 2>&1
  local rc=$?
  echo "leg $tag rc=$rc" | tee -a "$VERD"
}

run_leg A 0 /tmp/lc_idx_a.txt
run_leg B 1 /tmp/lc_idx_b.txt

{
  echo "============ int8 index_score SPEED A/B VERDICT ============"
  for L in A:0:/tmp/lc_idx_a.txt B:1:/tmp/lc_idx_b.txt; do
    tag=${L%%:*}; rest=${L#*:}; i8=${rest%%:*}; sent=${rest#*:}
    echo "--- leg $tag (DS4F_IDX_INT8=$i8) ---"
    grep -E 'LCTX_RC|argmax_distinct|nan=' "$sent" 2>/dev/null
    grep -iE 'decode:|prefill:|tb2prep|tb2scan|index|attn|warm' "$sent" 2>/dev/null | head -16
  done
} | tee -a "$VERD"
echo "SPEED_AB_DONE"
