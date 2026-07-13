#!/bin/bash
# Does the indexer-scan fix (DS4F_IDX_SCAN_MIN, 64 -> 8) also help FLASH?
#
# It should: ds4f_index_score is SHARED, and Flash uses the same indexer geometry (64 index heads x
# 128 dim, ds4f_default_config). So Flash's published 18.98 tok/s was almost certainly measured in
# the same serial-scalar band. But "should" is not a measurement -- Flash differs in the expert
# kernel (MXFP4/svtbl, ~84 Gmac/s vs base's int8-sdot), so its expert time is ~2x base's, which means
# the scan is a SMALLER fraction of its step and the win should be proportionally smaller. Measure it.
#
# A/B on a real 70-token prompt, gated on a coherent completion (the SVE scan reassociates vs the
# scalar loop, so top-k selection near ties could shift).
#
#   ./ab_scanmin_flash.sh
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"

export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f}
export LLM_THREADS=${LLM_THREADS:-47}      # NEVER 48
export DS4F_CMP_LOCAL=${DS4F_CMP_LOCAL:-1}

free_nodes() {
    pkill -x ds4f_ep_runner 2>/dev/null || true; pkill -x mpiexec 2>/dev/null || true
    pkill -x plexec 2>/dev/null || true
    for i in $(seq 1 60); do
        [ "$(ps -eo comm --no-headers | grep -cE 'ds4f_ep_runner|mpiexec|plexec' || true)" = "0" ] && return 0
        sleep 1
    done
}
trap free_nodes EXIT

PF="$LLM_DIR/sweep_k_prompt.txt"
OUT="$LLM_DIR/ab_scanmin_flash.txt"; : > "$OUT"

for SM in 64 8; do
    free_nodes
    L="$LLM_DIR/flash_sm_$SM.log"; S="$LLM_DIR/flash_sm_s_$SM.txt"; rm -f "$L" "$S"
    DS4F_IDX_SCAN_MIN=$SM PROMPT_FILE="$PF" MAX_NEW=32 \
      DS4F_GEN_LOG="$L" DS4F_GEN_SENTINEL="$S" \
      ./run_ds4f_gen_11n.sh > /dev/null 2>&1 || true
    {
      echo "### FLASH  DS4F_IDX_SCAN_MIN=$SM  (64 = old/broken, 8 = fixed)"
      grep -hE "^prefill:|^decode:" "$S" 2>/dev/null | sed 's/^/    /'
      grep -hE "^ *tb2scan|^ *tb2prep" "$L" 2>/dev/null | tail -2 | sed 's/^ */    /'
      echo "    completion (the gate):"
      sed -n '/<<<COMPLETION>>>/,/<<<END>>>/p' "$S" 2>/dev/null | sed '1d;$d' | head -4 | sed 's/^/    | /'
    } >> "$OUT"
done
echo "=== FLASH scan A/B ==="; cat "$OUT"
