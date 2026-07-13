#!/bin/bash
# HEADLINE BENCHMARK — every number in ds4f.md's CURRENT STATE table, measured in ONE allocation
# with the documented optimal flags, each one output-gated.
#
# WHY THIS EXISTS: the table's three numbers (single-stream decode, prefill, batched) were each
# measured in a DIFFERENT job. That violates the rule this repo learned the hard way -- tok/s is not
# comparable across allocations (the IDENTICAL FP8 config read 8.28 in one job and 11.34 in another).
# A table whose rows come from different allocations is not a table, it's three anecdotes.
#
# It also pins the flags. The "optimal" recipe is NOT the script defaults: DS4F_HC_SVE and
# DS4F_MV_FUSE are opt-in (HC_SVE is reassoc-class, so it is deliberately not a default). A run
# without them reads ~15 tok/s decode and looks like a regression when it is just a different config.
#
#   DS4F_STAGE_DIR=/local/ds4fbase_q8 ./bench_headline_12n.sh
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"

# ---- the documented optimum (a64fx/ds4f.md, "CURRENT STATE") ----
export DS4F_DENSE=${DS4F_DENSE:-q8pv}
export DS4F_EXPERTS=${DS4F_EXPERTS:-q8pv}
export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4fbase_q8}
export DS4F_TP_ATTN=${DS4F_TP_ATTN:-0}
export DS4F_CMP_LOCAL=${DS4F_CMP_LOCAL:-1}
export DS4F_HC_SVE=${DS4F_HC_SVE:-1}     # opt-in (reassoc class) -- part of the optimum, not a default
export DS4F_MV_FUSE=${DS4F_MV_FUSE:-1}
export LLM_THREADS=${LLM_THREADS:-47}    # NEVER 48: any other process on the node costs ~40%

OUT=$LLM_DIR/bench_headline.txt; : > "$OUT"
say() { echo "$*" | tee -a "$OUT"; }
# the serve/gate runs never return on their own; a leaked rank set makes the NEXT run die with
# "PLE 0054 plexec: number of processes exceed the limit on virtual coordinate"
cleanup() { pkill -x ds4f_ep_runner 2>/dev/null || true; sleep 2; pkill -x mpiexec 2>/dev/null || true; sleep 1; }
trap cleanup EXIT
cleanup

PF=$LLM_DIR/sweep_k_prompt.txt
[ -f "$PF" ] || cat > "$PF" <<'EOF'
Below is a complete, working implementation of an in-place quicksort in Python, together with a
short explanation of how the partition step works and why the average time complexity is O(n log n).
The implementation uses the Lomuto partition scheme and recurses on both halves.

def quicksort(arr, lo=0, hi=None):
EOF

say "=== HEADLINE, job=${PJM_JOBID:-?} $(date '+%F %T') ==="
say "cfg: dense=$DS4F_DENSE experts=$DS4F_EXPERTS TP_ATTN=$DS4F_TP_ATTN CMP_LOCAL=$DS4F_CMP_LOCAL HC_SVE=$DS4F_HC_SVE MV_FUSE=$DS4F_MV_FUSE threads=$LLM_THREADS"

# ---- 1. CORRECTNESS GATE first. A fast wrong number is worth nothing (see the DB_BENCH lesson). ----
say ""; say "--- [1/4] VERIFY_GATE: forward_verify == forward_token? (must be 16/16) ---"
# The gate's result is printed by logmsg(), which writes to ds4f_ep_rank00.txt -- NOT to stdout.
# Grab it BEFORE the next run, which overwrites that file. (Cost me a whole benchmark pass.)
# rc=1 from the gen wrapper is EXPECTED here: the gate exits before producing gen ids.
rm -f ds4f_ep_rank00.txt
DS4F_GEN_SENTINEL=/tmp/hl_vg_s.txt DS4F_GEN_LOG=/tmp/hl_vg.txt \
  PROMPT_FILE="$PF" MAX_NEW=4 DS4F_VERIFY_GATE=1 DS4F_VG_NTOK=16 \
  ./run_ds4fbase_gen_12n.sh > /dev/null 2>&1 || true
grep -hE "match, common prefix|token :|verify:" ds4f_ep_rank00.txt 2>/dev/null | tail -3 | sed 's/^/  /' | tee -a "$OUT"
cp -f ds4f_ep_rank00.txt "$LLM_DIR/bench_headline_gate_rank00.txt" 2>/dev/null || true
cleanup

# ---- 2+3. decode + prefill, one gen run (both reported per-phase), gated on the completion ----
say ""; say "--- [2/4] single-stream DECODE + [3/4] PREFILL (PREFILL_GEMM default K=32) ---"
rm -f /tmp/hl_gen_s.txt
DS4F_GEN_SENTINEL=/tmp/hl_gen_s.txt DS4F_GEN_LOG=/tmp/hl_gen.txt \
  PROMPT_FILE="$PF" MAX_NEW=${MAX_NEW:-64} \
  ./run_ds4fbase_gen_12n.sh > /dev/null 2>&1 || true
grep -hE "^prefill:|^decode:|^lockstep" /tmp/hl_gen_s.txt 2>/dev/null | sed 's/^/  /' | tee -a "$OUT"
say "  completion (the gate -- must be coherent):"
sed -n '/<<<COMPLETION>>>/,/<<<END>>>/p' /tmp/hl_gen_s.txt 2>/dev/null | sed '1d;$d' | head -8 | sed 's/^/  | /' | tee -a "$OUT"
cleanup

# ---- 4. batched decode. DB_BENCH only TIMES steps, so the VERIFY_GATE above is what makes it mean
#         anything -- it is the only evidence the batched forward is correct. ----
say ""; say "--- [4/4] batched decode (DB_BENCH sweep to M=16; correctness rests on gate [1]) ---"
DS4F_DB_BENCH=1 DS4F_DB_MAXM=16 DS4F_DB_NTOK=32 DS4F_PREFILL=32 DS4F_MAXGEN=4 \
  ./run_ds4fbase_12n.sh 2>&1 | grep -hiE "^ *M=|DB_BENCH|agg tok/s" | tail -8 | sed 's/^/  /' | tee -a "$OUT"
cleanup

say ""; say "=== all four measured in job ${PJM_JOBID:-?} -- comparable to each other ==="
cat "$OUT"
