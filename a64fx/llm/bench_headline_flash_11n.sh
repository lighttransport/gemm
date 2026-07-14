#!/bin/bash
# HEADLINE BENCHMARK — FLASH (ds4f, 11n). Sibling of bench_headline_12n.sh (base).
#
# Same contract: every number in ds4f.md's CURRENT STATE Flash row, measured in ONE allocation, with
# the gate run FIRST and the whole benchmark aborted if it fails. Flash's batched/arena figures were
# stale (never re-measured in a single job, never re-checked after the indexer-scan fix), so they
# were marked provisional. This is what removes that qualifier.
#
#   ./bench_headline_flash_11n.sh
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"

export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f}
export DS4F_CMP_LOCAL=${DS4F_CMP_LOCAL:-1}
export LLM_THREADS=${LLM_THREADS:-47}      # NEVER 48

# ---- THE REAL MODEL. run_ds4f_11n.sh defaults these to 0 (stand-in math) -- base's script defaults
# them to 1. Two consequences, both bad, and both bit me:
#   1. The batched path SEGFAULTS with EXACT=0: the RoPE tables are only built when exact is on
#      (ds4f_build_freqs: `if (!m->exact) return;`), but ds4f_forward_verify ropes unconditionally.
#      forward_token branches around it, so single-stream "works" and hides this.
#   2. Even without the crash, a benchmark of stand-in math measures NOTHING about the real model.
# The gen wrapper sets these; a bare run of run_ds4f_11n.sh does not. Always set them explicitly.
export DS4F_EXACT=${DS4F_EXACT:-1}
export DS4F_TIERB2=${DS4F_TIERB2:-1}
export DS4F_MHC=${DS4F_MHC:-1}

OUT="$LLM_DIR/bench_headline_flash.txt"; : > "$OUT"
say() { echo "$*" | tee -a "$OUT"; }

free_nodes() {
    pkill -x ds4f_ep_runner 2>/dev/null || true; pkill -x mpiexec 2>/dev/null || true
    pkill -x plexec 2>/dev/null || true
    for i in $(seq 1 60); do
        [ "$(ps -eo comm --no-headers | grep -cE 'ds4f_ep_runner|mpiexec|plexec' || true)" = "0" ] && return 0
        sleep 1
    done
}
trap free_nodes EXIT
free_nodes

PF="$LLM_DIR/sweep_k_prompt.txt"

say "=== FLASH HEADLINE, job=${PJM_JOBID:-?} $(date '+%F %T') ==="
say "cfg: 11n MXFP4 experts, TP off, CMP_LOCAL=$DS4F_CMP_LOCAL threads=$LLM_THREADS"
say "     IDX_SCAN_MIN=default(8)  PREFILL_GEMM=default(1,K=32)"

# ---- [1] CORRECTNESS GATE FIRST. A fast wrong number is worth nothing. ----
say ""; say "--- [1/4] VERIFY_GATE (must be 16/16) ---"
VOUT="$LLM_DIR/ds4f_verify_gate_flash.txt"
# RETRY on a LAUNCH failure, not on a gate failure -- they look identical from here and must not.
# If the previous run's ranks have not fully released the Tofu coordinates, mpiexec returns in ~3s
# having printed NOTHING (not even "PLE 0054"), the run never loads, and no verdict is written. That
# is a launcher hiccup, and reporting it as "GATE FAILED" is a false alarm that aborts the benchmark.
# A missing verdict => retry; a verdict that says FAIL => stop, that is real.
gate_ok=0
for attempt in 1 2 3; do
    rm -f "$VOUT"
    free_nodes; sleep 10                                  # let the fabric actually settle
    DS4F_VERIFY_GATE=1 DS4F_VG_NTOK=16 DS4F_VERIFY_GATE_OUT="$VOUT" \
      PROMPT_FILE="$PF" MAX_NEW=4 \
      DS4F_GEN_LOG="$LLM_DIR/fh_vg.log" DS4F_GEN_SENTINEL="$LLM_DIR/fh_vg_s.txt" \
      ./run_ds4f_gen_11n.sh > /dev/null 2>&1 || true      # rc=1 EXPECTED: gate exits before gen ids
    if [ -s "$VOUT" ]; then gate_ok=1; break; fi
    say "  (attempt $attempt: no verdict -- run never started; retrying)"
done
if [ "$gate_ok" != 1 ]; then
    say "  !! VERIFY_GATE never produced a verdict after 3 attempts -- the run is not starting."
    tail -5 "$LLM_DIR/fh_vg.log" | sed 's/^/  /' | tee -a "$OUT"
    exit 1
fi
if ! grep -q -- "-> PASS" "$VOUT"; then
    say "  !! VERIFY_GATE FAILED -- every number below would be suspect. Stopping."
    sed 's/^/  /' "$VOUT" | tee -a "$OUT"
    exit 3
fi
grep -hE "match, common prefix" "$VOUT" | sed 's/^/  /' | tee -a "$OUT"
free_nodes

# ---- [2]+[3] decode + prefill on a REAL prompt, gated on the completion ----
say ""; say "--- [2/4] DECODE + [3/4] PREFILL (real 70-tok prompt) ---"
S="$LLM_DIR/fh_gen_s.txt"; rm -f "$S"
PROMPT_FILE="$PF" MAX_NEW=${MAX_NEW:-64} \
  DS4F_GEN_LOG="$LLM_DIR/fh_gen.log" DS4F_GEN_SENTINEL="$S" \
  ./run_ds4f_gen_11n.sh > /dev/null 2>&1 || true
grep -hE "^prefill:|^decode:|^lockstep" "$S" 2>/dev/null | sed 's/^/  /' | tee -a "$OUT"
say "  completion (the gate -- must be coherent):"
sed -n '/<<<COMPLETION>>>/,/<<<END>>>/p' "$S" 2>/dev/null | sed '1d;$d' | head -8 | sed 's/^/  | /' | tee -a "$OUT"
free_nodes

# ---- [4] batched decode. DB_BENCH only TIMES steps -- its correctness rests on gate [1]. ----
say ""; say "--- [4/4] batched decode (DB_BENCH sweep to M=32) ---"
# logmsg() writes to ds4f_ep_rank00.txt, NOT stdout -- grepping the pipe finds nothing. And keep the
# full log: piping a run straight into grep leaves nothing to diagnose from when it comes back empty.
rm -f ds4f_ep_rank00.txt
# DS4F_MAXPOS matters A LOT here: the per-sequence decode-batch cache sets are sized by max_pos, so
# at M=32 x 43 layers the default 4096 reserves ~10 GB the bench never touches -- and that is what
# made M=32 die (an unchecked aligned_alloc -> NULL -> SIGSEGV; now a clean abort). This bench
# decodes 32 steps from pos 0, so size the context to that.
DS4F_DB_BENCH=1 DS4F_DB_MAXM=32 DS4F_DB_NTOK=32 DS4F_PREFILL=32 DS4F_MAXGEN=4 \
  DS4F_MAXPOS=${DS4F_MAXPOS:-128} \
  ./run_ds4f_11n.sh > "$LLM_DIR/fh_db.log" 2>&1 || true
if grep -qE "^ *M=" ds4f_ep_rank00.txt 2>/dev/null; then
    grep -hE "^ *M=" ds4f_ep_rank00.txt | sed 's/^/  /' | tee -a "$OUT"
    cp -f ds4f_ep_rank00.txt "$LLM_DIR/bench_headline_flash_db_rank00.txt"
else
    say "  !! no M= lines -- DB_BENCH did not run. Tail of fh_db.log:"
    tail -4 "$LLM_DIR/fh_db.log" | sed 's/^/  /' | tee -a "$OUT"
fi
free_nodes

say ""; say "=== all four measured in job ${PJM_JOBID:-?} -- comparable to each other ==="
