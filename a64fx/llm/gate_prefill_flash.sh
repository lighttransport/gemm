#!/bin/bash
# Enable DS4F_PREFILL_GEMM for FLASH — but GATE it first.
#
# Flash runs with TP entirely OFF by default (run_ds4f_11n.sh: TP_ATTN/SHARED/HEAD/EMBED all 0, no
# TP_OPROJ/TP_WOB), so the forward_verify x TP_WOB column-shard bug (f9daca59) never bit it -- the
# fix is a no-op in this config. That is a REASON TO EXPECT a pass, not a substitute for measuring
# one. PREFILL_GEMM drives prefill through ds4f_forward_verify, and turning that on ungated is
# exactly the mistake that produced months of fast garbage on base.
#
# Three steps, each blocking the next:
#   [1] VERIFY_GATE            forward_verify == forward_token?  must be 16/16
#   [2] A/B PREFILL_GEMM 0/1   speed + a coherent completion (GEMM reassociates: coherent, not
#                              bit-identical -- so the completion may legitimately differ from the
#                              control. It must still be COHERENT, and the decode it seeds must be.)
#   [3] report                 flip the default in run_ds4f_11n.sh only if [1] and [2] are green
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"

export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f}
export LLM_THREADS=${LLM_THREADS:-47}
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
OUT="$LLM_DIR/gate_prefill_flash.txt"; : > "$OUT"
say() { echo "$*" | tee -a "$OUT"; }

# ---------- [1] VERIFY_GATE ----------
say "--- [1] FLASH VERIFY_GATE (must be 16/16) ---"
free_nodes
VOUT="$LLM_DIR/ds4f_verify_gate_flash.txt"; rm -f "$VOUT"
DS4F_VERIFY_GATE=1 DS4F_VG_NTOK=16 DS4F_VERIFY_GATE_OUT="$VOUT" \
  PROMPT_FILE="$PF" MAX_NEW=4 \
  DS4F_GEN_LOG="$LLM_DIR/flash_vg.log" DS4F_GEN_SENTINEL="$LLM_DIR/flash_vg_s.txt" \
  ./run_ds4f_gen_11n.sh > /dev/null 2>&1 || true    # rc=1 EXPECTED (gate exits before gen ids)
if [ ! -s "$VOUT" ]; then
    say "  NO VERDICT -- the run died. Tail:"; tail -5 "$LLM_DIR/flash_vg.log" | sed 's/^/  /' | tee -a "$OUT"
    exit 1
fi
sed 's/^/  /' "$VOUT" | tee -a "$OUT"
grep -q -- "-> PASS" "$VOUT" || { say "  GATE FAILED -> NOT enabling PREFILL_GEMM on Flash."; exit 3; }

# ---------- [2] A/B ----------
for G in 0 1; do
    free_nodes
    L="$LLM_DIR/flash_pg_$G.log"; S="$LLM_DIR/flash_pg_s_$G.txt"; rm -f "$L" "$S"
    DS4F_PREFILL_GEMM=$G DS4F_PREFILL_K=32 PROMPT_FILE="$PF" MAX_NEW=32 \
      DS4F_GEN_LOG="$L" DS4F_GEN_SENTINEL="$S" ./run_ds4f_gen_11n.sh > /dev/null 2>&1 || true
    say ""
    say "--- [2] FLASH PREFILL_GEMM=$G ---"
    grep -hE "^prefill:|^decode:" "$S" 2>/dev/null | sed 's/^/  /' | tee -a "$OUT"
    grep -hE "^lockstep" "$S" 2>/dev/null | sed 's/^/  /' | tee -a "$OUT"
    say "  completion (the gate -- must be coherent):"
    sed -n '/<<<COMPLETION>>>/,/<<<END>>>/p' "$S" 2>/dev/null | sed '1d;$d' | head -6 | sed 's/^/  | /' | tee -a "$OUT"
done
say ""
say "=== enable PREFILL_GEMM on Flash only if [1] PASSed and both [2] completions are coherent ==="
