#!/bin/bash
# THE regression gate: is ds4f_forward_verify (the batched/serve forward) equivalent to the
# known-good ds4f_forward_token? Must be 16/16. Anything touching forward_verify, the TP shards,
# or the serve loops has to keep it green.
#
# Runs at the FULL 43 layers and the full production config by default -- there is no OOM (I once
# claimed there was; the gate had actually been PASSING and its result was being destroyed by the
# next run, which truncates ds4f_ep_rank00.txt). Two things make the verdict survive now:
#   * the runner writes it to $DS4F_VERIFY_GATE_OUT (default ds4f_verify_gate.txt), and
#   * the runner's exit code IS the verdict (0 = PASS, 3 = FAIL).
# NOTE the gen wrapper still reports rc=1 + "no gen_ids produced" -- EXPECTED, the gate exits
# before generating. Read the verdict file, not the wrapper's rc.
#
#   ./gate_verify_12n.sh            # 43 layers, full stack
#   DS4F_LAYERS=8 ./gate_verify_12n.sh
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"

export DS4F_DENSE=${DS4F_DENSE:-q8pv}
export DS4F_EXPERTS=${DS4F_EXPERTS:-q8pv}
export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4fbase_q8}
export DS4F_TP_ATTN=${DS4F_TP_ATTN:-0}
export DS4F_CMP_LOCAL=${DS4F_CMP_LOCAL:-1} DS4F_HC_SVE=${DS4F_HC_SVE:-1} DS4F_MV_FUSE=${DS4F_MV_FUSE:-1}
export LLM_THREADS=${LLM_THREADS:-47}
export DS4F_VERIFY_GATE=1 DS4F_VG_NTOK=${DS4F_VG_NTOK:-16}

VOUT="$LLM_DIR/ds4f_verify_gate.txt"
export DS4F_VERIFY_GATE_OUT="$VOUT"
rm -f "$VOUT"

# a leaked rank set makes the next run die with "PLE 0054 plexec: number of processes exceed the
# limit on virtual coordinate" -- and pkill does NOT free the coordinate instantly.
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

PF=${PROMPT_FILE:-$LLM_DIR/sweep_k_prompt.txt}
[ -f "$PF" ] || { echo "def quicksort(arr, lo=0, hi=None):" > "$PF"; }

PROMPT_FILE="$PF" MAX_NEW=4 \
  DS4F_GEN_LOG="$LLM_DIR/gate_verify.log" DS4F_GEN_SENTINEL="$LLM_DIR/gate_verify_sent.txt" \
  ./run_ds4fbase_gen_12n.sh > /dev/null 2>&1 || true    # rc=1 is EXPECTED (no gen ids)

if [ ! -s "$VOUT" ]; then
    echo "VERIFY_GATE: NO VERDICT -- the run really did die. Last log lines:"
    tail -6 "$LLM_DIR/gate_verify.log" 2>/dev/null
    exit 1
fi
echo "=== VERIFY_GATE (layers=${DS4F_LAYERS:-43}) ==="
cat "$VOUT"
grep -q -- "-> PASS" "$VOUT" && { echo "GATE: PASS"; exit 0; } || { echo "GATE: FAIL"; exit 3; }
