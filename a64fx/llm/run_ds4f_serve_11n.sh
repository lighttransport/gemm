#!/bin/bash
# DS4F HTTP serving (llama-server-like) on the 12-node interactive alloc = 11 EP nodes + controller.
#
# Architecture: the `ds4f_ep_runner` loads the model ONCE on 11 EP nodes and enters a persistent
# DS4F_SERVE loop; the python HTTP frontend (ds4f_serve.py) runs on the controller and drives it via
# shared-FS request/response files (all ranks read the same request -> lockstep, no broadcast).
# Prefill uses the batched-verify path (+65-74%); decode is the optimized bit-identical path.
#
# Prereq: weights staged (run_ds4f_stage_11n.sh). Usage (inside the live alloc, from a64fx/llm):
#     PORT=8080 ./run_ds4f_serve_11n.sh
# Then, from anywhere that can reach this node:
#     curl -s localhost:8080/v1/completions -d '{"prompt":"def quicksort(a):","max_tokens":128}'
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"

PORT=${PORT:-8080}
# BASE must be on the SHARED filesystem (all EP ranks + the controller read/write it) -- /tmp is
# node-local so rank 0 (an EP node) would never see the controller's request. Default to a home path.
BASE=${DS4F_SERVE_BASE:-$HOME/.ds4f_serve.${PJM_SUBJOBID:-local}}
LOG=${DS4F_SERVE_LOG:-/tmp/ds4f_serve_run.log}
rm -f "$BASE".req "$BASE".resp "$BASE".reqseq "$BASE".respseq   # start from a clean request seq (0)

# ---- serve env forwarded to every rank (mpiexec forwards exported env) ----
export DS4F_SERVE=1
export DS4F_SERVE_REQ="$BASE.req"       DS4F_SERVE_RESP="$BASE.resp"
export DS4F_SERVE_REQSEQ="$BASE.reqseq" DS4F_SERVE_RESPSEQ="$BASE.respseq"
# real-weight decode bundle (== --preset decode) + batched-verify prefill + serving ctx ceiling
export DS4F_REAL=1 DS4F_FP8_BF16=1 DS4F_Q8_DENSE=1 DS4F_TIERB2=1 DS4F_MHC=1 DS4F_HC_PAR=1 DS4F_HC_RMSPAR=1
export DS4F_PREFILL_GEMM=${DS4F_PREFILL_GEMM:-1}
export DS4F_MAXPOS=${DS4F_MAXPOS:-16384}    # KV/compressed cache ceiling: prompt+gen must fit (~16k safe)
export DS4F_NUMA=${DS4F_NUMA:-1}

# ---- launch the persistent 11-node runner in the background (loops until killed) ----
echo "[serve] launching 11-node runner (loads once, ~2-3 min)... log: $LOG"
( ./run_ds4f_11n.sh > "$LOG" 2>&1 ) &
RUNNER_PID=$!
trap 'echo "[serve] stopping"; kill $RUNNER_PID 2>/dev/null; pkill -f "org/mpiexec|ds4f_ep_runner" 2>/dev/null; exit 0' INT TERM

# the "SERVE ready" banner goes to the per-rank file (mpiexec does not forward rank stdout), not $LOG
READY=ds4f_ep_rank00.txt
for i in $(seq 1 180); do
    grep -q 'SERVE ready' "$READY" 2>/dev/null && break
    kill -0 $RUNNER_PID 2>/dev/null || { echo "[serve] runner died during load; see $LOG"; tail -20 "$LOG"; exit 1; }
    sleep 5
done
grep -q 'SERVE ready' "$READY" || { echo "[serve] runner did not become ready in time; see $LOG / $READY"; exit 1; }

echo "[serve] runner ready on 11 nodes. HTTP frontend -> :$PORT"
PORT=$PORT DS4F_SERVE_BASE="$BASE" TOK=${TOK:-$HOME/models/ds4f/tokenizer.json} exec python3 ds4f_serve.py
