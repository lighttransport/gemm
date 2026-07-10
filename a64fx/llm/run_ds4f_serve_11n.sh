#!/bin/bash
# DS4F HTTP serving (llama-server-like) on the 12-node interactive alloc = 11 EP nodes + controller.
#
# Architecture: the `ds4f_ep_runner` loads the model ONCE on 11 EP nodes and enters a persistent
# DS4F_SERVE loop; the python HTTP frontend (ds4f_serve.py) runs on the controller and drives it via
# shared-FS request/response files (all ranks read the same request -> lockstep, no broadcast).
# Prefill uses the batched-verify path (+65-74%); decode is the optimized bit-identical path.
#
# Prereq: weights staged (run_ds4f_stage_11n.sh). Usage (inside the live alloc, from a64fx/llm):
#     PORT=8080 ./run_ds4f_serve_11n.sh                       # default ~16k context, fast decode
#     CTX=65536 ./run_ds4f_serve_11n.sh                       # longer single context (see below)
#     CTX=1048576 CP=1 ./run_ds4f_serve_11n.sh                # extreme context (TP+CP sharded)
# Then, from anywhere that can reach this node:
#     curl -s localhost:8080/v1/completions -d '{"prompt":"def quicksort(a):","max_tokens":128}'
#   with sampling (default is greedy; temperature<=0 == greedy):
#     curl -s localhost:8080/v1/completions -d '{"prompt":"...","max_tokens":128,
#       "temperature":0.8,"top_p":0.95,"top_k":40,"repeat_penalty":1.1,"presence_penalty":0,"seed":42}'
#   OpenAI chat completions (the pi coding-agent path -- SSE stream + prompt-injected tools):
#     curl -s localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'
#     curl -s localhost:8080/v1/models      # lists model id "ds4f"
#   Drive pi against it (from ~/work/pi, .pi/extensions/ds4f-provider.ts registers the provider):
#     DS4F_BASE_URL=http://<this-node>:8080/v1 pi --model ds4f/ds4f
#
# LONGER SINGLE CONTEXT (CTX=<tokens>, prompt+gen must fit): past the ~16k fast ceiling the f32/bf16
# KV + compressed caches no longer fit, so CTX>16384 auto-enables the compressed ctx-cache levers --
# int8 KV, int8+int4 compressor, int4 indexer -- which shrink the per-position cache to ~2 KB/pos while
# keeping the fast coherent MHC decode (all token-identical to bf16 in prior A/B). This lifts the
# ceiling several-fold (dense weights, ~22 GB/node, become the floor). int8 KV forces token-by-token
# prefill (batched-verify aborts under int8 KV), so first-token latency grows ~linearly with the prompt.
#   CP=1: EXTREME context (hundreds of k -> millions) -- additionally TP-shard the replicated dense
#   across the 11 nodes and slot-shard the compressed caches (context-parallel, the validated 255k->12M
#   path). Requires FP8 dense (int8/bf16 dense is unsliceable) so decode is dequant-bound = slower.
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
# prefix cache (on by default): a request whose prompt EXTENDS the previous one skips re-prefilling
# the shared prefix (multi-turn TTFT win, byte-identical). Set 0 to always reprefill from scratch.
export DS4F_SERVE_PREFIX_CACHE=${DS4F_SERVE_PREFIX_CACHE:-1}
# SLOTS: number of independent conversation contexts (one live in the caches at a time; a request's
# "slot" field context-switches by snapshotting/restoring the per-slot KV+compressor state).
export DS4F_SERVE_SLOTS=${DS4F_SERVE_SLOTS:-1}
# SYSCACHE: preload a persisted context (built once with a cache_save request) into slot 0 so every
# conversation starts with the system prompt already prefilled -- instant TTFT, survives restarts.
[ -n "$DS4F_SERVE_SYSCACHE" ] && export DS4F_SERVE_SYSCACHE
# real-weight decode bundle (== --preset decode); dense stays int8 (Q8) for fast decode by default
export DS4F_REAL=1 DS4F_FP8_BF16=1 DS4F_Q8_DENSE=1 DS4F_TIERB2=1 DS4F_MHC=1 DS4F_HC_PAR=1 DS4F_HC_RMSPAR=1
export DS4F_NUMA=${DS4F_NUMA:-1}

# ---- context ceiling: CTX sets MAXPOS; >16k pulls in the compressed-cache memory levers ----
CTX=${CTX:-${DS4F_MAXPOS:-16384}}
export DS4F_MAXPOS=$CTX
if [ "$CTX" -gt 16384 ]; then
  echo "[serve] longer context: MAXPOS=$CTX -> compressed ctx-caches on (int8 KV/cmp, int4 cmp/idx)"
  export DS4F_INT8_KV=${DS4F_INT8_KV:-1} DS4F_INT8_CMP=${DS4F_INT8_CMP:-1} \
         DS4F_INT4_CMP=${DS4F_INT4_CMP:-1} DS4F_IDX_INT4=${DS4F_IDX_INT4:-1}
  export DS4F_PREFILL_GEMM=0            # batched-verify aborts under int8 KV -> token-by-token prefill
  if [ "${CP:-0}" = 1 ]; then          # extreme ctx: TP-shard dense (needs FP8) + CP-shard the caches
    echo "[serve]   CP=1: TP+CP sharding on (FP8 dense, dequant-bound decode)"
    export DS4F_FP8_BF16=0 DS4F_Q8_DENSE=0
    export DS4F_TP_ATTN=1 DS4F_TP_SHARED=1 DS4F_TP_HEAD=1 DS4F_TP_EMBED=1 DS4F_TP_OPROJ=1 DS4F_TP_WOB=1
    export DS4F_CP=1 DS4F_CP_SHARD=1 DS4F_CP_IDX=1 DS4F_CP_MERGE=1
  fi
else
  export DS4F_PREFILL_GEMM=${DS4F_PREFILL_GEMM:-1}   # fast ceiling: keep batched-verify prefill (+74%)
fi

# ---- 2026-07-10 session levers (all real-weight 11n A/B'd; see ds4f.md "2026-07-10 session") ----
export DS4F_HC_SVE=${DS4F_HC_SVE:-1}         # SVE mHC (decode +12%: mhc_pre 10.1->2.7 ms; SVE batch mHC in prefill)
export DS4F_PF_TP=${DS4F_PF_TP:-1}           # compute-shard verify shared+o-proj (prefill; decode untouched)
export DS4F_PREFILL_K=${DS4F_PREFILL_K:-64}  # verify chunk sweet spot (K=32 -> 64: comm /2; 128 payload-bound)
export DS4F_MV_FUSE=${DS4F_MV_FUSE:-1}       # wq_a+wkv one dispatch (bit-exact)
export DS4F_CMP_LOCAL=${DS4F_CMP_LOCAL:-1}   # reader-local Tier-B2 compressor weights: decode +6.6%, bit-exact (tb2lcmp 5.2->2.8 ms)
export DS4F_TP_HEAD=${DS4F_TP_HEAD:-1}       # vocab-shard head: greedy decode -1.5 ms via argmax-merge
export DS4F_TP_EMBED=${DS4F_TP_EMBED:-1}     # vocab-shard embed (-0.97 GB, bit-exact)

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

echo "[serve] runner ready on 11 nodes. HTTP frontend -> :$PORT (/v1/chat/completions /v1/completions /v1/models)"
PORT=$PORT DS4F_SERVE_BASE="$BASE" TOK=${TOK:-$HOME/models/ds4f/tokenizer.json} exec python3 ds4f_serve.py
