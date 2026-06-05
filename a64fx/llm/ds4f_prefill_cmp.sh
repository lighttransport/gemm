#!/bin/bash
# Paired prefill comparison on 11 nodes: token-at-a-time vs batched M-token GEMM,
# SAME exact+bf16 config so the only variable is batching. Extracts the rank0
# prefill line from each run. Synthetic weights (no staging). Uses SKIP_TOPO after
# the first run so topo is generated once.
set -e
cd "$(dirname "$0")"
LAYERS=${LAYERS:-32}
PREFILL=${PREFILL:-32}
MAXGEN=${MAXGEN:-4}
COMMON="DS4F_LAYERS=$LAYERS DS4F_PREFILL=$PREFILL DS4F_MAXGEN=$MAXGEN DS4F_EXACT=1 DS4F_FP8_BF16=1 DS4F_PROF=0"

echo "############ DS4F prefill comparison: layers=$LAYERS prefill=$PREFILL ############"

run() {  # $1=label  $2=extra-env  $3=skiptopo
    echo; echo "===== $1 ====="
    env $COMMON $2 SKIP_TOPO=$3 ./run_ds4f_11n.sh > "cmp_$1.log" 2>&1 || { echo "FAILED (see cmp_$1.log)"; tail -5 "cmp_$1.log"; return; }
    grep -E "^prefill:|^decode:" ds4f_ep_rank00.txt 2>/dev/null | sed 's/^/  /'
}

run "seq"      ""                      0
run "batch8"   "DS4F_PREFILL_BATCH=8"  1
run "batch16"  "DS4F_PREFILL_BATCH=16" 1
run "batch32"  "DS4F_PREFILL_BATCH=32" 1

echo; echo "############ done ############"
