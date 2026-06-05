#!/bin/bash
# Cluster prefill comparison at PREFILL=256 to exercise expert-grouping (Task #9):
# seq (token-at-a-time) vs batch32 (experts NOT batched, ~0.75 tok/expert) vs
# batch256 (experts batched, ~6 tok/expert at ep=11). SAME exact+bf16 config so
# the only variable is the tile size. Synthetic weights, run INSIDE the live alloc.
set -e
cd "$(dirname "$0")"
LAYERS=${LAYERS:-32}
PREFILL=${PREFILL:-256}
MAXGEN=${MAXGEN:-2}
export DS4F_LAYERS=$LAYERS DS4F_PREFILL=$PREFILL DS4F_MAXGEN=$MAXGEN
export DS4F_EXACT=1 DS4F_FP8_BF16=1 DS4F_PROF=0

echo "############ DS4F prefill comparison (expert-grouping): layers=$LAYERS prefill=$PREFILL ############"

run() {  # $1=label  $2=batch-tile  $3=skiptopo
    echo; echo "===== $1 ====="
    DS4F_PREFILL_BATCH=$2 SKIP_TOPO=$3 ./run_ds4f_11n.sh > "cmpx_$1.log" 2>&1 \
        || { echo "FAILED (see cmpx_$1.log)"; tail -8 "cmpx_$1.log"; return; }
    grep -E "^prefill:|^decode:" ds4f_ep_rank00.txt 2>/dev/null | sed 's/^/  /'
}

run "seq"      0   0
run "batch32"  32  1
run "batch256" 256 1

echo; echo "############ done ############"
