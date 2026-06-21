#!/bin/bash
# Stage-4 sparse-indexer attention: decode-attn cost vs context length,
# dense (DS4F_SPARSE=0) vs sparse (DS4F_SPARSE=1). Single node, synthetic
# ctx-warm fills the KV cache to each ctx without running real prefill.
# All output funneled to ONE file; caller reads it once after the sentinel.
set -e
export PATH="/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
OUT=/local/ds4f_sparse_sweep.out
[ -d /local ] || OUT="$HOME/tmp/ds4f_sparse_sweep.out"; mkdir -p "$(dirname "$OUT")"
: > "$OUT"

LAYERS=8
MAXPOS=70000
MAXGEN=6
EP=11
CTXS="2048 8192 32768 65536"

run() {  # $1=sparse $2=ctx
    LLM_THREADS=48 DS4F_EP_SIZE=$EP DS4F_EP_RANK=0 \
    DS4F_LAYERS=$LAYERS DS4F_MAXPOS=$MAXPOS DS4F_CTX_WARM=$2 DS4F_MAXGEN=$MAXGEN \
    DS4F_SPARSE=$1 DS4F_PREFILL=0 DS4F_PROF=1 \
    build/ds4f_runner 2>&1 | grep -E 'sparse indexer|ctx-warm|attn |TOTAL|decode:' \
      | sed "s/^/[sparse=$1 ctx=$2] /"
}

for ctx in $CTXS; do
    echo "================ ctx=$ctx ================" >> "$OUT"
    run 0 "$ctx" >> "$OUT"
    run 1 "$ctx" >> "$OUT"
done
echo "__SWEEP_DONE__" >> "$OUT"
