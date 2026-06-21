#!/bin/bash
# Per-layer-type sparse asymptote: dense vs all-CSA(R=4) vs all-HCA(R=128).
# FORCE_RATIO overrides the 0/1/last-dense rule so every layer is sparse,
# isolating the true O(topk)-capped per-layer cost from the mandatory-dense tax.
set -e
export PATH="/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
OUT=/local/ds4f_ratio_sweep.out
[ -d /local ] || OUT="$HOME/tmp/ds4f_ratio_sweep.out"; mkdir -p "$(dirname "$OUT")"
: > "$OUT"

LAYERS=8; MAXPOS=70000; MAXGEN=6; EP=11
CTXS="2048 8192 32768 65536"

run() {  # $1=label $2=sparse $3=force_ratio $4=ctx
    LLM_THREADS=48 DS4F_EP_SIZE=$EP DS4F_EP_RANK=0 \
    DS4F_LAYERS=$LAYERS DS4F_MAXPOS=$MAXPOS DS4F_CTX_WARM=$4 DS4F_MAXGEN=$MAXGEN \
    DS4F_SPARSE=$2 DS4F_FORCE_RATIO=$3 DS4F_PREFILL=0 DS4F_PROF=1 \
    build/ds4f_runner 2>&1 | grep -E '^  attn ' | awk -v l="$1" -v c="$4" \
      '{printf "[%-9s ctx=%-5s] attn %8.2f ms/tok\n", l, c, $2}'
}

for ctx in $CTXS; do
    echo "================ ctx=$ctx ================" >> "$OUT"
    run dense    0 0   "$ctx" >> "$OUT"
    run all-CSA4 1 4   "$ctx" >> "$OUT"
    run all-HCA128 1 128 "$ctx" >> "$OUT"
done
echo "__RATIO_DONE__" >> "$OUT"
