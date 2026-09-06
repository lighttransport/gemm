#!/bin/bash
# Capture greedy decode of fixed prompts from the CURRENT runner (KV precision set at its launch:
# bf16 KV = DS4F_INT8_KV=0, int8 KV = DS4F_INT8_KV=1). Run once per config, saving token ids to
# /tmp/kvacc_<label>/, then bench_kv_accuracy_diff.py compares (top-1 agreement, first divergence).
cd "$(dirname "$0")"
LABEL=${1:?usage: bench_kv_accuracy.sh <label>  (e.g. bf16 or int8); NGEN=64}
B=${SERVE_BASE:-$HOME/.ds4f_serve.lctx}; TOK=$HOME/models/ds4f/tokenizer.json
NGEN=${NGEN:-64}
OUT=/tmp/kvacc_$LABEL; mkdir -p "$OUT"
# fixed prompt set (short -> low comm risk; diverse -> exercises the KV path)
P=( "The theory of relativity, developed by Albert Einstein, changed physics. In summary,"
    "def binary_search(arr, target):"
    "Once upon a time in a distant kingdom, there lived a wise old queen who" )
enc(){ printf '%s' "$1" >/tmp/kvp.txt; python3 tools/ds4f_tokenizer.py encode --tokenizer $TOK --prompt-file /tmp/kvp.txt --out /tmp/kvp_ids.txt 2>/dev/null; }
for i in "${!P[@]}"; do
  enc "${P[$i]}"
  { echo "$NGEN 0 1 0 0 1 0 0 0"; cat /tmp/kvp_ids.txt; } > $B.req
  nx=$(( $(cat $B.reqseq 2>/dev/null||echo 0)+1 )); echo $nx > $B.reqseq
  until [ "$(cat $B.respseq 2>/dev/null || echo 0)" -ge "$nx" ]; do pgrep -f 'org/mpiexec'>/dev/null||{ echo "prompt$i DIED"; exit 1; }; sleep 1; done
  cp $B.resp "$OUT/p$i.txt"
  echo "[$LABEL p$i] $(python3 tools/ds4f_tokenizer.py decode --tokenizer $TOK --ids-file $B.resp 2>/dev/null | head -c 160)"
done
echo "[$LABEL] saved $(ls $OUT/*.txt|wc -l) prompt outputs to $OUT"
