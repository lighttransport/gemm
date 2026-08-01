#!/bin/bash
# decode tok/s: greedy vs sampling. reads the runner-internal tok/s from ds4f_ep_rank00.txt.
# liveness-guarded: if the runner (mpiexec) dies, abort the poll instead of hanging.
cd "$(dirname "$0")"
B=$HOME/.ds4f_serve.test; TOK=$HOME/models/ds4f/tokenizer.json
printf 'Once upon a time' > /tmp/rp.txt
python3 tools/ds4f_tokenizer.py encode --tokenizer $TOK --prompt-file /tmp/rp.txt --out /tmp/ri.txt 2>/dev/null
run() {  # $1=label ; $2.. = header (mnew temp top_p top_k pres rep seed)
  local label="$1"; shift
  { echo "$@"; cat /tmp/ri.txt; } > $B.req
  local nx=$(( $(cat $B.reqseq 2>/dev/null || echo 0) + 1 )); echo $nx > $B.reqseq
  local i
  for i in $(seq 1 2400); do
    [ "$(cat $B.respseq 2>/dev/null || echo 0)" -ge "$nx" ] && break
    pgrep -f 'org/mpiexec' >/dev/null || { echo "$label: RUNNER DIED (req#$nx)"; return 1; }
    sleep 0.25
  done
  echo "$label: $(grep "SERVE req#$nx:" ds4f_ep_rank00.txt | tail -1)"
}
echo "prompt=4tok ('Once upon a time'). runner-internal tok/s (wraps tiny prefill):"
run "greedy  m64 " 64  0   1    0  0 1   0
run "greedy  m64 " 64  0   1    0  0 1   0
run "sample  m64 " 64  0.8 0.95 40 0 1.1 42
run "sample  m64 " 64  0.8 0.95 40 0 1.1 43
run "sampNoTK m64" 64  0.8 1.0  0  0 1.0 44   # top_k=0,top_p=1 -> full-vocab sort, no cut (worst case)
run "greedy  m160" 160 0   1    0  0 1   0     # length check (did req#7 die from length or sampling?)
run "sample  m160" 160 0.8 0.95 40 0 1.1 45
