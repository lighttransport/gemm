#!/bin/bash
# Multi-slot context-switch overhead: prime two slots to length L, then trigger a switch and read the
# runner's logged "SERVE switch" save/restore ms (the snapshot memcpy). Repeat at several L.
cd "$(dirname "$0")"
B=$HOME/.ds4f_serve.lctx
# a big id file (distinct per slot) to slice prompts from
python3 - <<'PY'
ids0 = " ".join(str((i*2654435761) % 100000) for i in range(4000))   # slot0 token source
ids1 = " ".join(str((i*40503 + 17) % 100000) for i in range(4000))   # slot1 token source (distinct)
open("/tmp/src0.txt","w").write(ids0); open("/tmp/src1.txt","w").write(ids1)
PY
i=0; until grep -q 'SERVE ready' ds4f_ep_rank00.txt 2>/dev/null||[ $i -ge 60 ]; do i=$((i+1)); sleep 10; done
grep -q 'SERVE ready' ds4f_ep_rank00.txt || { echo NOTREADY; tail -6 /tmp/ds4f_lctx_serve.log; exit 1; }
echo "ready ($((i*10))s)"

req(){ # $1=header $2=ids-file
  { echo "$1"; cat "$2"; } > $B.req
  local nx=$(( $(cat $B.reqseq 2>/dev/null||echo 0)+1 )); echo $nx > $B.reqseq
  until [ "$(cat $B.respseq 2>/dev/null||echo 0)" -ge $nx ]; do pgrep -f 'org/mpiexec'>/dev/null||{ echo DIED;exit 1;}; sleep 1; done
}
slice(){ head -c 100000 "$1" | tr ' ' '\n' | head -n "$2" | tr '\n' ' ' > "$3"; }

echo "=== slot-switch overhead vs context length ==="
for L in 256 1024 2048; do
  slice /tmp/src0.txt $L /tmp/p0.txt; slice /tmp/src1.txt $L /tmp/p1.txt
  req "0 0 1 0 0 1 0 0 0" /tmp/p0.txt          # prime slot0 to L (prefill only)
  req "0 0 1 0 0 1 0 1 0" /tmp/p1.txt          # prime slot1 to L (switch 0->1: saves slot0@L)
  echo "3 2" > /tmp/tiny.txt
  req "0 0 1 0 0 1 0 0 0" /tmp/tiny.txt        # switch 1->0: saves slot1@L + restores slot0@L  <-- measured
  echo "L=$L : $(grep 'SERVE switch 1->0' ds4f_ep_rank00.txt | tail -1)"
done
echo "=== all switch log lines ==="; grep 'SERVE switch' ds4f_ep_rank00.txt
