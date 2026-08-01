#!/bin/bash
# Prefix-cache validation: an APPEND request must be (a) faster (skips the shared prefix) and
# (b) byte-identical to a FULL reprefill of the same prompt. Greedy so output is deterministic.
cd "$(dirname "$0")"
B=$HOME/.ds4f_serve.lctx; TOK=$HOME/models/ds4f/tokenizer.json
i=0; until grep -q 'SERVE ready' ds4f_ep_rank00.txt 2>/dev/null || [ $i -ge 60 ]; do i=$((i+1)); sleep 10; done
grep -q 'SERVE ready' ds4f_ep_rank00.txt || { echo "NOT READY"; tail -6 /tmp/ds4f_lctx_serve.log; exit 1; }
echo "ready. running prefix-cache A/B..."

req() {  # $1=ids-file -> echoes "seq | logline"; sets RESP to resp ids
  { echo "8 0 1 0 0 1 0"; cat "$1"; } > $B.req
  local nx=$(( $(cat $B.reqseq 2>/dev/null || echo 0) + 1 )); echo $nx > $B.reqseq
  until [ "$(cat $B.respseq 2>/dev/null || echo 0)" -ge $nx ]; do
    pgrep -f 'org/mpiexec' >/dev/null || { echo "RUNNER DIED"; exit 1; }; sleep 1
  done
  RESP=$(cat $B.resp); grep "SERVE req#$nx:" ds4f_ep_rank00.txt | tail -1
}

# Req1: full prefill of A, gen 8  -> cache = A + resp1
echo "--- Req1 (cold, full prefill A) ---"; req /tmp/A_ids.txt; R1="$RESP"
# build prompt2 = A + resp1 + extra  (strictly extends the cache)
cat /tmp/A_ids.txt > /tmp/P2_ids.txt; echo " $R1 $(cat /tmp/extra_ids.txt)" >> /tmp/P2_ids.txt
# Req2: APPEND (should show cached~=A+8, prefill ~= extra+ few) -> R2
echo "--- Req2 (APPEND: A+resp1+extra) ---"; req /tmp/P2_ids.txt; R2="$RESP"
# Req3: DIVERGENT prompt -> full reset (cache no longer holds P2's prefix)
echo "--- Req3 (divergent, forces reset) ---"; req /tmp/D_ids.txt
# Req4: same prompt as Req2, but now cache holds D-stuff -> FULL reprefill of P2 -> R4
echo "--- Req4 (FULL reprefill of P2, control) ---"; req /tmp/P2_ids.txt; R4="$RESP"

echo "=================================================="
echo "R2 (append) : $R2"
echo "R4 (full)   : $R4"
[ "$R2" = "$R4" ] && echo "RESULT: IDENTICAL  ==> prefix-cache append is byte-exact to full reprefill" \
                  || echo "RESULT: MISMATCH   ==> BUG"
