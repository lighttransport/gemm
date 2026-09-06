#!/bin/bash
# Validate: (1) system-prompt / KV disk snapshot is BYTE-EXACT (live-cache append == disk load+append);
#           (2) multi-slot keeps two independent contexts across a context switch.
cd "$(dirname "$0")"
B=$HOME/.ds4f_serve.lctx; TOK=$HOME/models/ds4f/tokenizer.json; SYS=$HOME/.ds4f_sys.bin
enc(){ printf '%s' "$2" >/tmp/e.txt; python3 tools/ds4f_tokenizer.py encode --tokenizer $TOK --prompt-file /tmp/e.txt --out "$1" 2>/dev/null; }
dec(){ echo "$1" >/tmp/d.txt; python3 tools/ds4f_tokenizer.py decode --tokenizer $TOK --ids-file /tmp/d.txt 2>/dev/null; }
req(){ # $1=header(9 fields) $2=path("" if none) $3=ids-file
  { echo "$1"; [ -n "$2" ] && echo "$2"; cat "$3"; } > $B.req
  local nx=$(( $(cat $B.reqseq 2>/dev/null||echo 0)+1 )); echo $nx > $B.reqseq
  until [ "$(cat $B.respseq 2>/dev/null||echo 0)" -ge $nx ]; do pgrep -f 'org/mpiexec'>/dev/null||{ echo DIED;exit 1;}; sleep 1; done
  RESP=$(cat $B.resp); RSEQ=$nx
}
i=0; until grep -q 'SERVE ready' ds4f_ep_rank00.txt 2>/dev/null||[ $i -ge 60 ]; do i=$((i+1)); sleep 10; done
grep -q 'SERVE ready' ds4f_ep_rank00.txt || { echo NOTREADY; tail -6 /tmp/ds4f_lctx_serve.log; exit 1; }
echo "ready ($((i*10))s)"

sed -n '1,10p' ../ds4f.md > /tmp/S.txt; enc /tmp/S_ids.txt "$(cat /tmp/S.txt)"
enc /tmp/U_ids.txt " In one sentence, the single most important takeaway is"
cat /tmp/S_ids.txt > /tmp/SU_ids.txt; echo " $(cat /tmp/U_ids.txt)" >> /tmp/SU_ids.txt
enc /tmp/D_ids.txt "Write a limerick about a robot."
echo "S=$(wc -w</tmp/S_ids.txt) U=$(wc -w</tmp/U_ids.txt) SU=$(wc -w</tmp/SU_ids.txt)"

echo "=== TEST 1: disk snapshot byte-exact ==="
echo "-- Req1: prefill S, SAVE to $SYS (mnew=0 ctl=2 slot0) --"; req "0 0 1 0 0 1 0 0 2" "$SYS" /tmp/S_ids.txt
echo "-- Req2: SU gen8 slot0 (live-cache append) --";           req "8 0 1 0 0 1 0 0 0" "" /tmp/SU_ids.txt; R_live="$RESP"
echo "-- Req3: divergent, evict live cache --";                 req "8 0 1 0 0 1 0 0 0" "" /tmp/D_ids.txt
echo "-- Req4: LOAD $SYS then SU gen8 (disk load+append) --";   req "8 0 1 0 0 1 0 0 1" "$SYS" /tmp/SU_ids.txt; R_disk="$RESP"
echo "R_live: $R_live"; echo "R_disk: $R_disk"
[ "$R_live" = "$R_disk" ] && echo "T1 PASS: disk snapshot round-trip BYTE-EXACT to live cache" || echo "T1 FAIL: MISMATCH"

echo "=== TEST 2: multi-slot independence ==="
enc /tmp/Q0_ids.txt "The capital of France is"; enc /tmp/Q1_ids.txt "def factorial(n):"
echo "-- Req5 slot0: Q0 gen8 --"; req "8 0 1 0 0 1 0 0 0" "" /tmp/Q0_ids.txt; R0="$RESP"; echo "  slot0 -> $(dec "$R0")"
echo "-- Req6 slot1: Q1 gen8 (switch to slot1) --"; req "8 0 1 0 0 1 0 1 0" "" /tmp/Q1_ids.txt; echo "  slot1 -> $(dec "$RESP")"
# slot0 turn2: Q0 + R0 + follow-up  (append; proves slot0 survived the slot1 interruption)
enc /tmp/F_ids.txt " The capital of Japan is"
cat /tmp/Q0_ids.txt > /tmp/Q0b.txt; echo " $R0 $(cat /tmp/F_ids.txt)" >> /tmp/Q0b.txt
echo "-- Req7 slot0: Q0+resp+' capital of Japan is' gen8 (switch back, append) --"; req "8 0 1 0 0 1 0 0 0" "" /tmp/Q0b.txt
echo "  slot0 turn2 -> $(dec "$RESP")"
echo "=== serve log ==="; grep 'SERVE ' ds4f_ep_rank00.txt | tail -10
