#!/bin/bash
# Under CP (sharded caches): per-rank shard save/load must be BYTE-EXACT (disk load+append == live
# append), and each rank must write its own <path>.rankNN shard. Also checks CP serve is coherent.
cd "$(dirname "$0")"
B=$HOME/.ds4f_serve.cp; TOK=$HOME/models/ds4f/tokenizer.json; SYS=$HOME/.ds4f_cpsys.bin
enc(){ printf '%s' "$2" >/tmp/e.txt; python3 tools/ds4f_tokenizer.py encode --tokenizer $TOK --prompt-file /tmp/e.txt --out "$1" 2>/dev/null; }
dec(){ echo "$1" >/tmp/d.txt; python3 tools/ds4f_tokenizer.py decode --tokenizer $TOK --ids-file /tmp/d.txt 2>/dev/null; }
req(){ { echo "$1"; [ -n "$2" ] && echo "$2"; cat "$3"; } > $B.req
  local nx=$(( $(cat $B.reqseq 2>/dev/null||echo 0)+1 )); echo $nx > $B.reqseq
  until [ "$(cat $B.respseq 2>/dev/null||echo 0)" -ge $nx ]; do pgrep -f 'org/mpiexec'>/dev/null||{ echo DIED;exit 1;}; sleep 1; done
  RESP=$(cat $B.resp); }
i=0; until grep -q 'SERVE ready' ds4f_ep_rank00.txt 2>/dev/null||[ $i -ge 60 ]; do i=$((i+1)); sleep 10; done
grep -q 'SERVE ready' ds4f_ep_rank00.txt || { echo NOTREADY; tail -8 /tmp/ds4f_cp_serve.log; exit 1; }
echo "ready ($((i*10))s)"; grep 'SERVE ready' ds4f_ep_rank00.txt

# reuse S/U/SU/D from the prior test if present, else rebuild
[ -f /tmp/S_ids.txt ] || { sed -n '1,10p' ../ds4f.md >/tmp/S.txt; enc /tmp/S_ids.txt "$(cat /tmp/S.txt)"; }
[ -f /tmp/SU_ids.txt ] || { enc /tmp/U_ids.txt " In one sentence, the single most important takeaway is"; cat /tmp/S_ids.txt >/tmp/SU_ids.txt; echo " $(cat /tmp/U_ids.txt)" >>/tmp/SU_ids.txt; }
[ -f /tmp/D_ids.txt ] || enc /tmp/D_ids.txt "Write a limerick about a robot."
echo "S=$(wc -w</tmp/S_ids.txt) SU=$(wc -w</tmp/SU_ids.txt)"

echo "-- coherence: plain gen --"; req "12 0 1 0 0 1 0 0 0" "" /tmp/S_ids.txt; echo "  cont-> $(dec "$RESP")"
echo "-- Req: prefill S, SAVE per-rank shards to $SYS --"; req "0 0 1 0 0 1 0 0 2" "$SYS" /tmp/S_ids.txt
echo "-- shard files written: --"; ls -1 ${SYS}.rank* 2>/dev/null | sed 's/^/  /'; echo "  count=$(ls ${SYS}.rank* 2>/dev/null | wc -l)"
echo "-- SU gen8 (live append) --"; req "8 0 1 0 0 1 0 0 0" "" /tmp/SU_ids.txt; R_live="$RESP"
echo "-- divergent (evict) --"; req "8 0 1 0 0 1 0 0 0" "" /tmp/D_ids.txt
echo "-- LOAD per-rank shards then SU gen8 --"; req "8 0 1 0 0 1 0 0 1" "$SYS" /tmp/SU_ids.txt; R_disk="$RESP"
echo "R_live: $R_live"; echo "R_disk: $R_disk"
[ "$R_live" = "$R_disk" ] && echo "PASS: per-rank CP shard save/load BYTE-EXACT" || echo "FAIL: MISMATCH"
echo "=== serve log ==="; grep 'SERVE ' ds4f_ep_rank00.txt | tail -8
