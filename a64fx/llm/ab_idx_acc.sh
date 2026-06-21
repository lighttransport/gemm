#!/bin/bash
# int8 index_score ACCURACY A/B: real long prompt (real rotate+fp4 idx_kv dist), full
# production path (REAL weights + mHC + TierB2). DS4F_IDX_INT8=0 (f32 ref) vs 1 (int8).
# Gate = generated token ids identical (argmax-exact). If they diverge, report WHERE and
# how many tokens matched (the de-risked probe showed 503/512 selected-set overlap, so a
# late single-token drift would be within quant noise; early/total drift is a NO-GO).
set -u
cd /vol0006/mdt0/data/hp250467/work/gemm/ds4f/a64fx/llm
PF=${PF:-prompt_long_idx.txt}
MAXNEW=${MAXNEW:-96}
VERD=/tmp/ab_idx_acc_verdict.txt
: > "$VERD"

run_leg () {  # $1=tag $2=int8val $3=gen_out $4=sentinel
  local tag=$1 i8=$2 out=$3 sent=$4
  : > "$sent"; rm -f "$out"
  echo "=== ACC leg $tag: DS4F_IDX_INT8=$i8 prompt=$PF max_new=$MAXNEW ===" | tee -a "$VERD"
  PROMPT_FILE=$PF MAX_NEW=$MAXNEW GEN_OUT=$out \
    DS4F_IDX_INT8=$i8 DS4F_GEN_SENTINEL=$sent DS4F_QUIET=1 \
    ./run_ds4f_gen_11n.sh > /tmp/ab_idx_acc_$tag.log 2>&1
  echo "leg $tag rc=$?" | tee -a "$VERD"
}

run_leg A 0 gen_idx_a.txt /tmp/gen_idx_a.txt
run_leg B 1 gen_idx_b.txt /tmp/gen_idx_b.txt

{
  echo "============ int8 index_score ACCURACY A/B VERDICT ============"
  na=$(wc -w < gen_idx_a.txt 2>/dev/null || echo 0)
  nb=$(wc -w < gen_idx_b.txt 2>/dev/null || echo 0)
  echo "f32 ids=$na  int8 ids=$nb"
  if [ "$na" -gt 0 ] && [ "$nb" -gt 0 ]; then
    # token-by-token first divergence + match count
    python3 - <<'PY'
a=open('gen_idx_a.txt').read().split()
b=open('gen_idx_b.txt').read().split()
n=min(len(a),len(b)); m=0; first=-1
for i in range(n):
    if a[i]==b[i]: m+=1
    elif first<0: first=i
if first<0 and len(a)==len(b):
    print(f"ARGMAX_EXACT: PASS  (all {len(a)} ids identical: f32 == int8)")
else:
    print(f"ARGMAX_EXACT: DIVERGE  matched {m}/{n}  first_diff_at_token={first}")
    lo=max(0,first-2); hi=min(n,first+4)
    print(f"  f32 [{lo}:{hi}] = {a[lo:hi]}")
    print(f"  int8[{lo}:{hi}] = {b[lo:hi]}")
PY
  else
    echo "  !! a leg produced no ids — see /tmp/ab_idx_acc_{A,B}.log"
  fi
  for L in A:0:/tmp/gen_idx_a.txt B:1:/tmp/gen_idx_b.txt; do
    tag=${L%%:*}; rest=${L#*:}; i8=${rest%%:*}; sent=${rest#*:}
    echo "--- leg $tag (DS4F_IDX_INT8=$i8) perf/lockstep ---"
    grep -iE 'GEN_RC|decode:|prefill:|NaN|argmax_distinct' "$sent" 2>/dev/null | head -8
  done
} | tee -a "$VERD"
echo "ACC_AB_DONE"
