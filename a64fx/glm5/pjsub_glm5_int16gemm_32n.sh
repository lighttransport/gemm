#!/bin/bash
# P1 — int16 GEMM (GLM5_GEMM_SDOT=2) full-model @32n: BATCHED DECODE + PREFILL A/Bs + quality logit A/B.
# The multi-token GEMM (glm5_gemm_int16sdot) is used by both prefill AND batched decode (bd=1) — the
# compute-bound M>=8 regime where int16 SDOT WINS (unlike the M=1 decode matvec GLM5_MV_SDOT, a net loss).
# One 78-layer stage; NUMA on (pin + interleave). Ref=w8a16 (sd 0), test=int16 (sd 2).
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=32,elapse=01:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=32"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=32; JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/int16gemm_run_${JOB_TAG}_32n"; NL=78
export GLM5_MODEL_DIR=$HOME/models/glm52-int8 GLM5_STAGE_DIR=/local/glm5_int8_$JOB_TAG
export GLM5_NSHARDS=233 GLM5_STAGE_LAYERS=$NL GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_MAXPOS=1024 GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
export LLM_THREADS=48 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores    # NUMA pin
echo "=== P1 int16 GEMM @${NP}n job=${PJM_JOBID:-?} ==="; date
mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2
rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np $NP "$UTOFU/tofu_topo_helper" && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge $NP ] && { topo_ok=1; echo "topo try $t"; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }
echo "--- staging 78L ($(date)) ---"
mpiexec -np $NP "$LLM/build/glm5_stage" 2>stage.err || { echo "FATAL: stage"; tail -20 stage.err; exit 4; }
echo "staged $(ls glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP ($(date))"
python3 -c "import struct;d=open('$HOME/eval_prompts/coding.bin','rb').read();a=struct.unpack('<%dI'%(len(d)//4),d);open('$WORK/prompt.ids','w').write(' '.join(map(str,a[:256])))"

echo "===== P1a: BATCHED-DECODE GEMM A/B (bd=1, GLM5_GEMM_SDOT 0=w8a16 vs 2=int16) ====="
for sd in 0 2; do echo "--- bd int16?=$sd ($(date)) ---"; rm -f glm5_ep_rank00.txt
  env GLM5_REAL=1 GLM5_LAYERS=$NL GLM5_BATCH_DECODE=1 GLM5_CBATCH_SLOTS=8 GLM5_MAX_NEW=24 \
      GLM5_GEMM_SDOT=$sd GLM5_CBATCH_PROMPTS="$GLM5/cbatch_prompts_32.txt" GLM5_CBATCH_OUT_PREFIX="$WORK/bd_g$sd" \
      mpiexec -np $NP numactl --interleave=all "$LLM/build/glm5_ep_runner" | grep -hE 'service decode'
  mv glm5_ep_rank00.txt "bd_g${sd}_rank00.txt" 2>/dev/null
done
echo "--- bd token identity (g0 vs g2; int16 near-lossless -> expect ~MATCH) ---"
for f in "$WORK"/bd_g0_*.txt; do b="$WORK/bd_g2_${f##*/bd_g0_}"; [ -f "$b" ] && { cmp -s "$f" "$b" || echo "DIFF ${f##*/bd_g0_}"; }; done

echo "===== P1b: PREFILL GEMM A/B (synth 512 tok, PCHUNK=128, GLM5_GEMM_SDOT 0 vs 2) ====="
for sd in 0 2; do echo "--- prefill int16?=$sd ($(date)) ---"; rm -f glm5_ep_rank00.txt
  env GLM5_REAL=1 GLM5_LAYERS=$NL GLM5_PREFILL_ONLY=1 GLM5_PREFILL_SYNTH=512 GLM5_PCHUNK=128 \
      GLM5_GEMM_SDOT=$sd \
      mpiexec -np $NP numactl --interleave=all "$LLM/build/glm5_ep_runner" | grep -hE 'prefill_synth:|gen_prefill|NaN'
  mv glm5_ep_rank00.txt "pf_g${sd}_rank00.txt" 2>/dev/null
  grep -hE "prefill_synth:|PROFILE (prefill|gen_prefill)" "pf_g${sd}_rank00.txt" | head -12
done

echo "===== P1 quality: int16-GEMM logit A/B (prefill teacher-forcing, replicated head) ====="
for sd in 0 2; do rm -f glm5_ep_rank00.txt
  env GLM5_REAL=1 GLM5_LAYERS=$NL GLM5_TP=1 GLM5_TP_HEAD=0 GLM5_PREFILL_ONLY=1 \
      GLM5_PROMPT_IDS="$WORK/prompt.ids" GLM5_GEMM_SDOT=$sd GLM5_LOGIT_DUMP="$WORK/logit_g$sd.bin" \
      mpiexec -np $NP numactl --interleave=all "$LLM/build/glm5_ep_runner" | grep -hE 'NaN' | head -1
done
NPOS=$(wc -w < "$WORK/prompt.ids")
python3 - "$WORK/logit_g0.bin" "$WORK/logit_g2.bin" "$NPOS" <<'PY'
import sys,struct,math
def load(f):
    try:d=open(f,'rb').read()
    except:return None
    n=len(d)//4;return struct.unpack('<%df'%n,d) if n else None
A=load(sys.argv[1]);B=load(sys.argv[2]);npos=int(sys.argv[3])
if not A or not B:print("EMPTY dump(s)");sys.exit()
vocab=len(A)//npos;nn=min(len(A),len(B))//vocab;cs=rs=0.0;t1=0
for p in range(nn):
    a=A[p*vocab:(p+1)*vocab];b=B[p*vocab:(p+1)*vocab]
    dot=sum(x*y for x,y in zip(a,b));na=math.sqrt(sum(x*x for x in a));nb=math.sqrt(sum(y*y for y in b))
    cs+=dot/(na*nb+1e-30);rs+=math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))/(na+1e-30)
    t1+= max(range(vocab),key=lambda i:a[i])==max(range(vocab),key=lambda i:b[i])
print(f"int16 GEMM vs w8a16: steps={nn} cosine={cs/nn:.6f} rms_rel={rs/nn:.4f} top1={100*t1/nn:.1f}%  (PASS if cosine>=0.999 & top1>=99%)")
PY
echo "SENTINEL glm5_int16gemm_32n=done"; date
