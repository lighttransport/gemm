#!/bin/bash
# MiniMax-M3 int4-KV QUALITY GATE (Lever 1) — REAL-weight gen, bf16-KV vs int4-KV, same prompt @24n fp8.
# int4-KV changes attention output (that's how we know it engages); this checks it still produces
# COHERENT text on real weights. Batched decode can't run real activations (it feeds synthetic X), but
# it uses the IDENTICAL codec (m3_q4_pack/dot/axpy) + per-(position,head) absmax quantization as the
# single-stream gen path — so single-stream real-weight coherence is the conclusive quality signal for
# the batched int4-KV too. Stage the fp8 model ONCE, gen the prompt twice (int4=0 then int4=1), compare.
#
# Submit: ssh -A fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory a64fx/m3/pjsub_m3_int4kv_quality_24n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=2x3x4:torus,elapse=01:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=24"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-24}
RUN="$M3/int4qual_${PJM_JOBID:-$$}"; mkdir -p "$RUN"; cd "$RUN" || exit 2
export M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_STAGE_DIR=/local/m3fp8 M3_STATUS_DIR="$RUN"
export M3_EP_SIZE=$NP M3_TP=1 M3_MSA=1 TP_AR_BF16=1
export M3_MAXPOS=${M3_MAXPOS:-256} M3_MAX_NEW=${M3_MAX_NEW:-64}
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3 int4-KV QUALITY GATE 24n fp8: NP=$NP job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
python3 "$M3/m3_tokenizer.py" encode "$(cat "$M3/prompt_m3.txt")" --bos > "$M3/prompt_ids.txt" || { echo "FATAL encode"; exit 3; }
echo "prompt_ids: $(cat "$M3/prompt_ids.txt")"

topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[q] topo try $t (1907?)"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

echo "--- staging fp8 ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/m3_stage" || { echo "FATAL stage"; exit 4; }
echo "staged: $(ls "$RUN"/m3_stage_rank*.txt 2>/dev/null | wc -l)/$NP"

gen_pass(){  # $1=label  $2=int4flag
  echo "--- gen KV=$1 (M3_INT4_KV=$2) ($(date)) ---"; rm -f m3_ep_rank00.txt
  M3_REAL=1 M3_INT4_KV=$2 M3_MSTREAM=1 M3_PROMPT_IDS="$M3/prompt_ids.txt" M3_GEN_OUT="$M3/gen_$1.txt" \
    mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || echo "[KV=$1] FATAL rc=$?"
  grep -hE "gen:|decode:|NaN" m3_ep_rank00.txt 2>/dev/null | sed "s/^/[KV=$1] /"
  echo "[KV=$1] ids: $(cat "$M3/gen_$1.txt" 2>/dev/null)"
  echo "[KV=$1] text: $(python3 "$M3/m3_tokenizer.py" decode-file "$M3/gen_$1.txt" 2>/dev/null)"
}
gen_pass bf16kv 0
gen_pass int4kv 1
echo "=== COMPARE (int4-KV coherent + close to bf16-KV? = quality PASS) ==="
echo "SENTINEL m3_int4kv_quality_24n=done"; echo "=== done $(date) ==="
