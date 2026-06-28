#!/bin/bash
# GLM-5.2 INT8 (compressed-tensors w8a16) bring-up smoke test on 24 A64FX nodes.
# Runs the first N (contiguous-downloaded) layers of the int8 checkpoint through prefill_synth,
# checking the new INT8 matvec path loads + runs NaN-free with a stable cross-rank argmax.
# model.norm (last shard, not downloaded) is sourced from the identical bf16 sibling via
# GLM5_BF16_EXTRA. TP off so every int8 col-shard is c0=0 (128-group aligned).
#PJM -g hp250467
#PJM -L "rscgrp=small,node=24,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=24"
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-24}
JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/int8_run_${JOB_TAG}"
NL=${GLM5_NL:-4}                                   # layers 0..NL-1 (>=4 includes MoE layer 3)
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm52-int8}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_int8_$JOB_TAG}
export GLM5_NSHARDS=${GLM5_NSHARDS:-233}
export GLM5_STAGE_LAYERS=$NL
export GLM5_BF16_DIR=$HOME/models/glm5.2 GLM5_BF16_NSHARDS=282 GLM5_BF16_EXTRA=model.norm.weight
export GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=0 GLM5_MAXPOS=${GLM5_MAXPOS:-512} GLM5_KV_BUDGET_GB=0
export GLM5_PREFILL_ONLY=1 GLM5_PREFILL_ROLLING=1 GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1
export GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
th=${GLM5_THREAD_SWEEP:-12}; pc=${GLM5_PCHUNK_SWEEP:-32}
RUNENV="LLM_THREADS=$th OMP_NUM_THREADS=$th GLM5_PCHUNK=$pc GLM5_REAL=1 GLM5_LAYERS=$NL GLM5_COMM_OVERLAP=1 TP_AR_BF16=1"

echo "=== GLM5.2 INT8 test: NP=$NP layers=0..$((NL-1)) model=$GLM5_MODEL_DIR job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; break; }; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }; echo "topo OK"
echo "--- staging int8 layers 0..$((NL-1)) + model.norm from bf16 ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
grep -hE "2nd pass" "$WORK"/../pjsub_glm5_int8_test_24n.sh.${PJM_JOBID}.out 2>/dev/null || true
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP  rank00: $(head -1 "$WORK"/glm5_stage_rank00.txt)"
grep -l "model.norm.weight" "$WORK"/rank00.manifest >/dev/null 2>&1 && echo "model.norm staged OK" || grep -c "model.norm" "$WORK"/rank00.manifest

SYN=${GLM5_SYNTH:-64}
echo "--- prefill_synth $SYN tok pchunk=$pc ($(date)) ---"; rm -f glm5_ep_rank00.txt
eval $RUNENV GLM5_PREFILL_SYNTH=$SYN GLM5_GEN_NEW=0 mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run"; tail -20 glm5_ep_rank00.txt 2>/dev/null; exit 5; }
echo "=== INT8 RESULT ==="
grep -hE "INT8|prefill_synth:|last argmax|NaNs|SENTINEL|owned|arena" glm5_ep_rank00.txt | head -20
echo "=== int8 test done $(date) ==="
