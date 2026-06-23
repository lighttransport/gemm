#!/bin/bash
# Cheap synthetic-weights correctness toy for data-parallel groups: 8 non-contiguous A64FX nodes,
# split into GLM5_PREFILL_GROUPS independent groups. No staging (GLM5_REAL=0), tiny model, short
# context (Tier A). Validates gbarrier + group-scoped tp_comm + per-group logging + NaNs=0 before
# spending a 384-node staged job.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=8,elapse=01:05:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=8"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"
GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-8}
JOB_TAG=${PJM_JOBID:-manual_$$}
WORK="$GLM5/group_toy_run_${JOB_TAG}"

export GLM5_PREFILL_GROUPS=${GLM5_PREFILL_GROUPS:-2}
export GLM5_TP=${GLM5_TP:-1}
export GLM5_TP_SHARED=${GLM5_TP_SHARED:-1}
export GLM5_MAXPOS=${GLM5_MAXPOS:-512}
export GLM5_PREFILL_SYNTH=${GLM5_PREFILL_SYNTH:-512}
export GLM5_PREFILL_ONLY=1
export GLM5_PREFILL_ROLLING=${GLM5_PREFILL_ROLLING:-1}

echo "=== GLM5 group toy: NP=$NP groups=$GLM5_PREFILL_GROUPS synth (REAL=0) layers=${GLM5_LAYERS:-4} experts=${GLM5_EXPERTS:-16} ==="
date
mkdir -p "$WORK" || exit 2
cd "$WORK" || exit 2
rm -f tofu_topo.txt glm5_ep_*.txt glm5_ep_rank00.txt

make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3

topo_ok=0
for t in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
    echo "[group-toy] topo try $t failed; retry"; sleep 3
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }
echo "topo OK ($(grep -vc '^#' tofu_topo.txt 2>/dev/null || echo 0) ranks)"

echo "--- group toy run ($(date)) ---"
COMMON="GLM5_PCHUNK=${GLM5_PCHUNK:-128} GLM5_REAL=0 GLM5_LAYERS=${GLM5_LAYERS:-4} GLM5_EXPERTS=${GLM5_EXPERTS:-16}"

# Run A: configured (groups + optional forced merge via GLM5_MERGE_AT). Survivor = group 0 = seq 0.
echo "--- run A: groups=$GLM5_PREFILL_GROUPS merge_at=${GLM5_MERGE_AT:-none} ($(date)) ---"
env $COMMON LLM_THREADS=${GLM5_THREAD_SWEEP:-12} OMP_NUM_THREADS=${GLM5_THREAD_SWEEP:-12} GLM5_MERGE_AT="${GLM5_MERGE_AT:-}" \
  mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run A"; exit 5; }
for f in glm5_ep_rank00.txt glm5_ep_g*_rank00.txt; do [ -f "$f" ] && { echo "--- A:$f ---"; grep -hE "group_merge:|prefill_synth:|SENTINEL" "$f"; }; done
A_ARG=$(grep -hoE "argmax=[0-9]+" glm5_ep_rank00.txt 2>/dev/null | tail -1)
mv glm5_ep_rank00.txt A_rank00.txt 2>/dev/null

# Run B: reference, single group (G=1), same seq 0, no merge -> ground truth for the survivor.
echo "--- run B: reference G=1 (seq 0, no merge) ($(date)) ---"
env $COMMON LLM_THREADS=${GLM5_THREAD_SWEEP:-12} OMP_NUM_THREADS=${GLM5_THREAD_SWEEP:-12} GLM5_PREFILL_GROUPS=1 GLM5_MERGE_AT="" \
  mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run B"; exit 5; }
echo "--- B:rank00 ---"; grep -hE "prefill_synth:|SENTINEL" glm5_ep_rank00.txt
B_ARG=$(grep -hoE "argmax=[0-9]+" glm5_ep_rank00.txt 2>/dev/null | tail -1)

echo "=== MERGE CHECK: survivor A $A_ARG vs reference B $B_ARG -> $([ -n "$A_ARG" ] && [ "$A_ARG" = "$B_ARG" ] && echo MATCH || echo MISMATCH) ==="
echo "=== group toy done $(date) ==="
