#!/bin/bash
# GLM-5.2 (text) REAL-WEIGHT run on 192 A64FX nodes, 16x12 torus. Stage the full
# model from $HOME/models/glm5.2 to each node's /local/glm5, then run GLM5_REAL=1 with the
# EP+TP forward. /local is wiped per job, so stage + run share ONE job.
#
# Mechanical real-weight validation (like ds4p's "load=64/64, NaN=0, lockstep"):
# synthetic ACTIVATIONS through REAL weights -> validates the loader at scale,
# per-node memory fit with real data, and cross-rank lockstep. (Coherent text
# generation needs the gen-mode + tokenizer path, a follow-on.)
#
# Per-rank blob is dominated by TP-staged dense tensors plus 1-2 owned experts/layer.
# Staging is shared-FS I/O bound; budget several hours for the first full 282-shard run.
#
# Submit: ssh fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory a64fx/glm5/pjsub_glm5_real_192n.sh'
# Smoke (faster): pjsub --no-check-directory -x GLM5_STAGE_LAYERS=12 -x GLM5_LAYERS=12 a64fx/glm5/pjsub_glm5_real_192n.sh

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=16x12:torus,elapse=03:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=192"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
cd "$GLM5" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-192}
STAGE_LAYERS=${GLM5_STAGE_LAYERS:-0}      # 0 = full 78 main layers
RUN_LAYERS=${GLM5_LAYERS:-0}              # match the stage (0 = full)
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm5.2}
export GLM5_STAGE_DIR=/local/glm5
export GLM5_NSHARDS=282 GLM5_EP_SIZE=$NP GLM5_STATUS_DIR="$GLM5"
export GLM5_TP=${GLM5_TP:-1} GLM5_MSA=${GLM5_MSA:-0}
export GLM5_MAXPOS=${GLM5_MAXPOS:-512} GLM5_PREFILL=${GLM5_PREFILL:-8} GLM5_DECODE=${GLM5_DECODE:-16}
export LLM_THREADS=${LLM_THREADS:-12} OMP_NUM_THREADS=${LLM_THREADS:-12}

echo "=== GLM5.2 REAL-weight 192n (16x12): NP=$NP stage_layers=$STAGE_LAYERS run_layers=$RUN_LAYERS job=${PJM_JOBID:-?} ==="
date

"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" || exit 2

# ---- build (native) ----
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

# ---- topology (retry the CODE=1907 uTofu flake) ----
topo_ok=0
for t in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
    echo "[real192] topo try $t failed (1907?); retry"; sleep 3
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed (1907 flake — resubmit)"; exit 3; }
echo "topo OK ($(wc -l < tofu_topo.txt) rows)"

# ---- stage (GLM5_STAGE_LAYERS truncation honored) ----
echo "--- staging full model -> /local/glm5 ($(date)) ---"
GLM5_STAGE_LAYERS=$STAGE_LAYERS mpiexec -np "$NP" "$LLM/build/glm5_stage" || { echo "FATAL: stage failed"; exit 4; }
echo "--- stage status ($(date)) ---"; cat "$GLM5"/glm5_stage_rank*.txt 2>/dev/null | sort | tail -5
echo "staged ranks: $(ls "$GLM5"/glm5_stage_rank*.txt 2>/dev/null | wc -l)/$NP"

# ---- run REAL weights ----
echo "--- run GLM5_REAL=1 ($(date)) ---"
GLM5_REAL=1 GLM5_LAYERS=$RUN_LAYERS mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run failed"; exit 5; }
echo "--- rank0 log ---"; cat glm5_ep_rank00.txt 2>/dev/null
echo "=== done $(date) ==="
