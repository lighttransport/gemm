#!/bin/bash
# MiniMax-M3 (text) REAL-WEIGHT run on 96 A64FX nodes, 8x12 torus. Stage the full
# model from $HOME/models/m3 to each node's /local/m3, then run M3_REAL=1 with the
# EP+TP forward. /local is wiped per job, so stage + run share ONE job.
#
# Mechanical real-weight validation (like ds4p's "load=64/64, NaN=0, lockstep"):
# synthetic ACTIVATIONS through REAL weights -> validates the loader at scale,
# per-node memory fit with real data, and cross-rank lockstep. (Coherent text
# generation needs the gen-mode + tokenizer path, a follow-on.)
#
# Per-rank blob ~38 GB (replicated dense whole + ~2 owned experts/layer); the
# loader TP-slices into a ~14 GB arena. Staging is shared-FS I/O bound (dense is
# replicated x96 -> heavy); budget ~30-60 min.
#
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_real_96n.sh'
# Smoke (faster): pjsub --no-check-directory -x M3_STAGE_LAYERS=12 -x M3_LAYERS=12 a64fx/m3/pjsub_m3_real_96n.sh

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=8x12:torus,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
cd "$M3" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-96}
STAGE_LAYERS=${M3_STAGE_LAYERS:-0}      # 0 = full 60 layers
RUN_LAYERS=${M3_LAYERS:-0}              # match the stage (0 = full)
export M3_MODEL_DIR=${M3_MODEL_DIR:-$HOME/models/m3}
export M3_STAGE_DIR=/local/m3
export M3_NSHARDS=59 M3_EP_SIZE=$NP M3_STATUS_DIR="$M3"
export M3_TP=${M3_TP:-1} M3_MSA=${M3_MSA:-1}
export M3_MAXPOS=${M3_MAXPOS:-512} M3_PREFILL=${M3_PREFILL:-8} M3_DECODE=${M3_DECODE:-16}
export LLM_THREADS=${LLM_THREADS:-48} OMP_NUM_THREADS=${LLM_THREADS:-48}

echo "=== M3 REAL-weight 96n (8x12): NP=$NP stage_layers=$STAGE_LAYERS run_layers=$RUN_LAYERS job=${PJM_JOBID:-?} ==="
date

# ---- build (native) ----
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

# ---- topology (retry the CODE=1907 uTofu flake) ----
topo_ok=0
for t in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
    echo "[real96] topo try $t failed (1907?); retry"; sleep 3
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed (1907 flake — resubmit)"; exit 3; }
echo "topo OK ($(wc -l < tofu_topo.txt) rows)"

# ---- stage (M3_STAGE_LAYERS truncation honored) ----
echo "--- staging full model -> /local/m3 ($(date)) ---"
M3_STAGE_LAYERS=$STAGE_LAYERS mpiexec -np "$NP" "$LLM/build/m3_stage" || { echo "FATAL: stage failed"; exit 4; }
echo "--- stage status ($(date)) ---"; cat "$M3"/m3_stage_rank*.txt 2>/dev/null | sort | tail -5
echo "staged ranks: $(ls "$M3"/m3_stage_rank*.txt 2>/dev/null | wc -l)/$NP"

# ---- run REAL weights ----
echo "--- run M3_REAL=1 ($(date)) ---"
M3_REAL=1 M3_LAYERS=$RUN_LAYERS mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "FATAL: run failed"; exit 5; }
echo "--- rank0 log ---"; cat m3_ep_rank00.txt 2>/dev/null
echo "=== done $(date) ==="
