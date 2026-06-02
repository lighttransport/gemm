#!/bin/bash
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#
# Pipeline-parallel decode of Qwen3.6-27B BF16 across N A64FX nodes (no MPI in
# the inference binary; mpiexec only places ranks). Each rank loads ONLY its
# contiguous layer range (lazy-mmap GGUF + range-aware panel build), so the 54GB
# model fits with ~1/N HBM per node.
#
# Submit on an N-node allocation (start with 4):
#   pjsub -g hp250467 -L "freq=2200,eco_state=2,rscgrp=small,node=4,elapse=00:30:00" \
#         --mpi "shape=4,proc=4" --no-check-directory run_pp_27b.sh
# Scale by changing node=N and shape=N,proc=N (model has 64 layers; N | 64 is cleanest:
# 4 -> 16 layers/node, 8 -> 8, 16 -> 4).
#
# Step 1 (MPI helper) writes each node's Tofu coordinates -> tofu_topo.txt.
# Step 2 (pure uTofu) is launched by mpiexec only for placement. stdout is
# swallowed by mpiexec, so rank 0 also writes pp_run_<coords>.txt.
#
# Env tunables: PP_PROMPT PP_MAXGEN PP_MAXSEQ LLM_THREADS  (see pp_runner.c).
set -e

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

LLM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LLM_DIR"
UTOFU_DIR="$LLM_DIR/../utofu-tests"

NP=${NP:-${PJM_NODE:-4}}
QWEN27B_NODES=${QWEN27B_NODES:-$NP}
QWEN27B_SOURCE=${QWEN27B_SOURCE:-$HOME/models/qwen36/27b}
QWEN27B_PACKAGE=${QWEN27B_PACKAGE:-$HOME/models/qwen36/27b/${QWEN27B_NODES}nodes}
MODEL=${MODEL:-$QWEN27B_PACKAGE/Qwen3.6-27B-BF16-00001-of-00002.gguf}
export QWEN27B_LOCAL_DIR=${QWEN27B_LOCAL_DIR:-/local/qwen36/27b}

if [ "${QWEN27B_PREPARE:-1}" != "0" ]; then
    chmod +x "$LLM_DIR/prepare_qwen36_27b_12nodes.sh"
    "$LLM_DIR/prepare_qwen36_27b_12nodes.sh" "$QWEN27B_SOURCE" "$QWEN27B_PACKAGE"
    if [ "$MODEL" = "$QWEN27B_SOURCE/$(basename "$MODEL")" ]; then
        MODEL="$QWEN27B_PACKAGE/$(basename "$MODEL")"
    fi
fi

export GGUF_LAZY_MMAP=1
export LLM_THREADS=${LLM_THREADS:-48}
export PP_PROMPT=${PP_PROMPT:-"Hello, who are you?"}
export PP_MAXGEN=${PP_MAXGEN:-64}
export OMP_NUM_THREADS=${LLM_THREADS}

echo "=== pipeline-parallel 27B decode on $NP node(s) ==="
if [ "${SKIP_STAGE:-0}" != "1" ]; then
    chmod +x "$LLM_DIR/stage_gguf_shards.sh"
    mpiexec -np "$NP" "$LLM_DIR/stage_gguf_shards.sh" "$MODEL" "$QWEN27B_LOCAL_DIR"
fi
MODEL_LOCAL="$QWEN27B_LOCAL_DIR/$(basename "$MODEL")"
if [ ! -f "$MODEL_LOCAL" ]; then
    echo "model local missing after stage: $MODEL_LOCAL" >&2
    exit 1
fi
echo "model=$MODEL_LOCAL  source=$MODEL  threads=$LLM_THREADS  maxgen=$PP_MAXGEN"
echo "prompt=\"$PP_PROMPT\""

# --- build helper (MPI) + runner (pure uTofu) if stale ---
echo "=== build ==="
make -C "$UTOFU_DIR" tofu_topo_helper
make -C "$LLM_DIR" pp_runner CC=fcc OPENMP=1

echo "=== step 1: discover topology (MPI helper) -> tofu_topo.txt ==="
mpiexec -np "$NP" "$UTOFU_DIR/tofu_topo_helper"
cat tofu_topo.txt

echo "=== step 2: pipeline-parallel decode (no MPI in the binary) ==="
mpiexec -np "$NP" "$LLM_DIR/build/pp_runner" "$MODEL_LOCAL"

echo "=== done (rank 0 output also in pp_run_*.txt) ==="
