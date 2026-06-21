#!/bin/bash
# DeepSeek-V4-Pro (ds4p) end-to-end validation on 96 A64FX nodes at 1M context.
#
# Stages real ds4p weights into per-rank EP blobs (ep_size=96 -> 4 experts/rank),
# then runs the EP harness with tensor-parallel dense sharding + lean long-ctx
# caches (int4 cmp_kv, int8 idx_kv), warming to ~1M positions under the MemFree
# guard. Validates: 96-rank staging, real-weight load, ep=96 fits 1M/rank,
# uTofu all-reduce lockstep across 96 ranks.
#
# 96 = 8x12 = 6x16 (multiple of 12 and 16). Alloc params below are BEST-GUESS
# (no prior >64-node job here, pjshowrsc blocked) — adjust rscgrp/shape if pjsub
# rejects. Submit from the frontend:
#   ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/llm/pjsub_ds4p_96n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=4x4x6:torus,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -eu

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
REPO="$HOME/work/gemm/ds4p"; LLM="$REPO/a64fx/llm"; cd "$LLM"

# ---- ds4p config, EP=96, model + staging ----
export DS4F_MODEL=ds4p
export NP=96
export EXCLUDE=none                       # use the whole 96-node alloc
export DS4F_EP_SIZE=96
export DS4F_MODEL_DIR="$HOME/models/ds4p"
export DS4F_STAGE_DIR="/local/ds4p"
export DS4F_NSHARDS=64
export DS4F_STAGE_FLUSH_GB=2

# ---- tensor-parallel dense sharding (mandatory to fit ds4p) + lean 1M caches ----
export DS4F_TP_ATTN=1 DS4F_TP_SHARED=1 DS4F_TP_HEAD=1 DS4F_TP_EMBED=1
export DS4F_REAL=1 DS4F_EXACT=1 DS4F_TIERB2=1
export DS4F_INT8_CMP=1 DS4F_INT4_CMP=1 DS4F_IDX_INT8=1

# ---- 1M context: warm to ~1M then decode a few tokens (memory + lockstep validation) ----
export DS4F_MAXPOS=1048576
export DS4F_CTX_WARM=1040000
export DS4F_MAXGEN=8
export DS4F_PREFILL=1
export DS4F_WARM_RSS_TRACE=1
export DS4F_WARM_MEMAVAIL_STOP_GB=2.0
export DS4F_QUIET=1
export LLM_THREADS=48

echo "=== ds4p 96n @1M : job=${PJM_JOBID:-unset} host=$(hostname) ==="
echo "model=$DS4F_MODEL_DIR ep_size=$DS4F_EP_SIZE maxpos=$DS4F_MAXPOS"

echo "=== [1/2] STAGE ds4p EP blobs (96 ranks, ~21 GB/rank to /local) ==="
DS4F_STAGE_COMPACT=1 ./run_ds4f_stage_11n.sh

echo "=== [2/2] RUN ds4p ep=96 @ ~1M ctx ==="
./run_ds4f_11n.sh

echo "SENTINEL ds4p_96n_1M=done"
