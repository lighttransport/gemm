#!/bin/bash
# ds4p @ 1M with CONTEXT-PARALLEL cache-sharding (DS4F_CP) on 112 nodes (ep=112, 96+16).
# CP shards the compressed Tier-B2 caches (cmp_q4, idx_kv8_4) by slot range across the 112
# EP ranks -> per-node cache ~/112 (vs the replicated ~5.4 GB), which fixes the MemFree
# cliff (0.38 GB -> decode all-reduce timeout) seen at ep=96 in the 108n co-serve (49228161).
# CP is validated for ds4f (255k->12M ctx, byte-identical to CP-off); this is the ds4p test.
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=4x4x7:torus,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=112"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -eu
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM="$HOME/work/gemm/ds4p/a64fx/llm"; cd "$LLM"

export DS4F_MODEL=ds4p NP=112 EXCLUDE=none DS4F_EP_SIZE=112
export DS4F_MODEL_DIR="$HOME/models/ds4p" DS4F_STAGE_DIR=/local/ds4p DS4F_NSHARDS=64 DS4F_STAGE_FLUSH_GB=2
export DS4F_TP_ATTN=1 DS4F_TP_SHARED=1 DS4F_TP_HEAD=1 DS4F_TP_EMBED=1
export DS4F_REAL=1 DS4F_EXACT=1 DS4F_TIERB2=1 DS4F_INT8_CMP=1 DS4F_INT4_CMP=1 DS4F_IDX_INT8=1
# --- context-parallel cache sharding (the lever under test) ---
export DS4F_CP=1 DS4F_CP_SHARD=1 DS4F_CP_IDX=1
export DS4F_MAXPOS=1048576 DS4F_CTX_WARM=1040000 DS4F_MAXGEN=8 DS4F_PREFILL=1
export DS4F_WARM_RSS_TRACE=1 DS4F_WARM_MEMAVAIL_STOP_GB=2.0 DS4F_QUIET=1 LLM_THREADS=48

echo "=== ds4p CP-sharded @1M ep=112 : job=${PJM_JOBID:-?} host=$(hostname) ==="
echo "=== [1/2] STAGE ds4p EP blobs (112 ranks) ==="
DS4F_STAGE_COMPACT=1 ./run_ds4f_stage_11n.sh
echo "=== [2/2] RUN ds4p ep=112 + CP cache-shard @ ~1M ctx ==="
./run_ds4f_11n.sh
echo "SENTINEL ds4p_cp_112n_1M=done"
