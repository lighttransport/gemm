#!/bin/bash
#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,retention_state=0,rscgrp=small-s2,node=48,elapse=02:00:00"
#PJM --mpi "proc=48"
#PJM --llio localtmp-size=80Gi
#PJM -j
#
# DeepSeek-V4-Pro full-model batch run on 48 A64FX nodes (self-contained:
# /local is wiped per job, so stage + run MUST share one job).
#
#   pjsub --no-check-directory pjsub_ds4p_48n.sh                 # full 61 layers + gen
#   pjsub --no-check-directory -x LAYERS=8 pjsub_ds4p_48n.sh     # smoke: 8 layers, no gen
#
# Phases (each gated; the job log is the single diagnostic artifact):
#   1. build (native fcc) + tofu topo
#   2. parallel stage from $HOME/models/ds4p (~46 GB/rank full; ~40 min at the
#      shared-FS aggregate read rate — dense is replicated x48)
#   3. synthetic smoke (LAYERS=4): arena/TP/comm at 64 ranks, no weights
#   4. real-weight mechanical run: load=48/48, NaN=0, lockstep
#   5. real GENERATION (only when LAYERS=0 i.e. full model): prompt -> text
#
# Budget @48 ranks, TP all on: weights ~22 GB/node — TIGHT (8 experts/layer/rank,
# 384%48=0; 128%48 uneven heads) -> ~2-3 GB headroom; prefer 64n for bring-up.
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
LLM="$REPO/a64fx/llm"
cd "$LLM" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-48}
LAYERS=${LAYERS:-0}                 # 0 = all 61 layers (+gen); N>0 = smoke, no gen
export DS4F_MODEL=ds4p
export DS4F_MODEL_DIR=${DS4F_MODEL_DIR:-$HOME/models/ds4p}
export DS4F_NSHARDS=64
export DS4F_STAGE_DIR=/local/ds4p
export NP EXCLUDE=none              # batch alloc: no co-located login node to spare
export DS4F_QUIET=1 DS4F_STAGE_COMPACT=1

echo "=== DS4P batch: NP=$NP layers=${LAYERS:-all} job=$PJM_JOBID ==="
date

# ---- 1. build ----
make ds4f_ep_runner CC=fcc OPENMP=1 || exit 2
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE \
    -I../../common -o build/ds4f_stage ds4f_stage.c || exit 2

# ---- 2. stage (each rank: its expert shard + replicated dense) ----
date; echo "--- phase 2: stage ---"
DS4F_STAGE_LAYERS=$LAYERS ./run_ds4f_stage_11n.sh || { echo "SENTINEL stage=FAIL"; exit 3; }
echo "SENTINEL stage=OK"

# ---- 3. synthetic smoke at full rank count (no weights) ----
date; echo "--- phase 3: synthetic smoke (LAYERS=4) ---"
DS4F_LAYERS=4 DS4F_PREFILL=4 DS4F_MAXGEN=4 SKIP_TOPO=1 \
DS4F_EXACT=1 DS4F_TIERB2=1 DS4F_MHC=1 \
DS4F_TP_HEAD=1 DS4F_TP_EMBED=1 DS4F_TP_SHARED=1 DS4F_TP_ATTN=1 DS4F_TP_OPROJ=1 DS4F_TP_WOB=1 \
./run_ds4f_11n.sh || { echo "SENTINEL smoke=FAIL"; exit 4; }
echo "SENTINEL smoke=$(grep -Eo 'NaNs=[0-9]+' ds4f_ep_rank00.txt | tail -1)"

# ---- 4. real-weight mechanical run ----
date; echo "--- phase 4: real-weight run ---"
export DS4F_REAL=1 DS4F_EXACT=1 DS4F_TIERB2=1 DS4F_MHC=1
export DS4F_TP_HEAD=1 DS4F_TP_EMBED=1 DS4F_TP_SHARED=1 DS4F_TP_ATTN=1 DS4F_TP_OPROJ=1 DS4F_TP_WOB=1
DS4F_LAYERS=$LAYERS DS4F_PREFILL=8 DS4F_MAXGEN=16 SKIP_TOPO=1 \
./run_ds4f_11n.sh || { echo "SENTINEL real=FAIL"; exit 5; }
echo "SENTINEL real=$(grep -Eo 'NaNs=[0-9]+' ds4f_ep_rank00.txt | tail -1) memfree=$(grep MemFree /proc/meminfo)"

# ---- 5. real generation (full model only) ----
if [ "$LAYERS" = "0" ]; then
    date; echo "--- phase 5: generation ---"
    TOK=$HOME/models/ds4p/tokenizer.json MAX_NEW=${MAX_NEW:-64} SKIP_TOPO=1 \
    ./run_ds4f_gen_11n.sh || { echo "SENTINEL gen=FAIL"; exit 6; }
    echo "SENTINEL gen=OK"
fi
date; echo "SENTINEL job=DONE"
