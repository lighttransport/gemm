#!/bin/bash
# Co-serving validation: ds4p (ep=96, REAL weights, 1M ctx) + ds4f (ep=12, synthetic)
# running CONCURRENTLY on disjoint subsets of ONE 108-node allocation. Proves two
# independent EP groups coexist: disjoint vcoordfiles, per-group cwd (so each group's
# tofu_topo.txt + ds4f_ep_*_rank*.txt don't collide), separate uTofu communicators.
# 108 = 96 + 12 (both multiples of 12). ds4f runs synthetic here — it exercises the
# full EP forward (alloc/TP/uTofu all-reduce/lockstep) concurrently with ds4p; real
# ds4f is a DS4F_REAL=1 + 12-node stage away (validated separately at 11n).

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=6x6x3:torus,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=108"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -eu

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM="$HOME/work/gemm/ds4p/a64fx/llm"; UTOFU="$LLM/../utofu-tests"; cd "$LLM"
mkdir -p cosrv/ds4p cosrv/ds4f
echo "=== cosrv 108n: ds4p(96,real,1M) + ds4f(12,synth) | job=${PJM_JOBID:-?} host=$(hostname) ==="

# --- allocation shape -> all coords -> disjoint vcoordfiles (first 96 ds4p, last 12 ds4f) ---
SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-6}}; SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-6}}; SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-3}}
: > allcoords.txt
for ((x=0;x<SX;x++)); do for ((y=0;y<SY;y++)); do for ((z=0;z<SZ;z++)); do echo "($x,$y,$z)" >> allcoords.txt; done; done; done
NTOT=$(wc -l < allcoords.txt)
echo "shape ${SX}x${SY}x${SZ} = $NTOT coords"
[ "$NTOT" -ge 108 ] || { echo "ERROR shape gives $NTOT < 108 coords"; exit 1; }
head -96 allcoords.txt > vcoord_ds4p.txt
sed -n '97,108p' allcoords.txt > vcoord_ds4f.txt
echo "ds4p nodes=$(wc -l < vcoord_ds4p.txt)  ds4f nodes=$(wc -l < vcoord_ds4f.txt)"

# --- build once ---
make -C "$UTOFU" tofu_topo_helper >/dev/null
make ds4f_ep_runner CC=fcc OPENMP=1 >/dev/null
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE -I../../common -o build/ds4f_stage ds4f_stage.c
BIN="$LLM/build/ds4f_ep_runner"; STG="$LLM/build/ds4f_stage"; HLP="$UTOFU/tofu_topo_helper"

# --- per-group topo (each helper runs with cwd=cosrv/<grp> so tofu_topo.txt is isolated) ---
mpiexec -np 96 -vcoordfile vcoord_ds4p.txt bash -c "cd $LLM/cosrv/ds4p && exec $HLP"
mpiexec -np 12 -vcoordfile vcoord_ds4f.txt bash -c "cd $LLM/cosrv/ds4f && exec $HLP"
echo "topo rows: ds4p=$(grep -vc '^#' cosrv/ds4p/tofu_topo.txt) ds4f=$(grep -vc '^#' cosrv/ds4f/tofu_topo.txt)"

# --- stage ds4p EP blobs on the 96 ds4p nodes (ds4f is synthetic -> no stage) ---
echo "=== stage ds4p (96 ranks -> /local/ds4p) ==="
env DS4F_MODEL=ds4p DS4F_EP_SIZE=96 DS4F_MODEL_DIR="$HOME/models/ds4p" DS4F_STAGE_DIR=/local/ds4p \
    DS4F_NSHARDS=64 DS4F_STAGE_FLUSH_GB=2 DS4F_STATUS_DIR="$LLM/cosrv/ds4p" \
    mpiexec -np 96 -vcoordfile vcoord_ds4p.txt "$STG"
echo "stage ds4p done=$(ls cosrv/ds4p/ds4f_stage_rank*.txt 2>/dev/null | wc -l)/96"

# --- run BOTH groups CONCURRENTLY (each in its own cwd; separate uTofu comm) ---
echo "=== launch ds4p(real,1M) on 96 + ds4f(synth) on 12 — concurrent ==="
( cd "$LLM/cosrv/ds4p"
  export DS4F_MODEL=ds4p DS4F_EP_SIZE=96 LLM_THREADS=48 OMP_NUM_THREADS=48 \
    DS4F_REAL=1 DS4F_EXACT=1 DS4F_TIERB2=1 DS4F_INT8_CMP=1 DS4F_INT4_CMP=1 DS4F_IDX_INT8=1 \
    DS4F_TP_ATTN=1 DS4F_TP_SHARED=1 DS4F_TP_HEAD=1 DS4F_TP_EMBED=1 \
    DS4F_STAGE_DIR=/local/ds4p DS4F_MAXPOS=1048576 DS4F_CTX_WARM=1040000 DS4F_PREFILL=1 DS4F_MAXGEN=8 \
    DS4F_WARM_MEMAVAIL_STOP_GB=2.0 DS4F_WARM_RSS_TRACE=1 TOFU_TOPO_PATH="$LLM/cosrv/ds4p/tofu_topo.txt"
  mpiexec -np 96 -vcoordfile "$LLM/vcoord_ds4p.txt" "$BIN" ) > cosrv/ds4p_run.log 2>&1 &
PIDP=$!
( cd "$LLM/cosrv/ds4f"
  export DS4F_EP_SIZE=12 LLM_THREADS=48 OMP_NUM_THREADS=48 \
    DS4F_REAL=0 DS4F_EXACT=1 DS4F_TIERB2=1 \
    DS4F_MAXPOS=8192 DS4F_CTX_WARM=4096 DS4F_PREFILL=1 DS4F_MAXGEN=8 \
    TOFU_TOPO_PATH="$LLM/cosrv/ds4f/tofu_topo.txt"
  mpiexec -np 12 -vcoordfile "$LLM/vcoord_ds4f.txt" "$BIN" ) > cosrv/ds4f_run.log 2>&1 &
PIDF=$!

set +e; wait "$PIDP"; RCP=$?; wait "$PIDF"; RCF=$?; set -e

echo "=== ds4p run tail ==="; tail -16 cosrv/ds4p_run.log
echo "=== ds4f run tail ==="; tail -16 cosrv/ds4f_run.log
echo "SENTINEL cosrv_108n ds4p_rc=$RCP ds4f_rc=$RCF"
