#!/bin/bash
# Single-context CTX-length sweep for ds4p with CP on 96 nodes (ep=96). Stages once,
# then re-runs the loader+warm at max_pos = 1M/2M/4M/8M, recording warmtb2-DONE MemFree
# at each to pin the CP per-position cache rate and the max single-context ceiling
# L_max(96). Caches commit at alloc-time (MemFree flat vs ctx_warm in the single-node
# probe), so a small ctx_warm gives the full max_pos footprint cheaply per point.
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=4x4x6:torus,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -uo pipefail
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM="$HOME/work/gemm/ds4p/a64fx/llm"; UTOFU="$LLM/../utofu-tests"; cd "$LLM"

# constant ds4p + CP config
export DS4F_MODEL=ds4p DS4F_EP_SIZE=96 LLM_THREADS=48 OMP_NUM_THREADS=48 DS4F_CMGS=4
export DS4F_REAL=1 DS4F_EXACT=1 DS4F_TIERB2=1 DS4F_INT8_CMP=1 DS4F_INT4_CMP=1 DS4F_IDX_INT8=1
export DS4F_CP=1 DS4F_CP_SHARD=1 DS4F_CP_IDX=1
export DS4F_TP_ATTN=1 DS4F_TP_SHARED=1 DS4F_TP_HEAD=1 DS4F_TP_EMBED=1
export DS4F_MODEL_DIR="$HOME/models/ds4p" DS4F_STAGE_DIR=/local/ds4p DS4F_NSHARDS=64 DS4F_STAGE_FLUSH_GB=2
export DS4F_WARM_RSS_TRACE=1 DS4F_WARM_MEMAVAIL_STOP_GB=2.0

# vcoordfile = whole 96-node alloc
VC="$LLM/vcoord_ds4p.txt"
SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-4}}; SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-4}}; SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-6}}
: > "$VC"; for ((x=0;x<SX;x++)); do for ((y=0;y<SY;y++)); do for ((z=0;z<SZ;z++)); do echo "($x,$y,$z)" >> "$VC"; done; done; done
head -96 "$VC" > "$VC.t" && mv "$VC.t" "$VC"

make -C "$UTOFU" tofu_topo_helper >/dev/null
make ds4f_ep_runner CC=fcc OPENMP=1 >/dev/null
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE -I../../common -o build/ds4f_stage ds4f_stage.c
BIN="$LLM/build/ds4f_ep_runner"

echo "=== STAGE ds4p ep=96 (once) ==="
DS4F_STAGE_COMPACT=1 NP=96 EXCLUDE=none ./run_ds4f_stage_11n.sh

echo "=== topo (96 nodes) ==="
mpiexec -np 96 -vcoordfile "$VC" "$UTOFU/tofu_topo_helper"

RES="$LLM/cp_sweep_results.txt"; : > "$RES"
for CTX in 1048576 2097152 4194304 8388608; do
  echo "=== CTX(max_pos)=$CTX ==="
  rm -f ds4f_ep_stderr_rank00.txt ds4f_ep_load_rank00.txt
  env DS4F_MAXPOS=$CTX DS4F_CTX_WARM=131072 DS4F_MAXGEN=2 DS4F_PREFILL=1 \
    mpiexec -np 96 -vcoordfile "$VC" "$BIN" > "run_ctx_${CTX}.log" 2>&1
  rc=$?
  mf=$(grep -hoE "warmtb2 DONE.*MemFree=[0-9.]+ GB" ds4f_ep_stderr_rank00.txt 2>/dev/null | grep -oE "MemFree=[0-9.]+" | tail -1)
  ar=$(awk -F'arena_used=' 'NF>1{split($2,a," ");print a[1]}' ds4f_ep_load_rank00.txt 2>/dev/null | tail -1)
  guard=$(grep -cE "MemAvailable too low|_exit|RSS over stop" ds4f_ep_stderr_rank00.txt 2>/dev/null)
  echo "max_pos=$CTX rc=$rc arena=${ar}GB warmtb2_${mf:-NONE} guard_stop=$guard" | tee -a "$RES"
done
echo "=== SWEEP RESULTS (ds4p CP, M=96) ==="; cat "$RES"
echo "SENTINEL ds4p_cp_sweep_96n=done"
