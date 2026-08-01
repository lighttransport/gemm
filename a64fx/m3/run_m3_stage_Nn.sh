#!/bin/bash
# Stage the MiniMax-M3 (text) weights to each node's /local/m3, one EP rank per
# node (mirrors run_ds4f_stage_11n.sh). Run INSIDE an allocation (no pjsub); the
# matching run uses M3_REAL=1 with the same NP/node set + DS4F_STAGE_DIR=/local/m3.
#
#   NP=11 ./run_m3_stage_Nn.sh                       # all 60 layers (large blob)
#   NP=11 M3_STAGE_LAYERS=8 ./run_m3_stage_Nn.sh     # truncated (fits small alloc)
#   NP=1  M3_STAGE_LAYERS=4 M3_SHARD_LIMIT=3 ./run_m3_stage_Nn.sh   # 1-rank smoke
#
# Each rank reads $HOME/models/m3 (shared FS) and writes rank<rr>.{blob,manifest}
# to /local/m3. Dense is replicated WHOLE per rank (the loader TP-slices); routed
# experts e%NP==rank only. A per-rank status file lands in the cwd.
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
M3_DIR="$(cd "$(dirname "$0")" && pwd)"; LLM_DIR="$M3_DIR/../llm"
cd "$M3_DIR"

NP=${NP:-11}
EXCLUDE=${EXCLUDE:-0,0,0}
VCOORD=${VCOORD:-vcoord_m3.txt}

MPI_PLACE=()
if [ "$EXCLUDE" != "none" ]; then
    SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}; SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}; SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
    : > "$VCOORD"; n=0
    for ((x=0;x<SX;x++)); do for ((y=0;y<SY;y++)); do for ((z=0;z<SZ;z++)); do
        [ "$x,$y,$z" = "$EXCLUDE" ] && continue; echo "($x,$y,$z)" >> "$VCOORD"; n=$((n+1)); done; done; done
    [ "$n" -ge "$NP" ] || { echo "shape minus ($EXCLUDE) = $n < NP=$NP" >&2; exit 1; }
    head -n "$NP" "$VCOORD" > "$VCOORD.tmp" && mv "$VCOORD.tmp" "$VCOORD"
    MPI_PLACE=(-vcoordfile "$VCOORD")
fi

export M3_MODEL_DIR=${M3_MODEL_DIR:-$HOME/models/m3}
export M3_STAGE_DIR=${M3_STAGE_DIR:-/local/m3}
export M3_NSHARDS=${M3_NSHARDS:-59}
export M3_EP_SIZE=$NP
export M3_STAGE_LAYERS=${M3_STAGE_LAYERS:-0}
export M3_SHARD_LIMIT=${M3_SHARD_LIMIT:-0}
export M3_STATUS_DIR="$M3_DIR"

make -C "$LLM_DIR" m3_stage CC=fcc OPENMP=1 >/dev/null
BIN="$LLM_DIR/build/m3_stage"
echo "--- staging M3 (NP=$NP layers=${M3_STAGE_LAYERS:-all} shard_limit=${M3_SHARD_LIMIT:-all}) -> $M3_STAGE_DIR ---"
# each rank derives its EP rank from PMIX_RANK; the stager reads it
mpiexec -np "$NP" "${MPI_PLACE[@]}" "$BIN"
echo "--- stage status ---"; cat "$M3_DIR"/m3_stage_rank*.txt 2>/dev/null | sort
