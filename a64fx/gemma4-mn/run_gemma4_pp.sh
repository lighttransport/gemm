#!/bin/sh
# Stage + run the pipeline-parallel (PP) Gemma-4 12B BF16 across N nodes.
# Each rank stages ONLY its layer shard to /local (<=1GB chunks, memory-safe), then the
# uTofu PP runner pipelines tokens stage->stage. One rank/node; mpiexec only PLACES ranks.
#
#   NP=11 ./run_gemma4_pp.sh                 # 11 ranks (excl login node 0,0,0)
#   NP=2  ./run_gemma4_pp.sh                 # 2-node smoke
#   SKIP_STAGE=1 NP=11 ./run_gemma4_pp.sh    # reuse already-staged /local shards
# Env: GGUF (shared source), STAGE_DIR (/local/gemma4_pp), PROMPT_IDS, MAXGEN, LLM_THREADS.
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

HERE=$(cd "$(dirname "$0")" && pwd)
UTOFU_DIR="$HERE/../utofu-tests"
GGUF=${GGUF:-$HOME/models/gemma4/12b/gemma-4-12b-it-BF16.gguf}   # shared FS (all nodes)
STAGE_DIR=${STAGE_DIR:-/local/gemma4_pp}                          # node-local
NP=${NP:-11}
EXCLUDE=${EXCLUDE:-0,0,0}        # drop the login node (where the agent runs) -> no OOM-kill
VCOORD=${VCOORD:-vcoord_g4pp.txt}
MAXGEN=${MAXGEN:-32}
export LLM_THREADS=${LLM_THREADS:-48}
export GEMMA4_STAGE_FLUSH_GB=${GEMMA4_STAGE_FLUSH_GB:-1}
export TOFU_TOPO_PATH=${TOFU_TOPO_PATH:-$HERE/tofu_topo_g4pp.txt}

# ---- vcoordfile: all alloc-shape coords except $EXCLUDE ----
MPI_PLACE=""
if [ "$EXCLUDE" != "none" ]; then
    SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}; SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}; SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
    : > "$VCOORD"; n=0
    for x in $(seq 0 $((SX-1))); do for y in $(seq 0 $((SY-1))); do for z in $(seq 0 $((SZ-1))); do
        [ "$x,$y,$z" = "$EXCLUDE" ] && continue
        echo "($x,$y,$z)" >> "$VCOORD"; n=$((n+1))
        [ "$n" -ge "$NP" ] && break 3
    done; done; done
    [ "$n" -lt "$NP" ] && { echo "shape ${SX}x${SY}x${SZ} minus ($EXCLUDE) = $n < NP=$NP" >&2; exit 1; }
    MPI_PLACE="-vcoordfile $VCOORD"
    echo "[g4pp] placing $NP ranks via $VCOORD (excl $EXCLUDE)"
fi

# ---- build ----
echo "[g4pp] building tofu_topo_helper + stager + runner..."
make -C "$UTOFU_DIR" tofu_topo_helper >/dev/null
fcc -Nclang -O2 -D_GNU_SOURCE -I../../common "$HERE/gemma4_stage.c" -o "$HERE/gemma4_stage"
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE -I../../common \
    "$HERE/gemma4_pp_runner.c" -lm -lpthread -lhwb -ltofucom -o "$HERE/gemma4_pp_runner"

# ---- topology ----
echo "[g4pp] generating tofu topo ($TOFU_TOPO_PATH)..."
mpiexec -np "$NP" $MPI_PLACE "$UTOFU_DIR/tofu_topo_helper"

# ---- stage (each rank stages its shard; memory-safe 1GB chunks) ----
if [ "${SKIP_STAGE:-0}" != "1" ]; then
    echo "[g4pp] staging shards to $STAGE_DIR (NP=$NP)..."
    mpiexec -np "$NP" $MPI_PLACE sh -c "mkdir -p $STAGE_DIR; exec $HERE/gemma4_stage $GGUF $STAGE_DIR \$PMIX_RANK $NP pp"
fi

# ---- run ----
PROMPT_ARG=""; [ -n "${PROMPT_IDS:-}" ] && PROMPT_ARG="$PROMPT_IDS"
echo "[g4pp] running PP pipeline (maxgen=$MAXGEN)..."
mpiexec -np "$NP" $MPI_PLACE "$HERE/gemma4_pp_runner" "$GGUF" "$STAGE_DIR" "$PROMPT_ARG" "$MAXGEN"
