#!/bin/sh
# Stage + run the tensor-parallel (TP) Gemma-4 12B BF16 runner.
# One rank/node holds a tensor slice of every layer.  EXCLUDE=none uses the
# scheduler's placement, which is required for the four-node non-shaped llmgr
# allocation.  GEMMA4_TP_MTP enables speculative draft/verify decoding.
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

HERE=$(cd "$(dirname "$0")" && pwd)
UTOFU_DIR="$HERE/../utofu-tests"
GGUF=${GGUF:-$HOME/models/gemma4/12b/gemma-4-12b-it-BF16.gguf}
STAGE_DIR=${STAGE_DIR:-/local/gemma4_tp}
NP=${NP:-4}
EXCLUDE=${EXCLUDE:-none}
VCOORD=${VCOORD:-vcoord_g4tp.txt}
PROMPT_IDS=${PROMPT_IDS:-}
MAXGEN=${MAXGEN:-32}
export LLM_THREADS=${LLM_THREADS:-48}
export GEMMA4_STAGE_FLUSH_GB=${GEMMA4_STAGE_FLUSH_GB:-1}
export TOFU_TOPO_PATH=${TOFU_TOPO_PATH:-$HERE/tofu_topo.txt}

MPI_PLACE=""
if [ "$EXCLUDE" != "none" ]; then
    SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}
    SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}
    SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
    : > "$VCOORD"; n=0
    for x in $(seq 0 $((SX-1))); do for y in $(seq 0 $((SY-1))); do for z in $(seq 0 $((SZ-1))); do
        [ "$x,$y,$z" = "$EXCLUDE" ] && continue
        echo "($x,$y,$z)" >> "$VCOORD"; n=$((n+1))
        [ "$n" -ge "$NP" ] && break 3
    done; done; done
    [ "$n" -lt "$NP" ] && { echo "shape minus $EXCLUDE has $n < NP=$NP" >&2; exit 1; }
    MPI_PLACE="-vcoordfile $VCOORD"
    echo "[g4tp] placing $NP ranks via $VCOORD (excl $EXCLUDE)"
else
    echo "[g4tp] using scheduler placement for $NP ranks (EXCLUDE=none)"
fi

OFP=""
[ -n "${MPIEXEC_OF_PROC:-}" ] && OFP="-of-proc $MPIEXEC_OF_PROC"

echo "[g4tp] building tofu_topo_helper + stager + runner..."
make -C "$UTOFU_DIR" tofu_topo_helper >/dev/null
fcc -Nclang -O2 -D_GNU_SOURCE -I../../common "$HERE/gemma4_stage.c" -o "$HERE/gemma4_stage"
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
    -I../../common "$HERE/gemma4_tp_runner.c" -lm -lpthread -lhwb -ltofucom \
    -o "$HERE/gemma4_tp_runner"

echo "[g4tp] generating tofu topo ($TOFU_TOPO_PATH)..."
mpiexec -np "$NP" $MPI_PLACE $OFP "$UTOFU_DIR/tofu_topo_helper"

if [ "${SKIP_STAGE:-0}" != "1" ]; then
    echo "[g4tp] staging TP shards to $STAGE_DIR (NP=$NP)..."
    mpiexec -np "$NP" $MPI_PLACE $OFP sh -c \
        "mkdir -p $STAGE_DIR; exec $HERE/gemma4_stage $GGUF $STAGE_DIR \$PMIX_RANK $NP tp"
fi

PROMPT_ARG=""; [ -n "$PROMPT_IDS" ] && PROMPT_ARG="$PROMPT_IDS"
echo "[g4tp] running TP pipeline (maxgen=$MAXGEN mtp=${GEMMA4_TP_MTP:-off})..."
mpiexec -np "$NP" $MPI_PLACE $OFP "$HERE/gemma4_tp_runner" \
    "$GGUF" "$STAGE_DIR" "$PROMPT_ARG" "$MAXGEN"
