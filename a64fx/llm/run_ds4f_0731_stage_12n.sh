#!/bin/bash
# Stage DeepSeek-V4-Flash-0731 real weights across all 12 nodes.
#
# Run inside an existing 12-node interactive allocation.  The rank/node order
# is shared with run_ds4f_0731_12n.sh; the interactive/login node is placed at
# EP rank 11 so it owns the smaller 21-expert shard.
set -euo pipefail

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"
UTOFU_DIR="$LLM_DIR/../utofu-tests"
cd "$LLM_DIR"

NP=${NP:-12}
LAST=${LAST:-0,0,0}
VCOORD=${VCOORD:-vcoord_ds4f_0731_12n.txt}
MODEL_DIR=${DS4F_MODEL_DIR:-$HOME/models/ds4f-0731}
JOB_TAG=${PJM_JOBID:-manual-$$}
STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f-0731-${PJM_JOBID:-manual}}

if [ "${PJM_MPI_PROC:-$NP}" -ne "$NP" ]; then
    echo "expected a ${NP}-process interactive allocation (PJM_MPI_PROC=${PJM_MPI_PROC:-unset})" >&2
    exit 2
fi
for ((s=1; s<=48; s++)); do
    f=$(printf '%s/model-%05d-of-00048.safetensors' "$MODEL_DIR" "$s")
    [ -s "$f" ] || { echo "missing weight shard: $f" >&2; exit 3; }
done

export DS4F_MODEL=""
export DS4F_MODEL_DIR="$MODEL_DIR"
export DS4F_STAGE_DIR="$STAGE_DIR"
export DS4F_EP_SIZE="$NP"
export DS4F_NSHARDS=48
export DS4F_STAGE_FLUSH_GB=${DS4F_STAGE_FLUSH_GB:-1}
export DS4F_STATUS_DIR="$LLM_DIR"

SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}
SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}
SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
: > "$VCOORD"
for ((x=0; x<SX; x++)); do
    for ((y=0; y<SY; y++)); do
        for ((z=0; z<SZ; z++)); do
            [ "$x,$y,$z" = "$LAST" ] || echo "($x,$y,$z)" >> "$VCOORD"
        done
    done
done
echo "($LAST)" >> "$VCOORD"
head -n "$NP" "$VCOORD" > "$VCOORD.tmp"
mv "$VCOORD.tmp" "$VCOORD"

echo "=== DS4F-0731 stage: ${NP} nodes, 48 shards ==="
echo "model=$MODEL_DIR"
echo "stage=$STAGE_DIR"
echo "vcoord=$VCOORD (last=$LAST -> EP rank $((NP-1)))"
cat -n "$VCOORD"

make -C "$LLM_DIR" ds4f_stage CC=fcc >/dev/null
rm -f ds4f_stage_rank*.txt
mpiexec -np "$NP" -vcoordfile "$VCOORD" build/ds4f_stage

done_count=$(find "$LLM_DIR" -maxdepth 1 -name 'ds4f_stage_rank*.txt' -type f | wc -l)
cat ds4f_stage_rank*.txt 2>/dev/null | sort || true
[ "$done_count" -eq "$NP" ] || {
    echo "stage incomplete: ${done_count}/${NP} ranks reported done" >&2
    exit 4
}
echo "DS4F_0731_STAGE_PASS ranks=$done_count stage=$STAGE_DIR"
