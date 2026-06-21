#!/bin/bash
# DeepSeek-V4-Flash sharded weight stager — fan out across the 11 EP nodes.
#
# Each node independently reads its EP-shard of the 46 safetensors shards from
# the shared filesystem and writes a packed blob + manifest to its node-local
# /local (fast scratch). No inter-node comm: embarrassingly parallel. The EP
# rank is taken from PMIX_RANK, which equals the runner's EP rank because the
# SAME vcoordfile pins PMIX_RANK r -> the same physical node that the runner's
# tofu_topo.txt assigns EP rank r. Run BEFORE run_ds4f_11n.sh (real weights).
#
# The stager streams file->file (bounded RSS, a few MB) so it does NOT load
# weights into memory and will NOT OOM a co-located login/claude session — but
# we still exclude relative-(0,0,0) by default so the staged node set matches
# the harness's 11-node set exactly (run_ds4f_11n.sh uses the same exclusion).
#
# Run INSIDE the existing allocation (NO pjsub):
#   ./run_ds4f_stage_11n.sh
#   DS4F_MODEL_DIR=/path/to/ds4f ./run_ds4f_stage_11n.sh
#   NP=12 EXCLUDE=none ./run_ds4f_stage_11n.sh   # stage the WHOLE alloc
set -e
# Prepend (do NOT replace) so mpiexec AND the native compiler stay on PATH.
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"

NP=${NP:-11}
EXCLUDE=${EXCLUDE:-0,0,0}           # relative coord to drop; "none" keeps full alloc
VCOORD=${VCOORD:-vcoord_ds4f.txt}   # SAME file run_ds4f_11n.sh uses (rank<->node binding)

# ---- harness knobs forwarded to every rank (mpiexec forwards EXPORTED env) ----
export DS4F_EP_SIZE=${DS4F_EP_SIZE:-$NP}
# DS4F_MODEL=ds4p selects the DeepSeek-V4-Pro config so the stager shards its 384 experts
# (forward to the stager ranks; empty = Flash). Pair with DS4F_MODEL_DIR/DS4F_NSHARDS=64.
export DS4F_MODEL=${DS4F_MODEL:-}
export DS4F_MODEL_DIR=${DS4F_MODEL_DIR:-$HOME/models/ds4f}
export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f}
export DS4F_NSHARDS=${DS4F_NSHARDS:-46}
# Bound the HBM /local page cache while writing the ~22 GB blob. Without this the
# blob's dirty pages pile up in HBM (the ~7 GB "/local caching" that OOM-segfaulted
# the prior 11-node stage); the stager fdatasync+fadvise(DONTNEED)s every FLUSH_GB
# and DONTNEEDs each source shard -> peak staging HBM ~= FLUSH_GB + one shard (well
# under the ~11 GB/node target). Lower it if a node is still tight.
export DS4F_STAGE_FLUSH_GB=${DS4F_STAGE_FLUSH_GB:-2}
export DS4F_STATUS_DIR="$LLM_DIR"   # per-rank DONE files land on the shared FS

# ---- generate the vcoordfile (all shape coords except $EXCLUDE) ----
MPI_PLACE=()
if [ "$EXCLUDE" != "none" ]; then
    SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}
    SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}
    SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
    : > "$VCOORD"
    n=0
    for ((x=0; x<SX; x++)); do for ((y=0; y<SY; y++)); do for ((z=0; z<SZ; z++)); do
        [ "$x,$y,$z" = "$EXCLUDE" ] && continue
        echo "($x,$y,$z)" >> "$VCOORD"
        n=$((n+1))
    done; done; done
    if [ "$n" -lt "$NP" ]; then
        echo "shape ${SX}x${SY}x${SZ} minus ($EXCLUDE) = $n nodes < NP=$NP" >&2; exit 1
    fi
    head -n "$NP" "$VCOORD" > "$VCOORD.tmp" && mv "$VCOORD.tmp" "$VCOORD"
    MPI_PLACE=(-vcoordfile "$VCOORD")
    echo "[stage] excluding relative ($EXCLUDE); staging $NP nodes via $VCOORD:"
    cat "$VCOORD" | sed 's/^/    /'
else
    echo "[stage] EXCLUDE=none: staging the whole alloc (-np $NP, no vcoordfile)"
fi

echo "=== DS4F sharded stage on $NP node(s) ==="
echo "model=$DS4F_MODEL_DIR  out=$DS4F_STAGE_DIR  ep_size=$DS4F_EP_SIZE  shards=$DS4F_NSHARDS"

# ---- build the stager (native fcc) ----
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE \
    -I../../common -o build/ds4f_stage ds4f_stage.c
BIN="$LLM_DIR/build/ds4f_stage"

# ---- clear stale per-rank status from any prior run ----
rm -f ds4f_stage_rank*.txt

# ---- fan out: each rank self-identifies via PMIX_RANK and stages its shard ----
echo "--- launching ds4f_stage (NP=$NP) — this reads ~25 GB/node from shared FS ---"
t0=$(date +%s)
mpiexec -np "$NP" "${MPI_PLACE[@]}" "$BIN"
t1=$(date +%s)

# ---- aggregate the per-rank DONE files (mpiexec drops stdout) ----
echo "=== per-rank stage status ($((t1-t0)) s wall) ==="
done=$(ls ds4f_stage_rank*.txt 2>/dev/null | wc -l)
# Per-rank stage lines are byte-identical except rank#/blob path; DS4F_STAGE_COMPACT=1
# prints only the count, keeping interactive sessions lean (full files on disk).
if [ "${DS4F_STAGE_COMPACT:-0}" = "1" ]; then
    echo "--- $done/$NP ranks reported DONE (compact; per-rank files: $LLM_DIR/ds4f_stage_rank*.txt) ---"
else
    cat ds4f_stage_rank*.txt 2>/dev/null | sort
    echo "--- $done/$NP ranks reported DONE ---"
fi
if [ "$done" -ne "$NP" ]; then
    echo "WARNING: only $done/$NP ranks finished — check node-local logs" >&2
    exit 1
fi
echo "OK: all $NP ranks staged to $DS4F_STAGE_DIR"
