#!/bin/bash
# DeepSeek-V4 BASE (~/models/ds4fbase) sharded weight stager — fan out across ALL 12 nodes.
#
# Sibling of run_ds4f_stage_11n.sh. Two things differ:
#
#  1. ALL 12 nodes are EP ranks (interactive is capped at 12), so the node running the
#     login/claude session hosts a rank too. We do NOT exclude it — we put it LAST in the
#     vcoordfile so it becomes EP rank 11. That matters: 256 experts % 12 means ranks 0-3
#     own 22 experts and ranks 4-11 own 21, so parking the shared node at rank 11 saves it
#     ~1.0 GB (43 layers x 24 MiB). The stager itself streams file->file (RSS a few MB), so
#     staging cannot OOM the session; the placement is for the RUN that follows.
#
#  2. The base model is 275 GB (vs Flash's 149 GB) because its experts are FP8-e4m3 rather
#     than MXFP4. Each node still reads every shard and keeps only its slice, so expect
#     ~30.5 GB/node staged (fits the 87 GB /local) and roughly 2x the Flash stage time.
#
# The stager needs NO base-specific flag: it keys off the source dtype. ds4fbase stores every
# *.scale as F32 (exact powers of two, scale_fmt="ue8m0"); ds4f_stage.c folds those losslessly
# to the E8M0 bytes the loader/kernels already consume. It reports the fold count when done.
#
# Run INSIDE the existing 12-node allocation (NO pjsub):
#   ./run_ds4fbase_stage_12n.sh
#
# Then: ./run_ds4fbase_12n.sh   (same vcoordfile => same rank<->node binding)
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"

NP=${NP:-12}
LAST=${LAST:-0,0,0}                       # coord to place LAST (the login/claude node -> rank 11)
VCOORD=${VCOORD:-vcoord_ds4fbase.txt}     # SAME file run_ds4fbase_12n.sh uses

export DS4F_EP_SIZE=${DS4F_EP_SIZE:-$NP}
export DS4F_MODEL=${DS4F_MODEL:-ds4fbase}
export DS4F_MODEL_DIR=${DS4F_MODEL_DIR:-$HOME/models/ds4fbase}
export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4fbase}
export DS4F_NSHARDS=${DS4F_NSHARDS:-46}
# Bound the /local dirty page cache in HBM while writing the ~30 GB blob. Lower than the
# ds4f default (2) because one of these nodes is also hosting the interactive session.
export DS4F_STAGE_FLUSH_GB=${DS4F_STAGE_FLUSH_GB:-1}
export DS4F_STATUS_DIR="$LLM_DIR"

# ---- vcoordfile: every coord in the shape, with $LAST pushed to the end (=> EP rank NP-1) ----
# tofu_topo_helper takes its rank from MPI_Comm_rank and the runner does ep_rank = MyRank,
# so vcoordfile LINE ORDER *is* the EP rank order.
SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}
SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}
SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
: > "$VCOORD"
n=0
for ((x=0; x<SX; x++)); do for ((y=0; y<SY; y++)); do for ((z=0; z<SZ; z++)); do
    [ "$x,$y,$z" = "$LAST" ] && continue
    echo "($x,$y,$z)" >> "$VCOORD"; n=$((n+1))
done; done; done
echo "($LAST)" >> "$VCOORD"; n=$((n+1))    # the shared node goes last -> fewest experts
if [ "$n" -lt "$NP" ]; then
    echo "shape ${SX}x${SY}x${SZ} = $n nodes < NP=$NP" >&2; exit 1
fi
head -n "$NP" "$VCOORD" > "$VCOORD.tmp" && mv "$VCOORD.tmp" "$VCOORD"
echo "[stage] placing $NP ranks via $VCOORD (rank order = line order; ($LAST) last = rank $((NP-1))):"
cat -n "$VCOORD" | sed 's/^/    /'

echo "=== DS4F-BASE sharded stage on $NP node(s) ==="
echo "model=$DS4F_MODEL_DIR  out=$DS4F_STAGE_DIR  ep_size=$DS4F_EP_SIZE  shards=$DS4F_NSHARDS"

fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE \
    -I../../common -o build/ds4f_stage ds4f_stage.c
BIN="$LLM_DIR/build/ds4f_stage"

rm -f ds4f_stage_rank*.txt

echo "--- launching ds4f_stage (NP=$NP) — each node reads ~275 GB from the shared FS ---"
t0=$(date +%s)
mpiexec -np "$NP" -vcoordfile "$VCOORD" "$BIN"
t1=$(date +%s)

echo "=== per-rank stage status ($((t1-t0)) s wall) ==="
done=$(ls ds4f_stage_rank*.txt 2>/dev/null | wc -l)
cat ds4f_stage_rank*.txt 2>/dev/null | sort
echo "--- $done/$NP ranks reported DONE ---"
if [ "$done" -ne "$NP" ]; then
    echo "WARNING: only $done/$NP ranks finished — check node-local logs" >&2
    exit 1
fi
echo "OK: all $NP ranks staged to $DS4F_STAGE_DIR"
