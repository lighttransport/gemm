#!/bin/bash
# Single-node, four-CMG native-Q8 decode for Qwen3.8-27B on A64FX.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
mkdir -p "$ROOT/tmp/fcc"
export TMPDIR=${TMPDIR:-$ROOT/tmp/fcc}
MODEL=${1:-/home/u14346/models/qwen38/27b/Qwen3.8-27B-Q8_0.gguf}
if [ "$#" -gt 0 ]; then shift; fi
STAGE_DIR=${Q38_CMG4_STAGE_DIR:-/local/u14346/qwen38-cmg4}
MPG=${Q38_CMG4_MPG:-/opt/FJSVxos/mmm/lib64/libmpg.so.1}

if [ ! -r "$MODEL" ]; then
    echo "qwen38-cmg4: model not readable: $MODEL" >&2
    exit 2
fi
if [ ! -r "$MPG" ]; then
    echo "qwen38-cmg4: Fugaku hugepage library not readable: $MPG" >&2
    exit 2
fi

case "$MODEL" in
    /local/*)
        LOCAL_MODEL=$MODEL
        echo "qwen38-cmg4: verified staged source $LOCAL_MODEL"
        ;;
    *)
        "$HERE/stage_gguf_shards.sh" "$MODEL" "$STAGE_DIR"
        LOCAL_MODEL="$STAGE_DIR/$(basename "$MODEL")"
        ;;
esac

case "$(readlink -f "$LOCAL_MODEL")" in
    /local/*) ;;
    *) echo "qwen38-cmg4: refusing non-local staged path: $LOCAL_MODEL" >&2; exit 2 ;;
esac

make -C "$HERE" qwen38_runner CC=fcc OPENMP=1

export LD_PRELOAD=$MPG
export XOS_MMM_L_HPAGE_TYPE=hugetlbfs
export XOS_MMM_L_HUGETLB_SZ=2M
export XOS_MMM_L_HUGE_MALLOC=1
export XOS_MMM_L_FORCE_MMAP_THRESHOLD=1
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export XOS_MMM_L_ARENA_FREE=2
export XOS_MMM_L_HUGETLB_FALLBACK=0
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export NUMA_DISTRIBUTE=1
export NUMA_N_CMGS=4
export NUMA_CMG_BUDGET_GB=7
export NUMA_ALIGNMENT=2097152
export TF_CMG4_STRICT=1
export TF_LOAD_KEEPCACHE=0
export TF_NO_PANEL=1
export TF_KV_DTYPE=${TF_KV_DTYPE:-f16}
export TF_HIER_BARRIER=${TF_HIER_BARRIER:-0}
export TF_BARRIER_BUSY_WAIT=${TF_BARRIER_BUSY_WAIT:-1}

exec numactl --physcpubind=12-59 --membind=4-7 \
    "$HERE/build/qwen38_runner" "$LOCAL_MODEL" --threads 48 \
    --q8-mode cmg4 "$@"
