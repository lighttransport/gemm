#!/bin/bash
# Qwen3.8-27B BF16 single-prompt prefill on twelve A64FX nodes (PP3 x TP4).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/home/u14346/models/qwen38/27b/bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf}
STAGE=${Q38_PREFILL_STAGE:-/local/u14346/qwen38-bf16-tp4}
TP_SIZE=${Q38_PREFILL_TP_SIZE:-4}
NODES=${Q38_PREFILL_NODES:-12}
MODE=${1:-bench}
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/bin"
case "$NODES" in 8|12) ;; *) echo "Q38_PREFILL_NODES must be 8 or 12" >&2; exit 2;; esac
export PJM_MPI_PROC=$NODES LLM_THREADS=${LLM_THREADS:-48}
export OMP_NUM_THREADS=$LLM_THREADS OMP_PROC_BIND=${OMP_PROC_BIND:-spread} OMP_PLACES=${OMP_PLACES:-cores}
export NUMA_DISTRIBUTE=1 NUMA_N_CMGS=4 TF_HIER_BARRIER=0 TF_NO_PANEL=1
export GGUF_LAZY_MMAP=1 TF_FORCE_MMAP=1 TF_LOAD_KEEPCACHE=0
export TF_PREFILL_KEEP_POOL_OFF=${TF_PREFILL_KEEP_POOL_OFF:-1}
export TF_PODD_CMG=${TF_PODD_CMG:-0}
export TF_SSM_FUSED_DOTS=${TF_SSM_FUSED_DOTS:-1}
export TF_SILU_SVE=${TF_SILU_SVE:-1}
export Q38_PREFILL_STAGE=$STAGE
export Q38_PREFILL_TP_SIZE=$TP_SIZE
export Q38_PREFILL_BF16=${Q38_PREFILL_BF16:-exact}
export TP_STAGE_BF16_PV=0
export TF_SOFTMAX_SVE=${TF_SOFTMAX_SVE:-1}
export TF_PODD_FFN_PIPE=${TF_PODD_FFN_PIPE:-0}
if [ "$Q38_PREFILL_BF16" = bf16-act ]; then
    export TF_SSM_SCAN4=${TF_SSM_SCAN4:-1}
    export TF_SSM_FAST_SCALARS=${TF_SSM_FAST_SCALARS:-1}
    export TF_SSM_PREEXP=${TF_SSM_PREEXP:-1}
else
    export TF_SSM_SCAN4=${TF_SSM_SCAN4:-0}
    export TF_SSM_FAST_SCALARS=${TF_SSM_FAST_SCALARS:-0}
    export TF_SSM_PREEXP=${TF_SSM_PREEXP:-0}
fi
if [ -z "${Q38_PREFILL_COMM:-}" ]; then
    if [ "$Q38_PREFILL_BF16" = bf16-act ]; then export Q38_PREFILL_COMM=utofu-i8
    else export Q38_PREFILL_COMM=mpi
    fi
fi
if [ "$Q38_PREFILL_BF16" = bf16-act ]; then
    export Q38_PREFILL_CUT1=${Q38_PREFILL_CUT1:-20}
    export Q38_PREFILL_CUT2=${Q38_PREFILL_CUT2:-42}
    export TF_TP_FUSED_NORM_PACK=${TF_TP_FUSED_NORM_PACK:-1}
    export TF_TP_PACKED_PROJ=${TF_TP_PACKED_PROJ:-1}
fi
if [ -z "${Q38_PREFILL_CHUNK:-}" ]; then
    if [ "$Q38_PREFILL_BF16" = bf16-act ]; then export Q38_PREFILL_CHUNK=252
    else export Q38_PREFILL_CHUNK=256
    fi
fi

case "$TP_SIZE" in 1|4|6|12) ;; *) echo "Q38_PREFILL_TP_SIZE must be 1, 4, 6, or 12" >&2; exit 2;; esac

cd "$HERE"
case "$MODE" in
    plan|stage)
        make qwen38_tp_stage CC=fcc OPENMP=1
        [ "$MODE" = plan ] && export Q38TP_PLAN=1
        # Every physical node builds its TP lane selected by world_rank%TP_SIZE into
        # its own node-local filesystem.  Duplicate logical filenames are safe
        # because /local is not shared between nodes.
        exec mpiexec -np "$NODES" sh -c '
            r=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}}
            export Q38TP_RANK=$((r % $3)) Q38TP_SIZE=$3
            exec ./build/qwen38_tp_stage "$1" "$2"
        ' sh "$MODEL" "$STAGE" "$TP_SIZE"
        ;;
    check|bench|profile)
        make qwen38_prefill_runner CC=fcc OPENMP=1
        case "${Q38_PREFILL_COMM:-mpi}" in
            utofu|utofu-rsag|utofu-bf16|utofu-i8|utofu-tree)
                make -C ../utofu-tests tofu_topo_helper >/dev/null
                mpiexec -np "$NODES" ../utofu-tests/tofu_topo_helper
                ;;
        esac
        rm -f q38_prefill_rank*.txt
        if [ "$MODE" = check ]; then
            export Q38_PREFILL_TOKENS=${Q38_PREFILL_TOKENS:-128}
            export Q38_PREFILL_CHUNK=${Q38_PREFILL_CHUNK:-64}
        elif [ "$MODE" = profile ]; then
            export Q38_PREFILL_TOKENS=${Q38_PREFILL_TOKENS:-1024}
            export Q38_PREFILL_CHUNK=${Q38_PREFILL_CHUNK:-256}
            export TF_PREFILL_PROF=1
        fi
        mpiexec -np "$NODES" ./build/qwen38_prefill_runner "$MODEL"
        cat q38_prefill_rank00.txt
        ;;
    sweep)
        make qwen38_prefill_runner CC=fcc OPENMP=1
        rm -f q38_prefill_rank*.txt
        for n in 128 512 1024 4096; do
            case "$n" in 128) c=64;; 512) c=128;; 4096) c=1024;; *) c=256;; esac
            Q38_PREFILL_TOKENS=$n Q38_PREFILL_CHUNK=$c \
                mpiexec -np "$NODES" ./build/qwen38_prefill_runner "$MODEL"
            cat q38_prefill_rank00.txt
        done
        ;;
    *)
        echo "usage: $0 {plan|stage|check|profile|bench|sweep}" >&2
        exit 2
        ;;
esac
