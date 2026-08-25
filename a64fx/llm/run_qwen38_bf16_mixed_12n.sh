#!/bin/bash
# One native-BF16 resident lifetime: exact PP3xTP4 prefill for three requests,
# in-memory state transpose, in-place PV8 conversion, then independent TP4 K=0 decode.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/home/u14346/models/qwen38/27b/bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf}
STAGE=${TP_STAGE_DIR:-/local/u14346/qwen38-bf16-tp4}
MODE=${1:-bench}
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/bin"
export PJM_MPI_PROC=12 TP_STAGE_DIR=$STAGE
export LLM_THREADS=${LLM_THREADS:-48} OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread} OMP_PLACES=${OMP_PLACES:-cores}
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive} KMP_BLOCKTIME=${KMP_BLOCKTIME:-0}
export OMP_DYNAMIC=${OMP_DYNAMIC:-false}
export NUMA_DISTRIBUTE=${NUMA_DISTRIBUTE:-1} NUMA_N_CMGS=${NUMA_N_CMGS:-4}
export NUMA_CMG_BUDGET_GB=${NUMA_CMG_BUDGET_GB:-7}
export NUMA_ALIGNMENT=${NUMA_ALIGNMENT:-2097152}
export TF_LOAD_KEEPCACHE=0 TF_NO_PANEL=1
export TF_KEEP_BF16_SRC=1 TF_SSM_FUSED_DOTS=${TF_SSM_FUSED_DOTS:-1}
export TF_HIER_BARRIER=${TF_HIER_BARRIER:-0}
export TF_ATTN_SEQ_SPLIT=${TF_ATTN_SEQ_SPLIT:-1}
export TF_SILU_SVE=${TF_SILU_SVE:-1} TF_BF16PV_PREFETCH=${TF_BF16PV_PREFETCH:-8}
unset GGUF_LAZY_MMAP TF_FORCE_MMAP TP_Q8_EXPAND_BF16 TP_Q8_MODE

cd "$HERE"
case "$MODE" in
    stage)
        export Q38_PREFILL_STAGE=$STAGE Q38_PREFILL_TP_SIZE=4 Q38_PREFILL_NODES=12
        exec ./run_qwen38_prefill_12n.sh stage
        ;;
    direct)
        export Q38_MIXED_DIRECT_DECODE=1
        export Q38_MIXED_PREFILL_TOKENS=1 Q38_MIXED_MAXSEQ=${Q38_MIXED_MAXSEQ:-320}
        export Q38_MIXED_MAXGEN=${Q38_MIXED_MAXGEN:-256}
        ;;
    smoke)
        export Q38_MIXED_PREFILL_TOKENS=${Q38_MIXED_PREFILL_TOKENS:-128}
        export Q38_MIXED_CHUNK=${Q38_MIXED_CHUNK:-64}
        export Q38_MIXED_MAXGEN=${Q38_MIXED_MAXGEN:-8}
        ;;
    bench)
        export Q38_MIXED_PREFILL_TOKENS=${Q38_MIXED_PREFILL_TOKENS:-4096}
        export Q38_MIXED_CHUNK=${Q38_MIXED_CHUNK:-256}
        export Q38_MIXED_MAXGEN=${Q38_MIXED_MAXGEN:-256}
        ;;
    *)
        echo "usage: $0 {stage|direct|smoke|bench}" >&2
        exit 2
        ;;
esac

make qwen38_mixed_runner CC=fcc OPENMP=1
make -C ../utofu-tests tofu_topo_helper >/dev/null
mpiexec -np 12 ../utofu-tests/tofu_topo_helper
exec mpiexec -np 12 ./build/qwen38_mixed_runner "$MODEL"
