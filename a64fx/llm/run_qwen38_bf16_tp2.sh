#!/bin/bash
# Stage and measure Qwen3.8-27B BF16 on exactly two A64FX nodes.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MODEL=${MODEL:-/home/u14346/models/qwen38/27b/bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf}
STAGE=${TP_STAGE_DIR:-/local/u14346/qwen38-bf16-tp2}
MODE=${1:-stage}

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/bin"
export TP_STAGE_DIR=$STAGE LLM_THREADS=${LLM_THREADS:-48} OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=${OMP_PLACES:-cores}
export NUMA_DISTRIBUTE=${NUMA_DISTRIBUTE:-1} NUMA_N_CMGS=${NUMA_N_CMGS:-4}
export NUMA_CMG_BUDGET_GB=${NUMA_CMG_BUDGET_GB:-7} NUMA_ALIGNMENT=${NUMA_ALIGNMENT:-2097152}
export GGUF_LAZY_MMAP=1 TF_FORCE_MMAP=1 TF_LOAD_KEEPCACHE=0

cd "$HERE"
case "$MODE" in
    stage)
        make qwen38_tp_stage CC=fcc OPENMP=1
        exec mpiexec -np 2 ./build/qwen38_tp_stage "$MODEL" "$STAGE"
        ;;
    stream|null|check|source-check|bench)
        make tp_runner CC=fcc OPENMP=1
        make -C ../utofu-tests tofu_topo_helper >/dev/null
        mpiexec -np 2 ../utofu-tests/tofu_topo_helper
        export TP_SYNTH_TOKENS=1 TP_SYNTH_TOKEN_ID=1 TP_MAXSEQ=${TP_MAXSEQ:-64}
        export TP_PREFILL_GEMM=0 TP_SPEC_K=0 TP_CACHE_LOAD=0 TP_CACHE_SAVE=0 TF_NO_PANEL=1
        case "$MODE" in
            stream)
                unset TF_NULL_GEMM
                export TP_NULL_STREAM_PASSES=${TP_NULL_STREAM_PASSES:-3} TP_MAXGEN=1
                ;;
            null) export TF_NULL_GEMM=1 TP_MAXGEN=${TP_MAXGEN:-32} ;;
            check) unset TF_NULL_GEMM; export TP_MAXGEN=${TP_MAXGEN:-1} TP_DUMP_TOKENS=1 ;;
            source-check)
                # Correctness-only control: use the original lazy GGUF tensors and
                # the runner's in-memory column repacks, bypassing staged shards.
                unset TP_STAGE_DIR TF_NULL_GEMM
                export TP_MAXGEN=${TP_MAXGEN:-1} TP_DUMP_TOKENS=1
                ;;
            bench) unset TF_NULL_GEMM; export TP_MAXGEN=${TP_MAXGEN:-64} TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1 ;;
        esac
        rm -f tp_run_*.txt tp_load_rank*.txt tp_perf_rank*.txt tp_stderr_rank*.txt tp_tokens_rank00.txt
        exec mpiexec -np 2 ./build/tp_runner "$MODEL"
        ;;
    *) echo "usage: $0 {stage|stream|null|check|source-check|bench}" >&2; exit 2 ;;
esac
