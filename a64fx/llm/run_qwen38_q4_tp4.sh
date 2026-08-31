#!/bin/bash
# Stage and benchmark the mixed Q4_K_XL Qwen3.8-27B model on A64FX TP4.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/home/u14346/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf}
TP_SIZE=${TP_SIZE:-4}
MODE=${1:-plan}

case "$TP_SIZE" in 2|4) ;; *) echo "Q4 TP_SIZE must be 2 or 4" >&2; exit 2 ;; esac
NEXTN_SUFFIX=
if [ "${TP_NEXTN_SHARD:-0}" != 0 ]; then NEXTN_SUFFIX=-nextnshard; fi
STAGE=${TP_STAGE_DIR:-/local/u14346/qwen38-q4-tp${TP_SIZE}${NEXTN_SUFFIX}}

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/bin"
export TP_STAGE_DIR=$STAGE PJM_MPI_PROC=$TP_SIZE
export LLM_THREADS=${LLM_THREADS:-48} OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread} OMP_PLACES=${OMP_PLACES:-cores}
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-active} OMP_DYNAMIC=${OMP_DYNAMIC:-false}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-1}
export NUMA_DISTRIBUTE=${NUMA_DISTRIBUTE:-1} NUMA_N_CMGS=${NUMA_N_CMGS:-4}
export TP_COMM_CMG_STRICT=${TP_COMM_CMG_STRICT:-1}
export NUMA_CMG_BUDGET_GB=${NUMA_CMG_BUDGET_GB:-7} NUMA_ALIGNMENT=${NUMA_ALIGNMENT:-2097152}
export TF_LOAD_KEEPCACHE=0 TF_NO_PANEL=1 TP_STAGE_BF16_PV=0
export TF_HIER_BARRIER=${TF_HIER_BARRIER:-0}
export TF_SSM_FUSED_DOTS=${TF_SSM_FUSED_DOTS:-1} TF_SILU_SVE=${TF_SILU_SVE:-1}
export TP_MTP_OMP_PARK=${TP_MTP_OMP_PARK:-1}
unset GGUF_LAZY_MMAP TF_FORCE_MMAP

if [ "${TP_NEXTN_SHARD:-0}" != 0 ]; then
    export TP_AR_A2A=${TP_AR_A2A:-1} TP_AR_A2A_MAX=${TP_AR_A2A_MAX:-8192}
fi

cd "$HERE"
case "$MODE" in
    plan|stage)
        make qwen38_tp_stage CC=fcc OPENMP=1
        if [ "$MODE" = plan ]; then export Q38TP_PLAN=1; fi
        exec mpiexec -np "$TP_SIZE" ./build/qwen38_tp_stage "$MODEL" "$STAGE"
        ;;
    check|bench|mtp-check|mtp-bench|profile)
        if [ "$MODE" = profile ]; then
            make tp_runner_prof CC=fcc OPENMP=1
            RUNNER=./build/tp_runner_prof
        else
            make tp_runner CC=fcc OPENMP=1
            RUNNER=./build/tp_runner
        fi
        make -C ../utofu-tests tofu_topo_helper >/dev/null
        mpiexec -np "$TP_SIZE" ../utofu-tests/tofu_topo_helper
        export TP_SYNTH_TOKEN_ID=1 TP_PREFILL_GEMM=0 TP_CACHE_LOAD=0 TP_CACHE_SAVE=0
        unset TF_NULL_GEMM
        case "$MODE" in
            check)
                export TP_SPEC_K=0 TP_MAXGEN=${TP_MAXGEN:-1} TP_DUMP_TOKENS=1
                ;;
            bench|profile)
                export TP_SPEC_K=0 TP_MAXGEN=${TP_MAXGEN:-256}
                export TP_PERF_WARMUP=${TP_PERF_WARMUP:-32} TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
            mtp-check)
                export TP_SPEC_K=${TP_SPEC_K:-4} TP_MTP_BATCH=${TP_MTP_BATCH:-1}
                export TP_MAXGEN=${TP_MAXGEN:-32} TP_PERF_WARMUP=0 TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
            mtp-bench)
                export TP_SPEC_K=${TP_SPEC_K:-4} TP_MTP_BATCH=${TP_MTP_BATCH:-1}
                export TP_MAXGEN=${TP_MAXGEN:-256} TP_PERF_WARMUP=${TP_PERF_WARMUP:-32}
                export TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
        esac
        export TP_MAXSEQ=${TP_MAXSEQ:-512}
        rm -f tp_run_*.txt tp_load_rank*.txt tp_perf_rank*.txt tp_stderr_rank*.txt tp_tokens_rank00.txt
        exec mpiexec -np "$TP_SIZE" "$RUNNER" "$MODEL"
        ;;
    *)
        echo "usage: TP_SIZE={2|4} $0 {plan|stage|check|bench|mtp-check|mtp-bench|profile}" >&2
        exit 2
        ;;
esac
