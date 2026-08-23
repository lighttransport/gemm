#!/bin/bash
# Stage and benchmark exact Qwen3.8-27B Q8_0 on two to four A64FX nodes.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/home/u14346/models/qwen38/27b/Qwen3.8-27B-Q8_0.gguf}
TP_SIZE=${TP_SIZE:-4}
MODE=${1:-plan}

# Reproducible accepted profiles.  Apply these before deriving the stage name:
# bf16-mtp needs the separately staged, TP-sharded NextN block.
case "$MODE" in
    bf16-bench)
        export TP_Q8_EXPAND_BF16=${TP_Q8_EXPAND_BF16:-1}
        ;;
    bf16-mtp)
        export TP_NEXTN_SHARD=${TP_NEXTN_SHARD:-1}
        export TP_Q8_EXPAND_BF16=${TP_Q8_EXPAND_BF16:-1}
        export TP_Q8_EXPAND_NEXTN_MASK=${TP_Q8_EXPAND_NEXTN_MASK:-53}
        export TF_BF16PV_PREFETCH=${TF_BF16PV_PREFETCH:-12}
        ;;
esac
case "$TP_SIZE" in 2|3|4) ;; *) echo "TP_SIZE must be 2, 3, or 4" >&2; exit 2 ;; esac
NEXTN_SUFFIX=
if [ "${TP_NEXTN_SHARD:-0}" != 0 ]; then
    [ "$TP_SIZE" != 3 ] || { echo "TP_NEXTN_SHARD requires TP_SIZE=2 or 4" >&2; exit 2; }
    NEXTN_SUFFIX=-nextnshard
fi
STAGE=${TP_STAGE_DIR:-/local/u14346/qwen38-q8-tp${TP_SIZE}${NEXTN_SUFFIX}}

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/bin"
export TP_STAGE_DIR=$STAGE LLM_THREADS=${LLM_THREADS:-48} OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
export PJM_MPI_PROC=$TP_SIZE
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread} OMP_PLACES=${OMP_PLACES:-cores}
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-active}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-1} OMP_DYNAMIC=${OMP_DYNAMIC:-false}
export TP_MTP_OMP_PARK=${TP_MTP_OMP_PARK:-1}
export NUMA_DISTRIBUTE=${NUMA_DISTRIBUTE:-1} NUMA_N_CMGS=${NUMA_N_CMGS:-4}
export NUMA_CMG_BUDGET_GB=${NUMA_CMG_BUDGET_GB:-7} NUMA_ALIGNMENT=${NUMA_ALIGNMENT:-2097152}
unset GGUF_LAZY_MMAP TF_FORCE_MMAP
export TF_LOAD_KEEPCACHE=0
export TF_NO_PANEL=1 TP_STAGE_BF16_PV=0
export TF_HIER_BARRIER=${TF_HIER_BARRIER:-0}
export TF_BF16PV_PREFETCH=${TF_BF16PV_PREFETCH:-8}
export TF_BF16PV_PREFETCH_MTP2=${TF_BF16PV_PREFETCH_MTP2:-12}
export TF_SSM_FUSED_DOTS=${TF_SSM_FUSED_DOTS:-1}
export TF_SILU_SVE=${TF_SILU_SVE:-1}
# Optional resident W8A8 conversion: row or block64.  An unset/empty value
# keeps the exact Q8_0 bytes already loaded from the rank stage.
# TP_Q8_EXPAND_BF16=1 instead expands the staged Q8 trunk into anonymous,
# pair-interleaved BF16 decode weights.  Its 3 GB HBM reserve guard is tunable
# with TP_Q8_BF16_RESERVE_GB, but never allowed below 2 GB.
# The accepted MTP profile expands NextN mask 53 (EH, attention output, FFN
# down, and LM head).  Other masks are diagnostic and may change draft tokens.
# TP_NEXTN_SHARD=1 requires a stage created with the same setting and trades
# two extra draft all-reduces for quarter-sized NextN attention/FFN weights.
if [ "${TP_Q8_EXPAND_BF16:-0}" != 0 ]; then
    export TP_Q8_EXPAND_NEXTN_MASK=${TP_Q8_EXPAND_NEXTN_MASK:-53}
fi
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
    stream|null|check|bench|mtp-check|profile|bf16-bench|bf16-mtp)
        make tp_runner CC=fcc OPENMP=1
        make -C ../utofu-tests tofu_topo_helper >/dev/null
        mpiexec -np "$TP_SIZE" ../utofu-tests/tofu_topo_helper
        export TP_SYNTH_TOKEN_ID=1
        export TP_PREFILL_GEMM=0 TP_CACHE_LOAD=0 TP_CACHE_SAVE=0
        case "$MODE" in
            stream) unset TF_NULL_GEMM; export TP_SPEC_K=0 TP_NULL_STREAM_PASSES=${TP_NULL_STREAM_PASSES:-10} TP_MAXGEN=1 ;;
            null) export TF_NULL_GEMM=1 TP_SPEC_K=0 TP_MAXGEN=${TP_MAXGEN:-32} ;;
            check) unset TF_NULL_GEMM; export TP_SPEC_K=0 TP_MAXGEN=${TP_MAXGEN:-1} TP_DUMP_TOKENS=1 ;;
            bench) unset TF_NULL_GEMM; export TP_SPEC_K=0 TP_MAXGEN=${TP_MAXGEN:-64} TP_PERF_WARMUP=${TP_PERF_WARMUP:-32} TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1 ;;
            bf16-bench)
                unset TF_NULL_GEMM
                export TP_SPEC_K=0 TP_MAXSEQ=${TP_MAXSEQ:-512}
                export TP_MAXGEN=${TP_MAXGEN:-32} TP_PERF_WARMUP=${TP_PERF_WARMUP:-8}
                export TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
            mtp-check)
                unset TF_NULL_GEMM
                export TP_SPEC_K=${TP_SPEC_K:-5} TP_MTP_BATCH=${TP_MTP_BATCH:-1}
                if [ -n "${TP_Q8_VERIFY:-}" ]; then export TP_Q8_VERIFY; fi
                export TP_MAXGEN=${TP_MAXGEN:-64} TP_PERF_WARMUP=${TP_PERF_WARMUP:-0}
                export TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
            bf16-mtp)
                [ "$TP_SIZE" = 4 ] || { echo "bf16-mtp is validated only with TP_SIZE=4" >&2; exit 2; }
                unset TF_NULL_GEMM
                export TP_SPEC_K=${TP_SPEC_K:-4} TP_MTP_BATCH=${TP_MTP_BATCH:-1}
                export TP_MAXSEQ=${TP_MAXSEQ:-512} TP_MAXGEN=${TP_MAXGEN:-64}
                export TP_PERF_WARMUP=${TP_PERF_WARMUP:-0} TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
            profile) unset TF_NULL_GEMM; export TP_SPEC_K=0 TF_DPROF=1 TP_MAXGEN=${TP_MAXGEN:-32} TP_PERF_WARMUP=${TP_PERF_WARMUP:-16} TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1 ;;
        esac
        export TP_MAXSEQ=${TP_MAXSEQ:-128}
        rm -f tp_run_*.txt tp_load_rank*.txt tp_perf_rank*.txt tp_stderr_rank*.txt tp_tokens_rank00.txt
        exec mpiexec -np "$TP_SIZE" ./build/tp_runner "$MODEL"
        ;;
    *) echo "usage: TP_SIZE={2|3|4} $0 {plan|stage|stream|null|check|bench|mtp-check|profile|bf16-bench|bf16-mtp}" >&2; exit 2 ;;
esac
