#!/bin/bash
# Stage and measure Qwen3.8-27B BF16 with TP4/TP6/TP12.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/home/u14346/models/qwen38/27b/bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf}
TP_SIZE=${TP_SIZE:-4}
MODE=${1:-stage}
# The accepted MTP profile uses a separately staged TP-sharded NextN block.
if [ "$MODE" = mtp-sustained ] || [ "$MODE" = stage-mtp ]; then
    [ "$TP_SIZE" = 4 ] || { echo "$MODE requires TP_SIZE=4" >&2; exit 2; }
    export TP_NEXTN_SHARD=${TP_NEXTN_SHARD:-1}
fi
case "$TP_SIZE" in 4|6|12) ;; *) echo "TP_SIZE must be 4, 6, or 12" >&2; exit 2 ;; esac
NEXTN_SUFFIX=
if [ "${TP_NEXTN_SHARD:-0}" != 0 ]; then NEXTN_SUFFIX=-nextnshard; fi
STAGE=${TP_STAGE_DIR:-/local/u14346/qwen38-bf16-tp${TP_SIZE}${NEXTN_SUFFIX}}

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/bin"
export TP_STAGE_DIR=$STAGE LLM_THREADS=${LLM_THREADS:-48} OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
# Fujitsu MPI consults PJM_MPI_PROC when it is present; make it agree with
# this launcher even when the allocation itself contains twelve nodes.
export PJM_MPI_PROC=$TP_SIZE
# Spread workers across the four CMGs.  Close placement can leave rank 0 as a
# compute outlier, charging the other ranks' wait as all-reduce time.  Keep it
# overrideable for topology-specific experiments.
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread} OMP_PLACES=${OMP_PLACES:-cores}
# Keep the verifier's many small OpenMP regions hot, then explicitly park those
# workers before the pthread-based native NextN pool runs.  Zero block time
# repeatedly sleeps/wakes the matrix teams; active wait without parking
# oversubscribes all 48 cores during draft generation.
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-active}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-1} OMP_DYNAMIC=${OMP_DYNAMIC:-false}
export TP_MTP_OMP_PARK=${TP_MTP_OMP_PARK:-1}
export NUMA_DISTRIBUTE=${NUMA_DISTRIBUTE:-1} NUMA_N_CMGS=${NUMA_N_CMGS:-4}
export TP_COMM_CMG_STRICT=${TP_COMM_CMG_STRICT:-1}
export NUMA_CMG_BUDGET_GB=${NUMA_CMG_BUDGET_GB:-7} NUMA_ALIGNMENT=${NUMA_ALIGNMENT:-2097152}
# TP_STAGE_DIR is a complete rank-local image.  The runner parses only GGUF
# metadata and explicitly reads the stage into anonymous HBM-resident memory.
unset GGUF_LAZY_MMAP TF_FORCE_MMAP
export TF_LOAD_KEEPCACHE=0
export TF_HIER_BARRIER=${TF_HIER_BARRIER:-0} TF_BF16PV_PREFETCH=${TF_BF16PV_PREFETCH:-8}
export TF_BF16PV_PREFETCH_MTP2=${TF_BF16PV_PREFETCH_MTP2:-12}
export TF_SSM_FUSED_DOTS=${TF_SSM_FUSED_DOTS:-1}
# Exact-token validated on both trunk and K=5 MTP.  Avoid scalar expf in the
# verifier and replicated NextN FFN activation passes.
export TF_SILU_SVE=${TF_SILU_SVE:-1}
# Packing only the replicated NextN embedding/hidden fusion projection retains
# the long-context 53/56 agreement and saves about one percent end-to-end.
if [ "${TP_NEXTN_SHARD:-0}" != 0 ]; then
    # EH + attention-output PV are exact under the sharded schedulers.  Q PV
    # changes draft argmaxes; K/V and gate/up cannot satisfy 8-row task bounds.
    export TP_NEXTN_PV_MASK=${TP_NEXTN_PV_MASK:-5}
    export TP_AR_A2A=${TP_AR_A2A:-1} TP_AR_A2A_MAX=${TP_AR_A2A_MAX:-8192}
else
    export TP_NEXTN_PV_MASK=${TP_NEXTN_PV_MASK:-1}
fi
# The pair-interleaved layout uses the same low/high accumulation order as the
# row-major kernel and now passes the long greedy-token gate.  Set PV=0 for the
# original source-equivalent layout.
export TP_STAGE_BF16_PV=${TP_STAGE_BF16_PV:-1}

cd "$HERE"
case "$MODE" in
    plan|stage|stage-mtp)
        make qwen38_tp_stage CC=fcc OPENMP=1
        if [ "$MODE" = plan ]; then export Q38TP_PLAN=1; fi
        exec mpiexec -np "$TP_SIZE" ./build/qwen38_tp_stage "$MODEL" "$STAGE"
        ;;
    stream|null|check|source-check|bench|mtp-check|mtp-sustained|prefill|handoff|profile)
        make tp_runner CC=fcc OPENMP=1
        make -C ../utofu-tests tofu_topo_helper >/dev/null
        mpiexec -np "$TP_SIZE" ../utofu-tests/tofu_topo_helper
        export TP_SYNTH_TOKEN_ID=1
        export TP_PREFILL_GEMM=${TP_PREFILL_GEMM:-0}
        export TP_CACHE_LOAD=${TP_CACHE_LOAD:-0} TP_CACHE_SAVE=${TP_CACHE_SAVE:-0} TF_NO_PANEL=1
        case "$MODE" in
            stream)
                unset TF_NULL_GEMM
                export TP_NULL_STREAM_PASSES=${TP_NULL_STREAM_PASSES:-10} TP_MAXGEN=1
                ;;
            null) export TF_NULL_GEMM=1 TP_MAXGEN=${TP_MAXGEN:-32} ;;
            check)
                unset TF_NULL_GEMM
                export TP_AR_DETERMINISTIC=${TP_AR_DETERMINISTIC:-1}
                export TP_STAGE_BF16_PV=${TP_STAGE_BF16_PV_CHECK:-0}
                export TP_MAXGEN=${TP_MAXGEN:-1} TP_DUMP_TOKENS=1
                ;;
            source-check)
                unset TP_STAGE_DIR TF_NULL_GEMM
                export TP_MAXGEN=${TP_MAXGEN:-1} TP_DUMP_TOKENS=1
                ;;
            bench)
                unset TF_NULL_GEMM
                export TP_MAXGEN=${TP_MAXGEN:-64} TP_PERF_WARMUP=${TP_PERF_WARMUP:-32}
                export TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
            mtp-check)
                # Diagnostic only until greedy agreement is nonzero.  This
                # runs the draft head alongside exact trunk decode; it does
                # not count unverified draft tokens as generated tokens.
                unset TF_NULL_GEMM
                export TP_AR_DETERMINISTIC=${TP_AR_DETERMINISTIC:-1}
                export TP_STAGE_BF16_PV=${TP_STAGE_BF16_PV_CHECK:-0}
                export TP_SPEC_K=${TP_SPEC_K:-1} TP_MTP_TRACE=${TP_MTP_TRACE:-1}
                export TP_MAXGEN=${TP_MAXGEN:-8} TP_PERF_WARMUP=0
                export TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
            mtp-sustained)
                unset TF_NULL_GEMM
                export TP_RAW_PROMPT=1
                export TP_PROMPT_FILE=${TP_PROMPT_FILE:-$HERE/qwen38_mtp_prompt.txt}
                export TP_PROMPT_REPEAT=${TP_PROMPT_REPEAT:-2}
                export TP_SPEC_K=${TP_SPEC_K:-2} TP_MTP_BATCH=${TP_MTP_BATCH:-1}
                export TP_MAXSEQ=${TP_MAXSEQ:-768} TP_MAXGEN=${TP_MAXGEN:-256}
                export TP_PERF_WARMUP=${TP_PERF_WARMUP:-0} TP_BUFFER_OUTPUT=${TP_BUFFER_OUTPUT:-1}
                export TP_MTP_TRACE=${TP_MTP_TRACE:-0} TP_MTP_PROFILE_DETAIL=${TP_MTP_PROFILE_DETAIL:-1}
                export TP_AR_DETERMINISTIC=${TP_AR_DETERMINISTIC:-0}
                export TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
            prefill)
                unset TF_NULL_GEMM
                export TP_SYNTH_TOKENS=${TP_SYNTH_TOKENS:-128}
                # Batched GEMM consumes row-major BF16; PV is decode-only.
                export TP_STAGE_BF16_PV=0
                export TP_PREFILL_GEMM=1 TP_PREFILL_ONLY=1 TP_MAXGEN=0 TP_DUMP_TOKENS=0
                export TP_CACHE_SAVE=1 TP_CACHE_SHARED=0
                ;;
            handoff)
                unset TF_NULL_GEMM
                export TP_PREFILL_GEMM=0 TP_CACHE_LOAD=1 TP_CACHE_SAVE=0
                export TP_CACHE_SHARED=0
                export TP_CACHE_AUTOSAVE=0
                export TP_CACHE_REPARTITION_FROM=${TP_CACHE_REPARTITION_FROM:-12}
                export TP_MAXGEN=${TP_MAXGEN:-32} TP_PERF_WARMUP=${TP_PERF_WARMUP:-8}
                export TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
            profile)
                unset TF_NULL_GEMM
                export TF_DPROF=1 TP_MAXGEN=${TP_MAXGEN:-32} TP_PERF_WARMUP=${TP_PERF_WARMUP:-16}
                export TP_IGNORE_EOS=1 TP_DUMP_TOKENS=1
                ;;
        esac
        export TP_MAXSEQ=${TP_MAXSEQ:-128}
        rm -f tp_run_*.txt tp_load_rank*.txt tp_perf_rank*.txt tp_stderr_rank*.txt \
              tp_tokens_rank00.txt tp_null_stream_rank*.txt
        exec mpiexec -np "$TP_SIZE" ./build/tp_runner "$MODEL"
        ;;
    *) echo "usage: TP_SIZE={4|6|12} $0 {plan|stage|stage-mtp|stream|null|check|source-check|bench|mtp-check|mtp-sustained|prefill|handoff|profile}" >&2; exit 2 ;;
esac
