#!/bin/bash
# Stage and measure Qwen3.8-27B BF16 with TP4/TP6/TP12.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/home/u14346/models/qwen38/27b/bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf}
TP_SIZE=${TP_SIZE:-4}
MODE=${1:-stage}
# The persistent-QKV proposer makes TP-sharding the NextN block profitable:
# reduced projection traffic now outweighs its two extra reductions per draft.
# Keep replicated NextN available explicitly for the historical control.
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
# Async K4 reserves local core slots 9..11 in every CMG for the 12-thread
# NextN pool.  Order the remaining places by CMG so static verifier row ranges
# retain the same row-quarter ownership as the staged weight first-touch.
if [ "${TP_MTP_ASYNC:-0}" != 0 ]; then
    export TP_MTP_VERIFY_THREADS=${TP_MTP_VERIFY_THREADS:-36}
    export TP_MTP_SHADOW_THREADS=${TP_MTP_SHADOW_THREADS:-12}
    export TP_MTP_SHADOW_STRIPED=${TP_MTP_SHADOW_STRIPED:-1}
    export TP_MTP_SHADOW_CORE_OFFSET=${TP_MTP_SHADOW_CORE_OFFSET:-9}
    # NextN executes on its private pthread pool.  Do not let its small SiLU
    # pass recursively launch a verifier-sized OpenMP team from the background.
    export TF_SILU_OMP=0
    export OMP_PROC_BIND=close
    export OMP_PLACES=${TP_MTP_VERIFY_PLACES:-\
'{12},{13},{14},{15},{16},{17},{18},{19},{20},{24},{25},{26},{27},{28},{29},{30},{31},{32},{36},{37},{38},{39},{40},{41},{42},{43},{44},{48},{49},{50},{51},{52},{53},{54},{55},{56}'}
fi
# Keep the verifier's many small OpenMP regions hot, then explicitly park those
# workers before the pthread-based native NextN pool runs.  Zero block time
# repeatedly sleeps/wakes the matrix teams; active wait without parking
# oversubscribes all 48 cores during draft generation.
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-active}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-1} OMP_DYNAMIC=${OMP_DYNAMIC:-false}
export TP_MTP_OMP_PARK=${TP_MTP_OMP_PARK:-1}
# Sequential drafting uses a private runtime context and persistent pthread
# pool while sharing immutable staged weights.  This avoids reusing the trunk
# context from inside the OpenMP parking region; it added no weight copy and
# improved the adjacent exact K=5 run by 11%.
if [ "$MODE" = mtp-sustained ]; then
    export TP_MTP_SHADOW_THREADS=${TP_MTP_SHADOW_THREADS:-48}
    # Give every verifier worker one 256-byte-aligned attention-score slice
    # and reuse it across all attention layers in the K=5 batch.  This avoids
    # 768 contended malloc/free pairs per speculative round while preserving
    # first-touch CMG ownership of every worker-private slice.
    export TF_BATCH_ATTN_SCORE_ARENA=${TF_BATCH_ATTN_SCORE_ARENA:-1}
    # Reuse one 256-byte-partitioned verifier scratch arena across rounds.
    # Stable addresses preserve first-touch ownership and avoid eleven aligned
    # allocation/free pairs per K=5 verification call.
    export TF_BATCH_SCRATCH_REUSE=${TF_BATCH_SCRATCH_REUSE:-1}
    # Keep the shadow workers active from attention output through gate/up,
    # SiLU, and down.  This removes two pool wakes and the nested OpenMP
    # activation region per draft step while preserving the exact BF16 rows.
    export TF_NEXTN_FFN_PERSIST=${TF_NEXTN_FFN_PERSIST:-1}
    export TF_NEXTN_BLOCK_PERSIST=${TF_NEXTN_BLOCK_PERSIST:-1}
    export TF_NEXTN_FULL_PERSIST=${TF_NEXTN_FULL_PERSIST:-1}
    export TF_NEXTN_ATTN_BLOCK_PERSIST=${TF_NEXTN_ATTN_BLOCK_PERSIST:-1}
    # Keep the same workers alive through QKV and distribute exact Q/K norm +
    # RoPE by head, avoiding the formerly serial preparation between pools.
    export TF_NEXTN_QKV_PERSIST=${TF_NEXTN_QKV_PERSIST:-1}
    export TF_BF16PV_PREFETCH_NEXTN=${TF_BF16PV_PREFETCH_NEXTN:-8}
    # The compact 4x5 verifier has five activation vectors competing with its
    # twenty accumulators. Distance 12 keeps its two weight streams ahead on
    # A64FX without the cache pressure measured at the old distance 16.
    export TF_BF16PV_PREFETCH_MTP5=${TF_BF16PV_PREFETCH_MTP5:-12}
    # K=6 remains a fallback, but its compact 2x6 kernel benefits from a
    # shorter distance than the historical 16 when explicitly selected.
    export TF_BF16PV_PREFETCH_MTP6=${TF_BF16PV_PREFETCH_MTP6:-8}
    # Fit three rows by six verifier tokens in the A64FX SVE register file.
    # Two 3x6 calls plus a 2x6 tail retain the exact PV row reductions while
    # reducing K=6 verifier time relative to four separate 2x6 calls.
    export TF_BF16PV_MTP6_3ROW=${TF_BF16PV_MTP6_3ROW:-1}
    # Each persistent head worker scans only the rows it just produced, then
    # the caller folds the 48 cached winners before the unchanged TP argmax.
    # This preserves CMG ownership and avoids a second serial vocabulary scan.
    export TF_NEXTN_LOCAL_ARGMAX=${TF_NEXTN_LOCAL_ARGMAX:-1}
    # Consume each eight-row BF16-PV head result while it is still on the
    # owning worker's stack. Only the worker winner is published; the full
    # local vocabulary slice is no longer stored and reread for greedy draft.
    export TF_NEXTN_INLINE_ARGMAX=${TF_NEXTN_INLINE_ARGMAX:-1}
fi
export NUMA_DISTRIBUTE=${NUMA_DISTRIBUTE:-1} NUMA_N_CMGS=${NUMA_N_CMGS:-4}
export TP_COMM_CMG_STRICT=${TP_COMM_CMG_STRICT:-1}
export NUMA_CMG_BUDGET_GB=${NUMA_CMG_BUDGET_GB:-7} NUMA_ALIGNMENT=${NUMA_ALIGNMENT:-2097152}
# The one-Put decode collective benefits from less frequent trailer-line
# invalidation.  MTP retains its independently validated cadence of eight.
if [ "$MODE" = mtp-sustained ]; then
    export TP_AR_POLL_SPINS=${TP_AR_POLL_SPINS:-8}
else
    export TP_AR_POLL_SPINS=${TP_AR_POLL_SPINS:-16}
fi
# TP_STAGE_DIR is a complete rank-local image.  The runner parses only GGUF
# metadata and explicitly reads the stage into anonymous HBM-resident memory.
unset GGUF_LAZY_MMAP TF_FORCE_MMAP
export TF_LOAD_KEEPCACHE=0
export TF_HIER_BARRIER=${TF_HIER_BARRIER:-0} TF_BF16PV_PREFETCH=${TF_BF16PV_PREFETCH:-6}
export TF_BF16PV_PREFETCH_MTP2=${TF_BF16PV_PREFETCH_MTP2:-12}
export TF_BF16PV_MTP5_FUSED=${TF_BF16PV_MTP5_FUSED:-1}
export TF_SSM_FUSED_DOTS=${TF_SSM_FUSED_DOTS:-1}
# Publish each depthwise-convolution channel slice to its recurrent ring and
# QKV buffer in the owning worker. This removes one global barrier per SSM
# layer without changing arithmetic; set 0 for the exact legacy A/B control.
export TF_SSM_CONV_INLINE_COPY=${TF_SSM_CONV_INLINE_COPY:-1}
# Prepare the six TP4-local Q heads concurrently after QKV projection. Each
# worker retains the exact per-head norm/bias/RoPE order; K/V stay on thread 0.
export TF_ATTN_PREP_HEADS=${TF_ATTN_PREP_HEADS:-1}
# The verifier's BF16 alpha/beta matrices have only 12 rows each. Compute both
# under one OpenMP team while retaining the exact established SVE row reduction.
export TF_SSM_AB_PAIR=${TF_SSM_AB_PAIR:-1}
# Decode-size TP4 reductions are faster as direct peer puts than as two
# recursive-doubling rounds.  Keep larger payloads on the tree and allow an
# explicit TP_AR_A2A=0 for reproducibility/control runs.
if [ "$TP_SIZE" = 4 ]; then
    ar_a2a_max_defaulted=0
    if [ -z "${TP_AR_A2A_MAX+x}" ]; then ar_a2a_max_defaulted=1; fi
    export TP_AR_A2A=${TP_AR_A2A:-1} TP_AR_A2A_MAX=${TP_AR_A2A_MAX:-8192}
    # Use the remote-completion notice as publication evidence and send one
    # payload Put per peer. This is exact for both decode and the K=5 verifier;
    # it avoids each peer's separate eight-byte trailer Put.
    export TP_AR_A2A_MRQ_ONEPUT=${TP_AR_A2A_MRQ_ONEPUT:-1}
fi
# Exact-token validated on both trunk and K=5 MTP.  Avoid scalar expf in the
# verifier and replicated NextN FFN activation passes.
export TF_SILU_SVE=${TF_SILU_SVE:-1}
# Packing only the replicated NextN embedding/hidden fusion projection retains
# the long-context 53/56 agreement and saves about one percent end-to-end.
if [ "${TP_NEXTN_SHARD:-0}" != 0 ]; then
    # EH + attention-output PV are exact under the sharded schedulers.  Q PV
    # changes draft argmaxes; K/V and gate/up cannot satisfy 8-row task bounds.
    export TP_NEXTN_PV_MASK=${TP_NEXTN_PV_MASK:-5}
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
            null) export TF_NULL_GEMM=${TF_NULL_GEMM:-1} TP_MAXGEN=${TP_MAXGEN:-32} ;;
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
                # K=5 is the validated 53.43 tok/s clean-node profile.  Keep
                # the explicit override for acceptance/prompt sweeps.
                export TP_SPEC_K=${TP_SPEC_K:-5} TP_MTP_BATCH=${TP_MTP_BATCH:-1}
                export TP_MAXSEQ=${TP_MAXSEQ:-768} TP_MAXGEN=${TP_MAXGEN:-256}
                export TP_PERF_WARMUP=${TP_PERF_WARMUP:-0} TP_BUFFER_OUTPUT=${TP_BUFFER_OUTPUT:-1}
                export TP_MTP_TRACE=${TP_MTP_TRACE:-0} TP_MTP_PROFILE_DETAIL=${TP_MTP_PROFILE_DETAIL:-1}
                export TP_AR_DETERMINISTIC=${TP_AR_DETERMINISTIC:-1}
                # A one-round peer all-gather folded as (r0+r1)+(r2+r3)
                # exactly reproduces the deterministic TP4 tree for the
                # 25,600-float K=5 residual batches.
                if [ "$TP_SIZE" = 4 ] && [ "$TP_AR_DETERMINISTIC" != 0 ]; then
                    export TP_AR_A2A_TREE=${TP_AR_A2A_TREE:-1}
                    if [ "$ar_a2a_max_defaulted" = 1 ]; then export TP_AR_A2A_MAX=32768; fi
                fi
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
