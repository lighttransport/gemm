#!/bin/bash
# Canonical GLM-5.3 Flash native A64FX launcher; see README.md.
set -euo pipefail
fail() { echo "error: $*" >&2; exit 2; }
usage() {
    printf '%s\n' \
        'Usage: bash a64fx/glm5/run_glm53f_12n.sh COMMAND [ARGS]' \
        '  run [TOKEN STEPS OPTIONS...]          build, stage, decode (default)' \
        '  build [runtime|check|all]             build only (default: check)' \
        '  stage                               stage/reuse this allocation weights' \
        '  decode [TOKEN=1 STEPS=128 OPTIONS...] reuse binaries and stages' \
        '  generate PROMPT_IDS OUTPUT_IDS N [OPTIONS...]' \
        '  benchmark PROMPT_IDS OUTPUT_IDS [OPTIONS...] snapshot-reset warm trials' \
        '  executor-check PROMPT_IDS [N=32]      compare full legacy/persistent state' \
        '  check                               build and run kernel/component/full-model checks' \
        'Defaults: Q4 native, 12 ranks, 47 threads; GLM53F_BUILD=0 skips run/check builds.' \
        'Paths: GLM53F_GGUF, GLM53F_MODEL_DIR, GLM53F_BIN_DIR, GLM53F_LOG_DIR.' \
        'Generation/prefill options are passed unchanged to the C runner. See README.md.'
}
command=${1:-run}
[ "$#" = 0 ] || shift
case "$command" in
    -h|--help|help) usage; exit 0 ;;
    run|build|stage|decode|generate|benchmark|executor-check|check) ;;
    *) usage >&2; fail "unknown command: $command" ;;
esac
source "$(dirname "$0")/scripts/glm53f_env.sh"
source "$glm53f_dir/scripts/glm53f_launch.sh"
build() { bash "$glm53f_dir/build_glm53f_integrated_12n.sh" "$1"; }
if [ "$command" = build ]; then
    [ "$#" -le 1 ] || fail 'build accepts one mode'
    build "${1:-check}"
    exit
fi
require_allocation
case "$command" in stage|check) [ "$#" = 0 ] || fail "$command takes no arguments";; esac
if [ "$command" = executor-check ]; then
    [ "$#" -ge 1 ] && [ "$#" -le 2 ] || fail 'executor-check requires PROMPT_IDS [N]'
    [ -s "$1" ] || fail "missing or empty prompt IDs: $1"
    count=${2:-32}
    [[ "$count" =~ ^[0-9]+$ ]] && [ "$count" -ge 1 ] && [ "$count" -le 512 ] || fail 'N must be 1..512'
fi
if [ "$command" = generate ] || [ "$command" = benchmark ]; then
    if [ "$command" = benchmark ]; then
        [ "$#" -ge 2 ] || fail 'benchmark requires PROMPT_IDS OUTPUT_IDS'
        [ ! -e "$2" ] || fail "benchmark output already exists: $2"
    else
        [ "$#" -ge 3 ] || fail 'generate requires PROMPT_IDS OUTPUT_IDS N'
        [[ "$3" =~ ^[0-9]+$ ]] && [ "$3" -ge 1 ] && [ "$3" -le 32768 ] || fail 'N must be 1..32768'
    fi
    [ -s "$1" ] || fail "missing or empty prompt IDs: $1"
    [ "$(realpath -m "$1")" != "$(realpath -m "$2")" ] || fail 'prompt and output paths must differ'
elif [ "$command" = run ] || [ "$command" = decode ]; then
    token=${1:-${GLM53F_TARGET_INPUT_TOKEN:-1}}
    steps=${2:-${GLM53F_TARGET_STEPS:-128}}
    [[ "$token" =~ ^[0-9]+$ ]] && [ "$token" -lt 154880 ] || fail 'TOKEN must be 0..154879'
    [[ "$steps" =~ ^[0-9]+$ ]] && [ "$steps" -ge 1 ] && [ "$steps" -le 2048 ] || fail 'STEPS must be 1..2048'
fi
mkdir -p "$logdir" "$GLM53F_BUILD_DIR"
if [ "$command" = run ] || [ "$command" = check ]; then
    if [ "${GLM53F_BUILD:-1}" = 1 ]; then
        if [ "$command" = check ]; then build check; else build runtime; fi
    fi
fi
if [ "$command" = run ] || [ "$command" = stage ]; then
    source "$glm53f_dir/scripts/glm53f_stage.sh"
    stage_weights
    [ "$command" != stage ] || exit 0
fi
prepare_runtime
if [ "$command" = executor-check ]; then
    [ -x "$GLM53F_BIN_DIR/glm53f_executor_check_12n" ] || fail 'executor check missing; build check or all first'
    mpi_run executor-check "$GLM53F_BIN_DIR/glm53f_executor_check_12n" "$model" "$routed" "$shared" "$1" "$logdir/executor-state-$run_tag" "$count"
    grep 'GLM53F_EXECUTOR_CHECK' "$last_log".*.0
    grep -q 'GLM53F_EXECUTOR_CHECK .*state=BIT_EXACT PASS' "$last_log".*.0 || fail 'executor state check failed'
    exit
fi
if [ "$command" = benchmark ]; then
    export GLM53F_PREWARM=${GLM53F_PREWARM:-1}
    [ -x "$GLM53F_BIN_DIR/bench_glm53f_run_12n" ] || fail 'benchmark executable missing; build check or all first'
    {
        printf 'job=%s quant=%s native=%s source_rev=%s\n' "$job" "$quant" "$native" "${GLM53F_BENCH_SOURCE_REV:-unrecorded}"
        printf 'arguments: '; printf '%q ' "$@"; printf '\n'
        sha256sum "$GLM53F_BIN_DIR/bench_glm53f_run_12n" "$1" "$topo_path"
        for key in GLM53F_PREWARM GLM53F_NUMA_INTERLEAVE GLM53F_IQ_FAST GLM53F_MHC_FAST \
            GLM53F_NATIVE_Q8_PANEL GLM53F_NATIVE_Q8_ROWS8 GLM53F_NATIVE_Q8_TILE2X8 GLM53F_MLA_FUSED_PROJECTION GLM53F_VERIFY_GROUPED GLM53F_MHC_BATCH_TEAM GLM53F_KDA_DECODE_COLUMNS GLM53F_KDA_PREFILL_COLUMNS GLM53F_MOE_GU_PAD GLM53F_ROUTER_FUSE GLM53F_DECODE_EXECUTOR \
            GLM53F_COMM_OWNER GLM53F_MOE_COMBINE GLM53F_INDEX_HEADS GLM53F_MLA_REGISTERS GLM53F_POOL_PARTITION_4K GLM53F_KDA_ASYNC GLM53F_SPARSE_ASYNC GLM53F_MTNI_DECODE \
            OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES FLIB_BARRIER XOS_MMM_L_PAGING_POLICY XOS_MMM_L_HPAGE_TYPE; do
            printf '%s=%s\n' "$key" "${!key-}"
        done
    } > "$logdir/benchmark-metadata-$run_tag.txt"
    mpi_run benchmark "$GLM53F_BIN_DIR/bench_glm53f_run_12n" "$model" "$routed" "$shared" "$@"
    grep '^GLM53F_BENCH_' "$last_log".*.0
    grep -q '^GLM53F_BENCH_COMPLETE .*"status":"PASS"' "$last_log".*.0 || fail 'benchmark did not complete'
    exit
fi
if [ "$command" = check ]; then
    for test in kquant native_batch moe_layout prefill_config team mhc_team mhc_batch kda_columns iq_scale_words lookup moe_combine index_heads index_keys4 pool_select mla_absorb mla_value mla_cache_f16 mla_attention mla_softmax; do "$GLM53F_BIN_DIR/test_glm53f_$test"; done
    for required in 1 0; do "$GLM53F_BIN_DIR/test_glm53f_repack_policy" "$logdir" "$required"; done
    for iq_mode in 0 1; do GLM53F_IQ_FAST=$iq_mode "$GLM53F_BIN_DIR/test_glm53f_iq_grouped"; done
    mpi_run check-lookup "$GLM53F_BIN_DIR/test_glm53f_lookup_spec"
    grep 'GLM53F_LOOKUP_SPEC PASS' "$last_log".*.0
    "$GLM53F_BIN_DIR/test_glm53f_state_io" "$logdir"
    export GLM53F_KDA_BATCH_TEAM=1 GLM53F_KDA_WIDE_TILE=1 GLM53F_KDA_PREFILL=1 GLM53F_SPARSE_BATCH_OP=1
    # The KDA exact gates compare against the sequential decode path: pin the faster (non bit-identical) prefill paths off;
    # they are validated separately by GLM53F_KDA_GEMM=2 and end-to-end token comparison.
    export GLM53F_KDA_GEMM=0 GLM53F_KDA_CONV_VEC=0 GLM53F_KDA_NATIVE_COLUMN=0 GLM53F_KDA_ASYNC=0
    mpi_run check-kda "$GLM53F_BIN_DIR/glm53f_kda_callback_check" "$model" 44 32
    # Tests reduce their status across ranks and return nonzero on failure.
    grep 'PASS' "$last_log".*.0
    for layer in 0 1 2; do
        mpi_run "check-dense-$layer" "$GLM53F_BIN_DIR/glm53f_dense_batch_check" "$model" "$layer"
        grep 'PASS' "$last_log".*.0
    done
    # Exact-path gate (per-token reference MLA), then the register-blocked batched MLA against the same reference.
    # The batched MLA is not bit-identical (a 1e-8 difference in the value accumulation can flip a Q8 tie), so it is
    # gated with an explicit tolerance instead of exactness.
    # Q8 projection GEMM changes accumulation/quantization relative to decode.
    # Disable it and fused decode projection in both MLA checks so they isolate
    # the MLA implementation; production GEMM is checked by generated IDs.
    (
        export GLM53F_SPARSE_GEMM=0 GLM53F_SPARSE_FUSE_FRONT=0 GLM53F_SPARSE_MLA_BATCH=0
        mpi_run check-sparse "$GLM53F_BIN_DIR/glm53f_sparse_batch_check" "$model" 3 2046 32
        grep 'GLM53F_SPARSE_BATCH .* PASS' "$last_log".*.0
    )
    (
        export GLM53F_SPARSE_GEMM=0 GLM53F_SPARSE_FUSE_FRONT=0 GLM53F_SPARSE_MLA_BATCH=1 GLM53F_SPARSE_CHECK_TOL=2e-4
        mpi_run check-sparse-mla-batch "$GLM53F_BIN_DIR/glm53f_sparse_batch_check" "$model" 3 2046 32
        grep 'GLM53F_SPARSE_BATCH .* PASS' "$last_log".*.0
    )
    export GLM53F_CHECK_WIDE_PREFILL=1
    # The exact scalar-vs-batch gates compare against the per-token decode kernels; the grouped native MoE path is
    # numerically different (closer to exact fp32), so it is validated separately with GLM53F_MOE_NATIVE_GROUPED=2.
    # The MoE combine reduction order (GLM53F_MOE_AR_SLAB) is likewise pinned to the decode collective here; the faster
    # collectives are validated by bench_glm53f_allreduce_12n (result check) and end-to-end generation.
    export GLM53F_MOE_NATIVE_GROUPED=0 GLM53F_SPARSE_MLA_BATCH=0 GLM53F_MOE_ROUTER_GEMM=0 GLM53F_MOE_SHARED_GEMM=0 GLM53F_MOE_AR_SLAB=0 GLM53F_IQ_FAST=0 GLM53F_MTNI_DECODE=0 GLM53F_MHC_FAST=0 GLM53F_MOE_FUSE_SHARED=0 GLM53F_KDA_ASYNC=0 GLM53F_SPARSE_GEMM=0 GLM53F_SPARSE_FUSE_FRONT=0
    mpi_run check-target "$GLM53F_BIN_DIR/glm53f_target_batch_check_12n" "$model" "$routed" "$shared" \
        --prefill-mode fast --prefill-features 27 --prefill-slab 16 --prefill-collective tree-packed
    grep 'PASS' "$last_log".*.0
    echo "SENTINEL glm53f_check_12n=OK logdir=$logdir"
    exit
fi
if [ "$command" = generate ]; then
    export GLM53F_PREWARM=${GLM53F_PREWARM:-1} # build the prefill GEMM panel copies at load time, not inside the first prefill
    mpi_run generate "$GLM53F_BIN_DIR/glm53f_target_decode_12n" "$model" "$routed" "$shared" --generate "$@"
else
    if [ "$#" -ge 2 ]; then shift 2; elif [ "$#" = 1 ]; then shift; fi
    mpi_run decode "$GLM53F_BIN_DIR/glm53f_target_decode_12n" "$model" "$routed" "$shared" "$token" "$steps" "$@"
fi
grep -E 'GLM53F_TARGET_(TIMING|DECODE_WINDOW|DECODE_12N|GENERATE_12N|RUN_MEMORY)' "$last_log".*.0
grep -Eq 'GLM53F_TARGET_(DECODE|GENERATE)_12N .* PASS' "$last_log".*.0 || fail 'missing completion record'
if [ "$command" != generate ]; then
    rate=$(sed -n 's/.*GLM53F_TARGET_DECODE_12N .* tok_s=\([0-9.]*\).*/\1/p' "$last_log".*.0)
    awk -v rate="$rate" -v target="${GLM53F_MIN_TOK_S:-20}" 'BEGIN { exit !(rate+0 >= target+0) }' || fail 'decode throughput gate failed'
fi
echo "SENTINEL glm53f_run_12n=OK quant=$quant native=$native logdir=$logdir"
