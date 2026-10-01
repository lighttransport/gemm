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
        '  check                               build and run kernel/component/full-model checks' \
        'Defaults: Q4 native, 12 ranks, 47 threads; GLM53F_BUILD=0 skips run/check builds.' \
        'Paths: GLM53F_GGUF, GLM53F_MODEL_DIR, GLM53F_BIN_DIR, GLM53F_LOG_DIR.' \
        'Generation/prefill options are passed unchanged to the C runner. See README.md.'
}
command=${1:-run}
[ "$#" = 0 ] || shift
case "$command" in
    -h|--help|help) usage; exit 0 ;;
    run|build|stage|decode|generate|check) ;;
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
if [ "$command" = generate ]; then
    [ "$#" -ge 3 ] || fail 'generate requires PROMPT_IDS OUTPUT_IDS N'
    [ -s "$1" ] || fail "missing or empty prompt IDs: $1"
    [ "$(realpath -m "$1")" != "$(realpath -m "$2")" ] || fail 'prompt and output paths must differ'
    [[ "$3" =~ ^[0-9]+$ ]] && [ "$3" -ge 1 ] && [ "$3" -le 32768 ] || fail 'N must be 1..32768'
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
if [ "$command" = check ]; then
    for test in kquant native_batch prefill_config; do "$GLM53F_BIN_DIR/test_glm53f_$test"; done
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
    (export GLM53F_SPARSE_MLA_BATCH=0; mpi_run check-sparse "$GLM53F_BIN_DIR/glm53f_sparse_batch_check" "$model" 3 2046 32)
    grep 'PASS' "$last_log".*.0
    (export GLM53F_SPARSE_MLA_BATCH=1 GLM53F_SPARSE_CHECK_TOL=2e-4; mpi_run check-sparse-mla-batch "$GLM53F_BIN_DIR/glm53f_sparse_batch_check" "$model" 3 2046 32)
    grep 'PASS' "$last_log".*.0
    export GLM53F_CHECK_WIDE_PREFILL=1
    # The exact scalar-vs-batch gates compare against the per-token decode kernels; the grouped native MoE path is
    # numerically different (closer to exact fp32), so it is validated separately with GLM53F_MOE_NATIVE_GROUPED=2.
    # The MoE combine reduction order (GLM53F_MOE_AR_SLAB) is likewise pinned to the decode collective here; the faster
    # collectives are validated by bench_glm53f_allreduce_12n (result check) and end-to-end generation.
    export GLM53F_MOE_NATIVE_GROUPED=0 GLM53F_SPARSE_MLA_BATCH=0 GLM53F_MOE_ROUTER_GEMM=0 GLM53F_MOE_SHARED_GEMM=0 GLM53F_MOE_AR_SLAB=0 GLM53F_IQ_FAST=0 GLM53F_MTNI_DECODE=0 GLM53F_MHC_FAST=0 GLM53F_MOE_FUSE_SHARED=0 GLM53F_KDA_ASYNC=0
    mpi_run check-target "$GLM53F_BIN_DIR/glm53f_target_batch_check_12n" "$model" "$routed" "$shared" \
        --prefill-mode fast --prefill-features 27 --prefill-slab 16 --prefill-collective tree-packed
    grep 'PASS' "$last_log".*.0
    echo "SENTINEL glm53f_check_12n=OK logdir=$logdir"
    exit
fi
if [ "$command" = generate ]; then
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
