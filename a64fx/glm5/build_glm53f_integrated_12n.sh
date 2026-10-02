#!/bin/bash
# Shared native build. "all" retains the historical developer tool set.
set -euo pipefail
mode=${1:-all}
case "$mode" in runtime|check|all) ;; *) echo "usage: $0 [runtime|check|all] [32|47|48|64] [512|1024|2048|4096]" >&2; exit 2;; esac
attention_panel=${2:-32}
case "$attention_panel" in 32|47|48|64) ;; *) echo "error: attention panel must be 32, 47, 48 or 64" >&2; exit 2;; esac
prefill_capacity=${3:-512}
case "$prefill_capacity" in 512|1024|2048|4096) ;; *) echo "error: prefill capacity must be 512, 1024, 2048 or 4096" >&2; exit 2;; esac
[ "$#" -le 3 ] || { echo "error: too many build arguments" >&2; exit 2; }
cd "$(dirname "$0")"
build_dir=${GLM53F_BUILD_DIR:-/local/glm53f-build-${PJM_JOBID:-manual}}
if [ ! -d /local ]; then build_dir=${GLM53F_BUILD_DIR:-../../tmp/glm53f-build}; fi
bin_dir=${GLM53F_BIN_DIR:-.}
mkdir -p "$build_dir" "$bin_dir"
export TMPDIR="$build_dir"

if [ -n "${GLM53F_MPICC:-}" ]; then
    cc=$GLM53F_MPICC
elif command -v mpiclang >/dev/null 2>&1; then
    cc=mpiclang
elif command -v mpiFCC >/dev/null 2>&1 && mpiFCC --showme:compile >/dev/null 2>&1; then
    cc=mpiFCC
elif command -v mpicc >/dev/null 2>&1; then
    cc=mpicc
else
    echo "error: set GLM53F_MPICC to a working MPI C wrapper" >&2
    exit 2
fi
cflags=("-DGLM53F_PREFILL_ATTN_PANEL=$attention_panel" "-DGLM53F_PREFILL_CAPACITY=$prefill_capacity" -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -Wall -Wextra -I. -I../../common)
case "$(basename "$cc")" in mpifcc|mpiFCC) cflags=(-Nclang "${cflags[@]}");; esac
[ "${GLM53F_FAST_MATH:-0}" != 1 ] || cflags+=(-ffast-math)
[ "${GLM53F_NO_MATH_ERRNO:-0}" != 1 ] || cflags+=(-fno-math-errno)
[ "${GLM53F_MHC_FUSED:-0}" != 1 ] || cflags+=(-DGLM53F_MHC_FUSED=1)
[ "${GLM53F_MHC_POST_FLOAT:-0}" != 1 ] || cflags+=(-DGLM53F_MHC_POST_FLOAT=1)
[ -z "${GLM53F_MOE_FUSED_WEIGHTED:-}" ] || cflags+=("-DGLM53F_MOE_FUSED_WEIGHTED=$GLM53F_MOE_FUSED_WEIGHTED")
external=(-DGLM53F_EXTERNAL_ST_IMPLEMENTATION)
obj() {
    local name=$1 source=$2
    shift 2
    "$cc" "${cflags[@]}" "$@" -c "$source" -o "$build_dir/$name.o"
}
bin() {
    local name=$1
    shift
    "$cc" "${cflags[@]}" "$@" -lm -lpthread -ltofucom -o "$bin_dir/$name"
}
obj collective glm53f_collective_12n.c
obj team glm53f_team.c
obj kda glm53f_kda_layer_12n.c -DGLM53F_KDA_NO_MAIN
obj sparse glm53f_sparse_layer_12n.c "${external[@]}" -DGLM53F_SPARSE_NO_MAIN
obj dense glm53f_dense_ffn_12n.c "${external[@]}" -DGLM53F_DENSE_NO_MAIN
obj moe glm53f_expert_decode_12n.c "${external[@]}" -DGLM53F_EXPERT_NO_MAIN
obj iq_bridge glm53f_iq_bridge.c
obj q8_panel kern/glm53f_kern_q8r16.c
obj gemm_kern kern/glm53f_kern_gemm.c
"$cc" "${cflags[@]}" -c kern/glm53f_kern_gemm_asm.S -o "$build_dir/gemm_asm.o"
obj head glm53f_target_head_12n.c "${external[@]}" -DGLM53F_TARGET_HEAD_NO_MAIN
obj embedding glm53f_embedding_12n.c "${external[@]}"
objects=("$build_dir/q8_panel.o" "$build_dir/gemm_kern.o" "$build_dir/gemm_asm.o")
for name in team collective kda sparse dense moe iq_bridge head embedding; do
    objects+=("$build_dir/$name.o")
done
kernels=("$build_dir/team.o" "$build_dir/iq_bridge.o" "$build_dir/q8_panel.o" "$build_dir/collective.o" "$build_dir/gemm_kern.o" "$build_dir/gemm_asm.o")
bin glm53f_target_decode_12n glm53f_target_decode_12n.c "${objects[@]}"
for name in q2_stage q2_embed_stage q2_dense_stage q2_sparse_stage q2_kda_stage \
            q2_shexp_stage q2_core_patch q2_shared_patch core_stage core_add_routers; do
    bin "glm53f_$name" "glm53f_$name.c"
done
bin glm53f_q2_head_stage '-DGLM53F_Q2_MATRIX_NAME="output.weight"' \
    '-DGLM53F_Q2_STAGE_LABEL="glm53f_q2_head_stage"' \
    '-DGLM53F_Q2_STAGE_MANIFEST="GLM53F_Q2_HEAD_V1"' glm53f_q2_embed_stage.c
bin tofu_topo_helper ../utofu-tests/tofu_topo_helper.c
# Older launchers expect the topology helper beside its source.
if [ "$bin_dir" = . ]; then bin ../utofu-tests/tofu_topo_helper ../utofu-tests/tofu_topo_helper.c; fi

if [ "$mode" != runtime ]; then
    obj target glm53f_target_decode_12n.c -DGLM53F_TARGET_MODEL_NO_MAIN
    bin test_glm53f_team test_glm53f_team.c "$build_dir/team.o"
    bin bench_glm53f_async_reduce bench_glm53f_async_reduce.c "$build_dir/collective.o"
    obj lookup_spec glm53f_lookup_spec_12n.c
    obj mtp glm53f_mtp_12n.c "${external[@]}"
    obj mtp_spec glm53f_mtp_spec_12n.c
    bin test_glm53f_lookup_spec test_glm53f_lookup_spec.c "$build_dir/lookup_spec.o"
    bin test_glm53f_mtp_spec test_glm53f_mtp_spec.c
    bin test_glm53f_head_hidden -ffunction-sections -fdata-sections -Wl,--gc-sections test_glm53f_head_hidden.c
    bin test_glm53f_mtp_cache_12n test_glm53f_mtp_cache_12n.c "${objects[@]}" "$build_dir/mtp.o"
    bin bench_glm53f_run_12n bench_glm53f_run_12n.c "$build_dir/lookup_spec.o" "$build_dir/mtp_spec.o" "$build_dir/mtp.o" "${objects[@]}" "$build_dir/target.o"
    bin test_glm53f_moe_layout test_glm53f_moe_layout.c "$build_dir/gemm_asm.o"
    bin test_glm53f_repack_policy test_glm53f_repack_policy.c
    for name in kquant native_batch prefill_config state_io iq_grouped mhc_team mhc_sync mhc_batch kda_columns lookup moe_combine index_heads index_keys4 pool_select mla_absorb mla_value mla_cache_f16 mla_attention; do
        bin "test_glm53f_$name" "test_glm53f_$name.c" "$build_dir/q8_panel.o" "$build_dir/team.o" "$build_dir/gemm_asm.o"
    done
    bin glm53f_kda_callback_check glm53f_kda_callback_check.c "$build_dir/kda.o" "${kernels[@]}"
    bin glm53f_dense_batch_check glm53f_dense_batch_check.c "$build_dir/dense.o" "$build_dir/kda.o" "${kernels[@]}"
    bin glm53f_sparse_batch_check glm53f_sparse_batch_check.c "$build_dir/sparse.o" "$build_dir/kda.o" "${kernels[@]}"
    bin glm53f_target_batch_check_12n glm53f_target_batch_check_12n.c "${objects[@]}" "$build_dir/target.o"
    bin glm53f_executor_check_12n glm53f_executor_check_12n.c "${objects[@]}" "$build_dir/target.o"
    bin glm53f_prefill_chunk_check_12n glm53f_prefill_chunk_check_12n.c "${objects[@]}"
fi
if [ "$mode" = all ]; then
    bin glm53f_decode_stage glm53f_decode_stage.c
    bin glm53f_repack_trace glm53f_repack_trace.c
    bin glm53f_spec_decode_12n glm53f_spec_decode_12n.c "${objects[@]}" "$build_dir/target.o" "$build_dir/mtp.o"
    bin glm53f_prefill_12n glm53f_prefill_12n.c "${objects[@]}" "$build_dir/target.o"
    bin test_glm53f_kda_reference_12n test_glm53f_kda_reference_12n.c "$build_dir/kda.o" "${kernels[@]}"
    bin test_glm53f_iq_bridge test_glm53f_iq_bridge.c "$build_dir/iq_bridge.o" "$build_dir/q8_panel.o"
    for name in glm53f_sparse_native_stage_probe_12n test_glm53f_sparse_reference_12n; do
        bin "$name" "$name.c" "$build_dir/sparse.o" "${kernels[@]}"
    done
    bin test_glm53f_sparse_prefix test_glm53f_sparse_prefix.c "$build_dir/sparse.o" "$build_dir/kda.o" "${kernels[@]}"
    bin glm53f_q8_stage glm53f_q8_stage.c
    obj q8_resident glm53f_q8_resident.c
    bin test_glm53f_q8_resident test_glm53f_q8_resident.c "$build_dir/q8_resident.o"
    for name in int8 int8_batch kda_prefill prefill_gemm mla_prefill mhc_prefill \
                index_score mhc_chain mhc_scalar_real; do
        bin "test_glm53f_$name" "test_glm53f_$name.c"
    done
    bin glm53f_mtp_expert_check glm53f_mtp_expert_check.c
    for name in sparse_prefill sparse_math sparse_cp; do
        bin "test_glm53f_$name" "test_glm53f_$name.c" "${kernels[@]}"
    done
    bin test_glm53f_collective_payload_12n test_glm53f_collective_payload_12n.c "$build_dir/collective.o"
    bin test_glm53f_quant_model_12n test_glm53f_quant_model_12n.c "${objects[@]}" "$build_dir/target.o"
    bin glm53f_expert_decode_12n glm53f_expert_decode_12n.c "${kernels[@]}"
fi
echo "SENTINEL glm53f_integrated_build_12n=OK mode=$mode cc=$cc bin_dir=$bin_dir"
