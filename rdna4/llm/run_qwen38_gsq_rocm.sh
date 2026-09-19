#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B GSQ-RCO IQ2_XS (qwen35) launcher for 16-GiB RDNA4 cards.
#
# The optimized profile stores the 16 full-attention K/V pairs as Q8 K/Q4 V.
# On the 16-GiB RX 9070 XT, 53,248 tokens was validated with 538 MiB free
# after model, KV, and decode work-buffer allocation (measured 504 MiB in the
# latest long-context run; leave margin for allocator variation).
# Set QWEN38_GSQ_ALLOW_UNSAFE_CONTEXT=1 only when deliberately testing a
# larger card or a separately patched KV implementation.
#
# Pass --qwen35-prefill-bf16 for the experimental prefill-only BF16 GEMM
# path, including SSM projections. The runner enables batched scheduling
# for this argument; no QWEN38_GSQ_PERF setting is required. Keep --ubatch
# and LLM_BMAX at 512 for the measured 4K profile. Decode dispatch is unchanged.

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# QWEN38_RUNNER_BIN is useful when the launcher is invoked through a mounted
# path whose physical /home alias is non-executable under AMD GPU escalation.
runner_bin="${QWEN38_RUNNER_BIN:-${runner_dir}/test_hip_llm}"
model="${QWEN38_MODEL:-/mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf}"
vram_profile="${QWEN38_VRAM_PROFILE:-16g}"
safe_context="${QWEN38_GSQ_SAFE_CONTEXT:-53248}"
allow_unsafe="${QWEN38_GSQ_ALLOW_UNSAFE_CONTEXT:-0}"

[[ -r "${model}" ]] || {
    echo "Qwen3.8 GSQ model not found: ${model}" >&2
    echo "Set QWEN38_MODEL to the GGUF path." >&2
    exit 2
}
[[ "${safe_context}" =~ ^[1-9][0-9]*$ ]] || {
    echo "QWEN38_GSQ_SAFE_CONTEXT must be a positive integer" >&2
    exit 2
}
case "${vram_profile}" in
    16g) ;;
    24g|32g) safe_context="${QWEN38_GSQ_SAFE_CONTEXT:-262144}" ;;
    *) echo "unknown QWEN38_VRAM_PROFILE=${vram_profile} (use 16g, 24g, or 32g)" >&2; exit 2 ;;
esac

args=()
requested_context=0
bench_depth=0
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    if [[ "${arg}" == "-s" || "${arg}" == "--max-seq-len" ]]; then
        j=$((i + 1))
        if (( j <= $# )); then
            requested_context="${!j}"
            args+=("${arg}" "${!j}")
            ((i++))
            continue
        fi
    elif [[ "${arg}" == "--bench-depth" ]]; then
        j=$((i + 1))
        if (( j <= $# )); then
            bench_depth="${!j}"
            args+=("${arg}" "${!j}")
            ((i++))
            continue
        fi
    fi
    args+=("${arg}")
done
if (( requested_context == 0 )); then
    requested_context="${safe_context}"
    args+=("-s" "${requested_context}")
fi
[[ "${requested_context}" =~ ^[1-9][0-9]*$ ]] || {
    echo "context after -s/--max-seq-len must be a positive integer" >&2
    exit 2
}
selected_context="${requested_context}"
selected_safe_context="${safe_context}"
# Q8/Q8 random-depth decode was measured with 66,560 allocated rows and
# 4.4 GiB still free for IQ2 on the 16-GiB profile. Keep the normal serving
# default conservative while allowing the explicit 64K benchmark through.
if [[ "${bench_depth}" =~ ^[1-9][0-9]*$ ]] && (( bench_depth <= 65536 )); then
    selected_safe_context=66560
fi
if [[ "${vram_profile}" == "16g" && "${allow_unsafe}" == "0" ]] &&
   (( requested_context > selected_safe_context )); then
    echo "Qwen3.8 GSQ: clamping context ${requested_context} to ${selected_safe_context} for the 16-GiB profile" >&2
    selected_context="${selected_safe_context}"
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[i]}" == "-s" || "${args[i]}" == "--max-seq-len" ]]; then
            args[i+1]="${selected_safe_context}"
            break
        fi
    done
fi

perf_profile="${QWEN38_GSQ_PERF:-0}"
# The performance profile uses BF16 staging for every IQ2 projection.  This
# is the path that clears 300 tok/s at 32K on gfx1201; normal mode remains the
# parity-oriented path, and either knob can still be explicitly set to 0.
fast_prefill="${QWEN38_GSQ_FAST_PREFILL:-${perf_profile}}"
fast_all_iq2="${QWEN38_GSQ_FAST_ALL_IQ2:-${perf_profile}}"
fast_iq2_max_layer="${QWEN38_GSQ_FAST_IQ2_MAX_LAYER:-${LLM_QWEN35_IQ2_BF16_MAX_LAYER:-}}"
fast_prefill_pins=""
if [[ "${fast_prefill}" != "0" ]]; then
    # gfx1201's first hipBLASLt heuristic is not consistently the fastest for
    # the large BF16 shapes used by the IQ2_XS fallback.  These pins were
    # measured with M=2048 on the RX 9070 XT; users can override them.
    fast_prefill_pins="2048x17408x5120:110451,2048x5120x17408:110451,2048x5120x6144:110451,2048x10240x5120:110451,2048x1024x5120:110451"
fi
# IQ3_XXS models contain a mixed set of IQ2 projections. The scalar Q8_1
# activation adapter is validated against the llama.cpp ROCm reference for
# both the mixed profile and standalone IQ2_XS.
iq2_q81_default=0
iq2_xs_q81_default=1
iq1_q81_scalar_default=1
iq1s_mmq_scales_default=1
iq1_ffn_gate_q81_default=1
iq1_ffn_up_q81_default=0
# llama.cpp's sequential decode uses the MMVQ Q8_1 kernel for the SSM IQ2_S
# output and for the SSM IQ3_S gate; the pure-IQ2 profile enables both. The
# mixed-IQ3 profile keeps them off until its own A/B is rerun.
iq2s_q81_scalar_default=0
ssm_gate_q81_default=0
# The mixed IQ3 profile may override individual families after its own ROCm
# A/B validation.
iq3_mixed_q81_default=0
qwen35_batch_default=1
decode_dp4a2_default=1
iq3_graph_disable_default=0
ssm_fused_default=0
# llama.cpp's RDNA4 MMQ path uses the Q8_1 activation contract for the gated
# Q/K/V projections. Fresh full-logit A/Bs improve both GSQ profiles with this
# route, so keep it as the production default; set it to 0 for a direct-F32
# projection control run.
attn_q81_default=1
attn_q81_wmma_default=1
iq3_xxs_q81_mmq_default=0
iq2_xs_q81_mmq_default=1
iq4_xs_q81_scalar_default=1
iq3s_f32_elem_default=0
q2k_q81_default=0
# Pure IQ2 uses llama.cpp's 32-lane GDA reduction order for numerical parity.
# The mixed IQ3 profile keeps its separately validated fused-GDN default.
gdn_ref_default=1
ssm_qkv_q81_default=1
ffn_iq1_q81_default=1
ssm_q81_default=0
# The llama.cpp-shaped F16-query vector attention path is the parity-safe
# Q8/Q4 prefill default for pure IQ2.  IQ3 keeps F32 KV below and therefore
# does not consume this switch.
attn_prefill_fattn_vec_default=1
if [[ "${model}" == *IQ3_XXS* ]]; then
    # IQ2_XXS blocks are part of the mixed IQ3 file.  The current mixed-file
    # parity capture keeps this family on direct F32; the pure IQ2_XS file
    # uses a different weight mix and retains its validated adapter below.
    # The current mixed-IQ3 scalar parity capture is closer with the
    # IQ2_XXS scalar adapter disabled; keep this family direct-F32 until its
    # mixed-layer contract is independently validated.
    iq2_q81_default=0
    # The mixed IQ3 GSQ model's IQ2_XS projections have a validated
    # llama.cpp Q8_1 contract and improve the ROCm logit comparison. Keep
    # IQ2_XXS and IQ3_XXS on direct F32 until their accumulated drift is
    # resolved independently.
    iq2_xs_q81_default=1
    iq3_mixed_q81_default=0
    # Fresh matched ROCm captures show the IQ3 direct-F32 batched schedule
    # adds ~0.016 relative logit L2 over the scalar schedule. Keep the
    # quality default scalar until its projection/reduction drift is isolated;
    # QWEN38_GSQ_BATCHED_PREFILL=1 remains an explicit performance A/B.
    qwen35_batch_default=0
    # IQ3's mixed IQ2_S projections stay on direct F32 dequant.  The global
    # DP4A2 knob also affects prefill; leave it opt-in for IQ3.
    decode_dp4a2_default=0
    iq3_graph_disable_default=1
    # Fresh matched scalar ROCm comparison shows the llama.cpp reference-order
    # GDA recurrence is closer for this IQ3 file (0.04143 vs 0.04670 rel-L2
    # against the sequential oracle). Keep it as the quality default; the
    # fused warp recurrence remains an explicit speed A/B.
    ssm_fused_default=1
    gdn_ref_default=1
    # The mixed-IQ3 reference was already validated with its selective
    # IQ2_XS adapter and direct-F32 IQ3_XXS path; the global SSM QKV adapter
    # is slightly worse there (0.03667 vs 0.03522 relative logit L2).
    ssm_qkv_q81_default=0
    ffn_iq1_q81_default=0
    attn_q81_default=0
    attn_q81_wmma_default=0
    # The batched Q8_1 attention adapter is not numerically uniform across
    # the mixed IQ3 layers: at layer 19 (IQ2_XXS Q / IQ3_S K / IQ3_XXS V)
    # it creates the first pre-attention mismatch. Keep the batch adapter an
    # explicit performance A/B; the validated scalar IQ2_XS adapter below is
    # independent and remains enabled.
    # IQ3's mixed IQ3_XXS and IQ4_XS projections stay on direct F32 for
    # numerical parity.  Their Q8_1 adapters remain explicit A/B controls.
    iq3_xxs_q81_mmq_default=0
    iq4_xs_q81_scalar_default=0
    # The elementwise-FMA IQ3_S dequant path follows llama.cpp's weight
    # accumulation order and improves the mixed-profile scalar comparison.
    iq3s_f32_elem_default=1
    # IQ3's Q2_K gate family benefits from llama.cpp's Q8_1 activation
    # contract; the pure IQ2 mix regresses with the same adapter.
    q2k_q81_default=1
    iq1_q81_scalar_default=0
    iq1s_mmq_scales_default=0
    iq1_ffn_gate_q81_default=0
    iq1_ffn_up_q81_default=0
    iq2s_q81_scalar_default=0
    ssm_gate_q81_default=0
    else
        # Pure IQ2_XS: match llama.cpp's per-projection Q8_1 contracts. The
        # SSM gate and IQ2_XXS/IQ2_S projections each improve the matched
        # q8q4 layer trace; see QWEN38_STATUS.md (bit-exact GDN port).
        attn_q81_default=1
        iq2_q81_default=1
        iq2s_q81_scalar_default=1
        ssm_gate_q81_default=1
fi
qwen35_batch_enable="${QWEN38_GSQ_BATCHED_PREFILL:-${qwen35_batch_default}}"
qwen35_batch_args=(--ubatch "${QWEN38_GSQ_UBATCH:-512}")
if [[ "${qwen35_batch_enable}" != "0" ]]; then
    qwen35_batch_args=(--qwen35-batched-prefill "${qwen35_batch_args[@]}")
fi
decode_kernel_mode="${QWEN38_GSQ_DECODE_KERNELS:-auto}"
if [[ "${model}" == *IQ3_XXS* && -z "${QWEN38_GSQ_DECODE_KERNELS+x}" ]]; then
    # The runner's `auto` load mode enables DP4A2 globally, including the
    # prefill graph. IQ3 mixed projections need the direct F32 path for parity.
    decode_kernel_mode=native
fi
decode_layout_mode="${QWEN38_GSQ_DECODE_LAYOUT:-auto}"
if [[ "${model}" == *IQ3_XXS* && -z "${QWEN38_GSQ_DECODE_LAYOUT+x}" ]]; then
    decode_layout_mode=native
fi
kv_cache_mode="${QWEN38_GSQ_KV_CACHE:-q8q4}"
# Q8 for both K and V is explicit: --kv-cache q8q8 (or QWEN38_GSQ_KV_CACHE=q8q8).
if [[ "${model}" == *IQ3_XXS* && -z "${QWEN38_GSQ_KV_CACHE+x}" ]]; then
    # The current q8/q4 Qwen3.5 attention path is not numerical-parity safe
    # for IQ3_XXS. F32 is the validated quality baseline; q8q4 remains an
    # explicit memory/performance experiment until its attention trace agrees.
    kv_cache_mode=f32
fi
# Match the runner's last-argument precedence in dry-run diagnostics too.
for ((i=0; i<${#args[@]}; ++i)); do
    if [[ "${args[i]}" == "--kv-cache" ]]; then
        if (( i + 1 >= ${#args[@]} )); then
            echo '--kv-cache requires a format' >&2
            exit 2
        fi
        kv_cache_mode="${args[i+1]}"
    fi
done
perf_bmax=512
if [[ "${perf_profile}" != "0" && "${selected_context}" -lt 40000 ]]; then
    # At 32K the Qwen3.8 Q8/Q4 KV allocation leaves enough headroom for the
    # validated 2048-row hybrid scratch set.  4096 rows faults gfx1201, while
    # at the 53K 16-GiB ceiling even 2048 leaves too little margin, so retain
    # the validated 512-row tile there.
    perf_bmax=2048
fi
selected_bmax="${LLM_BMAX:-${perf_bmax}}"
if [[ "${vram_profile}" == "16g" && "${allow_unsafe}" == "0" ]]; then
    bmax_cap=512
    if (( selected_context < 40000 )); then bmax_cap=2048; fi
    if [[ "${selected_bmax}" =~ ^[1-9][0-9]*$ ]] &&
       (( selected_bmax > bmax_cap )); then
        echo "Qwen3.8 GSQ: clamping batch size ${selected_bmax} to ${bmax_cap} for the 16-GiB profile" >&2
        selected_bmax="${bmax_cap}"
    fi
fi
if [[ "${QWEN38_DRY_RUN:-0}" != "0" ]]; then
    printf 'q38gsq kv_cache=%s\n' "${kv_cache_mode}"
    printf 'q38gsq profile: model=%s vram=%s requested_context=%s selected_context=%s safe_context=%s bmax=%s\n' \
        "${model}" "${vram_profile}" "${requested_context}" "${selected_context}" "${safe_context}" "${selected_bmax}"
    exit 0
fi

# IQ2_XXS/S BF16 dequant is approximate: it clears the 300 tok/s long-context
# target, but changes later-token numerics. It is therefore selected only by
# the explicit performance profile (or an explicit FAST_ALL_IQ2=1 override).
# Match llama.cpp's Q8_1 input contract for the active IQ2_XS/IQ3_S/IQ4_XS
# Qwen3.8 SSM projections. These remain overridable for parity A/B tests.
# Batched scheduling remains enabled, but native IQ projection kernels are not
# the quality-safe default yet.  A fresh q8/q4-KV A/B against llama.cpp's
# ROCm oracle measured 1.27 relative-L2 with the native family enabled versus
# 0.220 with it disabled, while prefill changed only ~35.3 -> ~35.0 tok/s.
# Keep the family opt-in until its row/scale contract is numerically equivalent
# to the scalar quantized path; each lower-level family remains independently
# overridable for kernel A/B work.
# Native IQ projections are therefore disabled by default for SSM, attention,
# and FFN, except for the validated scalar IQ2_XS adapter above. This preserves
# the device-resident batched state/attention schedule
# without allowing an unqualified projection approximation into production.
# IQ1_S/IQ1_M FFN weights use the llama.cpp-style single-term Q8_1 kernels;
# the generic IQ1 DP4A2 route remains an explicit regression override because
# its affine correction is not represented by the generic two-term contract.
# The Qwen3.5 SSM batch schedule remains diagnostic-only until its full
# pre-update staging path is parity validated; scalar SSM is the quality-safe
# production configuration.
# Q8_1 activation staging remains opt-in for IQ2_XXS/IQ3_XXS and for the
# recurrent SSM projection family. IQ2_XS's scalar adapter is enabled for
# pure IQ2 and the mixed IQ3 profile because fresh same-prompt A/B runs are
# closer to the llama.cpp ROCm logits.
# IQ4_XS is separately validated against the llama.cpp HIP Q8_1 contract and
# improves the final-logit comparison without changing the coding output.
# Q2_K G4 has a different reduction/order on gfx1201. The scalar warp-per-row
# path is measurably closer to llama.cpp for this Qwen3.8 quality profile;
# retain an explicit override for performance A/Bs.
# The remaining llama.cpp-compatible single-term Q8_1 adapters remain explicit
# A/B controls for the mixed IQ3 GSQ model until their accumulated drift is
# isolated.
# D4/MMQ remains opt-in.
# The warp-per-row fused GDN path changes the reduction order for the
# row-major runner state. Scalar GDN is the numerical-parity default;
# callers can restore the faster fused A/B with LLM_SSM_FUSED=1.
# For pure IQ2, use the llama.cpp column-oriented GDA reduction instead of
# the older row-oriented scalar kernel. This closes a measurable part of the
# fresh ROCm logit gap; it remains explicitly overridable for A/B testing.
# llama.cpp's IQ3_S MMVQ path uses the exact Q8_1 activation contract for
# the Qwen3.5 SSM QKV projection. This adapter is enabled for pure IQ2, where
# it improves the fresh comparison; the mixed-IQ3 profile overrides it below.
diag_env=()
for diag_name in LLM_LOGITS_PATH LLM_GEN_TEXT LLM_DEBUG_LAYERS \
                 LLM_DEBUG_DUMP_DIR LLM_DEBUG_ATTN_LAYER LLM_DEBUG_DUMP_SEQUENCE; do
    if [[ -v "${diag_name}" ]]; then
        diag_env+=("${diag_name}=${!diag_name}")
    fi
done
exec env QWEN38_MODEL="${model}" \
    LLM_SSM_FUSED="${LLM_SSM_FUSED:-${ssm_fused_default}}" \
    LLM_QWEN35_GDA_REF_SCALAR="${LLM_QWEN35_GDA_REF_SCALAR:-${gdn_ref_default}}" \
    LLM_BMAX="${selected_bmax}" \
    LLM_BENCH_STREAM_CHUNK="${LLM_BENCH_STREAM_CHUNK:-${selected_bmax}}" \
    LLM_MW_THREADS="${LLM_MW_THREADS:-64}" \
    LLM_DECODE_DP4A2="${LLM_DECODE_DP4A2:-${decode_dp4a2_default}}" \
    LLM_GRAPH_DISABLE="${LLM_GRAPH_DISABLE:-${iq3_graph_disable_default}}" \
    LLM_Q2K_G4="${LLM_Q2K_G4:-0}" \
    LLM_QWEN35_NATIVE_IQ2_BATCH="${LLM_QWEN35_NATIVE_IQ2_BATCH:-0}" \
    LLM_QWEN35_NATIVE_SSM="${LLM_QWEN35_NATIVE_SSM:-0}" \
    LLM_QWEN35_NATIVE_IQ2_DP4A_BATCH="${LLM_QWEN35_NATIVE_IQ2_DP4A_BATCH:-0}" \
    LLM_QWEN35_NATIVE_IQ4_DP4A_BATCH="${LLM_QWEN35_NATIVE_IQ4_DP4A_BATCH:-0}" \
    LLM_QWEN35_NATIVE_IQ1S_DP4A_BATCH="${LLM_QWEN35_NATIVE_IQ1S_DP4A_BATCH:-0}" \
    LLM_QWEN35_NATIVE_IQ1M_DP4A_BATCH="${LLM_QWEN35_NATIVE_IQ1M_DP4A_BATCH:-0}" \
    LLM_QWEN35_NATIVE_IQ3_DP4A_BATCH="${LLM_QWEN35_NATIVE_IQ3_DP4A_BATCH:-0}" \
    LLM_QWEN35_NATIVE_IQ3XXS_BATCH="${LLM_QWEN35_NATIVE_IQ3XXS_BATCH:-0}" \
    LLM_QWEN35_SSM_IN_Q81="${LLM_QWEN35_SSM_IN_Q81:-0}" \
    LLM_QWEN35_SSM_QKV_Q81="${LLM_QWEN35_SSM_QKV_Q81:-${ssm_qkv_q81_default}}" \
    LLM_QWEN35_SSM_Q81="${LLM_QWEN35_SSM_Q81:-${ssm_q81_default}}" \
    LLM_QWEN35_ATTN_Q81_BATCH="${LLM_QWEN35_ATTN_Q81_BATCH:-${attn_q81_default}}" \
    LLM_QWEN35_ATTN_Q81_D4="${LLM_QWEN35_ATTN_Q81_D4:-0}" \
    LLM_QWEN35_ATTN_Q81_WMMA="${LLM_QWEN35_ATTN_Q81_WMMA:-${attn_q81_wmma_default}}" \
    LLM_IQ2_XS_Q81_MMQ="${LLM_IQ2_XS_Q81_MMQ:-${iq2_xs_q81_mmq_default}}" \
    LLM_IQ3_XXS_Q81_MMQ="${LLM_IQ3_XXS_Q81_MMQ:-${iq3_xxs_q81_mmq_default}}" \
    LLM_IQ3S_F32_ELEM="${LLM_IQ3S_F32_ELEM:-${iq3s_f32_elem_default}}" \
    LLM_Q2K_Q81_SCALAR="${LLM_Q2K_Q81_SCALAR:-${q2k_q81_default}}" \
    LLM_IQ1S_MMQ_SCALES="${LLM_IQ1S_MMQ_SCALES:-${iq1s_mmq_scales_default}}" \
    LLM_QWEN35_FFN_IQ1_Q81="${LLM_QWEN35_FFN_IQ1_Q81:-${ffn_iq1_q81_default}}" \
    LLM_QWEN35_FFN_GATE_IQ1_Q81="${LLM_QWEN35_FFN_GATE_IQ1_Q81:-${iq1_ffn_gate_q81_default}}" \
    LLM_QWEN35_FFN_UP_IQ1_Q81="${LLM_QWEN35_FFN_UP_IQ1_Q81:-${iq1_ffn_up_q81_default}}" \
    LLM_QWEN35_NATIVE_FFN="${LLM_QWEN35_NATIVE_FFN:-0}" \
    LLM_QWEN35_NATIVE_IQ2XXS_FFN="${LLM_QWEN35_NATIVE_IQ2XXS_FFN:-0}" \
    LLM_QWEN35_NATIVE_IQ2XS_FFN="${LLM_QWEN35_NATIVE_IQ2XS_FFN:-0}" \
    LLM_QWEN35_NATIVE_IQ1_FFN="${LLM_QWEN35_NATIVE_IQ1_FFN:-0}" \
    LLM_QWEN35_NATIVE_IQ2S_FFN="${LLM_QWEN35_NATIVE_IQ2S_FFN:-0}" \
    LLM_QWEN35_NATIVE_IQ3S_FFN="${LLM_QWEN35_NATIVE_IQ3S_FFN:-0}" \
    LLM_QWEN35_NATIVE_IQ4XS_FFN="${LLM_QWEN35_NATIVE_IQ4XS_FFN:-0}" \
    LLM_QWEN35_IQ2XS_Q81_FFN="${LLM_QWEN35_IQ2XS_Q81_FFN:-0}" \
    LLM_QWEN35_IQ2XXS_Q81_FFN="${LLM_QWEN35_IQ2XXS_Q81_FFN:-0}" \
    LLM_QWEN35_IQ3XXS_Q81_FFN="${LLM_QWEN35_IQ3XXS_Q81_FFN:-0}" \
    LLM_QWEN35_IQ3S_Q81_FFN="${LLM_QWEN35_IQ3S_Q81_FFN:-1}" \
    LLM_QWEN35_IQ4XS_Q81_FFN="${LLM_QWEN35_IQ4XS_Q81_FFN:-1}" \
    LLM_QWEN35_IQ1S_Q81_BATCH="${LLM_QWEN35_IQ1S_Q81_BATCH:-1}" \
    LLM_QWEN35_IQ1M_Q81_BATCH="${LLM_QWEN35_IQ1M_Q81_BATCH:-1}" \
    LLM_QWEN35_SSM_Q81_EXACT="${LLM_QWEN35_SSM_Q81_EXACT:-1}" \
    LLM_QWEN35_SSM_Q81_IQ1S="${LLM_QWEN35_SSM_Q81_IQ1S:-1}" \
    LLM_QWEN35_SSM_Q81_IQ1M="${LLM_QWEN35_SSM_Q81_IQ1M:-1}" \
    LLM_QWEN35_SSM_GATE_Q81="${LLM_QWEN35_SSM_GATE_Q81:-${ssm_gate_q81_default}}" \
    LLM_IQ2S_Q81_SCALAR="${LLM_IQ2S_Q81_SCALAR:-${iq2s_q81_scalar_default}}" \
    LLM_IQ4_XS_Q81_SCALAR="${LLM_IQ4_XS_Q81_SCALAR:-${iq4_xs_q81_scalar_default}}" \
    LLM_IQ2_XXS_Q81_SCALAR="${LLM_IQ2_XXS_Q81_SCALAR:-${iq2_q81_default}}" \
    LLM_IQ2_XS_Q81_SCALAR="${LLM_IQ2_XS_Q81_SCALAR:-${iq2_xs_q81_default}}" \
    LLM_IQ1_Q81_SCALAR="${LLM_IQ1_Q81_SCALAR:-${iq1_q81_scalar_default}}" \
    LLM_IQ3_S_Q81_SCALAR="${LLM_IQ3_S_Q81_SCALAR:-${iq3_mixed_q81_default}}" \
    LLM_QWEN35_IQ2_MMQ_SINGLE="${LLM_QWEN35_IQ2_MMQ_SINGLE:-${perf_profile}}" \
    LLM_QWEN35_IQ2_BF16_GEMM="${LLM_QWEN35_IQ2_BF16_GEMM:-${fast_prefill}}" \
    LLM_QWEN35_IQ2XXS_BF16_GEMM="${LLM_QWEN35_IQ2XXS_BF16_GEMM:-${fast_all_iq2}}" \
    LLM_QWEN35_IQ2S_BF16_GEMM="${LLM_QWEN35_IQ2S_BF16_GEMM:-${fast_all_iq2}}" \
    LLM_QWEN35_IQ2_BF16_MAX_LAYER="${fast_iq2_max_layer}" \
    LLM_QWEN35_IQ2XXS_MMQ_FUSED="${LLM_QWEN35_IQ2XXS_MMQ_FUSED:-0}" \
    LLM_QWEN35_IQ2XS_MMQ_FUSED="${LLM_QWEN35_IQ2XS_MMQ_FUSED:-0}" \
    LLM_QWEN35_IQ2S_MMQ_FUSED="${LLM_QWEN35_IQ2S_MMQ_FUSED:-0}" \
    LLM_QWEN35_IQ3XXS_Q8_WMMA="${LLM_QWEN35_IQ3XXS_Q8_WMMA:-${perf_profile}}" \
    LLM_QWEN35_IQ3_Q8_WMMA="${LLM_QWEN35_IQ3_Q8_WMMA:-${perf_profile}}" \
    LLM_QWEN35_IQ1S_Q8_WMMA="${LLM_QWEN35_IQ1S_Q8_WMMA:-${perf_profile}}" \
    LLM_QWEN35_IQ1M_Q8_WMMA="${LLM_QWEN35_IQ1M_Q8_WMMA:-${perf_profile}}" \
    LLM_ATTN_DECODE_Q8Q4_WARPRED="${LLM_ATTN_DECODE_Q8Q4_WARPRED:-${perf_profile}}" \
    LLM_ATTN_DECODE_Q8Q4_SPLIT="${LLM_ATTN_DECODE_Q8Q4_SPLIT:-${perf_profile}}" \
    LLM_ATTN_DECODE_Q8Q4_SPLIT_CHUNK="${LLM_ATTN_DECODE_Q8Q4_SPLIT_CHUNK:-1024}" \
    LLM_ATTN_DECODE_Q8Q4_DP4A="${LLM_ATTN_DECODE_Q8Q4_DP4A:-${perf_profile}}" \
    LLM_ATTN_DECODE_Q8Q4_THREADS="${LLM_ATTN_DECODE_Q8Q4_THREADS:-512}" \
    LLM_ATTN_DECODE_Q8Q4_VECV="${LLM_ATTN_DECODE_Q8Q4_VECV:-${perf_profile}}" \
    LLM_ATTN_PREFILL_Q8Q4_FATTN_VEC="${LLM_ATTN_PREFILL_Q8Q4_FATTN_VEC:-${attn_prefill_fattn_vec_default}}" \
    LLM_QWEN35_BF16_FFN_ONLY="${LLM_QWEN35_BF16_FFN_ONLY:-${perf_profile}}" \
    LLM_QWEN35_BATCH_SSM="${LLM_QWEN35_BATCH_SSM:-0}" \
    LLM_QWEN35_BATCH_SSM_FAST="${LLM_QWEN35_BATCH_SSM_FAST:-1}" \
    LLM_SSM_BATCH_CONV="${LLM_SSM_BATCH_CONV:-1}" \
    LLM_SSM_BATCH_PREP_FUSE="${LLM_SSM_BATCH_PREP_FUSE:-0}" \
    LLM_SSM_BATCH_RECURRENCE="${LLM_SSM_BATCH_RECURRENCE:-1}" \
    LLM_SSM_BATCH_PARITY="${LLM_SSM_BATCH_PARITY:-1}" \
    LLM_SSM_BATCH_WARP="${LLM_SSM_BATCH_WARP:-0}" \
    MM_BLASLT_ALGO_PINS="${MM_BLASLT_ALGO_PINS:-${fast_prefill_pins}}" \
    "${diag_env[@]}" \
    "${runner_bin}" \
    "${model}" \
    --kv-cache "${kv_cache_mode}" --decode-kernels "${decode_kernel_mode}" --decode-layout "${decode_layout_mode}" \
    --decode-layout-budget-mib "${QWEN38_DECODE_LAYOUT_BUDGET_MIB:-1792}" \
    "${qwen35_batch_args[@]}" \
    "${args[@]}"
