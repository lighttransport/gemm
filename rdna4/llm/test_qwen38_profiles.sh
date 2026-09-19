#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
flash="${root_dir}/run_qwen38_flash_next_rocm.sh"
gsq="${root_dir}/run_qwen38_gsq_rocm.sh"
server="${root_dir}/run_qwen38_codex_server_rocm.sh"
nextn_forward="${root_dir}/qwen4_nextn_forward.h"
runner_c="${root_dir}/hip_llm_runner.c"

expect_contains() {
    local haystack="$1" needle="$2"
    [[ "${haystack}" == *"${needle}"* ]] || {
        printf 'profile test: expected %q in %s\n' "${needle}" "${haystack}" >&2
        return 1
    }
}

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g "${gsq}")"
expect_contains "${out}" 'Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf'
expect_contains "${out}" 'selected_context=53248'
expect_contains "${out}" 'kv_cache=q8q4'
out="$(QWEN38_DRY_RUN=1 QWEN38_GSQ_KV_CACHE=q8q8 "${gsq}" -s 8192)"
expect_contains "${out}" 'kv_cache=q8q8'
expect_contains "${out}" 'selected_context=8192'
out="$(QWEN38_DRY_RUN=1 QWEN38_GSQ_KV_CACHE=q8q4 "${gsq}" --kv-cache q8q8 -s 8192)"
expect_contains "${out}" 'kv_cache=q8q8'
# Numerical-parity policy is model-specific: pure IQ2 and mixed IQ3 enable the
# validated IQ2_XS Q8_1 adapter, while IQ2_XXS remains direct-F32.
grep -q 'iq2_xs_q81_default=1' "${gsq}" || {
    echo 'profile test: pure IQ2 IQ2_XS Q8_1 default missing' >&2
    exit 1
}
grep -q 'iq2_q81_default=0' "${gsq}" || {
    echo 'profile test: pure IQ2_XXS direct-F32 default missing' >&2
    exit 1
}
grep -q 'iq3_mixed_q81_default=0' "${gsq}" || {
    echo 'profile test: pure IQ2 IQ1 direct-F32 default missing' >&2
    exit 1
}
grep -q 'LLM_ATTN_DECODE_Q8Q4_VECV="\${LLM_ATTN_DECODE_Q8Q4_VECV:-\${perf_profile}}"' "${gsq}" || {
    echo 'profile test: vectorized Q4V performance selector missing' >&2
    exit 1
}
grep -q 'LLM_SSM_FUSED="\${LLM_SSM_FUSED:-\${ssm_fused_default}}"' "${gsq}" || {
    echo 'profile test: model-specific GDN default missing' >&2
    exit 1
}
grep -q 'ssm_fused_default=1' "${gsq}" || {
    echo 'profile test: IQ3 fused GDN default missing' >&2
    exit 1
}
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g "${gsq}" -s 262144)"
expect_contains "${out}" 'safe_context=53248'
expect_contains "${out}" 'selected_context=53248'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g "${gsq}" -s 66560 --bench-depth 65536)"
expect_contains "${out}" 'selected_context=66560'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g QWEN38_GSQ_ALLOW_UNSAFE_CONTEXT=1 "${gsq}" -s 262144)"
expect_contains "${out}" 'selected_context=262144'

grep -q 'One lane per 8-value codebook group' "${runner_c}" || {
    echo 'profile test: full-wave IQ2_XS decode kernel missing' >&2
    exit 1
}
grep -q 'matvec_q2_K_q81' "${runner_c}" || {
    echo 'profile test: llama.cpp Q2_K Q8_1 A/B kernel missing' >&2
    exit 1
}
grep -q 'QWEN38_GSQ_DECODE_TARGET:-30' "${root_dir}/bench_qwen38_gsq_decode.sh" || {
    echo 'profile test: GSQ 30 tok/s target gate missing' >&2
    exit 1
}

# The sidecar checkpoint ring must retain the pre-anchor state at slot zero;
# draft step i is written at slot i+1 so transaction commit cannot skip a
# token when resuming the accepted prefix.
grep -q 'size_t)(i + 1) \* hc_bytes' "${nextn_forward}" || {
    echo 'profile test: NextN checkpoint ring layout regressed' >&2
    exit 1
}

# Quantized KV scales belong to the target trunk only; the NextN sidecar owns an F16
# KV cache and must not index the target scale table at layer n_layers.
grep -q 'qwen4_kv_i8 || r->qwen4_kv_fp8).*trunk' "${runner_c}" || {
    echo 'profile test: quantized KV sidecar guard regressed' >&2
    exit 1
}

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g "${flash}" -s 262144)"
expect_contains "${out}" 'kv=none'
expect_contains "${out}" 'cache_mb=5900'
expect_contains "${out}" 'bmax=512'
expect_contains "${out}" 'copy=0'
expect_contains "${out}" 'cpu_misses=1'

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g "${flash}" -s 65536)"
expect_contains "${out}" 'kv=none'
expect_contains "${out}" 'copy=0'
expect_contains "${out}" 'prefill_copy_max=2048'
expect_contains "${out}" 'delayed=0'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_DELAYED_CACHE=1 "${flash}" -s 65536)"
expect_contains "${out}" 'delayed=1'

# The diagnostic benchmark must honor an explicit direct-copy override; this
# guards against accidentally turning every control into a pipeline run.
grep -q 'LLM_MOE_COPY_PIPELINE="\${LLM_MOE_COPY_PIPELINE:-1}"' \
    "${root_dir}/bench_qwen38_sub32_target.sh" || {
    echo 'profile test: benchmark pipeline override is not configurable' >&2
    exit 1
}
grep -q 'timeout --foreground' "${root_dir}/bench_qwen38_256k.sh" || {
    echo 'profile test: 256K benchmark lacks an internal timeout' >&2
    exit 1
}
grep -q 'QWEN38_SUB32_CONTEXT="\${max_seq}"' \
    "${root_dir}/test_qwen38_approx_coherence.sh" || {
    echo 'profile test: approximate coherence environment block regressed' >&2
    exit 1
}
grep -q 'LLM_QWEN4_DEVICE_REFRESH_START_LAYER=' \
    "${root_dir}/test_qwen38_approx_coherence.sh" || {
    echo 'profile test: approximate coherence refresh boundary missing' >&2
    exit 1
}
grep -q 'QWEN38_SUB32_CACHE_MB="\${cache_for_run}"' \
    "${root_dir}/test_qwen38_approx_coherence.sh" || {
    echo 'profile test: approximate coherence cache headroom missing' >&2
    exit 1
}
grep -q 'LLM_BMAX="\${bmax_for_run}"' \
    "${root_dir}/test_qwen38_approx_coherence.sh" || {
    echo 'profile test: approximate coherence tile headroom missing' >&2
    exit 1
}
grep -q 'approx_contexts' "${root_dir}/test_qwen38_approx_coherence.sh" || {
    echo 'profile test: approximate coherence context selection missing' >&2
    exit 1
}
grep -q 'QWEN38_SUB32_PROMPT_FILE' \
    "${root_dir}/test_qwen38_approx_coherence.sh" || {
    echo 'profile test: long-prompt file transport missing' >&2
    exit 1
}
grep -q 'QWEN38_APPROX_REFRESHES' "${root_dir}/test_qwen38_approx_coherence.sh" || {
    echo 'profile test: refresh sweep override missing' >&2
    exit 1
}
grep -q 'prompt_file' "${root_dir}/bench_qwen38_sub32_target.sh" || {
    echo 'profile test: benchmark prompt-file input missing' >&2
    exit 1
}
grep -q 'LLM_BENCH_STREAM_PUBLISH_CHUNK' "${root_dir}/bench_qwen38_sub32_target.sh" || {
    echo 'profile test: pipeline chunk publication missing' >&2
    exit 1
}
grep -q 'timeout --foreground' "${root_dir}/bench_qwen38_sub32_target.sh" || {
    echo 'profile test: sub32 benchmark lacks an internal timeout' >&2
    exit 1
}
grep -q 'QWEN38_SUB32_BATCH_SSM' "${root_dir}/bench_qwen38_sub32_target.sh" || {
    echo 'profile test: batched SSM A/B control missing' >&2
    exit 1
}
grep -q 'QWEN38_SUB32_BATCH_ATTN_MAX_LAYER' "${root_dir}/bench_qwen38_sub32_target.sh" || {
    echo 'profile test: batched attention-prefix control missing' >&2
    exit 1
}
grep -q 'bench-repeat' "${root_dir}/test_hip_llm.c" || {
    echo 'profile test: in-process bench-repeat option missing' >&2
    exit 1
}
grep -q 'hip_llm_reset_state(gpu)' "${root_dir}/test_hip_llm.c" || {
    echo 'profile test: bench-repeat must reset state between repeats' >&2
    exit 1
}
test -x "${root_dir}/bench_qwen38_target.sh" || {
    echo 'profile test: target repeatability gate script missing or not executable' >&2
    exit 1
}
grep -q 'sequence hash=' "${root_dir}/qwen38_target_result.sh" || {
    echo 'profile test: target gate must compare sequence hashes' >&2
    exit 1
}
grep -q 'nondeterministic sequence hash' "${root_dir}/qwen38_target_result.sh" || {
    echo 'profile test: target gate must fail on hash divergence' >&2
    exit 1
}
# The deterministic gate must default CPU cold experts off; the mixed CPU/GPU
# path changes arithmetic with cache warmth and is diagnostic-only.
out="$(QWEN38_DRY_RUN=1 QWEN38_TARGET_PROFILE=batch "${root_dir}/bench_qwen38_target.sh")"
expect_contains "${out}" 'cpu_prefill_jobs=0'
expect_contains "${out}" 'cpu_decode_misses=0'
out="$(QWEN38_DRY_RUN=1 QWEN38_TARGET_PROFILE=batch-cpu "${root_dir}/bench_qwen38_target.sh")"
expect_contains "${out}" 'cpu_prefill_jobs=160'
expect_contains "${out}" 'cpu_decode_misses=1'
# Diagnostic single-dispatch 4K profile.
out="$(QWEN38_DRY_RUN=1 QWEN38_TARGET_PROFILE=batch4k "${root_dir}/bench_qwen38_target.sh")"
expect_contains "${out}" 'bmax=4096'
expect_contains "${out}" 'stream_chunk=0'
expect_contains "${out}" 'batch=1'
expect_contains "${out}" 'multi=0'
expect_contains "${out}" 'force_multi=0'
expect_contains "${out}" 'cpu_prefill_jobs=0'
# Conservative defaults: pageable weights and serialized copies; overlap is opt-in.
expect_contains "${out}" 'copy=0'
expect_contains "${out}" 'reg=0'
out="$(QWEN38_DRY_RUN=1 LLM_MOE_REGISTER_HOST=1 \
    QWEN38_TARGET_PROFILE=batch4k "${root_dir}/bench_qwen38_target.sh")"
expect_contains "${out}" 'reg=1'
out="$(QWEN38_DRY_RUN=1 LLM_MOE_COPY_PIPELINE=1 \
    QWEN38_TARGET_PROFILE=batch4k "${root_dir}/bench_qwen38_target.sh")"
expect_contains "${out}" 'copy=1'
# Staged grouped-cold diagnostic preset.
out="$(QWEN38_DRY_RUN=1 QWEN38_TARGET_PROFILE=batch4k-stage \
    "${root_dir}/bench_qwen38_target.sh")"
expect_contains "${out}" 'staging=1'
expect_contains "${out}" 'cache_mb=4000'
expect_contains "${out}" 'bmax=4096'
grep -q -- '--qwen4-prefill-staging' "${root_dir}/bench_qwen38_target.sh" || {
    echo 'profile test: staged grouped-cold switch missing' >&2
    exit 1
}
grep -q 'LLM_QWEN4_STAGE_PROMOTE' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: staged promotion switch missing' >&2
    exit 1
}
grep -q 'LLM_BENCH_WARMUP' "${root_dir}/test_hip_llm.c" || {
    echo 'profile test: bench warmup hook missing' >&2
    exit 1
}
grep -q 'LLM_QWEN4_NATIVE_EXPERTS' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: native-vs-WMMA expert A/B switch missing' >&2
    exit 1
}
# Ordered MoE combine and synchronous CPU-result publication.
grep -q 'moe_scatter_accum_ordered' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: ordered MoE scatter missing' >&2
    exit 1
}
grep -q 'd_moe_assign_pos' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: MoE assignment-position map missing' >&2
    exit 1
}
if grep -q 'hipMemcpyAsync(r->d_xb2, r->h_moe_output' "${root_dir}/hip_llm_runner.c"; then
    echo 'profile test: CPU decode result publication must be synchronous' >&2
    exit 1
fi
grep -q 'hipMemcpy(r->d_xb2, r->h_moe_output' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: CPU decode result publication missing' >&2
    exit 1
}
grep -q 'LLM_QWEN4_RESET_MOE_CACHE' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: expert-cache reset gate missing' >&2
    exit 1
}
grep -q 'LLM_QWEN4_RESET_MOE_CACHE' "${root_dir}/bench_qwen38_target.sh" || {
    echo 'profile test: target gate must reset expert-cache state between repeats' >&2
    exit 1
}
grep -q 'group<groups' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: I8 KV scale writes lack inactive-group guard' >&2
    exit 1
}
grep -q 'n_heads / n_kv_heads == 12' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: GQA8 I8 ratio guard regressed' >&2
    exit 1
}
grep -q 'lane==0 && key<tn)red\[key\]=sc' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: GQA8 I8 key reduction indexing regressed' >&2
    exit 1
}
# Bank metadata and slot lifetimes are checked behaviorally by
# test_qwen4_moe_stage.c and test_copy_lifecycle.py.
# Per-row position publication must be stream-ordered (async from the stable
# host array); a blocking hipMemcpy on the null stream raced r->stream kernels
# and was the residual batched nondeterminism.
grep -q 'hipMemcpyAsync(r->d_position, &r->h_pos_batch\[m\]' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: per-row position publication not stream-ordered' >&2
    exit 1
}
grep -q 'h_pos_batch\[m\] = position_start + m' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: per-row position host array not precomputed' >&2
    exit 1
}
grep -q 'head_dim > 256' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: I8 KV geometry guard missing' >&2
    exit 1
}
test -x "${root_dir}/test_qwen4_i8_quality.sh" || {
    echo 'profile test: I8 quality regression script missing or not executable' >&2
    exit 1
}
grep -q 'f32_to_fp8_e4m3_dev' "${runner_c}" || {
    echo 'profile test: FP8 device encoder missing' >&2
    exit 1
}
grep -q 'qwen4_kv_fp8' "${runner_c}" || {
    echo 'profile test: FP8 KV format plumbing missing' >&2
    exit 1
}
grep -q 'disable_qsa_env' "${runner_c}" || {
    echo 'profile test: I8 QSA disable gate missing' >&2
    exit 1
}
grep -q 'LLM_QWEN4_BATCH_ROUTER_SCALAR' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: scalar-router parity diagnostic missing' >&2
    exit 1
}
grep -q 'hipMemcpy(r->d_position, &pos, sizeof(int), hipMemcpyHostToDevice)' "${root_dir}/hip_llm_runner.c" || {
    echo 'profile test: loop-local position publication must be synchronous' >&2
    exit 1
}
if grep -q 'hipMemcpyAsync(r->d_position, &pos,' "${root_dir}/hip_llm_runner.c"; then
    echo 'profile test: loop-local position still uses async host source' >&2
    exit 1
fi
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS=1024 "${flash}" -s 4096)"
expect_contains "${out}" 'prefill_copy_max=1024'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS=5000 "${flash}" -s 4096)"
expect_contains "${out}" 'prefill_copy_max=2048'

# An explicit MTP sidecar selects the validated MTP copy stream; ordinary
# launcher invocations and an explicit opt-out keep direct copies.
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    "${flash}" --qwen4-mtp sidecar.gguf -s 512)"
expect_contains "${out}" 'mtp_copy=1'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_MTP_COPY_PIPELINE=0 "${flash}" --qwen4-mtp sidecar.gguf -s 512)"
expect_contains "${out}" 'mtp_copy=0'

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g "${flash}" -s 131072)"
expect_contains "${out}" 'kv=none'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_KV_QUANT=i8 "${flash}" -s 131072)"
expect_contains "${out}" 'kv=i8'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_KV_QUANT=fp8 "${flash}" -s 131072)"
expect_contains "${out}" 'kv=fp8'

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=24g "${flash}" -s 262144)"
expect_contains "${out}" 'kv=none'
expect_contains "${out}" 'cache_mb=12000'
expect_contains "${out}" 'bmax=3072'

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_KV_QUANT=none QWEN38_MOE_CACHE_MB=3500 \
    "${flash}" -s 262144)"
expect_contains "${out}" 'kv=none'
expect_contains "${out}" 'cache_mb=3500'

out="$(QWEN38_DRY_RUN=1 QWEN38_FAST_PREFILL=1 QWEN38_VRAM_PROFILE=16g \
    "${flash}" -s 4096)"
expect_contains "${out}" 'cache_mb=7800'
expect_contains "${out}" 'bmax=2048'
expect_contains "${out}" 'batch=0'

# The explicit fast-prefill tile/cache must survive the 8K+ conservative
# approximate auto-selection; otherwise the documented 2048-row profile is
# silently downgraded to 512 rows.
for fast_seq in 8192 16384; do
    out="$(QWEN38_DRY_RUN=1 QWEN38_FAST_PREFILL=1 QWEN38_VRAM_PROFILE=16g \
        "${flash}" -s "${fast_seq}")"
    expect_contains "${out}" 'cache_mb=7800'
    expect_contains "${out}" 'bmax=2048'
done
# An explicit BMAX/cache override still wins over the fast-prefill defaults.
out="$(QWEN38_DRY_RUN=1 QWEN38_FAST_PREFILL=1 QWEN38_VRAM_PROFILE=16g \
    LLM_BMAX=1024 QWEN38_MOE_CACHE_MB=5000 "${flash}" -s 8192)"
expect_contains "${out}" 'cache_mb=5000'
expect_contains "${out}" 'bmax=1024'

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_APPROX_DECODE=1 "${flash}" -s 8192)"
expect_contains "${out}" 'cache_mb=7200'
expect_contains "${out}" 'bmax=512'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_APPROX_DECODE=1 "${flash}" -s 16384)"
expect_contains "${out}" 'cache_mb=5900'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    "${flash}" --qwen4-exact -s 8192)"
expect_contains "${out}" 'qsa_select=0'
expect_contains "${out}" 'qsa_warp=0'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_QSA_DEVICE_SELECT=1 LLM_QWEN4_QSA_WARP_ATTN=1 \
    "${flash}" --qwen4-exact -s 8192)"
expect_contains "${out}" 'qsa_select=1'
expect_contains "${out}" 'qsa_warp=1'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    "${flash}" --qwen4-exact -s 4096)"
expect_contains "${out}" 'qsa_select=0'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
    LLM_QWEN4_PREFILL_COPY_PIPELINE=1 "${flash}" -s 4096)"
expect_contains "${out}" 'balance=0'

out="$(QWEN38_DRY_RUN=1 QWEN38_BATCH_PREFILL=1 QWEN38_VRAM_PROFILE=16g \
    "${flash}" -s 4096)"
expect_contains "${out}" 'batch=1'
expect_contains "${out}" 'stateful=1'
expect_contains "${out}" 'multi=0'

out="$(QWEN38_DRY_RUN=1 QWEN38_BATCH_PREFILL=1 QWEN38_VRAM_PROFILE=16g \
    "${flash}" -s 262144)"
expect_contains "${out}" 'kv=none'
expect_contains "${out}" 'batch=1'
expect_contains "${out}" 'stateful=1'
expect_contains "${out}" 'multi=0'
expect_contains "${out}" 'cache_mb=4000'
expect_contains "${out}" 'bmax=512'

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g "${flash}" -s 262144)"
expect_contains "${out}" 'batch=0'


out="$(QWEN38_DRY_RUN=1 QWEN38_BATCH_PREFILL=1 QWEN38_VRAM_PROFILE=16g \
    QWEN38_CONTEXT=262144 "${server}")"
expect_contains "${out}" 'batch=1'
expect_contains "${out}" 'stateful=1'
expect_contains "${out}" 'multi=0'
expect_contains "${out}" 'cache_mb=4000'
expect_contains "${out}" 'bmax=512'

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=32g QWEN38_CONTEXT=262144 "${server}")"
expect_contains "${out}" 'kv=none'
expect_contains "${out}" 'cache_mb=18000'
expect_contains "${out}" 'bmax=4096'

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g QWEN38_CONTEXT=262144 "${server}")"
expect_contains "${out}" 'copy=0'
expect_contains "${out}" 'cpu_misses=0'
expect_contains "${out}" 'topk=1'
expect_contains "${out}" 'grouped=0'
expect_contains "${out}" 'balance=1'
expect_contains "${out}" 'prefill_copy=0'
expect_contains "${out}" 'prefill_copy_max=2048'
expect_contains "${out}" 'delayed=0'
expect_contains "${out}" 'publish_chunk=0'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g QWEN38_CONTEXT=4096 \
    LLM_QWEN4_PREFILL_COPY_PIPELINE=1 \
    LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS=5000 \
    LLM_QWEN4_PREFILL_COPY_PIPELINE_PUBLISH_CHUNK=1 \
    LLM_QWEN4_DELAYED_CACHE=1 "${server}")"
expect_contains "${out}" 'prefill_copy=1'
expect_contains "${out}" 'prefill_copy_max=2048'
expect_contains "${out}" 'delayed=1'
expect_contains "${out}" 'publish_chunk=1'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g QWEN38_CONTEXT=4096 \
    LLM_QWEN4_PREFILL_COPY_PIPELINE=1 "${server}")"
expect_contains "${out}" 'prefill_copy=1'
expect_contains "${out}" 'publish_chunk=1'
expect_contains "${out}" 'balance=0'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g QWEN38_CONTEXT=4096 \
    LLM_QWEN4_PREFILL_COPY_PIPELINE=1 \
    LLM_QWEN4_PREFILL_COPY_PIPELINE_PUBLISH_CHUNK=0 "${server}")"
expect_contains "${out}" 'prefill_copy=1'
expect_contains "${out}" 'publish_chunk=0'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g QWEN38_CONTEXT=4096 \
    LLM_QWEN4_PREFILL_COPY_PIPELINE=1 LLM_QWEN4_PREFILL_CACHE_BALANCE=1 \
    "${server}")"
expect_contains "${out}" 'balance=1'

out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g QWEN38_CONTEXT=4096 \
    LLM_QWEN4_APPROX_DECODE=1 env -u LLM_MOE_CPU_DECODE_MISSES "${server}")"
expect_contains "${out}" 'cpu_misses=1'
expect_contains "${out}" 'approx_start=24'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g QWEN38_CONTEXT=8192 \
    LLM_QWEN4_APPROX_DECODE=1 "${server}")"
expect_contains "${out}" 'cache_mb=7200'
expect_contains "${out}" 'bmax=512'
out="$(QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g QWEN38_CONTEXT=16384 \
    LLM_QWEN4_APPROX_DECODE=1 "${server}")"
expect_contains "${out}" 'cache_mb=5900'

# Large diagnostic requests must use bounded streamed prefill by default;
# forcing zero remains available for an explicit single-dispatch A/B run.
out="$(QWEN38_DRY_RUN=1 QWEN38_SUB32_PREFILL=4096 \
    "${root_dir}/bench_qwen38_sub32_target.sh")"
expect_contains "${out}" 'stream_chunk=512'
expect_contains "${out}" 'batch=0'
expect_contains "${out}" 'stateful=0'
out="$(QWEN38_DRY_RUN=1 QWEN38_SUB32_PREFILL=4096 \
    LLM_BENCH_STREAM_CHUNK=0 "${root_dir}/bench_qwen38_sub32_target.sh")"
expect_contains "${out}" 'stream_chunk=0'
out="$(QWEN38_DRY_RUN=1 QWEN38_SUB32_PREFILL=4096 \
    LLM_QWEN4_BATCH=1 "${root_dir}/bench_qwen38_sub32_target.sh")"
expect_contains "${out}" 'batch=1'
expect_contains "${out}" 'stateful=1'

# The faster deep-layer experiments stay explicit.  The launcher must pass an
# override through unchanged so tuning sweeps do not require editing it.
out="$(QWEN38_DRY_RUN=1 QWEN38_CONTEXT=4096 QWEN38_VRAM_PROFILE=16g \
  LLM_QWEN4_APPROX_DECODE=1 LLM_QWEN4_DEVICE_REFRESH_START_LAYER=8 \
  "${server}")"
expect_contains "${out}" 'cpu_misses=1'
expect_contains "${out}" 'approx_start=8'

if [[ -e /dev/kfd ]]; then
    echo 'profile test: /dev/kfd is present; skip missing-device guard check' >&2
else
    if "${flash}" -s 4096 >/dev/null 2>"${root_dir}/tmp_profile_guard.log"; then
        echo 'profile test: flash launcher unexpectedly succeeded without /dev/kfd' >&2
        exit 1
    fi
    expect_contains "$(<"${root_dir}/tmp_profile_guard.log")" '/dev/kfd is missing'
    rm -f "${root_dir}/tmp_profile_guard.log"
fi

echo 'qwen38 profile tests: PASS'
