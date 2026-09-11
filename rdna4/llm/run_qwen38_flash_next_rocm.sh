#!/usr/bin/env bash
set -euo pipefail

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="${QWEN38_RUNNER:-${runner_dir}/test_hip_llm}"
model="${QWEN38_MODEL:-/mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
# Keep CPU cold-expert work bounded and consistent with the HTTP launcher.
# Explicit OMP_NUM_THREADS remains authoritative for hardware-specific tuning.
omp_threads="${OMP_NUM_THREADS:-16}"
cpu_lib="${LLM_MOE_CPU_LIB:-/mnt/nvme02/work/llama.cpp/build-codex-hetero-dev2/bin/libggml-cpu.so.0.22.0}"
if [[ ! -r "${cpu_lib}" ]]; then
    cpu_lib=""
fi

# Fail before model loading when this shell was not granted AMD KFD access.
# Without this guard HIP can spend minutes compiling/loading before returning a
# device error, which is especially misleading for long-context benchmarks.
if [[ "${QWEN38_DRY_RUN:-0}" == "0" && ! -e /dev/kfd ]]; then
    echo "q38fn ROCm device unavailable: /dev/kfd is missing" >&2
    echo "Expose the AMD KFD device (and grant video/render access) before running the flash launcher." >&2
    exit 1
fi
if [[ "${QWEN38_DRY_RUN:-0}" == "0" && ( ! -r /dev/kfd || ! -w /dev/kfd ) ]]; then
    echo "q38fn ROCm device unavailable: /dev/kfd is not readable/writable by this process" >&2
    echo "Grant the container/user AMD KFD access (typically video/render groups) before running." >&2
    exit 1
fi
if [[ "${QWEN38_DRY_RUN:-0}" == "0" ]] && ! compgen -G '/dev/dri/renderD*' > /dev/null; then
    echo "q38fn ROCm device unavailable: no /dev/dri/renderD* node is visible" >&2
    echo "Expose an AMD render node before running the flash launcher." >&2
    exit 1
fi
if [[ "${QWEN38_DRY_RUN:-0}" == "0" ]]; then
    render_access=0
    for render_node in /dev/dri/renderD*; do
        if [[ -r "${render_node}" && -w "${render_node}" ]]; then render_access=1; break; fi
    done
    if [[ "${render_access}" -eq 0 ]]; then
        echo "q38fn ROCm device unavailable: no readable/writable AMD render node" >&2
        echo "Grant render-node access to the user/container before running." >&2
        exit 1
    fi
fi
if [[ "${QWEN38_DRY_RUN:-0}" == "0" && "${QWEN38_REQUIRE_EXCLUSIVE_GPU:-0}" != "0" ]] && command -v fuser >/dev/null 2>&1; then
    kfd_users="$(fuser /dev/kfd 2>/dev/null || true)"
    if [[ -n "${kfd_users//[[:space:]]/}" ]]; then
        echo "q38fn ROCm device busy: /dev/kfd users: ${kfd_users}" >&2
        echo "Stop competing ROCm workloads or set QWEN38_REQUIRE_EXCLUSIVE_GPU=0 to override." >&2
        exit 2
    fi
fi

# Approximate coding decode uses a six-token exact refresh, the validated
# quality/speed cadence at 256K.  Callers can explicitly override it with
# LLM_QWEN4_DEVICE_REFRESH_INTERVAL for throughput experiments.
# Host registration stalls initialization on the RX 9070 XT and is not
# needed by the resident-hit approximate decode path.
# RX 9070 XT's IQ3/SSM warp-per-row path is faster at 128 threads;
# retain LLM_MW_THREADS as an override for other devices.
# The coding profile intentionally disables MTP in the runner.  When the
# caller explicitly requests trusted MTP, omit only that profile flag so the
# sidecar path is selected; ordinary launcher calls remain coding-mode runs.
coding_args=(--coding)
exact_profile=0
fast_prefill="${QWEN38_FAST_PREFILL:-0}"
# Public opt-in for the parity-tested batched prefill route.  Keep the
# low-level LLM_QWEN4_BATCH knob for experiments, but make the launcher mode
# explicit and default its router to the host implementation: the GPU top-k
# path is faster, yet can change borderline routed-expert decisions.
batch_prefill_set=0
batch_prefill="${QWEN38_BATCH_PREFILL:-}"
if [[ -n "${QWEN38_BATCH_PREFILL+x}" ]]; then batch_prefill_set=1; fi
vram_profile="${QWEN38_VRAM_PROFILE:-16g}"
requested_max_seq=0
for ((ai=1; ai<=$#; ai++)); do
    arg="${!ai}"
    if [[ "${arg}" == "-s" || "${arg}" == "--max-seq-len" ]]; then
        ni=$((ai+1))
        if (( ni <= $# )); then requested_max_seq="${!ni}"; fi
    fi
done
# The 16-GiB profile uses F16 for ordinary contexts (quality control). Scaled
# I8 and E4M3 FP8 are explicit caller-selected experiments; automatic profile
# selection remains conservative until long-context quality is qualified.
if [[ -z "${LLM_QWEN4_KV_QUANT+x}" ]]; then
    kv_quant=none
else
    kv_quant="${LLM_QWEN4_KV_QUANT}"
fi
for arg in "$@"; do
    if [[ "${arg}" == "--qwen4-exact" ]]; then
        exact_profile=1
    fi
    if [[ "${arg}" == "--qwen4-mtp-trust-draft" ]]; then
        coding_args=()
        break
    fi
done
if [[ "${exact_profile}" -ne 0 ]]; then
    coding_args=()
    # Registered host mappings support the staged exact miss path. With exact
    # decode's one-row scratch profile, 9728 MiB is the largest validated
    # budget on a 16 GiB RX 9070 XT; 9856 MiB fails during sidecar load.
    # With LLM_BMAX=1 the exact path has enough headroom for the larger cache;
    # keep QWEN38_MOE_CACHE_MB as the explicit override for smaller cards.
    cache_mb="${QWEN38_MOE_CACHE_MB:-9728}"
    register_host="${LLM_MOE_REGISTER_HOST:-1}"
    mapped_misses="${LLM_QWEN4_MAPPED_MISSES:-1}"
    # Qwen3.8-Flash-Next uses Q6_K/Q8_0 routed-down experts; direct BAR
    # misses are not parity-safe for those encodings, so exact defaults to
    # staged misses. Keep the variable as an explicit diagnostic override.
    direct_misses="${LLM_QWEN4_DIRECT_MISSES:-0}"
    cpu_decode_misses="${LLM_MOE_CPU_DECODE_MISSES:-0}"
    qwen_batch="${LLM_QWEN4_BATCH:-0}"
    profile_cpu_lib="${LLM_QWEN4_EXACT_CPU_LIB:-}"
    approx_decode="${LLM_QWEN4_APPROX_DECODE:-0}"
    device_hits_only="${LLM_QWEN4_DEVICE_HITS_ONLY:-0}"
    refresh_interval="${LLM_QWEN4_DEVICE_REFRESH_INTERVAL:-6}"
    approx_cpu_min_weight="${LLM_QWEN4_APPROX_CPU_MIN_WEIGHT:-1.0}"
    pre_graphs="${LLM_QWEN_PRE_GRAPHS:-0}"
    plan_prewarm="${LLM_PLAN_PREWARM:-0}"
    # Grouped Qwen4 prefill remains experimental: repeated full-depth runs
    # can reset gfx1201, so keep it opt-in for both exact and approximate
    # profiles.  Set LLM_MOE_GROUPED_PREFILL=1 only for bounded diagnostics.
    grouped_prefill="${LLM_MOE_GROUPED_PREFILL:-0}"
    prefill_gpu_topk="${LLM_QWEN4_PREFILL_GPU_TOPK:-0}"
    # Exact decode does not use Qwen4 batched prefill.  Avoid reserving the
    # loader's 512-row hybrid scratch set so the recovered VRAM can be used by
    # the routed-expert cache instead.
    bmax="${LLM_BMAX:-1}"
else
    # Approximate resident-hit mode does not need host registration and can
    # use the larger device cache.
    # The RX 9070 XT cannot load the old 9.2-GiB cache together with the
    # batched-prefill scratch buffers.  7.2 GiB leaves headroom for BMAX=1024
    # while retaining better resident-expert decode throughput.
    cache_mb="${QWEN38_MOE_CACHE_MB:-7200}"
    register_host="${LLM_MOE_REGISTER_HOST:-0}"
    mapped_misses="${LLM_QWEN4_MAPPED_MISSES:-0}"
    direct_misses="${LLM_QWEN4_DIRECT_MISSES:-0}"
    cpu_decode_misses="${LLM_MOE_CPU_DECODE_MISSES:-1}"
    # Batched Qwen4 prefill is still an approximate route (router decisions
    # differ from the validated scalar path); keep quality-safe scalar mode
    # unless the caller explicitly opts in with LLM_QWEN4_BATCH=1.
    qwen_batch="${LLM_QWEN4_BATCH:-0}"
    profile_cpu_lib="${cpu_lib}"
    approx_decode="${LLM_QWEN4_APPROX_DECODE:-1}"
    device_hits_only="${LLM_QWEN4_DEVICE_HITS_ONLY:-1}"
    refresh_interval="${LLM_QWEN4_DEVICE_REFRESH_INTERVAL:-32}"
    approx_cpu_min_weight="${LLM_QWEN4_APPROX_CPU_MIN_WEIGHT:-0.20}"
    pre_graphs="${LLM_QWEN_PRE_GRAPHS:-${LLM_QWEN4_PRE_GRAPHS:-1}}"
    plan_prewarm="${LLM_PLAN_PREWARM:-1}"
    grouped_prefill="${LLM_MOE_GROUPED_PREFILL:-0}"
    prefill_gpu_topk="${LLM_QWEN4_PREFILL_GPU_TOPK:-0}"
    # Keep the default decode-safe profile; use LLM_BMAX=2048 with a 6-GiB
    # cache for the faster 4K prefill profile when decode headroom permits.
    bmax="${LLM_BMAX:-512}"
fi
if [[ "${exact_profile}" -eq 0 && "${fast_prefill}" != 0 ]]; then
    # Opt-in 4K+ profile: pinned expert tensors, asynchronous cold uploads,
    # and the largest stable tile on the 16-GiB RX 9070 XT.  The Qwen4
    # batched scratch path aliases its gathered-input/output buffers, making
    # a 7.8-GiB resident cache fit; explicit caller overrides still win.
    if [[ -z "${LLM_MOE_REGISTER_HOST+x}" ]]; then register_host=1; fi
    if [[ -z "${LLM_BMAX+x}" ]]; then bmax=2048; fi
    if [[ -z "${QWEN38_MOE_CACHE_MB+x}" ]]; then cache_mb=7800; fi
    if [[ -z "${LLM_QWEN4_PREFILL_GPU_TOPK+x}" ]]; then prefill_gpu_topk=1; fi
fi
# Optional larger-VRAM profile.  Keep the RX 9070 XT defaults untouched; this
# branch is explicit so a 24/32-GiB card can spend its extra memory on the
# expert cache and larger prefill tile without making 16-GiB startup fragile.
if [[ "${exact_profile}" -eq 0 && "${vram_profile}" != "16g" ]]; then
    case "${vram_profile}" in
        24g) default_cache=12000; default_bmax=3072 ;;
        32g) default_cache=18000; default_bmax=4096 ;;
        *) echo "unknown QWEN38_VRAM_PROFILE=${vram_profile} (use 16g, 24g, or 32g)" >&2; exit 2 ;;
    esac
    if [[ -z "${QWEN38_MOE_CACHE_MB+x}" ]]; then cache_mb="${default_cache}"; fi
    if [[ -z "${LLM_BMAX+x}" ]]; then bmax="${default_bmax}"; fi
fi
# The approximate coding path has a lower scratch headroom boundary at 8K on
# gfx1201. Select the validated 7.2-GiB/512-row profile below 16K; longer
# requests retain the conservative 5.9-GiB profile. Explicit overrides win.
# The explicit fast-prefill profile (2048-row tile / 7.8-GiB cache) must not be
# clobbered by this conservative selection; it is an opt-in caller choice.
if [[ "${exact_profile}" -eq 0 && "${vram_profile}" == "16g" &&
      "${fast_prefill}" == "0" &&
      "${requested_max_seq}" -ge 8192 && "${requested_max_seq}" -lt 131072 &&
      "${approx_decode}" != "0" ]]; then
    if [[ -z "${QWEN38_MOE_CACHE_MB+x}" ]]; then
        if [[ "${requested_max_seq}" -lt 16384 ]]; then cache_mb=7200; else cache_mb=5900; fi
    fi
    if [[ -z "${LLM_BMAX+x}" ]]; then bmax=512; fi
fi
# At 256K the I8 KV cache itself consumes about 3.2 GiB including scales.
# Keep enough headroom for BMAX=2048 and the per-layer dispatch buffers by
# shrinking only the *implicit* expert-cache default on the 16-GiB card.
# Explicit QWEN38_MOE_CACHE_MB remains authoritative for tuning experiments.
if [[ "${vram_profile}" == "16g" && "${requested_max_seq}" -ge 131072 &&
      -z "${QWEN38_MOE_CACHE_MB+x}" ]]; then
    # At 256K, a 5.9-GiB expert cache plus a 512-row scratch tile is the
    # largest reproducible scalar/batch-safe allocation on current gfx1201.
    # Do not apply this to the separate fast-prefill profile, whose scratch
    # reservation is intentionally explicit.
    if [[ "${fast_prefill}" == "0" ]]; then
        if [[ "${batch_prefill}" == "1" || "${qwen_batch}" == "1" ]]; then
            cache_mb=4000
        else
            cache_mb=5900
        fi
        if [[ -z "${LLM_BMAX+x}" ]]; then
            bmax=512
        fi
    if [[ -z "${LLM_QWEN_PRE_GRAPHS+x}" ]]; then pre_graphs=0; fi
    if [[ -z "${LLM_PLAN_PREWARM+x}" ]]; then plan_prewarm=0; fi
        if [[ "${qwen_batch}" != "1" && -z "${LLM_QWEN4_PREFILL_GPU_TOPK+x}" ]]; then
            prefill_gpu_topk=1
        fi
    else
        cache_mb=4000
    fi
fi
if [[ "${batch_prefill_set}" -ne 0 ]]; then
    case "${batch_prefill}" in
        0|1) qwen_batch="${batch_prefill}" ;;
        *) echo "QWEN38_BATCH_PREFILL must be 0 or 1" >&2; exit 2 ;;
    esac
    if [[ "${qwen_batch}" == "1" && -z "${LLM_QWEN4_PREFILL_GPU_TOPK+x}" ]]; then
        prefill_gpu_topk=0
    fi
fi
# Stateful batching is retained for explicit diagnostics.  A padded direct
# control passes, but heterogeneous multi-chunk HTTP parity is not proven;
# keep scalar and approximate defaults untouched unless explicitly overridden.
qwen_batch_multi="${LLM_QWEN4_BATCH_MULTI_CHUNK:-0}"
qwen_batch_stateful="${LLM_QWEN4_BATCH_STATEFUL:-0}"
qwen_batch_min="${LLM_QWEN4_BATCH_MIN_TOKENS:-128}"
# The exact MTP copy stream is validated on the RX 9070 XT (same greedy hash,
# ~5.5% better draft-4 decode). Enable it only for an explicit sidecar request;
# ordinary decode keeps its direct-copy default. An explicit environment value
# always wins.
mtp_copy_pipeline="${LLM_QWEN4_MTP_COPY_PIPELINE:-0}"
if [[ -z "${LLM_QWEN4_MTP_COPY_PIPELINE+x}" ]]; then
    for arg in "$@"; do
        if [[ "${arg}" == "--qwen4-mtp" ]]; then
            mtp_copy_pipeline=1
            break
        fi
    done
fi
# The depth-weighted table keeps the high-churn prefill layers resident.  On
# the RX 9070 XT scalar control it raised 512-token prefill from 19.04 to
# 21.64 tok/s and reduced H2D from 115.47 to 85.99 GiB.  Keep it enabled for
# the fast-prefill profile, but retain an explicit opt-out for decode-focused
# or compatibility runs.
qwen_prefill_balance="${LLM_QWEN4_PREFILL_CACHE_BALANCE:-${fast_prefill}}"
if [[ "${qwen_batch}" == "1" ]]; then
    # Multi-chunk recurrent batching remains diagnostic-only: padded direct
    # controls pass, but a real 2K HTTP prompt can close the runner after its
    # first 512-token chunk.
    if [[ -z "${LLM_QWEN4_BATCH_STATEFUL+x}" ]]; then qwen_batch_stateful=1; fi
fi
# Direct copies measured faster for exact 256K decode on gfx1201. Keep the
# asynchronous pipeline available as an explicit experiment.
# The event-ordered prefill copy stream is still diagnostic-only: it improves
# the 512-token control, but 2K+ runs can deadlock on gfx1201. Keep direct
# copies as the stable default and require explicit opt-in for the experiment.
copy_pipeline="${LLM_MOE_COPY_PIPELINE:-0}"
delayed_cache="${LLM_QWEN4_DELAYED_CACHE:-0}"
prefill_copy_pipeline="${LLM_QWEN4_PREFILL_COPY_PIPELINE:-0}"
prefill_copy_max="${LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS:-2048}"
if ! [[ "${prefill_copy_max}" =~ ^[0-9]+$ ]] ||
   (( prefill_copy_max < 1 || prefill_copy_max > 4096 )); then
    # Match the runner's bounded parser so dry-run output reflects the
    # effective safety limit rather than an ignored invalid request.
    prefill_copy_max=2048
fi
prefill_chunk="${LLM_MOE_CHUNK:-}"
if [[ "${exact_profile}" -eq 0 && "${fast_prefill}" != 0 &&
      -z "${LLM_MOE_CHUNK+x}" ]]; then prefill_chunk=2048; fi
if [[ "${prefill_copy_pipeline}" != "0" &&
      "${requested_max_seq}" -ge 4096 &&
      -z "${LLM_QWEN4_PREFILL_CACHE_BALANCE+x}" ]]; then
    # The explicit 4K overlap path measured better with uniform residency;
    # keep the depth-balanced policy for the ordinary short-request default.
    qwen_prefill_balance=0
fi
# Exact QSA selector/warp attention is parity-safe but still experimental.  A
# matched 2K exact run showed a severe decode regression (0.68 tok/s versus
# 14.53 tok/s with QSA disabled), so keep it opt-in until the selector and
# attention kernels are optimized.  Explicit settings remain available.
qsa_device_select="${LLM_QWEN4_QSA_DEVICE_SELECT:-0}"
qsa_warp_attn="${LLM_QWEN4_QSA_WARP_ATTN:-0}"
if [[ "${QWEN38_DRY_RUN:-0}" != "0" ]]; then
    printf 'q38fn profile: runner=%s vram=%s max_seq=%s kv=%s cache_mb=%s bmax=%s batch=%s stateful=%s multi=%s balance=%s chunk=%s copy=%s delayed=%s prefill_copy=%s prefill_copy_max=%s mtp_copy=%s cpu_misses=%s exact=%s fast_prefill=%s qsa_select=%s qsa_warp=%s\n' \
        "${runner}" "${vram_profile}" "${requested_max_seq}" "${kv_quant}" "${cache_mb}" \
        "${bmax}" "${qwen_batch}" "${qwen_batch_stateful}" "${qwen_batch_multi}" "${qwen_prefill_balance}" "${prefill_chunk:-auto}" "${copy_pipeline}" "${delayed_cache}" "${prefill_copy_pipeline}" "${prefill_copy_max}" "${mtp_copy_pipeline}" "${cpu_decode_misses}" "${exact_profile}" "${fast_prefill}" "${qsa_device_select}" "${qsa_warp_attn}"
    exit 0
fi
# LRU is the measured exact-production policy on 16 GiB; set
# LLM_MOE_LFU_CACHE=1 explicitly for LFU experiments.
exec env \
    OMP_NUM_THREADS="${omp_threads}" \
    LLM_MOE_REGISTER_HOST="${register_host}" \
    LLM_MOE_COPY_PIPELINE="${copy_pipeline}" \
    LLM_QWEN4_DELAYED_CACHE="${delayed_cache}" \
    LLM_QWEN4_MTP_COPY_PIPELINE="${mtp_copy_pipeline}" \
    LLM_QWEN4_PREFILL_COPY_PIPELINE="${prefill_copy_pipeline}" \
    LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS="${prefill_copy_max}" \
    LLM_MOE_CHUNK="${prefill_chunk}" \
    LLM_MOE_LFU_CACHE="${LLM_MOE_LFU_CACHE:-0}" \
    LLM_MOE_STREAM_SLOTS="${LLM_MOE_STREAM_SLOTS:-4}" \
    LLM_MW_THREADS="${LLM_MW_THREADS:-128}" \
    LLM_MOE_CPU_LIB="${profile_cpu_lib}" \
    LLM_MOE_CPU_DECODE_MISSES="${cpu_decode_misses}" \
    LLM_QWEN4_MAPPED_MISSES="${mapped_misses}" \
    LLM_QWEN4_DIRECT_MISSES="${direct_misses}" \
    LLM_QWEN4_PREFILL_GPU_TOPK="${prefill_gpu_topk}" \
    LLM_QWEN4_KV_QUANT="${kv_quant}" \
    LLM_QWEN4_QSA_DEVICE_SELECT="${qsa_device_select}" \
    LLM_QWEN4_QSA_WARP_ATTN="${qsa_warp_attn}" \
    LLM_ATTN_PREFILL_I8_WARP="${LLM_ATTN_PREFILL_I8_WARP:-0}" \
    LLM_ATTN_PREFILL_I8_GQA4="${LLM_ATTN_PREFILL_I8_GQA4:-0}" \
    LLM_MOE_CPU_MIN_WEIGHT="${LLM_MOE_CPU_MIN_WEIGHT:-0}" \
    LLM_QWEN4_APPROX_DECODE="${approx_decode}" \
    LLM_QWEN4_DEVICE_HITS_ONLY="${device_hits_only}" \
    LLM_QWEN4_DEVICE_REFRESH_INTERVAL="${refresh_interval}" \
    LLM_QWEN4_APPROX_CPU_MIN_WEIGHT="${approx_cpu_min_weight}" \
    LLM_QWEN_PRE_GRAPHS="${pre_graphs}" \
    LLM_PLAN_PREWARM="${plan_prewarm}" \
    LLM_QWEN4_BATCH="${qwen_batch}" \
    LLM_QWEN4_BATCH_MIN_TOKENS="${qwen_batch_min}" \
    LLM_QWEN4_BATCH_MULTI_CHUNK="${qwen_batch_multi}" \
    LLM_QWEN4_BATCH_STATEFUL="${qwen_batch_stateful}" \
    LLM_QWEN4_PREFILL_CACHE_BALANCE="${qwen_prefill_balance}" \
    LLM_QWEN4_NATIVE_BATCH_QKV="${LLM_QWEN4_NATIVE_BATCH_QKV:-0}" \
    LLM_MOE_GROUPED_PREFILL="${grouped_prefill}" \
    LLM_BMAX="${bmax}" \
    LLM_MOE_CPU_PREFILL_MAX_COUNT="${LLM_MOE_CPU_PREFILL_MAX_COUNT:-2}" \
    LLM_MOE_CPU_PREFILL_MAX_JOBS="${LLM_MOE_CPU_PREFILL_MAX_JOBS:-160}" \
    "${runner}" "${model}" \
    --gpu-only-bench "${coding_args[@]}" --moe-cache-mb "${cache_mb}" "$@"
