#!/usr/bin/env bash
set -euo pipefail

# Start the Qwen3.8-Flash-Next (q38fn) runner as a local OpenAI-compatible
# endpoint for Codex and other coding-agent clients.
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${QWEN38_MODEL:-/mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
# 7.0 GiB leaves room for the larger prefill scratch buffers while keeping
# enough routed experts resident for warm decode. 8 GiB exhausts VRAM during
# load with the current batched Qwen path.
cache_mb="${QWEN38_MOE_CACHE_MB:-7200}"
context="${QWEN38_CONTEXT:-65536}"
vram_profile="${QWEN38_VRAM_PROFILE:-16g}"
port="${QWEN38_API_PORT:-8080}"
host="${QWEN38_API_HOST:-127.0.0.1}"
# The HTTP shim emits the first SSE event after generation completes. Keep the
# interactive default bounded; raise QWEN38_MAX_OUTPUT for long code patches.
max_output="${QWEN38_MAX_OUTPUT:-512}"
# Public launcher alias for the batched prefill route. It remains opt-in until
# short request-level routed-MoE parity is complete; the server passes it
# through after selecting the VRAM profile.
batch_prefill_set=0
batch_prefill="${QWEN38_BATCH_PREFILL:-}"
if [[ -n "${QWEN38_BATCH_PREFILL+x}" ]]; then batch_prefill_set=1; fi
cpu_lib="${LLM_MOE_CPU_LIB:-/mnt/nvme02/work/llama.cpp/build-codex-hetero-dev2/bin/libggml-cpu.so.0.22.0}"
if [[ ! -r "${cpu_lib}" ]]; then
    cpu_lib=""
fi

if [[ ! -r "${model}" ]]; then
    echo "q38fn model not found: ${model}" >&2
    echo "Set QWEN38_MODEL to the first GGUF shard." >&2
    exit 1
fi

# Fail early with an actionable message when this is launched from a container
# or session that was not given the AMD device nodes. HIP otherwise reports a
# terse initialization error after loading the model and wasting startup time.
if [[ "${QWEN38_DRY_RUN:-0}" == "0" && ! -e /dev/kfd ]]; then
    echo "q38fn ROCm device unavailable: /dev/kfd is missing" >&2
    echo "Expose the AMD KFD device (and grant the user video/render access) before starting the server." >&2
    exit 1
fi
if [[ "${QWEN38_DRY_RUN:-0}" == "0" && ( ! -r /dev/kfd || ! -w /dev/kfd ) ]]; then
    echo "q38fn ROCm device unavailable: /dev/kfd is not readable/writable by this process" >&2
    echo "Grant the container/user AMD KFD access (typically video/render groups) before starting." >&2
    exit 1
fi
if [[ "${QWEN38_DRY_RUN:-0}" == "0" ]] && ! compgen -G '/dev/dri/renderD*' > /dev/null; then
    echo "q38fn ROCm device unavailable: no /dev/dri/renderD* node is visible" >&2
    echo "Expose an AMD render node before starting the server." >&2
    exit 1
fi
if [[ "${QWEN38_DRY_RUN:-0}" == "0" ]]; then
    render_access=0
    for render_node in /dev/dri/renderD*; do
        if [[ -r "${render_node}" && -w "${render_node}" ]]; then render_access=1; break; fi
    done
    if [[ "${render_access}" -eq 0 ]]; then
        echo "q38fn ROCm device unavailable: no readable/writable AMD render node" >&2
        echo "Grant render-node access to the user/container before starting." >&2
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

# Keep F16 KV for the measured 65K quality profile.  At 128K+ the 16-GiB
# card needs scaled-I8 KV and a smaller resident-expert budget to leave room
# for the context cache and batched scratch.  Explicit caller settings win.
case "${vram_profile}" in
    16g) profile_cache=7200; profile_bmax=512 ;;
    24g) profile_cache=12000; profile_bmax=3072 ;;
    32g) profile_cache=18000; profile_bmax=4096 ;;
    *) echo "unknown QWEN38_VRAM_PROFILE=${vram_profile} (use 16g, 24g, or 32g)" >&2; exit 2 ;;
esac
if [[ -z "${QWEN38_MOE_CACHE_MB+x}" ]]; then cache_mb="${profile_cache}"; fi
if [[ "${context}" -ge 131072 && "${vram_profile}" == "16g" ]]; then
    # Scaled-I8 KV remains explicit-only while kv_cache_store_i8_devp is
    # being repaired; automatic selection previously exposed a gfx1201 fault.
    if [[ -z "${LLM_QWEN4_KV_QUANT+x}" ]]; then LLM_QWEN4_KV_QUANT=none; fi
    if [[ -z "${QWEN38_MOE_CACHE_MB+x}" ]]; then
        if [[ "${batch_prefill:-${LLM_QWEN4_BATCH:-0}}" == "1" ]]; then
            cache_mb=4000
        else
            cache_mb=5900
        fi
    fi
    if [[ -z "${LLM_BMAX+x}" ]]; then
        LLM_BMAX=512
    fi
    # At full context the optional graph captures and plan warmup reserve
    # enough transient VRAM to make the otherwise useful 5.9-GiB cache fail
    # on gfx1201.  Disable them by default; explicit caller settings win.
    if [[ -z "${LLM_HC_GRAPHS+x}" ]]; then LLM_HC_GRAPHS=0; fi
    if [[ -z "${LLM_QWEN_PRE_GRAPHS+x}" ]]; then LLM_QWEN_PRE_GRAPHS=0; fi
    if [[ -z "${LLM_QWEN4_PRE_GRAPHS+x}" ]]; then LLM_QWEN4_PRE_GRAPHS=0; fi
    if [[ -z "${LLM_PLAN_PREWARM+x}" ]]; then LLM_PLAN_PREWARM=0; fi
else
    if [[ -z "${LLM_QWEN4_KV_QUANT+x}" ]]; then LLM_QWEN4_KV_QUANT=none; fi
fi
# The approximate coding path reaches a scratch-pressure boundary around 8K
# on gfx1201. Use the validated 7.2-GiB/512-row profile below 16K and retain
# 5.9 GiB for longer requests; explicit cache and BMAX settings win.
if [[ "${vram_profile}" == "16g" && "${context}" -ge 8192 &&
      "${context}" -lt 131072 && "${LLM_QWEN4_APPROX_DECODE:-0}" != "0" ]]; then
    if [[ -z "${QWEN38_MOE_CACHE_MB+x}" ]]; then
        if [[ "${context}" -lt 16384 ]]; then cache_mb=7200; else cache_mb=5900; fi
    fi
    if [[ -z "${LLM_BMAX+x}" ]]; then LLM_BMAX=512; fi
fi
if [[ "${batch_prefill_set}" -ne 0 ]]; then
    case "${batch_prefill}" in
        0|1) : ;;
        *) echo "QWEN38_BATCH_PREFILL must be 0 or 1" >&2; exit 2 ;;
    esac
fi
# Direct copies measured faster for exact 256K decode on gfx1201. Keep the
# asynchronous pipeline available as an explicit experiment.
copy_pipeline="${LLM_MOE_COPY_PIPELINE:-0}"
device_refresh_interval="${LLM_QWEN4_DEVICE_REFRESH_INTERVAL:-}"
if [[ -z "${device_refresh_interval}" ]]; then
    # Approximate decode is an explicit opt-in.  The validated coding-quality
    # cadence at 256K is a six-token exact refresh; exact mode keeps the
    # historical value for its separate refresh path.
    if [[ "${LLM_QWEN4_APPROX_DECODE:-0}" != "0" ]]; then
        device_refresh_interval=6
    else
        device_refresh_interval=32
    fi
fi
device_refresh_start="${LLM_QWEN4_DEVICE_REFRESH_START_LAYER:-}"
if [[ -z "${device_refresh_start}" ]]; then
    # Short coding requests are sensitive to early-layer cold routes. Keep
    # the first half exact and approximate only the deeper layers; the
    # validated long-context profile retains resident approximation throughout.
    if [[ "${LLM_QWEN4_APPROX_DECODE:-0}" != "0" && "${context}" -lt 32768 ]]; then
        device_refresh_start=24
    else
        device_refresh_start=0
    fi
fi
cpu_decode_misses="${LLM_MOE_CPU_DECODE_MISSES:-0}"
if [[ -z "${LLM_MOE_CPU_DECODE_MISSES+x}" &&
      "${LLM_QWEN4_APPROX_DECODE:-0}" != "0" &&
      "${context}" -lt 32768 && "${device_refresh_start}" -gt 0 ]]; then
    cpu_decode_misses=1
fi
# The stateful Qwen4 batch dispatcher is retained for explicit diagnostics.
# Real heterogeneous multi-chunk HTTP requests are not yet parity-safe; scalar
# production remains unchanged and callers can override either knob explicitly.
qwen_batch_multi="${LLM_QWEN4_BATCH_MULTI_CHUNK:-0}"
qwen_batch_stateful="${LLM_QWEN4_BATCH_STATEFUL:-0}"
qwen_batch_min="${LLM_QWEN4_BATCH_MIN_TOKENS:-128}"
qwen_prefill_gpu_topk="${LLM_QWEN4_PREFILL_GPU_TOPK:-0}"
# The depth-weighted table is parity-safe and improved the matched scalar
# 512-token prefill control (21.64 vs 19.04 tok/s, same hash). Keep it enabled
# for HTTP prefill too; callers can opt out for decode-focused A/B runs.
qwen_prefill_balance="${LLM_QWEN4_PREFILL_CACHE_BALANCE:-1}"
grouped_prefill="${LLM_MOE_GROUPED_PREFILL:-0}"
prefill_copy_pipeline="${LLM_QWEN4_PREFILL_COPY_PIPELINE:-0}"
prefill_copy_max="${LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS:-2048}"
prefill_publish_chunk="${LLM_QWEN4_PREFILL_COPY_PIPELINE_PUBLISH_CHUNK:-0}"
delayed_cache="${LLM_QWEN4_DELAYED_CACHE:-0}"
if [[ "${prefill_copy_pipeline}" != "0" &&
      -z "${LLM_QWEN4_PREFILL_COPY_PIPELINE_PUBLISH_CHUNK+x}" ]]; then
    # The overlap guard is keyed by the published per-chunk request length;
    # an explicit pipeline request must therefore publish its chunks unless
    # the caller deliberately disables publication.
    prefill_publish_chunk=1
fi
if [[ "${prefill_copy_pipeline}" != "0" &&
      "${context}" -ge 4096 &&
      -z "${LLM_QWEN4_PREFILL_CACHE_BALANCE+x}" ]]; then
    # Match the measured 4K overlap profile; short requests retain the
    # depth-balanced cache policy by default.
    qwen_prefill_balance=0
fi
if ! [[ "${prefill_copy_max}" =~ ^[0-9]+$ ]] ||
   (( prefill_copy_max < 1 || prefill_copy_max > 4096 )); then
    prefill_copy_max=2048
fi
if [[ "${context}" -ge 131072 &&
      "${batch_prefill:-${LLM_QWEN4_BATCH:-0}}" != "1" &&
      -z "${LLM_QWEN4_PREFILL_GPU_TOPK+x}" ]]; then
    qwen_prefill_gpu_topk=1
fi
if [[ "${batch_prefill:-${LLM_QWEN4_BATCH:-0}}" == "1" ]]; then
    # Real HTTP prompts exercise recurrent state and heterogeneous token
    # distributions; the padded direct benchmark is not sufficient parity
    # evidence.  Keep multi-chunk batching diagnostic-only until that route
    # survives a full request without a runner crash.
    if [[ -z "${LLM_QWEN4_BATCH_STATEFUL+x}" ]]; then qwen_batch_stateful=1; fi
fi
if [[ "${QWEN38_DRY_RUN:-0}" != "0" ]]; then
    printf 'q38fn server profile: vram=%s context=%s kv=%s cache_mb=%s bmax=%s batch=%s stateful=%s multi=%s topk=%s balance=%s copy=%s delayed=%s prefill_copy=%s prefill_copy_max=%s publish_chunk=%s grouped=%s cpu_misses=%s approx_start=%s\n' \
        "${vram_profile}" "${context}" "${LLM_QWEN4_KV_QUANT}" "${cache_mb}" \
        "${LLM_BMAX:-${profile_bmax}}" "${batch_prefill:-${LLM_QWEN4_BATCH:-0}}" "${qwen_batch_stateful}" "${qwen_batch_multi}" "${qwen_prefill_gpu_topk}" "${qwen_prefill_balance}" "${copy_pipeline}" "${delayed_cache}" "${prefill_copy_pipeline}" "${prefill_copy_max}" "${prefill_publish_chunk}" "${grouped_prefill}" \
        "${cpu_decode_misses}" "${device_refresh_start}"
    exit 0
fi

# Keep the measured 65K-context profile: 512-row grouped prefill sustains
# 100+ tok/s within this VRAM budget. Larger batches exhausted VRAM, and
# ungrouped pipelined prefill reduced subsequent decode throughput. See
# QWEN38_PREFILL_TUNING.md for the paired prefill/decode measurements.
# The 640-wide Q4_K experts run faster with one warp per output row.
# The matching Q5_1 down projection benefits from two warps per output row.
# On fast approximate steps retain cold routes with >=0.20 router weight on
# the CPU, then run an exact route refresh every four tokens.  This avoids the
# malformed coding completions observed with all-miss hit-only execution.
# The runner initializes ggml's CPU lookup tables before using these kernels.
# Grouped Qwen4 prefill remains diagnostic-only on gfx1201: full-depth grouped
# launches have reproduced late-layer VM resets. Keep the public server on the
# stable ungrouped route unless explicitly overridden.
# RX 9070 XT's IQ3/SSM warp-per-row path is faster at 128 threads; retain an
# environment override for other GPUs.
# Long-context Qwen4 batching is selected above after the streamed parity gate;
# set QWEN38_BATCH_PREFILL=0 to force the conservative scalar route.
exec env \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
    OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
    OMP_PLACES="${OMP_PLACES:-cores}" \
    LLM_MOE_REGISTER_HOST="${LLM_MOE_REGISTER_HOST:-0}" \
    LLM_MOE_COPY_PIPELINE="${copy_pipeline}" \
    LLM_MOE_LFU_CACHE="${LLM_MOE_LFU_CACHE:-0}" \
    LLM_MOE_CPU_DECODE_MISSES="${cpu_decode_misses}" \
    LLM_MOE_CPU_REFILLS_PER_LAYER="${LLM_MOE_CPU_REFILLS_PER_LAYER:-1}" \
    LLM_MOE_CPU_MIN_WEIGHT="${LLM_MOE_CPU_MIN_WEIGHT:-0}" \
    LLM_QWEN4_DELAYED_CACHE="${delayed_cache}" \
    LLM_QWEN4_DELAYED_REFILL_INTERVAL="${LLM_QWEN4_DELAYED_REFILL_INTERVAL:-2}" \
    LLM_QWEN4_DEVICE_REFRESH_INTERVAL="${device_refresh_interval}" \
    LLM_QWEN4_DEVICE_REFRESH_START_LAYER="${device_refresh_start}" \
    LLM_QWEN4_APPROX_DECODE="${LLM_QWEN4_APPROX_DECODE:-0}" \
    LLM_QWEN4_BATCH="${batch_prefill:-${LLM_QWEN4_BATCH:-0}}" \
    LLM_QWEN4_BATCH_MIN_TOKENS="${qwen_batch_min}" \
    LLM_QWEN4_PREFILL_GPU_TOPK="${qwen_prefill_gpu_topk}" \
    LLM_QWEN4_PREFILL_COPY_PIPELINE="${prefill_copy_pipeline}" \
    LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS="${prefill_copy_max}" \
    LLM_QWEN4_PREFILL_COPY_PIPELINE_PUBLISH_CHUNK="${prefill_publish_chunk}" \
    LLM_QWEN4_BATCH_MULTI_CHUNK="${qwen_batch_multi}" \
    LLM_QWEN4_BATCH_STATEFUL="${qwen_batch_stateful}" \
    LLM_QWEN4_PREFILL_CACHE_BALANCE="${qwen_prefill_balance}" \
    LLM_QWEN4_NATIVE_BATCH_QKV="${LLM_QWEN4_NATIVE_BATCH_QKV:-0}" \
    LLM_QWEN4_KV_QUANT="${LLM_QWEN4_KV_QUANT}" \
    LLM_ATTN_PREFILL_I8_WARP="${LLM_ATTN_PREFILL_I8_WARP:-0}" \
    LLM_ATTN_PREFILL_I8_GQA4="${LLM_ATTN_PREFILL_I8_GQA4:-0}" \
    LLM_Q4_2W="${LLM_Q4_2W:-0}" \
    LLM_Q5_DOWN_2W="${LLM_Q5_DOWN_2W:-1}" \
    LLM_HC_GRAPHS="${LLM_HC_GRAPHS:-1}" \
    LLM_QWEN_PRE_GRAPHS="${LLM_QWEN_PRE_GRAPHS:-1}" \
    LLM_QWEN4_PRE_GRAPHS="${LLM_QWEN4_PRE_GRAPHS:-1}" \
    LLM_PLAN_PREWARM="${LLM_PLAN_PREWARM:-1}" \
    LLM_MOE_STREAM_SLOTS="${LLM_MOE_STREAM_SLOTS:-4}" \
    LLM_MW_THREADS="${LLM_MW_THREADS:-128}" \
    LLM_MOE_CPU_LIB="${cpu_lib}" \
    LLM_MOE_CPU_PREFILL_MAX_COUNT="${LLM_MOE_CPU_PREFILL_MAX_COUNT:-2}" \
    LLM_MOE_CPU_PREFILL_MAX_JOBS="${LLM_MOE_CPU_PREFILL_MAX_JOBS:-160}" \
    LLM_QWEN4_PRE_GRAPHS="${LLM_QWEN4_PRE_GRAPHS:-1}" \
    LLM_BMAX="${LLM_BMAX:-${profile_bmax}}" \
    LLM_MOE_GROUPED_PREFILL="${grouped_prefill}" \
    python3 "${runner_dir}/codex_server.py" "${model}" \
    --runner "${runner_dir}/test_hip_llm" \
    --context "${context}" \
    --max-output "${max_output}" \
    --port "${port}" \
    --host "${host}" \
    --moe-cache-mb "${cache_mb}" \
    --coding --qwen4-coding-profile "$@"
