#!/usr/bin/env bash
set -euo pipefail

# Repeatability + throughput gate for the Qwen3.8-Flash-Next RDNA4 runner.
#
# Runs the same greedy request N times in a single runner process (the runner
# reloads the 48-GiB model once, then re-resets recurrent/KV/PLE state between
# repeats via --bench-repeat), and requires every repeat to produce the same
# first decoded token and the same full sequence hash.  It reports prefill,
# decode, and end-to-end tok/s plus peak VRAM and the GPU clock/temperature
# before and after so a throughput regression can be distinguished from clock
# variance.
#
# This is the Phase 0 gate: no profile may be promoted to a serving default
# until it passes here.  It is a throughput/stability smoke, not a quality
# claim (use the coding coherence harness for output quality).
#
# Profiles (QWEN38_TARGET_PROFILE):
#   scalar-exact  quality-safe scalar route, exact decode (default)
#   fast          pinned host + BMAX=2048 + 7.8-GiB cache + GPU prefill top-k
#   batch         explicit batched prefill (LLM_QWEN4_BATCH=1), exact decode,
#                 GPU-only cold experts (deterministic across cache states)
#   batch-cpu     same but with the mixed CPU/GPU cold-expert path (diagnostic:
#                 its arithmetic differs from the GPU path, so in-process
#                 repeats can hash-differ even though fresh processes match)
#   batch4k       deterministic single-dispatch 4K profile (BMAX=4096, no
#                 multi-chunk carry, pinned host, 5-GiB cache, GPU top-k,
#                 async cold uploads, expert cache reset per repeat)
#   batch4k-stage same, but groups cold experts through the staging banks with a
#                 4-GiB cache; deterministic and ~180 prefill tok/s (the
#                 grouped-routed profile)
#   approx        resident-hit approximate decode (quality-changing; explicit)
#
# Examples:
#   QWEN38_TARGET_PROFILE=fast QWEN38_TARGET_PREFILL=4096 \
#     QWEN38_TARGET_DECODE=64 rdna4/llm/bench_qwen38_target.sh
#   QWEN38_TARGET_PROFILE=batch QWEN38_TARGET_BATCH_SSM=1 \
#     QWEN38_TARGET_ATTN_MAX_LAYER=47 rdna4/llm/bench_qwen38_target.sh

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="${QWEN38_RUNNER:-${root_dir}/test_hip_llm}"
model="${QWEN38_MODEL:-/mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
cpu_lib="${LLM_MOE_CPU_LIB:-/mnt/nvme02/work/llama.cpp/build-codex-hetero-dev2/bin/libggml-cpu.so.0.22.0}"
if [[ ! -r "${cpu_lib}" ]]; then cpu_lib=""; fi

profile="${QWEN38_TARGET_PROFILE:-scalar-exact}"
prefill="${QWEN38_TARGET_PREFILL:-4096}"
decode="${QWEN38_TARGET_DECODE:-64}"
context="${QWEN38_TARGET_CONTEXT:-8192}"
repeats="${QWEN38_TARGET_REPEATS:-3}"
run_timeout="${QWEN38_TARGET_TIMEOUT:-1800}"
coding="${QWEN38_TARGET_CODING:-0}"
prompt="${QWEN38_TARGET_PROMPT:-}"
prompt_file="${QWEN38_TARGET_PROMPT_FILE:-}"
# CPU cold-expert execution changes arithmetic relative to the GPU resident
# path, and expert-cache warmth selects which path an expert takes.  That makes
# in-process repeats hash-different even though fresh-process runs match.  The
# gate therefore defaults CPU expert work off (GPU-only, deterministic); set
# QWEN38_TARGET_CPU_EXPERTS=1 to measure the mixed CPU/GPU path.
cpu_experts="${QWEN38_TARGET_CPU_EXPERTS:-0}"
if [[ "${cpu_experts}" == "0" ]]; then
    cpu_prefill_jobs=0
    cpu_decode_misses=0
else
    cpu_prefill_jobs="${LLM_MOE_CPU_PREFILL_MAX_JOBS:-160}"
    cpu_decode_misses="${LLM_MOE_CPU_DECODE_MISSES:-1}"
fi
log_file="${QWEN38_TARGET_LOG:-${root_dir}/tmp/qwen38_target_${profile}.log}"
export TMPDIR="${TMPDIR:-${root_dir}/tmp}"

if ! [[ "${repeats}" =~ ^[0-9]+$ ]] || (( repeats < 1 )); then
    echo "QWEN38_TARGET_REPEATS must be a positive integer" >&2
    exit 2
fi
if ! [[ "${run_timeout}" =~ ^[0-9]+$ ]] || (( run_timeout < 1 )); then
    echo "QWEN38_TARGET_TIMEOUT must be a positive integer" >&2
    exit 2
fi
if ! [[ "${prefill}" =~ ^[0-9]+$ ]] || (( prefill < 1 )); then
    echo "QWEN38_TARGET_PREFILL must be a positive integer" >&2
    exit 2
fi
(( decode >= 1 )) || { echo "QWEN38_TARGET_DECODE must be >= 1 (hash gate needs decoded tokens)" >&2; exit 2; }
if (( prefill + decode > context )); then
    echo "QWEN38_TARGET_CONTEXT (${context}) must cover prefill+decode ($((prefill + decode)))" >&2
    exit 2
fi

# Native/quantized batched-kernel switches.  Only the explicit `batch` profile
# turns these on by default; other profiles keep the parity-safe scalar bodies.
# The Q6_K SSM projections + fused recurrence are what make batched prefill
# fast, and they change arithmetic order (diagnostic until parity is proven).
copy_pipeline="${LLM_MOE_COPY_PIPELINE:-0}"
prefill_staging="${QWEN38_TARGET_PREFILL_STAGING:-0}"
native_batch_qkv="${LLM_QWEN4_NATIVE_BATCH_QKV:-0}"
ssm_batch_q6k="${LLM_SSM_BATCH_Q6K:-0}"
ssm_batch_conv="${LLM_SSM_BATCH_CONV:-0}"
ssm_batch_recur="${LLM_SSM_BATCH_RECURRENCE:-0}"
ssm_batch_parity="${LLM_SSM_BATCH_PARITY:-0}"
ssm_batch_warp="${LLM_SSM_BATCH_WARP:-0}"

case "${profile}" in
    scalar-exact)
        cache_mb="${QWEN38_MOE_CACHE_MB:-7200}"
        bmax="${LLM_BMAX:-512}"
        register_host="${LLM_MOE_REGISTER_HOST:-0}"
        gpu_topk="${LLM_QWEN4_PREFILL_GPU_TOPK:-0}"
        qwen_batch="${LLM_QWEN4_BATCH:-0}"
        batch_ssm="${QWEN38_TARGET_BATCH_SSM:-0}"
        attn_max="${QWEN38_TARGET_ATTN_MAX_LAYER:-2}"
        approx_decode="${LLM_QWEN4_APPROX_DECODE:-0}"
        device_hits_only="${LLM_QWEN4_DEVICE_HITS_ONLY:-0}"
        refresh="${LLM_QWEN4_DEVICE_REFRESH_INTERVAL:-2}"
        stream_chunk="${LLM_BENCH_STREAM_CHUNK:-512}"
        ;;
    fast)
        cache_mb="${QWEN38_MOE_CACHE_MB:-7800}"
        bmax="${LLM_BMAX:-2048}"
        register_host="${LLM_MOE_REGISTER_HOST:-1}"
        gpu_topk="${LLM_QWEN4_PREFILL_GPU_TOPK:-1}"
        qwen_batch="${LLM_QWEN4_BATCH:-0}"
        batch_ssm="${QWEN38_TARGET_BATCH_SSM:-0}"
        attn_max="${QWEN38_TARGET_ATTN_MAX_LAYER:-2}"
        approx_decode="${LLM_QWEN4_APPROX_DECODE:-0}"
        device_hits_only="${LLM_QWEN4_DEVICE_HITS_ONLY:-0}"
        refresh="${LLM_QWEN4_DEVICE_REFRESH_INTERVAL:-2}"
        stream_chunk="${LLM_BENCH_STREAM_CHUNK:-2048}"
        ;;
    batch|batch-cpu)
        cache_mb="${QWEN38_MOE_CACHE_MB:-5900}"
        bmax="${LLM_BMAX:-1024}"
        register_host="${LLM_MOE_REGISTER_HOST:-0}"
        gpu_topk="${LLM_QWEN4_PREFILL_GPU_TOPK:-0}"
        qwen_batch="${LLM_QWEN4_BATCH:-1}"
        batch_ssm="${QWEN38_TARGET_BATCH_SSM:-1}"
        attn_max="${QWEN38_TARGET_ATTN_MAX_LAYER:-47}"
        approx_decode="${LLM_QWEN4_APPROX_DECODE:-0}"
        device_hits_only="${LLM_QWEN4_DEVICE_HITS_ONLY:-0}"
        refresh="${LLM_QWEN4_DEVICE_REFRESH_INTERVAL:-2}"
        stream_chunk="${LLM_BENCH_STREAM_CHUNK:-1024}"
        native_batch_qkv="${LLM_QWEN4_NATIVE_BATCH_QKV:-1}"
        ssm_batch_q6k="${LLM_SSM_BATCH_Q6K:-1}"
        ssm_batch_conv="${LLM_SSM_BATCH_CONV:-1}"
        ssm_batch_recur="${LLM_SSM_BATCH_RECURRENCE:-1}"
        ssm_batch_parity="${LLM_SSM_BATCH_PARITY:-1}"
        ssm_batch_warp="${LLM_SSM_BATCH_WARP:-0}"
        if [[ "${profile}" == "batch-cpu" ]]; then
            cpu_prefill_jobs="${LLM_MOE_CPU_PREFILL_MAX_JOBS:-160}"
            cpu_decode_misses="${LLM_MOE_CPU_DECODE_MISSES:-1}"
        fi
        ;;
    batch4k|batch4k-stage)
        # Deterministic single-dispatch 4K profile: one 4096-row batched prefill
        # (no multi-chunk state carry), pinned host weights, async cold uploads,
        # GPU router top-k, and a resident cache on the 16-GiB card.  The
        # `-stage` variant groups cold experts through the staging banks and
        # uses a 4000-MiB cache to leave headroom (median ~180 prefill).
        if [[ "${profile}" == "batch4k-stage" ]]; then
            prefill_staging="${QWEN38_TARGET_PREFILL_STAGING:-1}"
            cache_mb="${QWEN38_MOE_CACHE_MB:-4000}"
        else
            cache_mb="${QWEN38_MOE_CACHE_MB:-5000}"
        fi
        bmax="${LLM_BMAX:-4096}"
        register_host="${LLM_MOE_REGISTER_HOST:-1}"
        gpu_topk="${LLM_QWEN4_PREFILL_GPU_TOPK:-1}"
        qwen_batch="${LLM_QWEN4_BATCH:-1}"
        batch_ssm="${QWEN38_TARGET_BATCH_SSM:-1}"
        attn_max="${QWEN38_TARGET_ATTN_MAX_LAYER:-47}"
        approx_decode="${LLM_QWEN4_APPROX_DECODE:-0}"
        device_hits_only="${LLM_QWEN4_DEVICE_HITS_ONLY:-0}"
        refresh="${LLM_QWEN4_DEVICE_REFRESH_INTERVAL:-2}"
        stream_chunk="${LLM_BENCH_STREAM_CHUNK:-0}"
        # The async cold-upload pipeline is faster (~147 vs ~132 prefill) and is
        # request-isolated because the cache reset drains the copy stream and
        # clears moe_pipeline_valid.  Set LLM_MOE_COPY_PIPELINE=0 for direct
        # copies (a lower-overhead A/B).
        copy_pipeline="${LLM_MOE_COPY_PIPELINE:-1}"
        native_batch_qkv="${LLM_QWEN4_NATIVE_BATCH_QKV:-1}"
        ssm_batch_q6k="${LLM_SSM_BATCH_Q6K:-1}"
        ssm_batch_conv="${LLM_SSM_BATCH_CONV:-1}"
        ssm_batch_recur="${LLM_SSM_BATCH_RECURRENCE:-1}"
        ssm_batch_parity="${LLM_SSM_BATCH_PARITY:-1}"
        ssm_batch_warp="${LLM_SSM_BATCH_WARP:-0}"
        ;;
    approx)
        cache_mb="${QWEN38_MOE_CACHE_MB:-7200}"
        bmax="${LLM_BMAX:-512}"
        register_host="${LLM_MOE_REGISTER_HOST:-0}"
        gpu_topk="${LLM_QWEN4_PREFILL_GPU_TOPK:-0}"
        qwen_batch="${LLM_QWEN4_BATCH:-0}"
        batch_ssm="${QWEN38_TARGET_BATCH_SSM:-0}"
        attn_max="${QWEN38_TARGET_ATTN_MAX_LAYER:-2}"
        approx_decode="${LLM_QWEN4_APPROX_DECODE:-1}"
        device_hits_only="${LLM_QWEN4_DEVICE_HITS_ONLY:-1}"
        refresh="${LLM_QWEN4_DEVICE_REFRESH_INTERVAL:-6}"
        stream_chunk="${LLM_BENCH_STREAM_CHUNK:-512}"
        ;;
    *)
        echo "unknown QWEN38_TARGET_PROFILE=${profile} (use scalar-exact, fast, batch, batch-cpu, batch4k, batch4k-stage, approx)" >&2
        exit 2
        ;;
esac

# The batched dispatcher refuses a request larger than BMAX unless the
# explicit stateful multi-chunk path is forced.  Mirror the sub-32K diagnostic
# so the `batch` profile actually exercises the batched route.
multi_chunk="${LLM_QWEN4_BATCH_MULTI_CHUNK:-0}"
stateful="${LLM_QWEN4_BATCH_STATEFUL:-0}"
force_multi="${LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE:-0}"
if [[ "${profile}" == "batch" || "${profile}" == "batch-cpu" ]]; then
    multi_chunk="${LLM_QWEN4_BATCH_MULTI_CHUNK:-1}"
    stateful="${LLM_QWEN4_BATCH_STATEFUL:-1}"
    if [[ -z "${LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE+x}" ]] && (( prefill > bmax )); then
        force_multi=1
    fi
fi

if [[ "${QWEN38_DRY_RUN:-0}" != "0" ]]; then
    printf 'target gate profile: profile=%s prefill=%s decode=%s context=%s repeats=%s cache_mb=%s bmax=%s batch=%s batch_ssm=%s attn_max=%s gpu_topk=%s approx=%s coding=%s stream_chunk=%s q6k=%s recur=%s conv=%s parity=%s native_qkv=%s multi=%s stateful=%s force_multi=%s cpu_prefill_jobs=%s cpu_decode_misses=%s copy=%s staging=%s\n' \
        "${profile}" "${prefill}" "${decode}" "${context}" "${repeats}" "${cache_mb}" \
        "${bmax}" "${qwen_batch}" "${batch_ssm}" "${attn_max}" "${gpu_topk}" "${approx_decode}" "${coding}" "${stream_chunk:-auto}" \
        "${ssm_batch_q6k}" "${ssm_batch_recur}" "${ssm_batch_conv}" "${ssm_batch_parity}" "${native_batch_qkv}" \
        "${multi_chunk}" "${stateful}" "${force_multi}" "${cpu_prefill_jobs}" "${cpu_decode_misses}" "${copy_pipeline}" "${prefill_staging}"
    exit 0
fi

[[ -r "${model}" ]] || { echo "model not readable: ${model}" >&2; exit 2; }
[[ -x "${runner}" ]] || { echo "runner not executable: ${runner} (build test_hip_llm first)" >&2; exit 2; }
[[ -r /dev/kfd && -w /dev/kfd ]] || { echo "AMD KFD access unavailable: /dev/kfd missing or not rw" >&2; exit 2; }

# A competing ROCm client makes throughput and MES-reset results unattributable.
if command -v fuser >/dev/null 2>&1; then
    kfd_users="$(fuser /dev/kfd 2>/dev/null || true)"
    if [[ -n "${kfd_users//[[:space:]]/}" ]]; then
        echo "warning: /dev/kfd is already in use by: ${kfd_users}" >&2
        if [[ "${QWEN38_TARGET_REQUIRE_EXCLUSIVE_GPU:-0}" != "0" ]]; then
            echo "target gate refused: GPU contention" >&2
            exit 2
        fi
    fi
fi

mkdir -p "$(dirname "${log_file}")"
# Each gate run is a fresh measurement; never let a prior run's footers leak
# into the determinism check.
: >"${log_file}"

clock_report() {
    local phase="$1"
    command -v rocm-smi >/dev/null 2>&1 || return 0
    local clocks temps
    clocks="$(rocm-smi --showclocks 2>/dev/null | tr '\n' ' ' || true)"
    temps="$(rocm-smi --showtemp 2>/dev/null | tr '\n' ' ' || true)"
    {
        echo "=== GPU ${phase} ==="
        [[ -n "${clocks}" ]] && echo "clocks: ${clocks}"
        [[ -n "${temps}" ]] && echo "temp: ${temps}"
    } >>"${log_file}"
}

echo "target gate: profile=${profile} prefill=${prefill} decode=${decode} context=${context} repeats=${repeats} cache_mb=${cache_mb} bmax=${bmax} batch=${qwen_batch} coding=${coding} stream_chunk=${stream_chunk} cpu_prefill_jobs=${cpu_prefill_jobs} cpu_decode_misses=${cpu_decode_misses}" | tee -a "${log_file}"

prompt_args=()
if [[ -n "${prompt_file}" ]]; then
    [[ -r "${prompt_file}" ]] || { echo "prompt file not readable: ${prompt_file}" >&2; exit 2; }
    prompt_args+=(--prompt-file "${prompt_file}")
elif [[ -n "${prompt}" ]]; then
    prompt_args+=(-t "${prompt}")
fi

bench_args=("${runner}" "${model}" -s "${context}" --gpu-only-bench --bench \
    -n "${prefill}" --prefill-len "${prefill}" --decode "${decode}" \
    --bench-repeat "${repeats}" --moe-cache-mb "${cache_mb}" "${prompt_args[@]}")
if [[ "${coding}" != "0" ]]; then bench_args+=(--coding); fi
if [[ "${prefill_staging}" != "0" ]]; then
    bench_args+=(--qwen4-prefill-staging)
fi

clock_report "before"
set +e
timeout --foreground "${run_timeout}s" env \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
    LLM_MOE_CPU_LIB="${cpu_lib}" \
    LLM_QWEN4_KV_QUANT="${LLM_QWEN4_KV_QUANT:-none}" \
    LLM_MOE_REGISTER_HOST="${register_host}" \
    LLM_MOE_CACHE_MB="${cache_mb}" \
    LLM_BMAX="${bmax}" \
    LLM_MOE_CHUNK="${LLM_MOE_CHUNK:-${bmax}}" \
    LLM_MOE_COPY_PIPELINE="${copy_pipeline}" \
    LLM_MOE_STREAM_SLOTS="${LLM_MOE_STREAM_SLOTS:-2}" \
    LLM_MOE_LFU_CACHE="${LLM_MOE_LFU_CACHE:-0}" \
    LLM_QWEN4_PREFILL_GPU_TOPK="${gpu_topk}" \
    LLM_QWEN4_BATCH="${qwen_batch}" \
    LLM_QWEN4_BATCH_MIN_TOKENS="${LLM_QWEN4_BATCH_MIN_TOKENS:-128}" \
    LLM_QWEN4_BATCH_MULTI_CHUNK="${multi_chunk}" \
    LLM_QWEN4_BATCH_STATEFUL="${stateful}" \
    LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE="${force_multi}" \
    LLM_QWEN4_BATCH_SSM="${batch_ssm}" \
    LLM_QWEN4_BATCH_ATTN_MAX_LAYER="${attn_max}" \
    LLM_QWEN4_NATIVE_BATCH_QKV="${native_batch_qkv}" \
    LLM_SSM_BATCH_Q6K="${ssm_batch_q6k}" \
    LLM_SSM_BATCH_CONV="${ssm_batch_conv}" \
    LLM_SSM_BATCH_RECURRENCE="${ssm_batch_recur}" \
    LLM_SSM_BATCH_PARITY="${ssm_batch_parity}" \
    LLM_SSM_BATCH_WARP="${ssm_batch_warp}" \
    LLM_QWEN4_PREFILL_CACHE_BALANCE="${LLM_QWEN4_PREFILL_CACHE_BALANCE:-${gpu_topk}}" \
    LLM_MOE_GROUPED_PREFILL="${LLM_MOE_GROUPED_PREFILL:-0}" \
    LLM_MOE_CPU_DECODE_MISSES="${cpu_decode_misses}" \
    LLM_MOE_CPU_PREFILL_MAX_JOBS="${cpu_prefill_jobs}" \
    LLM_QWEN4_RESET_MOE_CACHE="${LLM_QWEN4_RESET_MOE_CACHE:-1}" \
    LLM_QWEN4_STAGE_PROMOTE="${LLM_QWEN4_STAGE_PROMOTE:-0}" \
    LLM_BENCH_WARMUP="${LLM_BENCH_WARMUP:-0}" \
    LLM_QWEN4_APPROX_DECODE="${approx_decode}" \
    LLM_QWEN4_DEVICE_HITS_ONLY="${device_hits_only}" \
    LLM_QWEN4_DEVICE_REFRESH_INTERVAL="${refresh}" \
    LLM_HC_GRAPHS="${LLM_HC_GRAPHS:-0}" \
    LLM_QWEN_PRE_GRAPHS="${LLM_QWEN_PRE_GRAPHS:-0}" \
    LLM_QWEN4_PRE_GRAPHS="${LLM_QWEN4_PRE_GRAPHS:-0}" \
    LLM_PLAN_PREWARM="${LLM_PLAN_PREWARM:-0}" \
    LLM_BENCH_STREAM_CHUNK="${stream_chunk}" \
    "${bench_args[@]}" >>"${log_file}" 2>&1
run_rc=$?
set -e
clock_report "after"

if (( run_rc != 0 )); then
    echo "target gate run failed (rc=${run_rc}); see ${log_file}" >&2
    grep -E 'ROCm device unavailable|HIP error at|failed|Result:' "${log_file}" | tail -20 >&2 || true
    exit 1
fi
if grep -Eq 'Failed to (load weights to GPU|init HIP runner)|ROCm device unavailable|HIP error at' "${log_file}"; then
    echo "target gate failed: GPU init/load error present in ${log_file}" >&2
    exit 1
fi

result_lines="$(grep -c 'Result: PASS' "${log_file}" || true)"
mapfile -t first_tokens < <(grep -oE 'First decoded token id=-?[0-9]+' "${log_file}" | sed 's/.*=//')
mapfile -t hashes < <(grep -oE 'sequence hash=[0-9a-f]+' "${log_file}" | sed 's/sequence hash=//')

if (( ${#hashes[@]} != repeats )); then
    echo "target gate FAIL: expected ${repeats} sequence-hash footers, saw ${#hashes[@]} (timed out or crashed mid-repeat?)" >&2
    grep -E 'Prefill:|Decode:|End-to-end:|Result:' "${log_file}" | tail -20 >&2 || true
    exit 1
fi
if (( ${#first_tokens[@]} != repeats )); then
    echo "target gate FAIL: expected ${repeats} first-token footers, saw ${#first_tokens[@]}" >&2
    exit 1
fi
if (( result_lines != repeats )); then
    echo "target gate FAIL: ${result_lines}/${repeats} repeats returned 'Result: PASS'" >&2
    exit 1
fi

unique_hashes="$(printf '%s\n' "${hashes[@]}" | sort -u | wc -l)"
unique_firsts="$(printf '%s\n' "${first_tokens[@]}" | sort -u | wc -l)"
if (( unique_hashes != 1 )); then
    echo "target gate FAIL: nondeterministic sequence hash across ${repeats} repeats: ${hashes[*]}" >&2
    exit 1
fi
if (( unique_firsts != 1 )); then
    echo "target gate FAIL: nondeterministic first token across ${repeats} repeats: ${first_tokens[*]}" >&2
    exit 1
fi

# Throughput: summarize min + median over repeats (clock-variance safe) rather
# than reporting a single best sample.  tok/s is the value after "->".
tok_s_values() {
    grep -E "^$1:" "${log_file}" | sed -E 's/.*->[[:space:]]*([0-9.]+)[[:space:]]*tok\/s.*/\1/'
}
min_median() {
    awk '
        function med(a, n,   b,i,j,t){ for(i=0;i<n;i++)b[i]=a[i]; for(i=0;i<n-1;i++)for(j=i+1;j<n;j++)if(b[j]<b[i]){t=b[i];b[i]=b[j];b[j]=t} return (n%2)?b[int(n/2)]:(b[n/2-1]+b[n/2])/2 }
        { v[n++]=$1; if(n==1||$1<mn)mn=$1 }
        END { if(n) printf "%.2f %.2f", mn, med(v,n); else printf "0 0" }'
}
vram_line="$(grep -E '^VRAM:' "${log_file}" | tail -1 || true)"

pf_stats="$(tok_s_values Prefill | min_median)"
dec_stats="$(tok_s_values Decode | min_median)"
e2e_stats="$(tok_s_values End-to-end | min_median)"
read -r pf_min pf_med <<<"${pf_stats}"
read -r dec_min dec_med <<<"${dec_stats}"
read -r e2e_min e2e_med <<<"${e2e_stats}"

echo "target gate PASS: profile=${profile} repeats=${repeats} hash=${hashes[0]} first_token=${first_tokens[0]}"
echo "  prefill tok/s: min=${pf_min} median=${pf_med}"
echo "  decode  tok/s: min=${dec_min} median=${dec_med}"
echo "  e2e     tok/s: min=${e2e_min} median=${e2e_med}"
[[ -n "${vram_line}" ]] && echo "  ${vram_line}"
echo "  log=${log_file}"
