#!/usr/bin/env bash
set -euo pipefail

# Opt-in quality/throughput sweep for resident-expert approximate decode.
# Logs and extracted snippets stay in the repository-local tmp directory.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench="${root_dir}/bench_qwen38_sub32_target.sh"
out_dir="${QWEN38_APPROX_LOG_DIR:-${root_dir}/tmp/approx_coherence}"
mkdir -p "${out_dir}"
# The coherence sweep intentionally exercises refresh=8 as well as the
# production refresh=6 setting.  Keep enough VRAM/scratch headroom for that
# stress point on a 16-GiB RX 9070 XT; callers can override for cache/tile
# experiments.  The 8K loop applies its separately validated 5.9-GiB/512-row
# profile below.
approx_cache_mb="${QWEN38_APPROX_CACHE_MB:-7200}"
approx_bmax="${QWEN38_APPROX_BMAX:-1024}"
read -r -a approx_contexts <<< "${QWEN38_APPROX_CONTEXTS:-1024 4096 8192}"
read -r -a approx_refreshes <<< "${QWEN38_APPROX_REFRESHES:-4 6 8}"
approx_decode_tokens="${QWEN38_APPROX_DECODE_TOKENS:-64}"
approx_prompt="${QWEN38_APPROX_PROMPT:-}"
approx_batch="${QWEN38_APPROX_BATCH:-0}"
approx_batch_stateful="${QWEN38_APPROX_BATCH_STATEFUL:-0}"
approx_batch_multi="${QWEN38_APPROX_BATCH_MULTI_CHUNK:-0}"
approx_decode="${QWEN38_APPROX_DECODE:-1}"
approx_hits_only="${QWEN38_APPROX_DEVICE_HITS_ONLY:-1}"
case "${approx_batch}:${approx_batch_stateful}:${approx_batch_multi}" in
    0:0:0|1:0:0|1:1:0|1:1:1) ;;
    *) echo "approx coherence: invalid batch controls" >&2; exit 2 ;;
esac
case "${approx_decode}:${approx_hits_only}" in
    0:0|0:1|1:0|1:1) ;;
    *) echo "approx coherence: invalid decode controls" >&2; exit 2 ;;
esac
approx_prompt_default=0
        if [[ -z "${approx_prompt}" ]]; then
    # Keep the standalone benchmark's prompt identical to codex_server.py:
    # the runner does not apply a chat template itself.  In particular, the
    # explicit empty thinking frame is required for Qwen3.8 non-thinking
    # coding output; plain user text measures an unframed continuation and
    # can look incoherent even when the HTTP path is healthy.
    approx_prompt_default=1
fi

for context in "${approx_contexts[@]}"; do
    [[ "${context}" =~ ^[1-9][0-9]*$ ]] || {
        echo "approx coherence: invalid context ${context}" >&2
        exit 2
    }
    cache_for_run="${approx_cache_mb}"
    bmax_for_run="${approx_bmax}"
    if (( context >= 8192 )); then
        # At 8K the recurrent/attention scratch peak is the limiting
        # allocation. The validated 7.2-GiB/512-row profile is safe below
        # 16K; keep 5.9 GiB for longer contexts.
        if [[ -z "${QWEN38_APPROX_CACHE_MB+x}" ]]; then
            if (( context < 16384 )); then cache_for_run=7200; else cache_for_run=5900; fi
        fi
        if [[ -z "${QWEN38_APPROX_BMAX+x}" ]]; then bmax_for_run=512; fi
    fi
    for refresh in "${approx_refreshes[@]}"; do
        [[ "${refresh}" =~ ^[1-9][0-9]*$ ]] || {
            echo "approx coherence: invalid refresh ${refresh}" >&2
            exit 2
        }
        log="${out_dir}/c${context}-r${refresh}.log"
        max_seq=$((context + approx_decode_tokens + 128))
        prompt_for_run="${approx_prompt}"
        if (( approx_prompt_default )); then
            # Use real user-context tokens rather than --prefill-len's
            # last-token padding, which can make a quality check degenerate
            # into a repeated newline/marker continuation.  Keep the final
            # coding request at the end of the same ChatML frame used by the
            # HTTP server.
            context_text=""
            for ((i = 0; i < context / 12; i++)); do
                context_text+="The implementation must be portable, deterministic, and concise. "
            done
            prompt_for_run="<|im_start|>user\n${context_text}\nWrite a C function int clamp(int x, int lo, int hi) that clamps x to the inclusive range. Return only compilable C code.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        fi
        prompt_file=""
        if (( ${#prompt_for_run} > 100000 )); then
            # Avoid ARG_MAX when a real 32K+ prompt is passed to the runner.
            # Keep the file in the repository-local temporary tree per the
            # project policy; the benchmark consumes it before returning.
            prompt_file="${out_dir}/prompt-c${context}.txt"
            printf '%s' "${prompt_for_run}" >"${prompt_file}"
            prompt_for_run=""
        fi
        # The sweep is a quality gate, not a peak-throughput benchmark. Keep
        # the first 40 layers exact by default; callers can explicitly sweep
        # the faster layer-24 or full-depth profiles.
        QWEN38_SUB32_CONTEXT="${max_seq}" \
        QWEN38_SUB32_PREFILL="${context}" \
        QWEN38_SUB32_DECODE="${approx_decode_tokens}" \
        QWEN38_SUB32_NO_PAD=1 \
        QWEN38_SUB32_PROMPT="${prompt_for_run}" \
        QWEN38_SUB32_PROMPT_FILE="${prompt_file}" \
        QWEN38_SUB32_CODING=1 \
        LLM_QWEN4_APPROX_DECODE="${approx_decode}" \
        LLM_QWEN4_DEVICE_HITS_ONLY="${approx_hits_only}" \
        LLM_QWEN4_DEVICE_REFRESH_INTERVAL="${refresh}" \
        LLM_QWEN4_DEVICE_REFRESH_START_LAYER="${QWEN38_APPROX_START_LAYER:-40}" \
        LLM_QWEN4_APPROX_CPU_MIN_WEIGHT="${LLM_QWEN4_APPROX_CPU_MIN_WEIGHT:-0.20}" \
        LLM_QWEN4_BATCH="${approx_batch}" \
        LLM_QWEN4_BATCH_STATEFUL="${approx_batch_stateful}" \
        LLM_QWEN4_BATCH_MULTI_CHUNK="${approx_batch_multi}" \
        LLM_GEN_TEXT=1 \
        QWEN38_SUB32_CACHE_MB="${cache_for_run}" \
        LLM_BMAX="${bmax_for_run}" \
        QWEN38_SUB32_LOG="${log}" \
        "${bench}" >/dev/null

        python3 - "${log}" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text(errors="replace")
if "Result: PASS" not in s:
    raise SystemExit(f"benchmark failed: {p}")
try:
    body = s.split("=== Generated text ===", 1)[1].split("=== end ===", 1)[0]
except IndexError:
    raise SystemExit(f"generated text missing: {p}")
# test_hip_llm prints the tokenizer's visible UTF-8 markers rather than
# returning already-decoded text. Normalize the Qwen BPE whitespace markers
# before extracting C; otherwise a valid function is falsely rejected by gcc.
body = (body.replace("Ċ", "\n")
            .replace("Ġ", " ")
            .replace("ĉ", "\t")
            .replace("\\n", "\n")
            .replace("\\t", "\t"))
# A special-token burst is a model/prompt failure, not C source.  Reject it
# before the permissive brace search below; otherwise a random `return` or
# brace in corrupted text can make gcc accept an empty/comment-only snippet.
if any(marker in body for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
    raise SystemExit(f"special-token corruption in generated text: {p}")
for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
    body = body.replace(marker, "")
start = body.find("{")
end = body.rfind("}")
if (start < 0 or end <= start or "return" not in body or
        "int clamp" not in body):
    raise SystemExit(f"incoherent coding output: {p}")
if "```" in body:
    fenced = body.split("```", 2)[1]
    if fenced.startswith("c\n"):
        fenced = fenced[2:]
    snippet = fenced.strip()
else:
    fn_start = body.rfind("int ", 0, start)
    if fn_start < 0:
        fn_start = start
    snippet = body[fn_start:end + 1].strip()
tmp = p.with_suffix(".c")
tmp.write_text(snippet + "\n")
PY
        gcc -fsyntax-only "${log%.log}.c"
        echo "approx coherence PASS context=${context} refresh=${refresh} log=${log}"
    done
done
