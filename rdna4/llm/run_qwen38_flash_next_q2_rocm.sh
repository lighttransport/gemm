#!/usr/bin/env bash
set -euo pipefail

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="${QWEN38_Q2_RUNNER:-${runner_dir}/test_hip_llm}"
model="${QWEN38_Q2_MODEL:-/mnt/disk1/models/q38nf/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}"
rocm_lib="${QWEN38_ROCM_LIB:-/opt/rocm/lib}"

if [[ -r "${rocm_lib}/libamdhip64.so" ]]; then
    export ROCEW_ROCM_LIB="${ROCEW_ROCM_LIB:-${rocm_lib}}"
    export LD_LIBRARY_PATH="${rocm_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ ! -r "${model}" ]]; then
    echo "Qwen3.8 Q2_0 model shard is unavailable: ${model}" >&2
    exit 1
fi
if [[ ! -x "${runner}" ]]; then
    echo "Build the ROCm runner first: make -C rdna4/llm test_hip_llm" >&2
    exit 1
fi
if [[ ! -r /dev/kfd || ! -w /dev/kfd ]]; then
    echo "ROCm KFD device is unavailable to this process" >&2
    exit 1
fi

# Scalar prefill shares the exact CPU-miss path with decode. Keep its scratch
# bound small at 256K; the runner sizes the GPU expert cache from free VRAM.
export LLM_BMAX="${LLM_BMAX:-1}"
export LLM_QWEN4_BATCH=0
export LLM_MOE_REGISTER_HOST=0
export LLM_QWEN4_APPROX_DECODE=0
export LLM_QWEN4_MAPPED_MISSES=0
export LLM_QWEN4_DIRECT_MISSES=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

exec "${runner}" "${model}" --gpu-only --qwen4-exact \
    --qwen4-ple-backend ssd --qwen4-kv-quant i8 -s 262144 "$@"
