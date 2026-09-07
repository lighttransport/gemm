#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash CPU+HIP coding/research server.
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${GLM53F_MODEL:-/home/syoyo/models/glm53f/iq3/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf}"
context="${GLM53F_CONTEXT:-4096}"
port="${GLM53F_API_PORT:-8080}"
host="${GLM53F_API_HOST:-127.0.0.1}"
max_output="${GLM53F_MAX_OUTPUT:-512}"

if [[ ! -r "${model}" ]]; then
    echo "GLM5.3-Flash model not found: ${model}" >&2
    echo "Set GLM53F_MODEL to the first GGUF shard." >&2
    exit 1
fi
if [[ ! -e /dev/kfd ]]; then
    echo "ROCm device unavailable: /dev/kfd is missing" >&2
    exit 1
fi
if ! compgen -G '/dev/dri/renderD*' > /dev/null; then
    echo "ROCm device unavailable: no /dev/dri/renderD* node is visible" >&2
    exit 1
fi

exec env \
    GLM5NEXT_HIP_DSA="${GLM5NEXT_HIP_DSA:-1}" \
    GLM5NEXT_HIP_KDA="${GLM5NEXT_HIP_KDA:-1}" \
    GLM5NEXT_HIP_DENSE="${GLM5NEXT_HIP_DENSE:-1}" \
    GLM5NEXT_HIP_DENSE_CACHE="${GLM5NEXT_HIP_DENSE_CACHE:-1}" \
    GLM5NEXT_HIP_MOE="${GLM5NEXT_HIP_MOE:-1}" \
    GLM5NEXT_HIP_MOE_THREADS="${GLM5NEXT_HIP_MOE_THREADS:-512}" \
    GLM5NEXT_HIP_IQ1_DP4A="${GLM5NEXT_HIP_IQ1_DP4A:-1}" \
    GLM5NEXT_CUDA_MOE="${GLM5NEXT_CUDA_MOE:-1}" \
    GLM5NEXT_CUDA_IQ1="${GLM5NEXT_CUDA_IQ1:-1}" \
    GLM5NEXT_CUDA_IQ1_SPLIT="${GLM5NEXT_CUDA_IQ1_SPLIT:-2}" \
    GLM5NEXT_CUDA_CACHE_MB="${GLM5NEXT_CUDA_CACHE_MB:-12000}" \
    GLM5NEXT_CUDA_IQ4XS="${GLM5NEXT_CUDA_IQ4XS:-1}" \
    GLM5NEXT_HIP_MOE_MAPPED="${GLM5NEXT_HIP_MOE_MAPPED:-1}" \
    GLM5NEXT_HIP_MOE_MAPPED_DOWN_CACHE="${GLM5NEXT_HIP_MOE_MAPPED_DOWN_CACHE:-1}" \
    GLM5NEXT_HIP_KDA_CACHE="${GLM5NEXT_HIP_KDA_CACHE:-1}" \
    GLM5NEXT_HIP_DSA_CACHE="${GLM5NEXT_HIP_DSA_CACHE:-1}" \
    GLM5NEXT_HIP_OUTPUT="${GLM5NEXT_HIP_OUTPUT:-1}" \
    GLM5NEXT_HIP_NEXTN="${GLM5NEXT_HIP_NEXTN:-1}" \
    GLM5NEXT_HIP_INDEXER="${GLM5NEXT_HIP_INDEXER:-1}" \
    GLM5NEXT_HIP_MHC="${GLM5NEXT_HIP_MHC:-1}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}" \
    OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
    OMP_PLACES="${OMP_PLACES:-cores}" \
    python3 "${runner_dir}/codex_server.py" "${model}" \
    --runner "${runner_dir}/test_hip_llm" \
    --context "${context}" \
    --max-output "${max_output}" \
    --port "${port}" \
    --host "${host}" \
    --coding "$@"
