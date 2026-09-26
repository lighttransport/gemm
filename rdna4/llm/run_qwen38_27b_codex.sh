#!/usr/bin/env bash
set -euo pipefail

# Serve Qwen3.8-27B (IQ2_XS GSQ + DFlash2) as a local OpenAI-compatible
# endpoint for Codex and install a matching Codex profile.
#
#   rdna4/llm/run_qwen38_27b_codex.sh            # start server (foreground)
#   CODEX_HOME=$HOME/.codex-qwen38 codex exec --skip-git-repo-check "..."
#
# The profile directory defaults to $HOME/.codex-qwen38 (QWEN38_CODEX_HOME).
# It is written only when absent (or QWEN38_CODEX_PROFILE_FORCE=1), so an
# existing Codex login/config is never touched.
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${QWEN38_MODEL:-/mnt/disk1/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf}"
dflash="${QWEN38_DFLASH2:-/mnt/disk1/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
port="${QWEN38_API_PORT:-8090}"
host="${QWEN38_API_HOST:-127.0.0.1}"
context="${QWEN38_CONTEXT:-65536}"
max_output="${QWEN38_MAX_OUTPUT:-16384}"
thinking="${QWEN38_THINKING:-auto}"
# Host-side conversation snapshots: the shared system prefix plus the live
# state of conversations that lost the GPU to another one.  A 64K-token
# snapshot is about 2.3 GiB.
cache_entries="${QWEN38_CONTEXT_CACHE_ENTRIES:-8}"
cache_mib="${QWEN38_CONTEXT_CACHE_MIB:-12288}"
snapshot_tokens="${QWEN38_SNAPSHOT_MAX_TOKENS:-${context}}"
codex_home="${QWEN38_CODEX_HOME:-${HOME}/.codex-qwen38}"

for f in "${model}" "${dflash}"; do
    if [[ ! -r "${f}" ]]; then
        echo "missing model file: ${f} (set QWEN38_MODEL / QWEN38_DFLASH2)" >&2
        exit 1
    fi
done
if [[ ! -x "${runner_dir}/test_hip_llm" ]]; then
    echo "build the runner first: make -C ${runner_dir}" >&2
    exit 1
fi

if [[ ! -e "${codex_home}/config.toml" || "${QWEN38_CODEX_PROFILE_FORCE:-0}" != "0" ]]; then
    mkdir -p "${codex_home}"
    sed -e "s|@CATALOG@|${runner_dir}/codex/qwen38_model_catalog.json|" \
        -e "s|http://127.0.0.1:8090/v1|http://${host}:${port}/v1|" \
        "${runner_dir}/codex/config.toml" > "${codex_home}/config.toml"
    echo "installed Codex profile: CODEX_HOME=${codex_home}" >&2
fi

export ROCEW_ROCM_LIB="${ROCEW_ROCM_LIB:-/opt/rocm/lib}"
cd "${runner_dir}"
exec python3 codex_server.py "${model}" --runner "${runner_dir}/test_hip_llm" \
    --host "${host}" --port "${port}" --context "${context}" \
    --max-output "${max_output}" --thinking "${thinking}" \
    --qwen35-dflash2 "${dflash}" \
    --context-cache-entries "${cache_entries}" \
    --context-cache-max-mib "${cache_mib}" \
    --qwen35-snapshot-max-tokens "${snapshot_tokens}" "$@"
