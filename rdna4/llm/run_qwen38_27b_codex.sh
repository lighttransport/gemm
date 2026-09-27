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
# 96K is the largest context that leaves VRAM for the GSQ profile's decode
# layout, the prefill scratch and DFlash2's verify workspace on a 16 GB card
# (above it the runner serves without DFlash2).  Claude Code needs well over
# 64K: it compacts ~30K tokens below the window.
context="${QWEN38_CONTEXT:-98304}"
max_output="${QWEN38_MAX_OUTPUT:-16384}"
thinking="${QWEN38_THINKING:-auto}"
# Host-side conversation snapshots: the shared system prefix plus the live
# state of conversations that lost the GPU to another one.  A 64K-token
# snapshot is about 2.3 GiB, a 100K-token one about 3.5 GiB.
cache_entries="${QWEN38_CONTEXT_CACHE_ENTRIES:-16}"
cache_mib="${QWEN38_CONTEXT_CACHE_MIB:-12288}"
snapshot_tokens="${QWEN38_SNAPSHOT_MAX_TOKENS:-${context}}"
codex_home="${QWEN38_CODEX_HOME:-${HOME}/.codex-qwen38}"

for f in "${model}" "${dflash}"; do
    if [[ ! -r "${f}" ]]; then
        echo "missing model file: ${f} (set QWEN38_MODEL / QWEN38_DFLASH2)" >&2
        exit 1
    fi
done
# The GSQ profile wrapper selects the tuned kernels (see the script);
# QWEN38_RUNNER=path/to/test_hip_llm runs the binary with its defaults.
runner="${QWEN38_RUNNER:-${runner_dir}/qwen38_gsq_stdio_runner.sh}"
if [[ ! -x "${runner_dir}/test_hip_llm" ]]; then
    echo "build the runner first: make -C ${runner_dir}" >&2
    exit 1
fi

# The profile's model catalog carries the context window Codex compacts
# against; regenerate it on every start so it follows --context.
mkdir -p "${codex_home}"
python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    catalog = json.load(f)
for entry in catalog.get("models", []):
    entry["context_window"] = entry["max_context_window"] = int(sys.argv[3])
with open(sys.argv[2], "w") as f:
    json.dump(catalog, f, indent=1)
' "${runner_dir}/codex/qwen38_model_catalog.json" \
    "${codex_home}/qwen38_model_catalog.json" "${context}"
if [[ ! -e "${codex_home}/config.toml" || "${QWEN38_CODEX_PROFILE_FORCE:-0}" != "0" ]]; then
    sed -e "s|@CATALOG@|${codex_home}/qwen38_model_catalog.json|" \
        -e "s|http://127.0.0.1:8090/v1|http://${host}:${port}/v1|" \
        "${runner_dir}/codex/config.toml" > "${codex_home}/config.toml"
    echo "installed Codex profile: CODEX_HOME=${codex_home}" >&2
fi

export ROCEW_ROCM_LIB="${ROCEW_ROCM_LIB:-/opt/rocm/lib}"
cd "${runner_dir}"
exec python3 codex_server.py "${model}" --runner "${runner}" \
    --host "${host}" --port "${port}" --context "${context}" \
    --max-output "${max_output}" --thinking "${thinking}" \
    --served-model-name "${QWEN38_SERVED_MODEL:-qwen3.8-27b}" \
    --prefix-store "${QWEN38_PREFIX_STORE:-${HOME}/.cache/qwen38-server/prefixes.json}" \
    --qwen35-dflash2 "${dflash}" \
    --context-cache-entries "${cache_entries}" \
    --context-cache-max-mib "${cache_mib}" \
    --qwen35-snapshot-max-tokens "${snapshot_tokens}" "$@"
