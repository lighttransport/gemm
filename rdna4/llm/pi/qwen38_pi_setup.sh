#!/usr/bin/env bash
# Point pi's built-in llama.cpp extension at the local Qwen3.8 server started
# by rdna4/llm/run_qwen38_27b_codex.sh, which speaks the llama.cpp router
# endpoints pi uses (/models with status, /props, /models/load|unload) and
# OpenAI chat completions.
#
#   rdna4/llm/pi/qwen38_pi_setup.sh
#   PI_CODING_AGENT_DIR=~/.pi-qwen38 pi --provider llama.cpp --model qwen3.8-27b
#
# Interactively, `/login llama.cpp` (URL http://127.0.0.1:8090) followed by
# `/model` does the same for an existing pi setup.  This script writes the
# credential into a dedicated agent directory (PI_CODING_AGENT_DIR, default
# ~/.pi-qwen38) so an existing ~/.pi configuration is left alone, then runs
# pi once in RPC mode: only interactive and RPC modes refresh extension model
# catalogs from the network; `pi -p` uses the stored catalog.
set -euo pipefail
agent_dir="${PI_CODING_AGENT_DIR:-${HOME}/.pi-qwen38}"
url="http://${QWEN38_API_HOST:-127.0.0.1}:${QWEN38_API_PORT:-8090}"
mkdir -p "${agent_dir}"
python3 - "${agent_dir}/auth.json" "${url}" <<'EOF'
import json, os, sys
path, url = sys.argv[1], sys.argv[2]
data = {}
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
data["llama.cpp"] = {"type": "api_key", "env": {"LLAMA_BASE_URL": url}}
with open(path, "w") as f:
    json.dump(data, f, indent=2)
os.chmod(path, 0o600)
EOF
curl -fsS "${url}/models" > /dev/null || {
    echo "server not reachable at ${url}; start rdna4/llm/run_qwen38_27b_codex.sh first" >&2
    exit 1
}
(sleep 8) | PI_CODING_AGENT_DIR="${agent_dir}" timeout 30 pi --mode rpc > /dev/null 2>&1 || true
PI_CODING_AGENT_DIR="${agent_dir}" pi --list-models 2>/dev/null | grep -i "llama.cpp" || {
    echo "pi did not list the llama.cpp model; check ${url}/models" >&2
    exit 1
}
echo "ready: PI_CODING_AGENT_DIR=${agent_dir} pi --provider llama.cpp --model <id above>"
