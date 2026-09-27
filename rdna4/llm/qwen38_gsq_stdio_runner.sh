#!/usr/bin/env bash
# Resident stdio runner for codex_server.py with the tuned Qwen3.8 GSQ kernel
# profile (run_qwen38_gsq_rocm.sh): decode layout/kernels, batched 512-token
# prefill and the Q8_1 kernel selections the 4K benchmark is validated with.
# Launching test_hip_llm directly skips all of these; measured on agent-sized
# prompts that costs ~1/3 of prefill throughput and ~12% of long-context decode.
#
#   codex_server.py MODEL --runner rdna4/llm/qwen38_gsq_stdio_runner.sh ...
#
# codex_server passes the model path first; the profile script reads it from
# QWEN38_MODEL.  The server chooses its own context (and the runner clamps it
# to free VRAM), so the profile's 16-GiB benchmark context cap does not apply.
set -euo pipefail
export QWEN38_MODEL="$1"
shift
export QWEN38_GSQ_ALLOW_UNSAFE_CONTEXT="${QWEN38_GSQ_ALLOW_UNSAFE_CONTEXT:-1}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen38_gsq_rocm.sh" "$@"
