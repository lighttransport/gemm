#!/bin/bash
# Exact native-Q8_0 single-sequence prefill on twelve A64FX nodes (PP3 x TP4).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
export MODEL=${MODEL:-/home/u14346/models/qwen38/27b/Qwen3.8-27B-Q8_0.gguf}
export Q38_PREFILL_TP_SIZE=${Q38_PREFILL_TP_SIZE:-4}
export Q38_PREFILL_STAGE=${Q38_PREFILL_STAGE:-/local/u14346/qwen38-q8-prefill-tp${Q38_PREFILL_TP_SIZE}}
export Q38_PREFILL_PV48=0
export Q38_PREFILL_QUANT=none
export Q38_PREFILL_Q8=${Q38_PREFILL_Q8:-q8v2}
export Q38_PREFILL_CHUNK=${Q38_PREFILL_CHUNK:-256}
export TF_SSM_FUSED_DOTS=${TF_SSM_FUSED_DOTS:-1}
export TF_SILU_SVE=${TF_SILU_SVE:-1}
export TF_SSM_PREEXP=${TF_SSM_PREEXP:-0}
export TF_ATTN_GATE_SVE=${TF_ATTN_GATE_SVE:-0}

exec "$HERE/run_qwen38_prefill_12n.sh" "${1:-bench}"
