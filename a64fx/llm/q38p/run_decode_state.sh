#!/usr/bin/env bash
# Reuse a completed producer snapshot to benchmark decode independently.
set -euo pipefail
cd "$(dirname "$0")/../../.."
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
if (( $# < 4 )); then
    echo "usage: $0 RUN_DIR STATE_DIR PROMPT_TOKENS GEN [PROMPT_FILE]" >&2
    exit 2
fi
run=$1
mkdir -p "$run"
export HANDOFF_RUN="$run" HANDOFF_STATE="$2" HANDOFF_PN="$3" HANDOFF_GN="$4" HANDOFF_PROMPT="${5:-}"
export HANDOFF_KV_I6=${HANDOFF_KV_I6:-0}
export HANDOFF_MODEL=${HANDOFF_MODEL:-$HOME/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf}
export HANDOFF_IMAGE=${HANDOFF_IMAGE:-/local/q38/fp4.image}
export HANDOFF_DECODER=${HANDOFF_DECODER:-a64fx/llm/build/q38d}
export TOFU_TOPO_PATH=${TOFU_TOPO_PATH:-$PWD/tofu_topo.txt}
for tp in ${HANDOFF_TPS:-4 2}; do
    case "$tp" in 1|2|4) ;; *) echo "decode TP must be 1, 2 or 4" >&2; exit 2;; esac
    if [[ -e $run/decode.tp${tp}.rank0.log ]]; then
        echo "refusing to overwrite $run/decode.tp${tp}.rank0.log" >&2; exit 2
    fi
    export HANDOFF_TP="$tp"
    mpiexec -n "$tp" bash -c '
        args=(); [[ -z $HANDOFF_PROMPT ]] || args+=(--prompt-file "$HANDOFF_PROMPT")
        [[ -z ${HANDOFF_PREFIX_TOKENS:-} ]] || args+=(--state-prefix-tokens "$HANDOFF_PREFIX_TOKENS")
        [[ $HANDOFF_KV_I6 == 1 ]] && args+=(--kv-i6 --prune-model)
        Q38D_TP="$HANDOFF_TP" /usr/bin/time -f "phase_resource: wall_s=%e max_rss_kib=%M" "$HANDOFF_DECODER" "$HANDOFF_MODEL" \
          --image "$HANDOFF_IMAGE" --fmt fp4 --act a16 \
          --prompt-tokens "$HANDOFF_PN" --gen "$HANDOFF_GN" "${args[@]}" --state-in "$HANDOFF_STATE" \
          > "$HANDOFF_RUN/decode.tp${HANDOFF_TP}.rank${PMIX_RANK}.log" 2>&1
    '
    grep -E 'q38d_state:|q38d: (prefill |imported_prefix=)|RESULT' "$run/decode.tp${tp}.rank0.log"
done
