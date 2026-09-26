#!/usr/bin/env bash
# Exercise three independent Qwen NVFP4 TP4 decode groups on one 12-node job.
set -euo pipefail

cd "$(dirname "$0")/../../.."
run=${1:?usage: run_tp4_groups_12n.sh NEW_RUN_DIR [PROMPT_TOKENS=128] [GEN=16] [CONTEXTS=1]}
pn=${2:-128}
gn=${3:-16}
contexts=${4:-1}
(( pn > 0 && gn > 0 && contexts >= 1 && contexts <= 32 ))
mkdir "$run"
run=$(realpath "$run")
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
export TMPDIR=/local/q38/tmp
export Q38_GROUP_RUN="$run" Q38_GROUP_PN="$pn" Q38_GROUP_GN="$gn" Q38_GROUP_CONTEXTS="$contexts"
export Q38_GROUP_PRUNE=${Q38_GROUP_PRUNE:-0}
export Q38_GROUP_KV_I8=${Q38_GROUP_KV_I8:-0}
export Q38_GROUP_KV_I6=${Q38_GROUP_KV_I6:-0}
export Q38_GROUP_ROTATE=${Q38_GROUP_ROTATE:-0}
export Q38_GROUP_WARM_DEPTH=${Q38_GROUP_WARM_DEPTH:-0}
export Q38_GROUP_CONTEXTS_BY_GROUP=${Q38_GROUP_CONTEXTS_BY_GROUP:-}
export Q38_GROUP_CONTEXT_PROMPT_PREFIX=${Q38_GROUP_CONTEXT_PROMPT_PREFIX:-}
export Q38_GROUP_CONTEXT_STATE_PREFIX=${Q38_GROUP_CONTEXT_STATE_PREFIX:-}
export Q38_GROUP_MODEL=${Q38_GROUP_MODEL:-$HOME/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf}
export Q38_GROUP_IMAGE=${Q38_GROUP_IMAGE:-/local/q38/fp4.image}

make -C a64fx/utofu-tests tofu_topo_helper MPICC=mpifcc
mpiexec -n 12 a64fx/utofu-tests/tofu_topo_helper
export TOFU_TOPO_PATH="$PWD/tofu_topo.txt"
mpiexec -n 12 bash -c '
    group=$((PMIX_RANK / 4))
    group_contexts=$Q38_GROUP_CONTEXTS
    if [[ -n $Q38_GROUP_CONTEXTS_BY_GROUP ]]; then
        IFS=, read -r c0 c1 c2 <<< "$Q38_GROUP_CONTEXTS_BY_GROUP"
        case $group in 0) group_contexts=$c0 ;; 1) group_contexts=$c1 ;; 2) group_contexts=$c2 ;; esac
    fi
    case $group in
        0) prompt="Review C: int add(int a, int b) { return a + b; }" ;;
        1) prompt="Review C: size_t bound(size_t n) { return n < 4096 ? n : 4096; }" ;;
        2) prompt="Review C: unsigned hash(unsigned x) { return x * 2654435761u; }" ;;
    esac
    set --
    if [[ $Q38_GROUP_PRUNE == 1 ]]; then set -- "$@" --prune-model; fi
    if [[ $Q38_GROUP_KV_I8 == 1 ]]; then set -- "$@" --kv-i8; fi
    if [[ $Q38_GROUP_KV_I6 == 1 ]]; then set -- "$@" --kv-i6; fi
    if [[ -n $Q38_GROUP_CONTEXT_PROMPT_PREFIX ]]; then
        set -- "$@" --context-prompt-list "${Q38_GROUP_CONTEXT_PROMPT_PREFIX}.group${group}.txt"
    fi
    if [[ -n $Q38_GROUP_CONTEXT_STATE_PREFIX ]]; then
        set -- "$@" --context-state-list "${Q38_GROUP_CONTEXT_STATE_PREFIX}.group${group}.txt"
    fi
    Q38D_TP=4 Q38D_TP_GROUP=$group \
      /usr/bin/time -f "Q38_GROUP_RESOURCE wall_s=%e max_rss_kib=%M" \
      a64fx/llm/build/q38d "$Q38_GROUP_MODEL" --image "$Q38_GROUP_IMAGE" \
      --fmt fp4 --act a16 --prompt "$prompt" \
      --prompt-tokens "$Q38_GROUP_PN" --gen "$Q38_GROUP_GN" --contexts "$group_contexts" \
      --prompt-rotate "$Q38_GROUP_ROTATE" --warm-kv-depth "$Q38_GROUP_WARM_DEPTH" "$@" \
      > "$Q38_GROUP_RUN/rank${PMIX_RANK}.log" 2>&1
'
if (( contexts == 1 )); then
    python3 a64fx/llm/q38d/check_tp4_groups_12n.py "$run" "$gn"
else
    python3 a64fx/llm/q38d/check_q38d_batch.py "$run" "${Q38_GROUP_CONTEXTS_BY_GROUP:-$contexts}" "$gn"
fi
