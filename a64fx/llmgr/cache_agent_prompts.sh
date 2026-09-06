#!/bin/sh
# Warm and save system/tool prefixes for one or more captured agent requests.
# Usage: cache_agent_prompts.sh --model MODEL [options] agent=request.json ...
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
server=${LLMGR_CACHE_SERVER:-${QWEN36_SERVER:-http://127.0.0.1:8081}}
output_dir=${LLMGR_CACHE_DIR:-${LLMGR_QWEN36_CACHE_DIR:-"$HOME/.cache/llmgr/qwen36"}}
model=${LLMGR_CACHE_MODEL:-${QWEN36_MODEL:-}}
python_bin=${LLMGR_CACHE_PYTHON:-python3}
warm_user=${LLMGR_CACHE_WARM_USER:-hello}
reuse=0

usage() {
    printf '%s\n' "Usage: $0 --model MODEL [--server URL] [--output-dir DIR] [--reuse] agent=request.json ..."
    printf '%s\n' "Example: $0 --model model.gguf codex=codex.json claude-code=claude.json"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            model=$2; shift 2 ;;
        --server)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            server=$2; shift 2 ;;
        --output-dir)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            output_dir=$2; shift 2 ;;
        --python)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            python_bin=$2; shift 2 ;;
        --warm-user)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            warm_user=$2; shift 2 ;;
        --reuse)
            reuse=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift; break ;;
        -* )
            printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
        * )
            break ;;
    esac
done

[ -n "$model" ] || { printf '%s\n' 'missing --model (or LLMGR_CACHE_MODEL)' >&2; exit 2; }
[ "$#" -gt 0 ] || { printf '%s\n' 'provide at least one agent=request.json pair' >&2; usage >&2; exit 2; }

mkdir -p "$output_dir"
for pair in "$@"; do
    case "$pair" in
        *=*) agent=${pair%%=*}; request=${pair#*=} ;;
        *) printf 'expected agent=request.json, got: %s\n' "$pair" >&2; exit 2 ;;
    esac
    [ -n "$agent" ] || { printf '%s\n' 'agent label cannot be empty' >&2; exit 2; }
    [ -f "$request" ] || { printf 'request file not found: %s\n' "$request" >&2; exit 2; }
    if [ "$reuse" -eq 1 ]; then
        "$python_bin" "$script_dir/qwen36_prompt_cache.py" \
            --agent "$agent" --request-file "$request" --model "$model" \
            --server "$server" --output-dir "$output_dir" \
            --warm-user "$warm_user" --reuse
    else
        "$python_bin" "$script_dir/qwen36_prompt_cache.py" \
            --agent "$agent" --request-file "$request" --model "$model" \
            --server "$server" --output-dir "$output_dir" \
            --warm-user "$warm_user"
    fi
done
