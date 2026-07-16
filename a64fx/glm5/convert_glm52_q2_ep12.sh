#!/bin/bash
# Convert the seven-shard GLM-5.2 mixed-IQ GGUF into twelve A64FX rank blobs.
if [ -z "${BASH_VERSION:-}" ] || shopt -oq posix; then exec /bin/bash "$0" "$@"; fi
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LLM="$REPO/a64fx/llm"
SOURCE="${GLM52_MODEL_DIR:-$HOME/models/glm52-2bit}"
OUTPUT="${GLM52_CONVERT_DIR:-$SOURCE/a64fx-ep12-v1}"
EP_SIZE="${GLM52_EP_SIZE:-12}"
LAYERS="${GLM5_LAYERS:-78}"
MAXPOS="${GLM5_MAXPOS:-2304}"

make -C "$LLM" glm52_gguf_convert HOSTCC="${HOSTCC:-cc}" >/dev/null
BIN="$LLM/build/glm52_gguf_convert"
args=(--source "$SOURCE" --output "$OUTPUT" --ep-size "$EP_SIZE"
      --layers "$LAYERS" --maxpos "$MAXPOS")
rank=""
dry=0
force=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --rank) rank="$2"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --force) force=1; shift ;;
        *) echo "usage: $0 [--rank R] [--dry-run] [--force]" >&2; exit 2 ;;
    esac
done
[ "$dry" = 1 ] && args+=(--dry-run)
[ "$force" = 1 ] && args+=(--force)
[ "${GLM52_SHARD2:-0}" = 1 ] && args+=(--shard2)
[ "${GLM52_MTP:-0}" = 1 ] && args+=(--mtp)

if [ -n "$rank" ]; then
    exec "$BIN" "${args[@]}" --rank "$rank"
fi
if [ "${GLM52_CONVERT_PARALLEL:-0}" = 1 ]; then
    exec mpiexec -np "$EP_SIZE" "$BIN" "${args[@]}"
fi
for ((r=0; r<EP_SIZE; r++)); do
    "$BIN" "${args[@]}" --rank "$r"
done
