#!/bin/bash
# Experimental draft/verify runner; shares the native target launch contract.
set -euo pipefail
fail() { echo "error: $*" >&2; exit 2; }
if [ "$#" -lt 2 ]; then
    fail "usage: $0 PROMPT_IDS OUTPUT_IDS [cycles=128] [drafts=1] [OPTIONS...]"
fi
prompt=$1 output=$2
shift 2
cycles=128 drafts=1
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then cycles=$1; shift; fi
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then drafts=$1; shift; fi
[[ "$cycles" =~ ^[0-9]+$ ]] && [ "$cycles" -ge 1 ] && [ "$cycles" -le 32768 ] || fail 'cycles must be 1..32768'
[[ "$drafts" =~ ^[0-9]+$ ]] && [ "$drafts" -ge 1 ] && [ "$drafts" -le 4 ] || fail 'drafts must be 1..4'
[ -s "$prompt" ] || fail "missing or empty prompt IDs: $prompt"
[ "$(realpath -m "$prompt")" != "$(realpath -m "$output")" ] || fail 'prompt and output paths must differ'
source "$(dirname "$0")/scripts/glm53f_env.sh"
source "$glm53f_dir/scripts/glm53f_launch.sh"
require_allocation
logdir=$(realpath -m "${GLM53F_SPEC_LOG_DIR:-$logdir}")
mkdir -p "$logdir" "$GLM53F_BUILD_DIR"
if [ "${GLM53F_BUILD:-1}" = 1 ]; then
    bash "$glm53f_dir/build_glm53f_integrated_12n.sh" all
fi
prepare_runtime
mtp_routed=${GLM53F_MTP_STAGE_DIR:-/local/glm53f-mtp-routed-$job}
mtp_shared=${GLM53F_MTP_SHARED_STAGE_DIR:-/local/glm53f-mtp-shared-$job}
mpi_run mtp-preflight bash -c '
    rank=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
    for directory; do
        printf -v manifest "%s/rank%02d.manifest" "$directory" "$rank"
        [ -s "$manifest" ] || { echo "missing stage: $manifest" >&2; exit 2; }
    done
    echo GLM53F_MTP_PREFLIGHT=PASS
' bash "$mtp_routed" "$mtp_shared"
require_ranks 'GLM53F_MTP_PREFLIGHT=PASS'
export GLM53F_PROFILE=1
# The MTP draft layer (layer 45) is read from the safetensors model directory, not from the compact core image;
# glm53f_env.sh forces REPACK_REQUIRE for native targets, so relax it for the speculative binary.
export GLM53F_REPACK_REQUIRE=${GLM53F_REPACK_REQUIRE_MTP:-0}
export GLM53F_KDA_BATCH_TEAM=${GLM53F_KDA_BATCH_TEAM:-1}
export GLM53F_Q4_BATCH_SHARED=${GLM53F_Q4_BATCH_SHARED:-1}
export GLM53F_SPARSE_BATCH_OP=${GLM53F_SPARSE_BATCH_OP:-0}
export GLM53F_SPEC_PROMPT_IDS=$prompt GLM53F_SPEC_OUTPUT_IDS=$output
mpi_run speculate "${GLM53F_SPEC_BINARY:-$GLM53F_BIN_DIR/glm53f_spec_decode_12n}" \
    "$model" "$routed" "$shared" "$mtp_routed" "$mtp_shared" 1 "$cycles" "$drafts" 0 "$@"
grep -E 'GLM53F_SPEC_(PHASE|DECODE|REFERENCE|VARIANT|TRIAL)' "$last_log".*.0
