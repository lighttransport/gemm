#!/bin/sh
# Start llmgr with DS4F selected as the default semantic provider.
# Model and staged-weight locations remain explicit/configurable.
set -eu

: "${DS4F_MODEL_DIR:=}"
: "${DS4F_STAGE_DIR:=}"
: "${DS4F_WORK_DIR:=}"
: "${DS4F_NP:=11}"
export LLMGR_DEFAULT_MODEL=ds4f DS4F_NP
[ -z "$DS4F_MODEL_DIR" ] || export DS4F_MODEL_DIR
[ -z "$DS4F_STAGE_DIR" ] || export DS4F_STAGE_DIR
[ -z "$DS4F_WORK_DIR" ] || export DS4F_WORK_DIR

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$HERE/run_llmgr.sh" "$@"
