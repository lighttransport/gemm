#!/bin/bash
# DeepSeek-V4-Pro layer-truncated run on the 12-node interactive alloc (11 EP ranks).
#
# Mechanical validation of the DS4P config on real weights: load=11/11, NaN=0,
# cross-rank lockstep, perf + MemFree. The model is TRUNCATED to the staged
# layer count, so generated text is garbage by construction — the full model
# only fits at 48-64 nodes (pjsub_ds4p_*.sh).
#
#   ./run_ds4p_12n.sh                 # real weights, layers 0..9 (must be staged)
#   LAYERS=4 ./run_ds4p_12n.sh        # match a LAYERS=4 stage
#   SYNTH=1 ./run_ds4p_12n.sh         # synthetic weights (no staging needed)
#
# Feature set mirrors the DS4F production path (EXACT + Tier-B2 + mHC) plus the
# full dense TP stack — TP is MANDATORY for DS4P at scale (un-TP'd dense alone
# is ~33 GB/node at 61 layers), so it is always exercised here.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

export DS4F_MODEL=ds4p
export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4p}
export DS4F_LAYERS=${LAYERS:-10}
export DS4F_REAL=$([ "${SYNTH:-0}" = "1" ] && echo 0 || echo 1)
export DS4F_EXACT=1 DS4F_TIERB2=1 DS4F_MHC=1
export DS4F_TP_HEAD=1 DS4F_TP_EMBED=1 DS4F_TP_SHARED=1
export DS4F_TP_ATTN=1 DS4F_TP_OPROJ=1 DS4F_TP_WOB=1
export DS4F_PREFILL=${DS4F_PREFILL:-8}
export DS4F_MAXGEN=${DS4F_MAXGEN:-16}
export DS4F_QUIET=${DS4F_QUIET:-1}

echo "[ds4p run] layers=$DS4F_LAYERS real=$DS4F_REAL stage_dir=$DS4F_STAGE_DIR"
exec "$DIR/../llm/run_ds4f_11n.sh"
