#!/bin/bash
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
export DS4F_STAGE_DIR=/local/ds4f DS4F_CMP_LOCAL=1 LLM_THREADS=47 DS4F_BACKTRACE=1 DS4F_TRACE=1 DS4F_ROPE_GUARD=1
pkill -x ds4f_ep_runner 2>/dev/null; pkill -x mpiexec 2>/dev/null
for i in $(seq 1 60); do [ "$(ps -eo comm --no-headers|grep -cE 'ds4f_ep_runner|mpiexec'||true)" = "0" ] && break; sleep 1; done
DS4F_DB_BENCH=1 DS4F_DB_MAXM=1 DS4F_DB_NTOK=4 DS4F_PREFILL=32 DS4F_MAXGEN=4 \
  ./run_ds4f_11n.sh > flash_bt.log 2>&1 || true
pkill -x ds4f_ep_runner 2>/dev/null; pkill -x mpiexec 2>/dev/null
