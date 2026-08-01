#!/bin/bash
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
export DS4F_STAGE_DIR=/local/ds4f DS4F_CMP_LOCAL=1 LLM_THREADS=47
export DS4F_EXACT=1 DS4F_TIERB2=1 DS4F_MHC=1      # <-- THE FIX: run_ds4f_11n.sh defaults these to 0
free_nodes() { pkill -x ds4f_ep_runner 2>/dev/null||true; pkill -x mpiexec 2>/dev/null||true; pkill -x plexec 2>/dev/null||true
  for i in $(seq 1 60); do [ "$(ps -eo comm --no-headers|grep -cE 'ds4f_ep_runner|mpiexec|plexec'||true)" = "0" ] && return 0; sleep 1; done; }
free_nodes; rm -f ds4f_ep_rank00.txt
DS4F_DB_BENCH=1 DS4F_DB_MAXM=32 DS4F_DB_NTOK=32 DS4F_PREFILL=32 DS4F_MAXGEN=4 DS4F_MAXPOS=128 \
  ./run_ds4f_11n.sh > flash_db_fixed.log 2>&1 || true
{ grep -hE "^ *M=" ds4f_ep_rank00.txt 2>/dev/null || { echo "NO M LINES"; grep -hE "REQUIRED|sig=|OUT OF MEM" flash_db_fixed.log | head -2; }; } > flash_db_fixed.txt
free_nodes
