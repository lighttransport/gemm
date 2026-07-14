#!/bin/bash
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
export DS4F_STAGE_DIR=/local/ds4f DS4F_CMP_LOCAL=1 LLM_THREADS=47
free_nodes() { pkill -x ds4f_ep_runner 2>/dev/null||true; pkill -x mpiexec 2>/dev/null||true; pkill -x plexec 2>/dev/null||true
  for i in $(seq 1 60); do [ "$(ps -eo comm --no-headers|grep -cE 'ds4f_ep_runner|mpiexec|plexec'||true)" = "0" ] && return 0; sleep 1; done; }
: > flash_mladder.txt
# Which M crashes? DB_BENCH sweeps 1,2,4,... up to MAXM, so MAXM=n tells us "everything <= n is fine".
for M in 1 2 4 8 16; do
  free_nodes; rm -f ds4f_ep_rank00.txt
  DS4F_DB_BENCH=1 DS4F_DB_MAXM=$M DS4F_DB_NTOK=8 DS4F_PREFILL=32 DS4F_MAXGEN=4 DS4F_MAXPOS=128 \
    ./run_ds4f_11n.sh > mlad_$M.log 2>&1 || true
  n=$(grep -cE "^ *M=" ds4f_ep_rank00.txt 2>/dev/null || echo 0)
  sig=$(grep -oE "sig=[0-9]+" mlad_$M.log | head -1)
  echo "MAXM=$M -> $n M-lines ${sig:+CRASH($sig)}" >> flash_mladder.txt
  grep -hE "^ *M=" ds4f_ep_rank00.txt 2>/dev/null | sed 's/^/    /' >> flash_mladder.txt
done
free_nodes
