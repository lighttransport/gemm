#!/bin/bash
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
export DS4F_DENSE=q8pv DS4F_EXPERTS=q8pv DS4F_STAGE_DIR=/local/ds4fbase_q8 DS4F_TP_ATTN=0
export DS4F_CMP_LOCAL=1 DS4F_HC_SVE=1 DS4F_MV_FUSE=1 LLM_THREADS=47
free_nodes() { pkill -x ds4f_ep_runner 2>/dev/null; pkill -x mpiexec 2>/dev/null; pkill -x plexec 2>/dev/null
  for i in $(seq 1 60); do [ "$(ps -eo comm --no-headers|grep -cE 'ds4f_ep_runner|mpiexec|plexec'||true)" = "0" ] && return 0; sleep 1; done; }
: > ab_scanmin.txt
# (a) synthetic ctx=64 -- the worst case in the sweep
for SM in 64 8; do
  free_nodes
  DS4F_IDX_SCAN_MIN=$SM DS4F_PREFILL=64 DS4F_MAXGEN=16 DS4F_MAXPOS=128 \
    ./run_ds4fbase_12n.sh > sm_syn_$SM.log 2>&1 || true
  { echo "### SCAN_MIN=$SM  synthetic ctx=64"
    grep -hE "^decode:" sm_syn_$SM.log | tail -1 | sed 's/^/    /'
    grep -hE "^ *tb2scan|^ *tb2prep" sm_syn_$SM.log | tail -2 | sed 's/^ */    /'
  } >> ab_scanmin.txt
done
# (b) REAL 70-token prompt -- gated on a coherent completion
for SM in 64 8; do
  free_nodes
  DS4F_IDX_SCAN_MIN=$SM DS4F_GEN_LOG=$PWD/sm_gen_$SM.log DS4F_GEN_SENTINEL=$PWD/sm_gen_s_$SM.txt \
    PROMPT_FILE=sweep_k_prompt.txt MAX_NEW=32 ./run_ds4fbase_gen_12n.sh >/dev/null 2>&1 || true
  { echo "### SCAN_MIN=$SM  REAL 70-tok prompt"
    grep -hE "^decode:|^prefill:" sm_gen_s_$SM.txt 2>/dev/null | sed 's/^/    /'
    grep -hE "^ *tb2scan" sm_gen_$SM.log 2>/dev/null | tail -1 | sed 's/^ */    /'
    echo "    completion:"
    sed -n '/<<<COMPLETION>>>/,/<<<END>>>/p' sm_gen_s_$SM.txt 2>/dev/null | sed '1d;$d' | head -4 | sed 's/^/    | /'
  } >> ab_scanmin.txt
done
free_nodes
