#!/bin/bash
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
export DS4F_DENSE=q8pv DS4F_EXPERTS=q8pv DS4F_STAGE_DIR=/local/ds4fbase_q8 DS4F_TP_ATTN=0
export DS4F_CMP_LOCAL=1 DS4F_HC_SVE=1 DS4F_MV_FUSE=1 LLM_THREADS=47
pkill -x ds4f_ep_runner 2>/dev/null; pkill -x mpiexec 2>/dev/null
for i in $(seq 1 60); do [ "$(ps -eo comm --no-headers|grep -cE 'ds4f_ep_runner|mpiexec'||true)" = "0" ] && break; sleep 1; done
rm -f ds4f_ep_rank00.txt
DS4F_VERIFY_GATE=1 DS4F_VG_NTOK=16 PROMPT_FILE=sweep_k_prompt.txt MAX_NEW=4 \
  DS4F_GEN_LOG=$PWD/gate43.log DS4F_GEN_SENTINEL=$PWD/gate43_s.txt ./run_ds4fbase_gen_12n.sh > gate43_outer.log 2>&1
echo "EXIT=$?" >> gate43_outer.log
pkill -x ds4f_ep_runner 2>/dev/null; sleep 2; pkill -x mpiexec 2>/dev/null
