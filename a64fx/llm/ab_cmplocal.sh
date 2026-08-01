#!/bin/bash
# A/B DS4F_CMP_LOCAL to explain tb2prep = 24.4 ms (docs claim 8.3 with CMP_LOCAL=1).
# tb2prep is 53.6% of decode compute in job 49556601, and is the ENTIRE gap between the
# measured 16.29 tok/s and the documented headline 22.45.
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
export DS4F_DENSE=q8pv DS4F_EXPERTS=q8pv DS4F_STAGE_DIR=/local/ds4fbase_q8 DS4F_TP_ATTN=0
export DS4F_HC_SVE=1 DS4F_MV_FUSE=1 LLM_THREADS=47

# The serve/gate/bench runs LEAK their MPI ranks, and a pkill does not free the Tofu coordinate
# instantly -- launching 3s later gives "PLE 0054 plexec: number of processes exceed the limit on
# virtual coordinate (0,0,0)" and the run dies at 3s having measured nothing. WAIT for release.
free_nodes() {
    pkill -x ds4f_ep_runner 2>/dev/null; pkill -x mpiexec 2>/dev/null; pkill -x plexec 2>/dev/null
    for i in $(seq 1 60); do
        [ "$(ps -eo comm --no-headers | grep -cE 'ds4f_ep_runner|mpiexec|plexec' || true)" = "0" ] && return 0
        sleep 1
    done
    echo "WARN: nodes still busy after 60s" >&2
}
: > ab_cmplocal.txt
for CL in 0 1; do
  free_nodes
  DS4F_CMP_LOCAL=$CL DS4F_GEN_LOG=$PWD/cl_$CL.log DS4F_GEN_SENTINEL=$PWD/cl_s_$CL.txt \
    PROMPT_FILE=sweep_k_prompt.txt MAX_NEW=32 ./run_ds4fbase_gen_12n.sh >/dev/null 2>&1 || true
  { echo "### CMP_LOCAL=$CL"
    grep -hE "^decode:|^prefill:" cl_s_$CL.txt 2>/dev/null | sed 's/^/    /'
    grep -hE "tb2prep|mhc_pre|experts " cl_$CL.log 2>/dev/null | tail -3 | sed 's/^/    /'
    grep -hcE "PLE 0054" cl_$CL.log 2>/dev/null | grep -q '^0$' || echo "    !! PLE 0054 -- run never started"
  } >> ab_cmplocal.txt
done
free_nodes
