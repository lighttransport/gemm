#!/bin/bash
# Full-model multi-node prefill throughput: token-serial vs chunked (Lever 1) on 48 nodes.
# ~1020-token prompt (prompt_ids tiled; throughput test, coherence not the point). Measures
# prefill tok/s with M3_PCHUNK=0 (serial) and M3_PCHUNK=256 (chunked: M=256 GEMMs + ONE
# all-reduce/layer/chunk). Target: chunked >= 100 tok/s.
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_prefill_48n.sh'
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=4x4x3:torus,elapse=01:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=48"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=48
RUN="$M3/pfrun_${PJM_JOBID:-$$}"; mkdir -p "$RUN"; cd "$RUN" || exit 2
export M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_STAGE_DIR=/local/m3fp8 M3_STATUS_DIR="$RUN"
export M3_EP_SIZE=$NP M3_TP=1 M3_MSA=1 M3_MAXPOS=2048 M3_MAX_NEW=2
export LLM_THREADS=12 OMP_NUM_THREADS=12
echo "=== M3 prefill 48n: $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 20); do rm -f tofu_topo.txt
  if mpiexec -np $NP "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null||echo 0)" -ge $NP ]; then topo_ok=1; break; fi
  echo "topo try $t"; sleep 8; done
[ "$topo_ok" = 1 ] || { echo FATAL topo; exit 3; }
echo "--- staging ($(date)) ---"; mpiexec -np $NP "$LLM/build/m3_stage" || { echo FATAL stage; exit 4; }
# build a ~1020-token prompt by tiling the 6-token prompt_ids (throughput test)
awk 'BEGIN{for(i=0;i<170;i++){while((getline l < "'"$M3"'/prompt_ids.txt")>0) printf "%s ",l; close("'"$M3"'/prompt_ids.txt")}}' > "$RUN/prompt_long.txt"
echo "prompt tokens: $(wc -w < "$RUN/prompt_long.txt")"
for PC in 0 256; do
  echo "--- M3_PCHUNK=$PC ($(date)) ---"
  M3_REAL=1 M3_PCHUNK=$PC M3_PROMPT_IDS="$RUN/prompt_long.txt" \
    mpiexec -np $NP "$LLM/build/m3_ep_runner" 2>&1 | grep -iE "prompt=|prefill|chunked|decode |NaN|FATAL" | head -6
done
echo "=== done $(date) ==="
