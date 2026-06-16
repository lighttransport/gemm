#!/bin/bash
# 16384-token prefill benchmark, real weights, full 60-layer model. WEIGHTS=bf16, NODES=48.
# Chunked prefill (Lever 1), M=256, 48 threads/node (all 4 CMGs - prefill is compute-bound).
# KV cache bf16 (same for bf16/mxfp8 variants so only the WEIGHT format differs). MSA on
# (16384 >> 2304 -> sparse attention). Reports prefill tok/s + per-rank arena + NaN.
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_prefill16k_bf16_48n.sh'
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=8x12:torus,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-48}
RUN="$M3/pf16krun_${PJM_JOBID:-$$}"; mkdir -p "$RUN"; cd "$RUN" || exit 2
export M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_STAGE_DIR=/local/m3fp8 M3_STATUS_DIR="$RUN"
export M3_EP_SIZE=$NP M3_TP=1 M3_MSA=1 M3_MAXPOS=20480 M3_MAX_NEW=2
export LLM_THREADS=48 OMP_NUM_THREADS=48
NTOK=16384; WTAG=mxfp8
echo "=== M3 prefill16k WEIGHTS=$WTAG NODES=$NP: $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 30); do rm -f tofu_topo.txt
  if mpiexec -np $NP "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null||echo 0)" -ge $NP ]; then topo_ok=1; break; fi
  echo "topo try $t"; sleep 15; done
[ "$topo_ok" = 1 ] || { echo FATAL topo; exit 3; }
echo "--- staging $WTAG ($(date)) ---"; mpiexec -np $NP "$LLM/build/m3_stage" || { echo FATAL stage; exit 4; }
# 16384-token prompt (tile the 6-token prompt; throughput test)
awk -v n=$NTOK 'BEGIN{c=0;while(c<n){while((getline l < "'"$M3"'/prompt_ids.txt")>0){printf "%s ",l;c++;if(c>=n)break} close("'"$M3"'/prompt_ids.txt")}}' > "$RUN/prompt_16k.txt"
echo "prompt tokens: $(wc -w < "$RUN/prompt_16k.txt")"
export M3_REAL=1 M3_PROMPT_IDS="$RUN/prompt_16k.txt" M3_PCHUNK=256
echo "--- chunked prefill M=256, $NTOK tok ($(date)) ---"
mpiexec -np $NP "$LLM/build/m3_ep_runner" 2>&1 | grep -iE "prompt=|prefill|chunked|FATAL" | head -4
echo "--- rank0 ---"; grep -iE "prompt=|prefill|chunked|NaN|GEN_IDS" m3_ep_rank00.txt 2>/dev/null | head -5
echo "--- load arena ---"; grep -iE "arena_used|owned" m3_ep_load_rank00.txt 2>/dev/null | head -2
echo "=== done $(date) ==="
