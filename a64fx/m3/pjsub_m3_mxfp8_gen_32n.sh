#!/bin/bash
# MiniMax-M3 MXFP8 (~/models/m3-fp8, 451 GB ~half bf16) REAL-WEIGHT generation on 96
# A64FX nodes (8x12 torus). Stage the full MXFP8 model to /local/m3fp8, then generate.
# MXFP8 halves the WEIGHT memory (experts/dense FP8 + E8M0 scale); KV is still bf16 at
# runtime (so 1M ctx needs CP/int4-KV separately). Coherence gate: should match the bf16
# "Paris" output. Reports per-rank arena (the memory win) + decode tok/s.
# Detokenize: python3 a64fx/m3/m3_tokenizer.py decode-file a64fx/m3/gen_mxfp8_32.txt
#
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_mxfp8_gen_96n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=4x4x2:torus,elapse=01:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=32"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-96}
RUN="$M3/mxfp8run_${PJM_JOBID:-$$}"; mkdir -p "$RUN"; cd "$RUN" || exit 2   # isolate tofu_topo.txt + logs per job
export M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_STAGE_DIR=/local/m3fp8 M3_STATUS_DIR="$RUN"
export M3_EP_SIZE=$NP M3_TP=1 M3_MSA=1
export M3_MAXPOS=${M3_MAXPOS:-256} M3_MAX_NEW=${M3_MAX_NEW:-48}
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3-MXFP8 gen 32n (4x4x2): NP=$NP job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[mxfp8] topo try $t (1907?)"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

echo "--- staging MXFP8 ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/m3_stage" || { echo "FATAL stage"; exit 4; }
echo "staged: $(ls "$M3"/m3_stage_rank*.txt 2>/dev/null | wc -l)/$NP  rank0: $(cat "$M3"/m3_stage_rank00.txt 2>/dev/null)"

echo "--- generate ($(date)) ---"
M3_REAL=1 M3_PROMPT_IDS="$M3/prompt_ids.txt" M3_GEN_OUT="$M3/gen_mxfp8_32.txt" \
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "FATAL gen"; exit 5; }
echo "--- rank0 ---"; cat m3_ep_rank00.txt 2>/dev/null
echo "--- load (arena) ---"; cat m3_ep_load_rank00.txt 2>/dev/null
echo "--- gen ids ---"; cat "$M3/gen_mxfp8_32.txt" 2>/dev/null
echo "=== done $(date) ==="
