#!/bin/bash
# MiniMax-M3 (text) REAL-WEIGHT GENERATION on 96 A64FX nodes, 8x12 torus. Stage the
# full model to /local/m3, then run M3_REAL=1 in gen-mode: embed the prompt token ids
# (a64fx/m3/prompt_ids.txt), prefill, greedy-decode M3_MAX_NEW tokens, write the
# generated ids to a64fx/m3/gen_ids.txt. Detokenize offline:
#   python3 a64fx/m3/m3_tokenizer.py decode-file a64fx/m3/gen_ids.txt
#
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_gen_96n.sh'

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
cd "$M3" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-96}
STAGE_LAYERS=${M3_STAGE_LAYERS:-0}; RUN_LAYERS=${M3_LAYERS:-0}
export M3_MODEL_DIR=${M3_MODEL_DIR:-$HOME/models/m3} M3_STAGE_DIR=/local/m3
export M3_NSHARDS=59 M3_EP_SIZE=$NP M3_STATUS_DIR="$M3"
export M3_TP=${M3_TP:-1} M3_MSA=${M3_MSA:-1}
export M3_MAXPOS=${M3_MAXPOS:-256} M3_MAX_NEW=${M3_MAX_NEW:-48}
export LLM_THREADS=${LLM_THREADS:-12} OMP_NUM_THREADS=${LLM_THREADS:-12}   # 12 = 1 CMG (sweet spot)

echo "=== M3 REAL gen 96n (8x12): NP=$NP layers=$RUN_LAYERS prompt=$(cat "$M3/prompt_ids.txt") job=${PJM_JOBID:-?} ==="
date
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
    echo "[gen96] topo try $t failed (1907?); retry"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }

echo "--- staging ($(date)) ---"
M3_STAGE_LAYERS=$STAGE_LAYERS mpiexec -np "$NP" "$LLM/build/m3_stage" || { echo "FATAL: stage"; exit 4; }
echo "staged ranks: $(ls "$M3"/m3_stage_rank*.txt 2>/dev/null | wc -l)/$NP ($(date))"

echo "--- generate ($(date)) ---"
M3_REAL=1 M3_LAYERS=$RUN_LAYERS \
  M3_PROMPT_IDS="$M3/prompt_ids.txt" M3_GEN_OUT="$M3/gen_ids.txt" \
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "FATAL: gen"; exit 5; }
echo "--- rank0 log ---"; cat m3_ep_rank00.txt 2>/dev/null
echo "--- gen ids ---"; cat "$M3/gen_ids.txt" 2>/dev/null
echo "=== done $(date) ==="
