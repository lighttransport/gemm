#!/bin/bash
# MiniMax-M3 BF16 DECODE-PERF probe P3 (OPTIONAL, EXPENSIVE ~72 node-h) — confirm the P1/P2-winning
# config on REAL bf16 weights @ 48n (the bf16 minimal node count), with COHERENT generation. Stages the
# full 796 GB model to /local/m3, then runs gen-mode. Run this ONLY after P1 (mstream x AR) and P2
# (int4-KV) pick a winner — set the knobs below to that winner.
#
# Purpose: the synthetic probes (P1/P2) measure the structural levers (dispatch/comm/memory) which are
# weight-value-independent; this validates that the winning config (a) loads real bf16 weights, fits
# 48n (arena < 27 GB/node — check via MemFree), (b) still produces coherent text (the "Paris" gate),
# (c) delivers the projected aggregate tok/s on real weights.
#
# Winning-config knobs (defaults = current known-best M=8 / f32-AR / bf16-KV; OVERRIDE per P1/P2):
#   M3_MSTREAM (=8)  TP_AR_BF16 (=0)  M3_INT4_KV (=0)
#
# Submit (set the winner): ssh -A fugaku 'cd ~/work/gemm/glm5-1 && \
#   M3_MSTREAM=16 M3_INT4_KV=1 pjsub --no-check-directory a64fx/m3/pjsub_m3_bf16_confirm_48n.sh'
# Detokenize: python3 a64fx/m3/m3_tokenizer.py decode-file a64fx/m3/gen_bf16_48.txt   # expect "... Paris"

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=8x6:torus,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=48"
#PJM --llio localtmp-size=64Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-48}
RUN="$M3/bf16confirm_${PJM_JOBID:-$$}"; mkdir -p "$RUN"; cd "$RUN" || exit 2
export M3_MODEL_DIR=$HOME/models/m3 M3_STAGE_DIR=/local/m3 M3_NSHARDS=59 M3_STATUS_DIR="$RUN"
export M3_EP_SIZE=$NP M3_TP=1 M3_MSA=1
export M3_MSTREAM=${M3_MSTREAM:-8} TP_AR_BF16=${TP_AR_BF16:-0} M3_INT4_KV=${M3_INT4_KV:-0}
export M3_MAXPOS=${M3_MAXPOS:-256} M3_MAX_NEW=${M3_MAX_NEW:-48}
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3 bf16 CONFIRM 48n (8x6): NP=$NP mstream=$M3_MSTREAM ar_bf16=$TP_AR_BF16 int4_kv=$M3_INT4_KV job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

python3 "$M3/m3_tokenizer.py" encode "$(cat "$M3/prompt_m3.txt")" --bos > "$M3/prompt_ids.txt" \
  || { echo "FATAL encode"; exit 3; }
echo "prompt_ids: $(cat "$M3/prompt_ids.txt")"

topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[p3] topo try $t (1907?)"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

echo "--- staging bf16 (~45 GB/node blob; ~20 min) ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/m3_stage" || { echo "FATAL stage"; exit 4; }
echo "staged: $(ls "$RUN"/m3_stage_rank*.txt 2>/dev/null | wc -l)/$NP"

echo "--- generate (winning config) ($(date)) ---"
M3_REAL=1 M3_PROMPT_IDS="$M3/prompt_ids.txt" M3_GEN_OUT="$M3/gen_bf16_48.txt" \
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "FATAL gen"; exit 5; }
echo "--- rank0 ---"; cat m3_ep_rank00.txt 2>/dev/null
echo "--- load (arena / MemFree) ---"; cat m3_ep_load_rank00.txt 2>/dev/null
echo "--- gen ids ---"; cat "$M3/gen_bf16_48.txt" 2>/dev/null
echo "SENTINEL m3_bf16_confirm_48n=done"; echo "=== done $(date) ==="
