#!/bin/bash
# MiniMax-M3 CP (context-parallel KV) + int4-KV @ 1M context on 96 A64FX nodes (bf16 KV = full-precision quality reference) (MXFP8 weights).
# CP shards the KV cache block-cyclic across the 48 ranks (block 128 == MSA block); each decode
# does a cross-rank MAX of block scores (global top-k) + a flash-style combine of the per-rank
# partial attention. int4-KV stores k/v/idx at ~3.9x. Together they make 1M context fit.
#  Run1: gen "Paris" with MAXPOS=1M  -> coherence (combine+int4 correct) + per-rank arena (1M FIT).
#  Run2: synthetic prefill 6200 tok (> 48*128 -> shards across ALL ranks; > 2304 -> MSA on)
#        -> lockstep argmax across ranks + NaN=0 (the cross-rank block-merge + combine stress test).
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_cp_1m_96n_bf16.sh'

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
NP=${PJM_MPI_PROC:-96}
RUN="$M3/cprun_${PJM_JOBID:-$$}"; mkdir -p "$RUN"; cd "$RUN" || exit 2
export M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_STAGE_DIR=/local/m3fp8 M3_STATUS_DIR="$RUN"
export M3_EP_SIZE=$NP M3_TP=1 M3_MSA=1 M3_CP=1 M3_INT4_KV=0 M3_KV_FP16=0
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3 CP+int4 1M 96n bf16-KV (4x4x3): NP=$NP job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in $(seq 1 40); do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[cp] topo try $t (1907 episode?)"; sleep 20; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

echo "--- staging MXFP8 ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/m3_stage" || { echo "FATAL stage"; exit 4; }
echo "staged: $(ls "$M3"/m3_stage_rank*.txt 2>/dev/null | wc -l)/$NP"

echo "--- Run1: gen Paris @ MAXPOS=1M (CP+int4 coherence + 1M arena fit) ($(date)) ---"
M3_REAL=1 M3_MAXPOS=1048576 M3_MAX_NEW=48 M3_PROMPT_IDS="$M3/prompt_ids.txt" M3_GEN_OUT="$M3/gen_cp_1m_96bf16.txt" \
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || echo "Run1 nonzero rc=$?"
echo "--- Run1 rank0 ---"; cat m3_ep_rank00.txt 2>/dev/null
echo "--- Run1 load (arena @1M) ---"; cat m3_ep_load_rank00.txt 2>/dev/null
echo "--- Run1 gen ids ---"; cat "$M3/gen_cp_1m_96bf16.txt" 2>/dev/null

echo "--- Run2: synthetic prefill 6200 (CP cross-rank merge + MSA) ($(date)) ---"
M3_REAL=1 M3_MAXPOS=8192 M3_PREFILL=6200 M3_DECODE=8 \
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" 2>&1 | grep -iE "CP ON|prefill|decode|argmax|NaN|SENTINEL|lockstep|FATAL" | head -12
echo "--- Run2 lockstep (all ranks must share one argmax) ---"
grep -h "last argmax" "$RUN"/m3_ep_perf_rank*.txt 2>/dev/null | sed 's/ NaN.*//' | sort | uniq -c
echo "=== done $(date) ==="
