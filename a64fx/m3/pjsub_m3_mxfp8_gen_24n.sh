#!/bin/bash
# MiniMax-M3 MXFP8 (~/models/m3-fp8, ~444 GB) REAL-WEIGHT generation on the MINIMAL node count:
# 24 A64FX nodes (2x3x4 torus). fp8 won't fit a 12-node interactive alloc, so this is a batch job.
#
# 24n is the minimal CLEAN-torus config that fits SHORT-context fp8 gen: the busiest EP rank owns
# ceil(128/24)=6 experts (~21.7 GB weight arena; ~26.2 GB operating estimate incl. page-cache/THP).
# It is TIGHT (~0.8 GB headroom) -> keep M3_MAXPOS small and VALIDATE VIA MemFree, not RSS. An OOM
# SIGKILL degrades PMIx and costs the whole alloc. If MemFree is tight, use the 32n script instead
# (4 experts/rank, comfortable). KV grows the floor (8k->26n, 32k->32n); for long ctx add
# M3_CP=1 + M3_INT4_KV=1. Sizing: python3 a64fx/m3/m3_sim.py.
#
# Submit:    ssh -A fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory a64fx/m3/pjsub_m3_mxfp8_gen_24n.sh'
# Detokenize: python3 a64fx/m3/m3_tokenizer.py decode-file a64fx/m3/gen_mxfp8_24.txt   # expect "... Paris"

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=2x3x4:torus,elapse=01:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=24"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-24}
RUN="$M3/mxfp8run_${PJM_JOBID:-$$}"; mkdir -p "$RUN"; cd "$RUN" || exit 2   # isolate tofu_topo.txt + logs per job
export M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_STAGE_DIR=/local/m3fp8 M3_STATUS_DIR="$RUN"
export M3_EP_SIZE=$NP M3_TP=1 M3_MSA=1
# TP_AR_BF16=1: halve the EP all-reduce payload. P3-confirmed LOSSLESS on real weights (coherent
# "Paris"), a free +2-3%. On by default. M3_MSTREAM=1 = single-prompt coherence gen; for BATCHED
# SERVING set M3_MSTREAM=48 (the throughput peak = ~15.25 tok/s real @48n; see m3_decode_sim.py).
export TP_AR_BF16=${TP_AR_BF16:-1}
export M3_MAXPOS=${M3_MAXPOS:-256} M3_MAX_NEW=${M3_MAX_NEW:-48} M3_MSTREAM=${M3_MSTREAM:-1}
export LLM_THREADS=12 OMP_NUM_THREADS=12   # M3 runs 1 CMG / 12 threads (pool-vs-OpenMP; do NOT bump)

echo "=== M3-MXFP8 gen 24n (2x3x4 torus, MINIMAL): NP=$NP mstream=$M3_MSTREAM maxpos=$M3_MAXPOS job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

# ---- prompt: encode the coherence-gate prompt (chat model completes "... Paris") ----
python3 "$M3/m3_tokenizer.py" encode "$(cat "$M3/prompt_m3.txt")" --bos > "$M3/prompt_ids.txt" \
  || { echo "FATAL encode"; exit 3; }
echo "prompt_ids: $(cat "$M3/prompt_ids.txt")"

topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[mxfp8] topo try $t (1907?)"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

echo "--- staging MXFP8 ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/m3_stage" || { echo "FATAL stage"; exit 4; }
echo "staged: $(ls "$RUN"/m3_stage_rank*.txt 2>/dev/null | wc -l)/$NP  rank0: $(cat "$RUN"/m3_stage_rank00.txt 2>/dev/null)"

echo "--- generate ($(date)) ---"
M3_REAL=1 M3_PROMPT_IDS="$M3/prompt_ids.txt" M3_GEN_OUT="$M3/gen_mxfp8_24.txt" \
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "FATAL gen"; exit 5; }
echo "--- rank0 ---"; cat m3_ep_rank00.txt 2>/dev/null
echo "--- load (arena / MemFree — the memory fit) ---"; cat m3_ep_load_rank00.txt 2>/dev/null
echo "--- gen ids ---"; cat "$M3/gen_mxfp8_24.txt" 2>/dev/null
echo "SENTINEL m3_mxfp8_24n=done"; echo "=== done $(date) ==="
