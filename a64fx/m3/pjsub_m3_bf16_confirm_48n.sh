#!/bin/bash
# MiniMax-M3 BF16 DECODE-PERF probe P3 (EXPENSIVE ~72 node-h) — confirm the P1/P1b winner
# (M3_MSTREAM=48 + TP_AR_BF16=1 = 17.75 tok/s synth) on REAL bf16 weights @ 48n. Stages the full 796 GB
# model to /local/m3 once, then runs TWO passes (the runner routes mstream>1 to synthetic batched
# throughput, so coherence and throughput can't share one run):
#   Pass A — COHERENCE: real weights, mstream=1, bf16-AR, greedy gen of the "Paris" prompt. Confirms
#            bf16-AR (argmax-identical on synth) preserves REAL-weight output + the arena/MemFree fit.
#   Pass B — THROUGHPUT: real weights, M3_MSTREAM=48, bf16-AR, maxpos=512 (synthetic activations —
#            throughput/memory only). Confirms the 17.75 synth number on real weights AND that M=48's
#            per-stream KV fits 48n with real-weight page cache (the m3_decode_sim open question).
#
# Knobs: TP_AR_BF16 (=1, the free P1 win), M3_INT4_KV (=0). Validate fit via MemFree, not RSS.
# Submit: ssh -A fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory a64fx/m3/pjsub_m3_bf16_confirm_48n.sh'
# Detokenize (Pass A): python3 a64fx/m3/m3_tokenizer.py decode-file a64fx/m3/gen_bf16_48.txt  # "... Paris"

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
export TP_AR_BF16=${TP_AR_BF16:-1} M3_INT4_KV=${M3_INT4_KV:-0}   # bf16-AR = the free P1 win, on
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3 bf16 CONFIRM 48n (8x6): NP=$NP ar_bf16=$TP_AR_BF16 int4_kv=$M3_INT4_KV job=${PJM_JOBID:-?} $(date) ==="
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

# ---- Pass A: COHERENCE — real weights + bf16-AR, single-stream greedy gen ("Paris" gate). Confirms
#      bf16-AR (argmax-identical on synth) preserves REAL-weight output + the arena/MemFree fit. ----
echo "--- Pass A: coherence gen (real, mstream=1, bf16-AR) ($(date)) ---"
rm -f m3_ep_rank00.txt m3_ep_load_rank00.txt
M3_REAL=1 M3_MSTREAM=1 M3_MAXPOS=256 M3_MAX_NEW=${M3_MAX_NEW:-48} \
  M3_PROMPT_IDS="$M3/prompt_ids.txt" M3_GEN_OUT="$M3/gen_bf16_48.txt" \
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || echo "FATAL gen rc=$?"
echo "--- A rank0 ---"; cat m3_ep_rank00.txt 2>/dev/null
echo "--- A load (arena / MemFree) ---"; cat m3_ep_load_rank00.txt 2>/dev/null
echo "--- A gen ids ---"; cat "$M3/gen_bf16_48.txt" 2>/dev/null
echo "detok:"; python3 "$M3/m3_tokenizer.py" decode-file "$M3/gen_bf16_48.txt" 2>/dev/null

# ---- Pass B: THROUGHPUT — real weights, M=48 batched decode (the winning config). Confirms the 17.75
#      synth number on REAL weights AND that M=48's per-stream KV fits 48n with real-weight page cache
#      (the m3_decode_sim open question). Synthetic activations (throughput/memory only). maxpos=512.
#      LAST pass: an OOM here (if real page cache caps M<48) can't poison Pass A. ----
echo "--- Pass B: M=48 throughput (real weights, bf16-AR, maxpos=512) ($(date)) ---"
rm -f m3_ep_rank00.txt m3_ep_load_rank00.txt
M3_REAL=1 M3_MSTREAM=48 M3_MAXPOS=512 M3_PREFILL=8 M3_DECODE=32 \
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || echo "[Pass B] FATAL rc=$? (M=48 real-weight OOM? -> int4-KV / more nodes)"
echo "--- B rank0 (AGG tok/s) ---"; cat m3_ep_rank00.txt 2>/dev/null
echo "--- B load (arena / MemFree) ---"; cat m3_ep_load_rank00.txt 2>/dev/null
echo "SENTINEL m3_bf16_confirm_48n=done"; echo "=== done $(date) ==="
