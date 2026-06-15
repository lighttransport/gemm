#!/bin/bash
# MiniMax-M3 REAL-WEIGHT multi-task generation on 96 A64FX nodes (8x12 torus).
# Stage the full model ONCE, then run two instruct tasks (summarization w/ a long
# input, and a coding task) with long outputs, logging prefill + decode tok/s each.
# Greedy decode (= temperature 0, the deterministic / "lowest temperature" setting,
# appropriate for coding). Detokenize offline with m3_tokenizer.py decode-file.
#
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_tasks_96n.sh'

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
export M3_MODEL_DIR=${M3_MODEL_DIR:-$HOME/models/m3} M3_STAGE_DIR=/local/m3
export M3_NSHARDS=59 M3_EP_SIZE=$NP M3_STATUS_DIR="$M3"
export M3_TP=1 M3_MSA=1 M3_MAXPOS=${M3_MAXPOS:-1024} M3_MAX_NEW=${M3_MAX_NEW:-256}
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3 multi-task 96n (8x12) job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[tasks96] topo try $t (1907?); retry"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

echo "--- staging ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/m3_stage" || { echo "FATAL stage"; exit 4; }
echo "staged: $(ls "$M3"/m3_stage_rank*.txt 2>/dev/null | wc -l)/$NP"

run_task(){  # $1=name $2=prompt_file $3=out_file
  echo "=== TASK $1 ($(date)) prompt_tokens=$(wc -w < "$2") ==="
  M3_REAL=1 M3_PROMPT_IDS="$2" M3_GEN_OUT="$3" \
    mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "FATAL gen $1"; return 1; }
  echo "[$1] $(grep -E 'gen:' m3_ep_rank00.txt | tail -2)"
  echo "[$1] GEN_IDS -> $3 ($(wc -w < "$3") tokens)"
}
run_task summ "$M3/prompt_summ.txt" "$M3/gen_summ.txt"
run_task code "$M3/prompt_code.txt" "$M3/gen_code.txt"
echo "=== done $(date) ==="
