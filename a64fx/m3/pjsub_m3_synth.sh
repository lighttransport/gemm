#!/bin/bash
#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,retention_state=0,rscgrp=small-s2,node=8,elapse=01:01:00"
#PJM --mpi "proc=8"
#PJM --llio localtmp-size=80Gi
#PJM -j
#
# MiniMax-M3 (text) SYNTHETIC expert-parallel run as a BATCH job (robust to the
# interactive int-queue CODE=1907 topo-helper flake — a batch alloc picks fresh
# nodes when scheduled, and the run script retries the helper). No weight staging.
#
#   pjsub --no-check-directory a64fx/m3/pjsub_m3_synth.sh
#   pjsub --no-check-directory -x M3_LAYERS=8 -x M3_EXPERTS=0 a64fx/m3/pjsub_m3_synth.sh
#
# Validates the M3 EP forward at full dims (truncated layers) on N ranks: per-node
# memory fit, cross-rank lockstep (identical synthetic argmax), prefill/decode tok/s.
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
cd "$REPO" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-8}
export NP EXCLUDE=none                 # batch alloc: no co-located login node to spare
export M3_LAYERS=${M3_LAYERS:-8}       # truncate to fit a small alloc (0 = full 60)
export M3_EXPERTS=${M3_EXPERTS:-0}     # 0 = full 128 (uneven e%NP ok)
export M3_MAXPOS=${M3_MAXPOS:-512}
export M3_PREFILL=${M3_PREFILL:-8}
export M3_DECODE=${M3_DECODE:-16}
export LLM_THREADS=${LLM_THREADS:-48}

echo "=== M3 synth EP batch: NP=$NP layers=$M3_LAYERS experts=$M3_EXPERTS job=${PJM_JOBID:-?} ==="
date
bash a64fx/m3/run_m3_synth_Nn.sh
echo "=== done $(date) ==="
