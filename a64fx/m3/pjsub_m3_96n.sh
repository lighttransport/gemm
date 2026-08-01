#!/bin/bash
# MiniMax-M3 (text) SYNTHETIC EP+TP smoke on 96 A64FX nodes, 8x12 torus (compact,
# same-rack -> low Tofu-hop all-reduce). No weight staging: each rank fills its
# e%96 expert shard + TP-sharded dense in HBM, then runs synthetic prefill+decode
# with the per-MoE-layer tp_allreduce_sum combine + TP_HEAD argmax merge. Validates
# at full scale: per-node memory fit (~14-15 GB/rank with TP on, full 60 layers),
# cross-rank lockstep (identical synthetic argmax), and prefill/decode tok/s.
#
# NOTE: the multi-rank topo helper has been hitting the system-wide CODE=1907
# uTofu flake; run_m3_synth_Nn.sh retries it x5. If it still fails, the job log
# says so -> resubmit on a fresh alloc when uTofu is healthy.
#
# Submit:  ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_96n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=8x12:torus,elapse=01:05:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=20Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
cd "$REPO" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-96}
export NP EXCLUDE=none                 # batch alloc: -np 96, no vcoordfile
export M3_LAYERS=${M3_LAYERS:-0}       # 0 = full 60 layers
export M3_EXPERTS=${M3_EXPERTS:-0}     # 0 = full 128 (uneven e%96: 32 ranks own 2, 64 own 1)
export M3_TP=${M3_TP:-1}               # TP_ATTN/SHARED/FFN/HEAD on (mandatory to fit full dims)
export M3_MSA=${M3_MSA:-1}
export M3_MAXPOS=${M3_MAXPOS:-512}
export M3_PREFILL=${M3_PREFILL:-8}
export M3_DECODE=${M3_DECODE:-16}
export LLM_THREADS=${LLM_THREADS:-48}

echo "=== M3 synth EP+TP 96n (8x12 torus): NP=$NP layers=$M3_LAYERS tp=$M3_TP job=${PJM_JOBID:-?} ==="
date
bash a64fx/m3/run_m3_synth_Nn.sh
echo "=== done $(date) ==="
