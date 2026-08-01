#!/bin/bash
# MiniMax-M3 COMM-OVERLAP test on 96 A64FX nodes (8x12 torus). SYNTHETIC. N=8 multi-
# stream, TP_SHARED OFF (shared expert replicated -> independent of the routed reduce).
# Compares the routed-expert EP all-reduce SYNCHRONOUS vs OVERLAPPED with the shared-
# expert compute on a dedicated comm-driver thread. Same config both passes -> the
# delta is the overlap benefit. Reports AGG tok/s + comm%.
#
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_overlap_96n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=8x12:torus,elapse=01:05:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=20Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
cd "$M3" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-96}
export M3_LAYERS=0 M3_EXPERTS=0 M3_MSA=1
# TP set with SHARED OFF (replicated shared -> overlap-eligible)
export M3_TP=0 M3_TP_ATTN=1 M3_TP_FFN=1 M3_TP_HEAD=1 M3_TP_EMBED=1 M3_TP_SHARED=0
export M3_MAXPOS=${M3_MAXPOS:-1024} M3_DECODE=${M3_DECODE:-32} M3_MSTREAM=${M3_MSTREAM:-8}
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3 comm-overlap 96n (8x12) N=$M3_MSTREAM job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[ov] topo try $t (1907?)"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

run_pass(){  # $1=label $2=overlap
  export M3_COMM_OVERLAP=$2
  echo "=== $1 (M3_COMM_OVERLAP=$2) $(date) ==="
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "FATAL $1"; return 1; }
  grep -E "MSTREAM|AGG|overlap" m3_ep_rank00.txt | sed "s/^/[$1] /"
}
run_pass sync    0
run_pass overlap 1
echo "=== done $(date) ==="
