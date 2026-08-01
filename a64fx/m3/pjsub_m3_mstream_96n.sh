#!/bin/bash
# MiniMax-M3 MULTI-STREAM batched decode on 96 A64FX nodes (8x12 torus). SYNTHETIC
# (no staging). Compares single-stream vs N concurrent streams: dense GEMMs become
# M=N and the EP all-reduce fires ONCE per layer for all N tokens -> dispatch+comm
# amortized N-fold. Reports AGGREGATE tok/s (the throughput that matters for serving).
#
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_mstream_96n.sh'

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
export M3_LAYERS=0 M3_EXPERTS=0 M3_TP=1 M3_MSA=1
export M3_MAXPOS=${M3_MAXPOS:-1024} M3_DECODE=${M3_DECODE:-32}
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3 mstream 96n (8x12) job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[ms] topo try $t (1907?)"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

run_pass(){  # $1=N
  export M3_MSTREAM=$1
  echo "=== N=$1 ($(date)) ==="
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "FATAL N=$1"; return 1; }
  grep -E "decode:|MSTREAM|AGG|per-stream" m3_ep_rank00.txt | sed "s/^/[N=$1] /"
}
run_pass 1
run_pass 4
run_pass 8
echo "=== done $(date) ==="
