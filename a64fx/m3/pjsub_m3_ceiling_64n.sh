#!/bin/bash
# MiniMax-M3 PRACTICAL-CEILING estimate on 96 A64FX nodes (8x12 torus). SYNTHETIC
# weights (no staging -> fast). Runs two passes of the full 60-layer EP+TP forward:
#   actual  (M3_DUMMY=0): real M=1 bf16 matvecs (the measured rate)
#   ceiling (M3_DUMMY=1): matvecs STREAM weight bytes at full HBM BW, NO FMA, but
#                         REAL EP all-reduce comm + REAL dispatch -> the comm+mem+
#                         dispatch floor = the practical tok/s ceiling if matvec were
#                         BW-perfect (compute idealized).
# Both at full 60 layers / 128 experts / TP on. Context is moderate (weights+comm+
# dispatch are ctx-independent; the 1M index-scan term is estimated separately --
# 1M KV is ~120 GB/rank replicated, needs CP/int4 to physically hold).
#
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_ceiling_96n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=4x4x4:torus,elapse=01:05:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=64"
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
export M3_MAXPOS=${M3_MAXPOS:-8192} M3_PREFILL=${M3_PREFILL:-16} M3_DECODE=${M3_DECODE:-64}
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3 ceiling 64n (4x4x4) job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[ceil] topo try $t (1907?)"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

run_pass(){  # $1=label  $2=dummy
  export M3_DUMMY=$2
  echo "=== PASS $1 (M3_DUMMY=$2) $(date) ==="
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "FATAL run $1"; return 1; }
  grep -E "prefill:|decode:" m3_ep_rank00.txt | sed "s/^/[$1] /"
}
run_pass actual  0
run_pass ceiling 1
echo "=== done $(date) ==="
