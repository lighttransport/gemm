#!/bin/bash
# MiniMax-M3 BF16 DECODE-PERF probe P1b — higher-M plateau. P1 found batching still climbing at M=32
# (16.68 tok/s @48n, bf16-AR); P2's single-node compute ceiling is ~17.6. This probes M=32/48/64 to
# find where aggregate throughput plateaus. bf16-AR fixed on (P1's free +2-3% win, lockstep-preserving).
#
# MEMORY: per-stream KV grows x M and is NOT in arena_used. At maxpos=1024, M=64 KV ~8 GB + arena 21.5
# ~= 29.5 GB (risky at 48n -> OOM SIGKILL degrades PMIx, costs the alloc). So P1b uses **maxpos=512**
# (KV halved -> M=64 ~25.5 GB, safe) and includes M=32 as a control: if M=32@512 ~= P1's M=32@1024
# (16.68), tok/s is maxpos-independent and the M=48/64 numbers are comparable to P1. M=64 runs LAST so
# an OOM (if any) can't poison earlier passes.
#
# Submit: ssh -A fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory a64fx/m3/pjsub_m3_bf16_mstream_hiM_48n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=8x6:torus,elapse=01:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=48"
#PJM --llio localtmp-size=20Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
cd "$M3" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-48}
export M3_LAYERS=0 M3_EXPERTS=0 M3_TP=1 M3_MSA=1        # full 60L synthetic bf16, TP on
export M3_MAXPOS=${M3_MAXPOS:-512} M3_DECODE=${M3_DECODE:-32}   # maxpos 512 keeps M=64 KV in bounds
export TP_AR_BF16=1                                      # P1's free win, fixed on
export LLM_THREADS=12 OMP_NUM_THREADS=12

echo "=== M3 bf16 mstream hi-M 48n (8x6, maxpos=$M3_MAXPOS, bf16-AR): NP=$NP job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[p1b] topo try $t (1907?)"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

run_pass(){  # $1=M
  export M3_MSTREAM=$1
  echo "=== M=$1 bf16-AR maxpos=$M3_MAXPOS ($(date)) ==="
  rm -f m3_ep_rank00.txt m3_ep_load_rank00.txt
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "[M=$1] FATAL (OOM? per-stream KV x M)"; return 0; }
  grep -iE "decode:|MSTREAM|AGG|per-stream|comm|arena|argmax|NaN" m3_ep_load_rank00.txt m3_ep_rank00.txt | sed "s/^/[M=$1] /"
}
run_pass 32   # control vs P1's M=32@1024 (16.68) — confirms maxpos-independence
run_pass 48
run_pass 64   # last: if it OOMs, earlier passes are safe
echo "SENTINEL m3_bf16_mstream_hiM_48n=done"; echo "=== done $(date) ==="
