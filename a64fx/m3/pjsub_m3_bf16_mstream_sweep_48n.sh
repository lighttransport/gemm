#!/bin/bash
# MiniMax-M3 BF16 DECODE-PERF probe P1 — multi-stream throughput past N=8 x comm-payload.
# SYNTHETIC full-60L bf16 @ 48n (the bf16 minimal-node deployment scale; tok/s is ~flat in N so this
# is representative and cheaper than 96n). No staging (synthetic weights).
#
# Best known: ~14.2 tok/s aggregate @ N=8 (multi-stream M3_MSTREAM=8 + expert grouping, TP_SHARED=1).
# Single-stream ceiling ~3.3 is STRUCTURAL (dispatch+comm+serial, not compute/BW). Comm grows 25->42%
# as M rises. This probe answers two open questions:
#   (a) does aggregate decode throughput keep scaling past M=8 (to 16/32)? — the primary lever.
#   (b) does TP_AR_BF16=1 (halve the per-layer EP all-reduce payload) cut the comm that dominates at
#       high M? bf16-AR reassociates the reduce -> NOT bit-identical to f32-AR; lockstep argmax must
#       still hold (all ranks identical) and output must stay coherent.
# Expert grouping is on (default). Watch per-rank arena at M=32 (per-stream KV x M): if it OOMs, that
# is the int4-KV probe's job (P2) — the sweep continues past a failed pass.
#
# Submit: ssh -A fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory a64fx/m3/pjsub_m3_bf16_mstream_sweep_48n.sh'

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
export M3_MAXPOS=${M3_MAXPOS:-1024} M3_DECODE=${M3_DECODE:-32}
export LLM_THREADS=12 OMP_NUM_THREADS=12                # 1 CMG sweet spot (full-48/pool did not pan out)

echo "=== M3 bf16 mstream x AR sweep 48n (8x6): NP=$NP job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[p1] topo try $t (1907?)"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

run_pass(){  # $1=M  $2=AR_bf16
  export M3_MSTREAM=$1 TP_AR_BF16=$2
  echo "=== M=$1 AR_bf16=$2 ($(date)) ==="
  rm -f m3_ep_rank00.txt m3_ep_load_rank00.txt    # so a failed pass shows empty, not stale numbers
  mpiexec -np "$NP" "$LLM/build/m3_ep_runner" || { echo "[M=$1 AR=$2] FATAL (OOM at high M? -> P2 int4-KV)"; return 0; }
  grep -iE "decode:|MSTREAM|AGG|per-stream|comm|arena|argmax|NaN" m3_ep_load_rank00.txt m3_ep_rank00.txt | sed "s/^/[M=$1 AR=$2] /"
}
# f32-AR: does mstream scale past 8?
run_pass 8  0
run_pass 16 0
run_pass 32 0
# bf16-AR: does halving the reduce payload help at the comm-bound high-M points?
run_pass 8  1
run_pass 16 1
run_pass 32 1
echo "SENTINEL m3_bf16_mstream_sweep_48n=done"; echo "=== done $(date) ==="
