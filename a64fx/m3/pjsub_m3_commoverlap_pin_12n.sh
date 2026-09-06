#!/bin/bash
# FU2 — Lever-2 threading de-risk: does PINNING the comm-driver thread (M3_COMM_CORE + OMP=cores-1) make
# the per-layer comm-overlap viable, or is it structurally capped? SYNTHETIC 12n (full 60L, 24 experts =
# 2/rank), M3_MSTREAM=8. Three passes:
#   A sync         : OMP=12, TP_SHARED=1, overlap OFF  -> the best-config baseline
#   C overlap/unpin: OMP=12, TP_SHARED=0, overlap ON   -> m3.md's net-negative config (oversubscribed)
#   B overlap/pin  : OMP=11, TP_SHARED=0, overlap ON, comm pinned to core 23 -> the de-risk
# Reads: does B not hang (thread stability)? is B > C (pinning recovers the oversubscription cost)? is
# B >= A (would make per-layer overlap net-positive -> Lever 2 cheap)? Expect B>C but B<A (TP_SHARED=0
# penalty is structural -> Lever 2 needs the cross-layer restructure, not just pinning).
#
# Submit: ssh -A fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory a64fx/m3/pjsub_m3_commoverlap_pin_12n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small,node=3x4:torus,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=20Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
cd "$M3" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-12}
export M3_LAYERS=0 M3_EXPERTS=24 M3_MSA=1 M3_MSTREAM=${M3_MSTREAM:-8}
export M3_MAXPOS=${M3_MAXPOS:-1024} M3_DECODE=${M3_DECODE:-32}

echo "=== M3 comm-overlap pin de-risk 12n (3x4): NP=$NP job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
  echo "[o] topo try $t"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo"; exit 3; }

run(){  # $1=label  $2..=env
  echo "=== PASS $1 ($(date)) ==="
  rm -f m3_ep_rank00.txt
  env "${@:2}" timeout 300 mpiexec -np "$NP" "$LLM/build/m3_ep_runner" >/dev/null 2>&1
  rc=$?; [ $rc -eq 124 ] && echo "[$1] TIMEOUT/HANG (rc=124)"; [ $rc -ne 0 ] && [ $rc -ne 124 ] && echo "[$1] rc=$rc"
  grep -hE "AGG|comm-driver pinned|MSTREAM|NaN" m3_ep_rank00.txt 2>/dev/null | sed "s/^/[$1] /"
}
# M3 uses only CMG0 (12 threads); CMG1-3 (cores 24-59) are IDLE. Pin comm to an idle-CMG core (24) so
# compute keeps all 12 CMG0 cores AND comm gets a dedicated core -> the correct overlap test.
run A_sync           M3_TP=1                 M3_COMM_OVERLAP=0 OMP_NUM_THREADS=12 LLM_THREADS=12
run C_overlap_unpin  M3_TP=1 M3_TP_SHARED=0  M3_COMM_OVERLAP=1 OMP_NUM_THREADS=12 LLM_THREADS=12
run B_overlap_idlecmg M3_TP=1 M3_TP_SHARED=0 M3_COMM_OVERLAP=1 OMP_NUM_THREADS=12 LLM_THREADS=12 \
                     OMP_PROC_BIND=close OMP_PLACES=cores M3_COMM_CORE=24
echo "SENTINEL m3_commoverlap_pin_12n=done"; echo "=== done $(date) ==="
