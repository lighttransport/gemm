#!/bin/bash
# CP plumbing smoke on 4 nodes (SYNTHETIC weights, no staging): validates the context-parallel
# KV path -- block-cyclic position sharding, kv_combine_cb (flash combine), blk_reduce_cb
# (block-score MAX), and lockstep argmax across ranks -- cheaply before the 1M real job.
# prefill 700 (> 4*128=512 -> positions shard across all 4 ranks; dense, fast O(n^2)).
# Key check: noCP-bf16 and CP-bf16 must give the SAME argmax (the flash combine is EXACT
# reassembly) -> proves the CP machinery. CP-int4 may differ slightly (quant). All: NaN=0,
# lockstep (all ranks share one argmax). The MSA block top-k merge is tested at 6200 in the
# 48n real job. Pass = noCP==CP-bf16 argmax, NaN=0, rc=0.
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_cp_smoke_4n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small,node=4,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=4"
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; M3="$REPO/a64fx/m3"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=4
RUN="$M3/cpsmoke_${PJM_JOBID:-$$}"; mkdir -p "$RUN"; cd "$RUN" || exit 2
export M3_EP_SIZE=$NP M3_TP=0 M3_MSA=1 M3_LAYERS=4 M3_MAXPOS=4096
export LLM_THREADS=12 OMP_NUM_THREADS=12
echo "=== CP smoke 4n (synthetic): $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0
for t in $(seq 1 12); do rm -f tofu_topo.txt
  if mpiexec -np $NP "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null||echo 0)" -ge $NP ]; then topo_ok=1; break; fi
  echo "topo try $t (1907?)"; sleep 8; done
[ "$topo_ok" = 1 ] || { echo "FATAL topo (1907 persistent)"; exit 3; }

for KV in "noCP M3_CP=0 M3_INT4_KV=0" "CP-bf16 M3_CP=1 M3_INT4_KV=0" "CP-int4 M3_CP=1 M3_INT4_KV=1"; do
  set -- $KV; lbl=$1; shift
  echo "--- [$lbl] prefill 700 (CP position-shard + combine) ($(date)) ---"
  env "$@" M3_PREFILL=700 M3_DECODE=6 mpiexec -np $NP "$LLM/build/m3_ep_runner" 2>&1 | grep -iE "CP ON|prefill:|decode:|argmax=|NaN|SENTINEL|FATAL" | head -6
  echo "  lockstep argmax across ranks:"; grep -h "last argmax" "$RUN"/m3_ep_perf_rank*.txt 2>/dev/null | sed 's/ NaN.*//' | sort | uniq -c
done
echo "=== done $(date) ==="
