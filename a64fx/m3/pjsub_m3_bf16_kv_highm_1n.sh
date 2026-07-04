#!/bin/bash
# MiniMax-M3 BF16 DECODE-PERF probe P2 — int4-KV as the high-M memory enabler. 1 NODE, SYNTHETIC bf16,
# truncated (12 layers / 16 experts) so it fits one node; the KV cache is per-layer/per-stream and
# REPLICATED per rank, so its memory scaling is node-count-independent — 1 node is representative and
# dirt cheap (~0.4 node-h).
#
# P1 shows aggregate throughput scaling with M; but per-stream KV grows x M, so high M can OOM at the
# minimal node count. This probe answers: does M3_INT4_KV=1 (int4 K/V, ~4x smaller than bf16) shrink the
# arena enough to deploy P1's high-M configs at the min nodes, WITHOUT hurting per-stream decode tok/s?
# Reports arena/rank + decode tok/s per (M, KV-format). The bf16->int4 arena DELTA at fixed M is the KV
# saving; extrapolate x(60/12) for the full model. (KV *quality* is a separate real-weight test —
# pjsub_m3_kv_mstream_1n.sh; this is memory + speed only.)
#
# Submit: ssh -A fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory a64fx/m3/pjsub_m3_bf16_kv_highm_1n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --llio localtmp-size=8Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; M3="$REPO/a64fx/m3"; UTOFU="$REPO/a64fx/utofu-tests"
cd "$M3" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
# small synthetic that fits 1 node (1 rank owns ALL experts): 12 layers / 16 experts. TP off (1 rank).
export M3_LAYERS=12 M3_EXPERTS=16 M3_TP=0 M3_MSA=1
export M3_MAXPOS=${M3_MAXPOS:-1024} M3_DECODE=${M3_DECODE:-32}
export OMP_NUM_THREADS=12 LLM_THREADS=12

echo "=== M3 bf16 KV x high-M enabler 1n (12L/16E synth): job=${PJM_JOBID:-?} $(date) ==="
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" m3_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
# the runner read_topo()s tofu_topo.txt and exit(1)s without it — generate it even for 1 rank
rm -f tofu_topo.txt
mpiexec -np 1 "$UTOFU/tofu_topo_helper" || { echo "FATAL topo"; exit 3; }
echo "topo lines: $(wc -l < tofu_topo.txt 2>/dev/null)"

run_pass(){  # $1=M  $2=label  $3=int4flag(0/1)
  echo "=== M=$1 KV=$2 ($(date)) ==="
  rm -f m3_ep_rank00.txt m3_ep_load_rank00.txt
  # the runner writes perf to m3_ep_rank00.txt and arena to m3_ep_load_rank00.txt (NOT stdout)
  M3_MSTREAM=$1 M3_INT4_KV=$3 mpiexec -np 1 "$LLM/build/m3_ep_runner" > "pass_${1}_${2}.log" 2>&1 \
    || echo "[M=$1 KV=$2] runner rc=$?"
  grep -iE "arena|decode:|prefill:|MSTREAM|AGG|per-stream|NaN|cannot open|FATAL|Error" \
    m3_ep_load_rank00.txt m3_ep_rank00.txt "pass_${1}_${2}.log" 2>/dev/null | sed "s/^/[M=$1 KV=$2] /"
}
for M in 8 32 64; do
  run_pass "$M" bf16 0
  run_pass "$M" int4 1
done
echo "SENTINEL m3_bf16_kv_highm_1n=done"; echo "=== done $(date) ==="
