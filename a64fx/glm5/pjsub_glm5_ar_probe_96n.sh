#!/bin/bash
# GLM-5.2 decode CALIBRATION probe, 96 nodes / ~30 min — feeds the two local simulators
# (a64fx/glm5/decode_sim.py and the qlair A64FX sim) with REAL-machine numbers so decode
# optimization keeps happening locally instead of burning node-hours. See CALIBRATION.md
# for exactly where each output line plugs in.
#
# Four phases (all synthetic weights; NO staging):
#   1. ARPROBE   — production tp_allreduce anatomy at 8 ranks: robust {1,2-lean,0} x
#                  {fp32,bf16} x M in {1,2,8,16,32}, argmax/argmax_n, 78-AR decode-token.
#   2. KERNBENCH — decode-shape bf16 GEMM bandwidth vs batch M (single node, 48 threads).
#   3. SELFCHECK — GLM5_BATCH_SELFCHECK: batched-MLA kernel vs single-token forward on
#                  REAL SVE hardware with the real 8-rank comm (argmax must MATCH).
#   4. CBATCH A/B — synthetic continuous-batch decode, GLM5_BATCH_DECODE=0 vs 1:
#                  identical CBATCH_IDS token streams + the batched-vs-per-slot agg tok/s
#                  ratio at 8 nodes (compare against decode_sim.pred_tok_s(8, M=8)).

#PJM -g hp250467
#PJM -L "rscgrp=small,node=96,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"
GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-8}
WORK="$GLM5/ar_probe_run_${PJM_JOBID:-manual}"
mkdir -p "$WORK" || exit 2
cd "$WORK" || exit 2
export LLM_THREADS=${LLM_THREADS:-48}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}

echo "=== GLM5.2 decode calibration probe ${NP}n job=${PJM_JOBID:-?} ==="
echo "workdir=$WORK"; date

rm -f tofu_topo.txt glm5_ep_*.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE -fopenmp \
    -I "$REPO/common" -o bdecode_kern_bench "$GLM5/bdecode_kern_bench.c" -lm || exit 3

# Fujitsu MPI: the first mpiexec after a fresh allocation can fail coll-select init
# ("system-rule-file is not found"); retry until tofu_topo.txt has >= NP lines (matches the
# working cbatch/int8 scripts). coll-select stderr per-rank is benign noise.
topo_ok=0
for t in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" 2>/dev/null && \
       [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then
        topo_ok=1; echo "topo window on try $t"; break
    fi
    echo "[ar_probe] topo try $t failed; retry"; sleep 5
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }

echo "--- phase 1: ARPROBE (production all-reduce anatomy, $NP ranks) ---"
## GLM5_PREFILL_GROUPS=1 forces ONE group of all NP ranks (else pick_groups splits into small
## data-parallel groups and the AR spans only 2 ranks -> N-ladder meaningless).
GLM5_AR_PROBE=1 GLM5_PROBE_REPS=50 GLM5_PREFILL_GROUPS=1 \
GLM5_REAL=0 GLM5_DUMMY=1 GLM5_LAYERS=1 GLM5_EXPERTS=8 GLM5_MAXPOS=32 GLM5_AR_TOKENS=32 \
GLM5_PREFILL=1 GLM5_DECODE=0 \
  mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || echo "WARN: ar_probe rc=$?"
grep -h "ARPROBE" glm5_ep_rank00.txt 2>/dev/null
mv glm5_ep_rank00.txt ar_probe_rank00.txt 2>/dev/null

echo "--- phase 2: KERNBENCH (decode-shape bf16 GEMM GB/s vs M, 1 node) ---"
./bdecode_kern_bench || echo "WARN: kern bench rc=$?"

echo "--- phase 3: BATCH_SELFCHECK on real hardware ($NP ranks, synthetic weights) ---"
GLM5_BATCH_SELFCHECK=7 \
GLM5_REAL=0 GLM5_LAYERS=4 GLM5_EXPERTS=8 GLM5_MAXPOS=64 GLM5_TP=1 \
GLM5_PREFILL=2 GLM5_DECODE=2 \
  mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || echo "WARN: selfcheck rc=$?"
grep -h "BATCH_SELFCHECK\|BATCH_DECODE AR" glm5_ep_rank00.txt 2>/dev/null
mv glm5_ep_rank00.txt selfcheck_rank00.txt 2>/dev/null

echo "--- phase 4: synthetic cbatch decode, per-slot (bd=0) vs batched (bd=1) ---"
# 8 layers x 64 experts fits 8 nodes; comm/token = 8-9 ARs, directly scalable in decode_sim.
PROMPTS="$WORK/probe_prompts.txt"
python3 - "$PROMPTS" <<'EOF' 2>/dev/null || { for i in $(seq 0 15); do echo "$((i+2)) $((i+3)) $((i+5)) $((i+7))"; done > "$PROMPTS"; }
import sys, random
random.seed(7)
with open(sys.argv[1], "w") as f:
    for r in range(16):
        f.write(" ".join(str(random.randrange(2, 5000)) for _ in range(8)) + "\n")
EOF
for BD in 0 1; do
  echo "--- cbatch GLM5_BATCH_DECODE=$BD ---"
  GLM5_BATCH_DECODE=$BD \
  GLM5_REAL=0 GLM5_LAYERS=8 GLM5_EXPERTS=64 GLM5_MAXPOS=128 GLM5_TP=1 \
  GLM5_CBATCH_PROMPTS="$PROMPTS" GLM5_CBATCH_SLOTS=8 GLM5_MAX_NEW=12 \
  GLM5_CBATCH_OUT_PREFIX="$WORK/cb${BD}" \
    mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || echo "WARN: cbatch bd=$BD rc=$?"
  grep -h "cbatch:\|CBATCH_IDS" glm5_ep_rank00.txt 2>/dev/null
  mv glm5_ep_rank00.txt cbatch_bd${BD}_rank00.txt 2>/dev/null
done
echo "--- A/B token-stream identity (must be no diff lines) ---"
for f in "$WORK"/cb0_*.txt; do
  b="$WORK/cb1_${f##*/cb0_}"
  cmp -s "$f" "$b" || echo "TOKEN DIFF: $f vs $b"
done
echo "SENTINEL glm5_ar_probe_job=done"
date