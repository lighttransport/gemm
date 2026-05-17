#!/bin/bash
# pjsub_run.sh -- submit copy+read LLIO benchmark on one A64FX node.
#
# Submit from a login/cross node:
#   pjsub a64fx/llio/pjsub_run.sh
#
# Requests:
#   localtmp-size=87Gi   -- $PJM_LOCALTMP for `cp` staging
#   sharedtmp-size=80Gi  -- backs `llio_transfer` common-file cache
#                           (without this, llio_transfer fails with
#                            "Not enough disk space")

#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:60:00"
#PJM -L "freq=2000,eco_state=0"
#PJM --llio localtmp-size=87Gi
#PJM --llio sharedtmp-size=80Gi
#PJM --no-check-directory
#PJM -j

set -u

DIR=/vol0006/mdt0/data/hp250467/work/gemm/localtemp/a64fx/llio
cd "$DIR"

echo "============================================================"
echo "host:           $(hostname)"
echo "date:           $(date -u +%FT%TZ)"
echo "PJM_LOCALTMP:  ${PJM_LOCALTMP:-<unset>}"
echo "PJM_SHAREDTMP: ${PJM_SHAREDTMP:-<unset>}"
echo "============================================================"

# Build native on the A64FX compute node.
echo
echo "### build (fcc native) ###"
make clean
make COMPILER=fcc

echo
echo "### copy: \$HOME/models/qwen35/9b -> \$PJM_LOCALTMP ###"
bash ./copy_bench.sh

echo
echo "### read benchmarks (localtmp staged via cp) ###"
# Run reads against the Q4 (smallest), Q8, and BF16 files.
for f in Qwen3.5-9B-UD-Q4_K_XL.gguf Qwen3.5-9B-UD-Q8_K_XL.gguf Qwen3.5-9B-BF16.gguf; do
    echo
    echo "=========================================================="
    echo "TARGET (localtmp): $f"
    echo "=========================================================="
    bash ./run_bench.sh "$PJM_LOCALTMP/llio_bench/$f"
done

echo
echo "### llio_transfer bench (sharedtmp L2 common-file cache) ###"
# llio_transfer reads the file from its ORIGINAL Lustre path but caches
# it on the I/O node's SSD. Test the Q4 file (smallest, fastest turnaround).
# NOTE: per llio_transfer(1), the file must not have been opened in this
# job before --sync. Run this step BEFORE copy_bench.sh if you want a
# strictly-clean measurement; running it after means the per-CN OS page
# cache may already be warm.
bash ./llio_transfer_bench.sh "$HOME/models/qwen35/9b" \
    Qwen3.5-9B-UD-Q4_K_XL.gguf

echo
echo "### final usage ###"
du -sh "$PJM_LOCALTMP"/* 2>/dev/null || true
df -h "$PJM_LOCALTMP" "$PJM_SHAREDTMP" 2>/dev/null || true
