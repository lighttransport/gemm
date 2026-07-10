#!/bin/bash
# 2026-07-10 session: batched-decode throughput sweep with the new PF_TP + HC_SVE levers
# (dbbench config mirrors pjsub_ds4f.sh's MODE=dbbench: bf16 dense — Q8 GEMM is flat/M).
cd "$(dirname "$0")" || exit 2
SCR=${SCR:-/tmp}
DS4F_REAL=1 \
DS4F_FP8_BF16=1 DS4F_Q8_DENSE=0 DS4F_TIERB2=1 DS4F_MHC=1 DS4F_HC_PAR=1 DS4F_HC_RMSPAR=1 \
DS4F_ATTN_SVE=1 DS4F_OPROJ_FUSE=1 \
DS4F_DB_BENCH=1 DS4F_DB_MAXM=${DB_MAXM:-16} DS4F_DB_NTOK=${DB_NTOK:-24} \
DS4F_PF_TP=${DS4F_PF_TP:-1} DS4F_HC_SVE=${DS4F_HC_SVE:-1} \
DS4F_TP_HEAD=${DS4F_TP_HEAD:-1} DS4F_TP_EMBED=${DS4F_TP_EMBED:-1} \
DS4F_PREFILL=8 DS4F_MAXGEN=2 DS4F_MAXPOS=2048 \
  ./run_ds4f_11n.sh > "$SCR/dbbench.log" 2>&1
rc=$?
{ echo "DBBENCH rc=$rc"
  grep -E 'DECODE_BATCH|M=|agg' "$SCR/dbbench.log" ds4f_ep_rank00.txt 2>/dev/null | head -20
  echo DBBENCH_END
} | tee "$SCR/dbbench_sentinel.txt"
