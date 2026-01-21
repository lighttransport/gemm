#!/bin/bash
#PJM -L "rscgrp=small"
#PJM -L "node=1"
#PJM -L "elapse=00:05:00"
#PJM -L "freq=2000,eco_state=0"
#PJM -g hp250467
#PJM -j

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/mem-pattern

echo "Q@K^T Only Memory Access Test"
echo "=============================="
./bench_qkt_only 10000
