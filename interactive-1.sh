#!/bin/sh
# hp250467
# pjsub --interact -g hp250467 -L "node=1" -L "rscgrp=int" -L "elapse=1:00:00" --sparam "wait-time=600"

# run interactive job with 1 hour limit.
# freq=2000 : normal mode
# freq=2200 : boost mode(eco_state=2 only: boost-eco mode)
# eco_state = 0 -> eco mode off(use 2 FPUs. default 2. it use only 1 FPU = up to 50% of peak FLOPS)
# pjsub --interact -g hp250467 -L "freq=2000,eco_state=0,rscgrp=int,node=1,elapse=06:00:00" --sparam "wait-time=600" --no-check-directory -x PJM_LLIO_GFSCACHE=/vol0004
pjsub --interact -g hp250467 -L "freq=2000,eco_state=0,rscgrp=int,node=1,elapse=06:00:00" --sparam "wait-time=600" --no-check-directory -x PJM_LLIO_GFSCACHE=/vol0004 --llio localtmp-size=87Gi
