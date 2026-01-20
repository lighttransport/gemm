
# compiler for a64fx fugaku

- fccpx: C cross(Intel host) compiler
- FCCpx: C++ cross(Intel host) compiler
- fcc: C native(a64fx) compiler
- FCC: C++ native(a64fx) compiler

# submit job for a64fx fugaku

// freq=2000 : normal mode
// freq=2200 : boost mode(eco_state=2 only: boost-eco mode)
// eco_state = 0 -> eco mode off(use 2 FPUs. default 2. it use only 1 FPU = up to 50% of peak FLOPS)
pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00" --no-check-directory <script.sh>

change elapse time according to the expected job execution time.
