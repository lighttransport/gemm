
# compiler for a64fx fugaku

- fccpx: C cross(Intel host) compiler
- FCCpx: C++ cross(Intel host) compiler
- fcc: C native(a64fx) compiler
- FCC: C++ native(a64fx) compiler

# running on a64fx

This host is a native A64FX (aarch64) node. Use native compilers (fcc/FCC, not fccpx/FCCpx) and run binaries directly (no pjsub needed).

Example:
```
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -o bench bench.c kernel.S -lm
OMP_NUM_THREADS=1 ./bench
```

# submit job (only when using login/cross-compile node)

// freq=2000 : normal mode
// freq=2200 : boost mode(eco_state=2 only: boost-eco mode)
// eco_state = 0 -> eco mode off(use 2 FPUs. default 2. it use only 1 FPU = up to 50% of peak FLOPS)
pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00" --no-check-directory <script.sh>

change elapse time according to the expected job execution time.

## a64fx architecture

See a64fx/doc/A64FX_ARCHITECTURE_SUMMARY.md

## a64fx profiling using fapp

See a64fx/doc/profiler.md

## a64fx SVE instructions

See a64fx/doc/SVE_INSTRUCTION_REFERENCE.md a64fx/doc/SVE_INSTRUCTION_LATENCY.md
