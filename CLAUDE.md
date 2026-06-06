
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


## tempdir

use /local as tempdir
if /local not available, use ~/tmp

## interactive sessions: keep multi-node output compact (context hygiene)

Multi-node DS4F runs emit per-rank status files (×11) that are **lockstep-identical**
(perf/argmax/RSS match across all ranks by design), so reading all of them into an
interactive session burns context for no information. Keep runs lean:

- `DS4F_QUIET=1 ./run_ds4f_11n.sh` — prints only counts (`load=11/11 perf=11/11`) + the
  rank0 head instead of `cat`-ing every `ds4f_ep_{load,perf}_rank*.txt`. Full files stay on
  disk for inspection.
- `DS4F_STAGE_COMPACT=1 ./run_ds4f_stage_11n.sh` — prints only `done=N/NP`, not all 11
  byte-identical stage lines.
- `run_ds4f_longctx_11n.sh` / `run_ds4f_gen_11n.sh` already redirect the full run to a log
  and print only a compact **sentinel** block — use them as the model for new runners.
- General practice: **redirect a run's full output to a log file, then read back only the
  sentinel / a few `grep`'d lines** — never `cat` all per-rank files into the conversation.
  Rank0 + a cross-rank lockstep count check is sufficient (the scripts already verify
  `argmax_distinct_across_ranks==1`).

