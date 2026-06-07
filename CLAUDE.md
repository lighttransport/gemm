
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

## ctx ceiling & memory: measure MemFree, NOT process RSS

OOM-kill on these nodes is driven by **true physical memory** (fires at MemFree≈2.0 GB;
MemTotal=31.81 GB, no cgroup limit, no swap). Some large allocations — notably the Tier-B2
caches `cmp_kv`/`idx_kv` — are **physically resident (cost real MemFree) but DO NOT appear
in `/proc/self/statm` RSS** (THP-backed; the one big alloc without `MADV_NOHUGEPAGE`). At
ctx=131072 they cost **2.72 GB physical with zero RSS change**. So:

- **For any ctx-ceiling / OOM / memory-reduction work, validate against MemFree (or
  MemAvailable), never process RSS.** RSS-based footprint numbers undercount by the
  Tier-B2 amount, and an RSS A/B test will falsely report a real memory lever as "no effect".
- **Measured ceiling (FP8-on-demand dense `DS4F_FP8_BF16=0` + int8 KV `DS4F_INT8_KV=1`):
  ~200k tokens** (131072 fits w/ ~4.65 GB MemFree spare; 262144 OOMs at warm_kv layer
  ~27/43). This is ~6× the old "ctx=32768 OOMs" (that was the +6 GB bf16-predequant config).
- **`DS4F_INT8_CMP=1` (default off) lifts the ceiling to ~255k hard / ~242k safe**: int8s the
  Tier-B2 `cmp_kv` cache (the THP physical dominator above) via the same S5 static-per-channel
  scheme as int8 KV. MemFree @131072 4.52→5.57 GB (+1.05 GB; combined slope 36.7→28.7 KB/pos).
  Real-gen A/B token-identical to f32 cmp_kv. `DS4F_INT8CMP_CAL` = calib window in SLOTS
  (default 64; cmp freezes at slot≥CAL ⇒ pos≥CAL×ratio). For a robust 262144 margin the next
  lever is int8-ing `idx_kv` too (~+2 KB/pos). NOTE: a short eos-terminated gen (~160 pos) is
  BELOW the CAL=64 freeze point (256 pos) — to validate the int8 frozen path in a short gen,
  lower CAL (e.g. CAL=8) or it only exercises the pre-freeze bf16 calbuf.
- **Safe ctx-probing on a shared alloc:** build with the warm-phase guard and set
  `DS4F_WARM_RSS_TRACE=1 DS4F_WARM_MEMAVAIL_STOP_GB=2.4` — every rank `_exit(42)` cleanly
  before the OOM-killer fires (verified rc=42, not 137 → **no PMIx/plexec degradation**).
  Per-layer RSS/MemAvail/MemFree trace lands in `ds4f_ep_stderr_rank00.txt` (the runner
  freopen's per-rank stderr there, not the .log).

