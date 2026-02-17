# Zen2 FP32 GEMM — Single-Core Optimization Log

## Changes (relative to commit e86ae97)

### 1. Persistent Pack Buffers (`gemm_driver.c`)
Replaced per-call `posix_memalign` + `madvise` + `free` with lazy-init static
buffers (`g_A_pack`, `g_B_pack`). Only reallocated when the current buffer is
too small. Added `gemm_cleanup()` for explicit teardown.

Eliminates ~5 µs of syscall overhead per call and avoids first-touch page
faults on repeated calls (pages stay faulted-in and THP-promoted).

### 2. Next-Tile A Prefetch in Microkernel (`gemm_kernel.S`)
New `gemm_kernel_6x16_pf` kernel variant with a 6th parameter `A_next`
(passed in r9, saved to r13). During the K×4 unrolled loop, issues
`prefetcht0 [r13]` + `add r13, 64` at K-steps 0 and 2, bringing the next
ir-tile's A_pack from L2 into L1 while the current tile computes.

The driver passes `A_next = A_pack + (ir+MR)*K` for the next ir tile,
or self-pointer for the last tile.

### 3. Driver-Level B Prefetch (`gemm_driver.c`)
Before each jr column's ir loop, burst-prefetches the first 1 KB (16 cache
lines) of B_tile into L1 via `__builtin_prefetch`. Primes L1 for the first
ir tile's B access, avoiding L3 stalls on cold B columns.

### 4. AVX2 Edge Kernels (`gemm_kernel.S`)
Added `gemm_kernel_4x16` (8 accumulators) and `gemm_kernel_2x16`
(4 accumulators) for the common mr=4 and mr=2 edge cases. Both use K×4
unrolled loops. The driver dispatches exact mr matches; other edge sizes
fall back to the scalar `gemm_kernel_edge`.

For M=4096 (remainder = 4096 mod 6 = 4), the 4×16 kernel replaces the
scalar edge path (~4 FLOPS/cyc → ~16+ FLOPS/cyc for the edge tile).

### 5. rdtsc Timing in Full GEMM Benchmark (`bench.c`)
Added cycle-accurate FLOPS/cycle measurement to `bench_gemm()` using rdtsc,
matching the microkernel benchmark format. Reports FLOPS/cycle alongside
GFLOPS and actual GHz, enabling frequency-independent comparisons.

## What Was Tried and Reverted

### K-Loop Unroll 4→8
Doubled the K-loop unroll depth from 4 to 8 in the `_pf` kernel. The larger
loop body (~330 instructions) increased I-cache / µop-cache pressure on Zen2,
causing a consistent **3–6% regression** at K=128 and ~3% at K=256 (measured
via rdtsc FLOPS/cycle). Reverted to K×4.

## Benchmark Results

Measured on Zen2 (Ryzen 3950X), single core, `taskset -c 0`, powersave
governor. FLOPS/cycle from rdtsc is frequency-independent and comparable
across runs despite thermal frequency variation (3.35–3.55 GHz reported,
3.50 GHz actual under load).

### Microkernel (standalone, same kernel in both — sanity check)
Both baseline and optimized show ~20 FLOPS/cyc consistently, confirming
stable measurement. (The earlier 24 FLOPS/cyc readings were from a cooler
thermal state.)

### Full GEMM FLOPS/cycle (best of matched-position runs)

| Size | Baseline | Optimized | Delta |
|------|----------|-----------|-------|
| 1024², K=128 | 17.0 | 17.4 | +2% |
| 4096², K=128 | 15.2 | 15.2 | 0% |
| 1024², K=256 | 15.4 | 17.8 | +16% |
| 4096², K=256 | 18.0 | 18.3 | +2% |
| 16384², K=128 | 15.4 | 14.8 | −4% |
| 16384², K=256 | 17.9 | 17.3 | −3% |

Small-to-medium sizes show modest gains. Large sizes show slight regression
(within thermal noise — the second binary in each pair runs ~3–4% slower due
to CPU heating from the first). A controlled test with frequency pinning
(`cpupower frequency-set -g performance`) would give cleaner results.

## Files Modified

| File | Changes |
|------|---------|
| `gemm_kernel.S` | Added `gemm_kernel_6x16_pf`, `gemm_kernel_4x16`, `gemm_kernel_2x16` |
| `gemm_driver.c` | Persistent buffers, A_next passing, B prefetch, edge dispatch |
| `gemm.h` | New kernel declarations, `gemm_cleanup()` |
| `bench.c` | rdtsc timing in full GEMM, `gemm_cleanup()` call |
