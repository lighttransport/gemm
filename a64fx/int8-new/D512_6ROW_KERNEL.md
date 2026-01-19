# D=512 Fused Attention Kernels for A64FX

## Overview

This document describes the optimized fused attention kernels for D=512 (head dimension 512) on A64FX SVE.

## Key Parameters

- **Head dimension (D)**: 512
- **Sequence lengths tested**: L=4096, L=8192
- **Load latency**: 11 cycles (A64FX L1 cache)
- **SDOT throughput**: 2 per cycle (with 2 FPUs)

## Kernel Variants

### 5-row Kernel (`kernel_fused_d512_5row.S`)

Processes 5 query rows per kernel call.

**Register allocation:**
- z0-z19: 20 accumulators (5 rows x 4 K vectors)
- z20-z23: 4 K vectors
- z24-z27: 5 Q vectors (broadcast loaded)
- z28-z31: temporaries

**Compute density:**
- 20 SDOTs per 9 loads = 2.22 SDOT/load
- Insufficient to fully hide 11-cycle load latency

### 6-row Kernel (`kernel_fused_d512_6row.S`)

Processes 6 query rows per kernel call using split Q loading technique.

**Key optimization: Split Q Loading**

To fit 6 rows with limited registers, Q vectors are loaded in two phases:
1. Load Q[0-3] into z28-z31, compute rows 0-3
2. Reuse z28-z29 for Q[4-5], compute rows 4-5

This exploits the fact that once a load is issued, the register can be reused
for another load before the first load completes.

**Register allocation:**
- z0-z23: 24 accumulators (6 rows x 4 K vectors)
- z24-z27: 4 K vectors
- z28-z31: Q vectors (phase 1: rows 0-3)
- z28-z29: Q vectors (phase 2: rows 4-5, reusing after rows 0-1 done)

**Compute density:**
- 24 SDOTs per 10 loads = 2.40 SDOT/load
- 8% better than 5-row kernel

## Memory Layout

### K matrix (interleaved for Q@K^T)
```
K_int[D_group=128, N=64, 4]
- D_group stride: 256 bytes
- Total per N-chunk: 128 * 64 * 4 = 32KB
```

### V matrix (transposed for P@V)
```
V_t[D_tile=8, N_group=16, 64, 4]
- D_tile stride: 4KB
- Total per N-chunk: 8 * 16 * 64 * 4 = 32KB
```

## Performance Results

Benchmark: M=1800, D=512

| Sequence Length | 5-row GFLOPS | 6-row GFLOPS | Speedup |
|-----------------|--------------|--------------|---------|
| L=4096          | 249.1        | 265.5        | 1.07x   |
| L=8192          | 250.5        | 267.1        | 1.07x   |

**Per-call cycles:**
- 5-row: ~261-263 cycles for 81920 SDOTs (fused Q@K^T + P@V)
- 6-row: ~294-296 cycles for 98304 SDOTs

**Cycles per SDOT:**
- 5-row: 0.0032
- 6-row: 0.0030 (6% better)

## Analysis

The 6-row kernel achieves a consistent 7% speedup over the 5-row kernel,
closely matching the theoretical 8% improvement from the better SDOT/load ratio.

To fully hide the 11-cycle load latency at 2 SDOT/cycle throughput, we need
~22 SDOTs per load group. The 6-row kernel with 24 SDOTs per 10 loads (2.4 ratio)
comes closer to this requirement than the 5-row kernel with 20 SDOTs per 9 loads (2.22 ratio).

## Files

- `kernel_fused_d512_5row.S` - 5-row D=512 fused kernel
- `kernel_fused_d512_6row.S` - 6-row D=512 fused kernel with split Q loading
- `bench_6row_large.c` - Benchmark comparing 5-row vs 6-row
- `test_6row.c` - Correctness test
- `run_6row_large.sh` - Run script for benchmark
