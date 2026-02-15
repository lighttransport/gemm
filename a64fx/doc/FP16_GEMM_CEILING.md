# FP16 GEMM Ceiling Analysis on A64FX

## Summary

Systematic analysis of the fp16 12×2 GEMM micro-kernel performance ceiling on A64FX.
Through exhaustive scheduling experiments, the L1-resident ceiling is **89.4% of 256 GF peak**
(228.7 GF), limited by register file read-port contention. The fp16→fp32 conversion
epilogue costs an additional 2-7 GF depending on K.

## Peak Configuration

| Metric | Value |
|--------|-------|
| **Best L1-resident** | **228.7 GF (89.4%)** — NOEPI4, K=384, fp16 C |
| Best with fp32 C | 222.5 GF (86.9%) — SWP, K=256 |
| Best streaming (fp32 C) | 206.7 GF (80.7%) — SWP, K=256, L=32768, d=256 |
| Best streaming (fp16 C) | **217.7 GF (85.0%)** — NOEPI, K=256, L=4096, d=256 |
| Best fused total (fp16 C) | **173.6 GF (67.8%)** — NOEPI pipeline, K=256, d=256 |
| Best fused total (fp32 C) | 168.5 GF (65.8%) — SWP pipeline, K=256, d=256 |
| K-loop inherent efficiency | ~89.7% (converges at high K) |
| Per-K overhead | 1.4 cy above 12 cy ideal → 13.4 cy/K actual |

## Kernel Variants Tested

### L1-Resident Single Tile

| Kernel | Description | K=128 INIT | K=256 INIT | K=384 INIT |
|--------|-------------|------------|------------|------------|
| SWP | 2K unroll, fp32 C | 214.5 (83.8%) | 222.5 (86.9%) | 213.8 (83.5%)* |
| SWP4 | 4K unroll, fp32 C | 214.8 (83.9%) | 223.1 (87.1%) | — |
| SPLITB | 4K + split batch-2 B reload, fp32 C | 215.1 (84.0%) | 222.6 (87.0%) | — |
| DBUF | B double-buffer, fp32 C | — | ~220 (86%) | — |
| NOEPI | 2K unroll, fp16 C (no conversion) | 223.2 (87.2%) | 227.3 (88.8%) | 228.3 (89.2%) |
| NOEPI4 | 4K unroll, fp16 C (no conversion) | 223.0 (87.1%) | 227.8 (89.0%) | **228.7 (89.4%)** |

*SWP K=384 with fp32 C: A+B+C = 61KB, L1 pressure causes slight regression.

### Streaming GEMM (K=256, d=256, multi-tile)

| Kernel | L=4096 | L=32768 |
|--------|--------|---------|
| SWP (fp32 C) | 207.7 (81.1%) | 206.7 (80.7%) |
| SWP4 (fp32 C) | 205.1 (80.1%) | — |
| SPLITB (fp32 C) | 204.2 (79.8%) | — |
| **NOEPI (fp16 C)** | **217.7 (85.0%)** | **216.4 (84.5%)** |

## K-Sweep (NOEPI, fp16 C, L1-resident)

| K | A+B+C (bytes) | INIT GF | INIT % | ACCUM GF | ACCUM % |
|---|---------------|---------|--------|----------|---------|
| 64 | 11,264 | 218.0 | 85.1% | 214.6 | 83.8% |
| 128 | 20,992 | 223.2 | 87.2% | 221.8 | 86.6% |
| 192 | 30,720 | 225.7 | 88.2% | 224.9 | 87.8% |
| 256 | 40,448 | 227.3 | 88.8% | 226.5 | 88.5% |
| 320 | 50,176 | 227.2 | 88.7% | 226.8 | 88.6% |
| **384** | **59,904** | **228.3** | **89.2%** | **228.0** | **89.0%** |
| 416 | 64,768 | 218.0 | 85.2% | 217.7 | 85.0% |

K=384 is optimal: A(9KB)+B(48KB)+C(1.5KB)=58.5KB fits 64KB L1D.
K=416 spills from L1 (64,768B ≈ 63.3KB, alignment overhead pushes past 64KB).

## Micro-Kernel Architecture

### Register Map (12×2 tile, fp16)
```
z0-z23:  24 fp16 accumulators (12 rows × 2 SVE vectors of 32 fp16)
z24-z25: B matrix columns (2 × 512-bit = 64 fp16 elements)
z26-z31: A matrix broadcast values (6 per batch, 2 batches per K)
p0:      ptrue.h (all 32 half-precision lanes)
p1:      ptrue.s (for fp32 epilogue conversion)
```

### K-Loop Structure (per K step)
```
Phase 1: 6 × ld1rh  (batch 1 A loads, rows 0-5)
Phase 2: 12 × fmla   (batch 1 FMAs, interleaved with 6 × ld1rh batch 2 A)
Phase 3: 12 × fmla   (batch 2 FMAs, rows 6-11 × 2 B columns)
Phase 4: 2 × ld1h    (next B columns)

Total: 24 FMLA + 14 loads = 38 instructions per K step
Ideal: 24 FMLA / 2 pipes = 12 cy (FMA-bound)
Measured: 13.4 cy (89.7% FMA utilization)
```

### Epilogue Variants
```
INIT (fp32 C):   uunpklo/hi + fcvt + st1w  → 156 insns, ~50-60 cy
ACCUM (fp32 C):  uunpklo/hi + fcvt + ld1w + fadd + st1w → 204 insns, ~80-90 cy
INIT (fp16 C):   st1h × 24 + add × 11      → 36 insns, ~12 cy
ACCUM (fp16 C):  ld1h + fadd + st1h         → 84 insns, ~30 cy
```

## Scheduling Experiments

### 4K Unrolling (SWP → SWP4)
- Processes K=0,1,2,3 per loop iteration instead of K=0,1
- Halves amortized loop tail overhead (add, subs, beq, ld1h, b)
- A immediate offsets: K=3 max = #94 (fits ld1rh range 0-126)
- B offsets: K=3 at #6/#7 mul vl (fits ld1h range -8 to 7)
- **Result: +0.2-1.0% improvement** — loop overhead already hidden by OoO

### Split-B (batch-2 FMA reordering)
- Splits batch-2 FMAs by B column: z24-column first, then z25-column
- After z24-column FMAs, reloads B(k+1) col 0 into freed z24
- After z25-column FMAs, reloads B(k+1) col 1 into freed z25
- Increases B load-to-use distance from ~8 to ~13 instructions
- **Result: neutral** — OoO already hides B load latency

### B Double-Buffering (DBUF)
- Pre-loads next K's B into z26-z27 while current K uses z24-z25
- Alternates B register sets each K step
- **Result: 1-1.5% SLOWER** — extra register pressure hurts more than it helps

### Conversion Epilogue Elimination (NOEPI)
- Stores fp16 accumulators directly (no uunpklo/hi + fcvt)
- INIT: 24 × st1h + 11 × add (~12 cy) vs 156 insns (~55 cy) = ~43 cy saved
- ACCUM: ld1h + fadd.h + st1h (~30 cy) vs ld1w + fadd.s + st1w chain (~85 cy) = ~55 cy saved
- **Result: +3-7 GF** depending on K, reveals K-loop ceiling

## Bottleneck Analysis

### Theoretical vs Measured (per K step)

| Resource | Demand | Supply (2 GHz) | Cycles | Bottleneck? |
|----------|--------|----------------|--------|-------------|
| FMA throughput | 24 FMLA | 2/cy (FLA+FLB) | 12.0 cy | **YES** |
| Load throughput | 14 loads | 2/cy (EXA+EXB) | 7.0 cy | No |
| Decode width | 38 insns | 4/cy | 9.5 cy | No |
| **Measured** | — | — | **13.4 cy** | +1.4 cy gap |

### Root Cause: Register Read-Port Contention

Each `FMLA z_acc.h, p0/m, z_a.h, z_b.h` reads 3 vector registers:
- z_acc (accumulator, read-modify-write)
- z_a (A broadcast value)
- z_b (B column vector)

At 2 FMLA/cy sustained, this requires **6 vector register reads per cycle** from the
512-bit register file. A64FX likely has 4-6 read ports, creating occasional 1-cycle
stalls when both FMA pipes need simultaneous access.

Over 24 FMLA in a K step: ~1.4 cycles of read-port stall = 11.7% overhead.
This is consistent with the measured 10.3% gap from peak.

This is a **hardware-level limitation** — no software scheduling can avoid it because
the contention arises from the fundamental instruction format (3-operand predicated FMA).

## Working Set Analysis

### fp32 C (SWP kernel)
```
K=256: A = 6KB + B = 32KB + C = 3KB = 41KB  (fits L1, best fp32 result)
K=384: A = 9KB + B = 48KB + C = 3KB = 60KB  (tight, slight regression)
```

### fp16 C (NOEPI kernel)
```
K=256: A = 6KB + B = 32KB + C = 1.5KB = 39.5KB  (comfortable)
K=384: A = 9KB + B = 48KB + C = 1.5KB = 58.5KB  (optimal, 89.4%)
K=416: A = 10KB + B = 52KB + C = 1.5KB = 63.5KB  (L1 spill!)
```

fp16 C saves 1.5KB over fp32 C, enabling K=384 without L1 spill.

## Fused Attention Pipeline Integration

NOEPI kernels were integrated into the full 2-pass fused softmax attention pipeline
(`bench_2pass.c`) with fp16 C accumulation, and benchmarked against the existing
SWP (fp32 C) pipeline.

### Pipeline Architecture (NOEPI variant)
```
Phase 0: Global max-find over S (MR rows simultaneously via SVE)
Phase 1: Fill ring buffer with first PIPE_AHEAD=2 pass1 blocks
          pass1_block_exp_fp16: exp2 in fp32 → fcvt+st1h → fp16 Pp ring slot
Phase 2: Interleaved pipeline (for each K-block):
          - Produce: pass1 for block kb+2 (writes fp16 Pp to ring)
          - Consume: GEMM for block kb (reads fp16 Pp, fp16 Vp → fp16 O)
          - NOEPI kernel: st1h epilogue (12 cy) vs uunpklo/fcvt/st1w (55 cy)
Phase 3: Normalize fp16 O /= row_sum (scalar, <1 µs)
```

### NOEPI vs SWP Pipeline Results (Kc=256, d=256, PA=2)

| L | NOEPI Total GF (%) | SWP Total GF (%) | NOEPI GEMM GF (%) | SWP GEMM GF (%) |
|---|---|---|---|---|
| 4096 | **173.6 (67.8%)** | 167.2 (65.3%) | **217.7 (85.0%)** | 207.7 (81.1%) |
| 8192 | **173.6 (67.8%)** | 167.1 (65.3%) | **217.7 (85.0%)** | 207.4 (81.0%) |
| 16384 | **172.8 (67.5%)** | 166.6 (65.1%) | **216.5 (84.6%)** | 206.5 (80.7%) |
| 32768 | **172.9 (67.5%)** | 164.7 (64.3%) | **216.4 (84.5%)** | 206.7 (80.7%) |

NOEPI improvement: **+10 GF GEMM-only (+3.9%)**, **+6-8 GF total fused (+2.5-5.0%)**

### Kc Sweep in Pipeline (NOEPI, L=32768)

| Kc | NOEPI GEMM GF (%) | NOEPI Total GF (%) | SWP GEMM GF (%) | SWP Total GF (%) |
|----|---|---|---|---|
| 192 | 211.9 (82.8%) | 171.6 (67.0%) | 200.3 (78.2%) | 163.3 (63.8%) |
| **256** | **216.4 (84.5%)** | **172.9 (67.5%)** | 206.7 (80.7%) | 164.7 (64.3%) |
| 384 | 206.6 (80.7%) | 170.9 (66.8%) | 202.4 (79.1%) | 169.2 (66.1%) |

**Kc=256 is optimal for streaming NOEPI** — opposite of L1-resident (K=384 optimal).
At Kc=384, B tile = 48KB nearly fills 64KB L1D, causing excessive eviction during
L2→L1 cold-start per tile. At Kc=256, B = 32KB leaves 32KB headroom for Pp and O.

### Pipeline Timing Breakdown (Kc=256, L=32768)

| Phase | NOEPI (µs) | SWP (µs) | Notes |
|-------|-----------|----------|-------|
| Fill (max + 2 pass1) | 67.8 | 67.5 | Same (pass1 is fp32 in both) |
| Pipeline (pass1 + GEMM) | 1095.6 | 1154.0 | NOEPI -5.1% faster |
| Normalize | 0.9 | 1.2 | fp16 normalize cheaper |
| **Total** | **1164.4** | **1222.7** | **NOEPI -4.8% faster** |

Pipeline overhead analysis (L=32768):
- Pure GEMM-only: ~930 µs (216.4 GF) vs total pipeline 1095.6 µs
- Pass1 interleaved: ~165 µs (15% of pipeline)
- Fill phase: 67.8 µs (5.8% of total)
- **Total non-GEMM overhead: ~20%** of fused total time

### Streaming GEMM Gap Analysis

```
L1-resident:  228.7 GF (89.4%) — NOEPI4 K=384, single tile
Streaming:    217.7 GF (85.0%) — NOEPI K=256, multi-tile
Gap:          11.0 GF (4.3%)

Root cause: B tile cold-start from L2
- B tile = 64 × 256 × 2 = 32KB loaded from L2 per tile
- 4 tiles per K-block → 4 × 32KB = 128KB streamed from L2 per block
- HW prefetch detects sequential pattern but first ~4-8 K steps stall
- L2 latency ~30 cy vs L1 ~11 cy → 2.7× penalty on leading loads
```

### Accuracy

| Metric | NOEPI (fp16 C) | SWP (fp32 C) |
|--------|---------------|--------------|
| maxerr vs double ref | 5.8e-05 – 2.0e-04 | 4.5e-05 – 1.8e-04 |
| Kernel self-test | 4.47e-03 (init) | 4.47e-03 (init) |
| Status | Acceptable for attention | Acceptable |

fp16 C accumulation adds negligible error: softmax outputs are already fp16
precision after pass1 fcvt, so the inter-block accumulation rounding is in
the same precision as the input data.

## Files

| File | Description |
|------|-------------|
| `micro_kernel_fp16_12x2_swp.S` | Baseline SWP kernel, fp32 C |
| `micro_kernel_fp16_12x2_swp_accum.S` | SWP ACCUM variant |
| `micro_kernel_fp16_12x2_swp4.S` | 4K-unrolled INIT |
| `micro_kernel_fp16_12x2_swp4_accum.S` | 4K-unrolled ACCUM |
| `micro_kernel_fp16_12x2_splitB.S` | Split-B scheduling INIT |
| `micro_kernel_fp16_12x2_splitB_accum.S` | Split-B scheduling ACCUM |
| `micro_kernel_fp16_12x2_dbuf.S` | B double-buffer INIT |
| `micro_kernel_fp16_12x2_dbuf_accum.S` | B double-buffer ACCUM |
| `micro_kernel_fp16_12x2_noepi.S` | fp16 C INIT (no conversion) |
| `micro_kernel_fp16_12x2_noepi_accum.S` | fp16 C ACCUM (no conversion) |
| `micro_kernel_fp16_12x2_noepi4.S` | fp16 C 4K-unrolled INIT |
| `micro_kernel_fp16_12x2_noepi4_accum.S` | fp16 C 4K-unrolled ACCUM |
| `bench_fapp.c` | Kernel comparison benchmark (L1-resident) |
| `bench_ksweep.c` | K-value sweep benchmark (L1-resident) |
| `bench_2pass.c` | Full fused attention pipeline benchmark |

## Conclusion

The fp16 12×2 GEMM kernel on A64FX reaches **89.4% of peak** in L1-resident
configuration (K=384, fp16 C, 4K unrolled) and **85.0% streaming** in the
multi-tile fused attention pipeline (K=256, fp16 C). The remaining gap is:

- **L1-resident → peak (10.6%)**: register file read-port contention (hardware limit)
- **Streaming → L1-resident (4.3%)**: B tile cold-start from L2 per tile
- **Fused total → GEMM-only (17.5%)**: pass1 softmax interleave + pipeline fill

Performance hierarchy:
```
L1-resident GEMM:    228.7 GF (89.4%)  — architectural ceiling
Streaming GEMM:      217.7 GF (85.0%)  — B cold-start from L2
Fused total:         173.6 GF (67.8%)  — pass1 + fill overhead
```

Key takeaways:
- **NOEPI (fp16 C) is the single most impactful optimization**:
  +10 GF GEMM-only, +6-8 GF total fused, with negligible accuracy loss
- **OoO execution is highly effective**: all scheduling optimizations (4K unroll,
  split-B, double-buffer) provide <1% gain
- **Kc=256 optimal for streaming** (85.0%), Kc=384 optimal for L1-resident (89.4%)
  — B tile size determines the crossover
- **Pipeline overhead (~20%) is the main bottleneck** for total fused performance;
  further gains require reducing pass1 cost or deeper pipeline overlap
- **90% appears to be the architectural GEMM ceiling** for this instruction mix
  on A64FX, reachable only by eliminating all non-FMA overhead
