# Memory Access Pattern Analysis for Fused Attention on A64FX

## Summary

Analysis of memory access patterns for INT8 fused attention kernel `(Q@K^T)@V` on A64FX.
Goal: Isolate memory subsystem behavior by replacing compute (SDOT) with NOPs.

## A64FX Memory Hierarchy

| Level | Size | Latency | Bandwidth |
|-------|------|---------|-----------|
| L1D | 64 KB | 11 cycles | 128 B/cy load, 64 B/cy store (2 pipes) |
| L2 | 8 MB/CMG | 27-37 cycles | ~42.6 B/cy/core |
| HBM2 | 32 GB | ~200 cycles | ~1 TB/s aggregate |

## Key Finding: `ld1rw` Broadcast Load is the Bottleneck

### Problem
The Q matrix loads use `ld1rw` (load-replicate word) to broadcast 4-byte values across SVE vectors.
Original implementation achieved only **21% efficiency** (~27 B/cy vs 128 B/cy peak).

### Root Cause Analysis

| Load Type | Measured Throughput | Notes |
|-----------|---------------------|-------|
| `ld1b` sequential | 30 B/cy | Poor - address dependency |
| `ld1b` stride256 (4x) | **93.6 B/cy** | Good - 73% of peak |
| `ld1rw` sequential | 1.9 B/cy | **Very slow** - ~2 cy/load |
| `ld1rw` stride256 (4x) | 4.1 B/cy | Still slow |
| Mixed ld1b + ld1rw | 55.9 B/cy | Port contention |

**`ld1rw` takes ~2 cycles per load** with naive addressing, causing the bottleneck.

### Solution: Optimized `ld1rw` Addressing

Pre-compute row base pointers and use immediate offsets:

```asm
// Setup: pre-compute Q row pointers
mov x5, Q_base          // Row 0
add x6, x5, #256        // Row 1 (Q_stride = 256)
add x7, x5, #512        // Row 2
add x8, x5, #768        // Row 3

// Loop: use immediate offsets for multiple columns
ld1rw {z0.s}, p1/z, [x5]       // Col 0
ld1rw {z1.s}, p1/z, [x6]
ld1rw {z2.s}, p1/z, [x7]
ld1rw {z3.s}, p1/z, [x8]

ld1rw {z4.s}, p1/z, [x5, #4]   // Col 1
ld1rw {z5.s}, p1/z, [x6, #4]
ld1rw {z6.s}, p1/z, [x7, #4]
ld1rw {z7.s}, p1/z, [x8, #4]

ld1rw {z8.s}, p1/z, [x5, #8]   // Col 2
...
ld1rw {z12.s}, p1/z, [x5, #12] // Col 3
...

// Single increment per 4 columns
add x5, x5, #16
add x6, x6, #16
add x7, x7, #16
add x8, x8, #16
```

### Optimization Results

| Approach | Cycles (256 loads) | cy/load |
|----------|-------------------|---------|
| Original (C ptr arithmetic) | 270 | 1.05 |
| Pre-computed base + post-inc | 218 | 0.85 |
| **Unroll4 + immediate offset** | **131** | **0.51** |

**2x throughput improvement** with optimized addressing.

### Alternative: `ldr` + `dup` - NOT Recommended

```asm
ldr w14, [x9], #4    // Scalar load
dup z0.s, w14        // Broadcast to vector
```

**Problem**: `dup` consumes FPU port, competing with SDOT compute.
When compute is added back, this approach will hurt overall throughput.

**Conclusion**: Stick with optimized `ld1rw`, avoid `ldr` + `dup`.

## Q@K^T Kernel Performance

| Implementation | Cycles | Bandwidth | Efficiency |
|----------------|--------|-----------|------------|
| Original (with 24 NOPs) | ~640 | 27 B/cy | 21% |
| Original (no NOPs) | 298 | 58 B/cy | 46% |
| Optimized unroll4 | 277 | 63 B/cy | 49% |
| **Optimized unroll8** | **267** | **65 B/cy** | **51%** |
| Theoretical minimum | 136 | 128 B/cy | 100% |

### Analysis

Memory traffic for Q@K^T [4,256] @ [256,64]:
- K loads: 64 iterations × 4 × 64B = 16,384 bytes
- Q loads: 64 iterations × 4 × 4B = 1,024 bytes
- Total: 17,408 bytes

Theoretical minimum: 17,408 / 128 = **136 cycles**

Achieved: **267 cycles (51% efficiency)**

Remaining gap due to:
1. Load port contention between `ld1b` (64B) and `ld1rw` (4B)
2. Loop overhead (8 iterations remain)
3. Pointer increment instructions

## L1 vs L2 Cache Performance

| Config | Working Set | Cycles | Load B/cy | Efficiency |
|--------|-------------|--------|-----------|------------|
| L1_small (N=64) | 37 KB | 2329 | 15.0 | 12% (vs 128 peak) |
| L1_edge (N=128) | 70 KB | 4266 | 12.2 | 29% (vs 42.6 peak) |
| L2_small (N=256) | 134 KB | 7942 | 11.0 | 26% |
| L2_medium (N=512) | 263 KB | 13057 | 12.0 | 28% |
| L2_large (N=1024) | 521 KB | 22763 | 13.0 | 31% |

Note: L1 test shows lower efficiency because it uses non-optimized `ld1rw`.
After optimization, L1-resident kernels achieve ~50% efficiency.

## Recommendations for Fused Attention Kernel

1. **Use optimized `ld1rw` addressing**:
   - Pre-compute row base pointers before inner loop
   - Use immediate offsets: `[base], [base, #4], [base, #8], [base, #12]`
   - Single pointer increment per 4 columns

2. **Avoid `ldr` + `dup`**: Consumes FPU port, conflicts with SDOT

3. **Unroll K dimension by 4-8**: Amortize loop overhead

4. **Use `#N, mul vl` for K loads**:
   ```asm
   ld1b {z0.b}, p0/z, [x9]
   ld1b {z1.b}, p0/z, [x9, #1, mul vl]
   ld1b {z2.b}, p0/z, [x9, #2, mul vl]
   ld1b {z3.b}, p0/z, [x9, #3, mul vl]
   ```

5. **Expected efficiency**: ~50% of L1 peak bandwidth for memory-access-only kernel.
   With SDOT compute overlapped, should approach higher utilization.

## Fused INT32 Attention (No Quantization)

Simplified fused kernel: Q@K^T -> S(INT32) -> P@V -> O(INT32)
No INT8 quantization of S, directly use INT32 for P@V.

### Memory Traffic

| Phase | Load | Store |
|-------|------|-------|
| Q@K^T | K(16KB) + Q(1KB) | S(256B) |
| P@V | V(16KB) + S(256B) | O(1KB) |
| **Total** | **33.5KB** | **1.25KB** |

Theoretical minimum: 266 cycles (load-bound at 128 B/cy)

### Current Results

| Kernel | Cycles | Load BW | Efficiency |
|--------|--------|---------|------------|
| Baseline (sequential) | 547 | 62.2 B/cy | **48.6%** |
| Theoretical | 266 | 128 B/cy | 100% |

### Efficiency Analysis

The ~50% efficiency is consistent with our single-phase results:
- Q@K^T alone: 267 cycles, 65 B/cy, 51%
- P@V has similar structure

**The bottleneck is NOT ST→LD**, it's the **mixed ld1b + ld1rw pattern**:
- `ld1b` (64B vector load): ~0.5 cy/load when grouped
- `ld1rw` (4B broadcast load): ~0.5 cy/load with optimized addressing
- Mixed access: ~50% of peak due to port contention

**Practical peak for this access pattern: ~50-55% of theoretical**

### ST→LD Hazard Analysis

The baseline kernel stores S[4,64] then immediately loads it for P@V.
This creates a **store-to-load forwarding** dependency.

**Potential solutions:**
1. **Tiling**: Process multiple M-tiles, overlap Phase 1 of tile N with Phase 2 of tile N-1
2. **Software pipelining**: Interleave S stores with V loads
3. **Double buffering**: Use two S buffers, alternate between them

### Tiling Strategy to Avoid ST→LD

```
For tile i = 0 to N:
    // Phase 1: Q@K^T for tile i -> store to S[i % 2]
    // Phase 2: P@V for tile i-1 using S[(i-1) % 2] (if i > 0)
```

This creates sufficient distance (hundreds of cycles) between S store and S load.

## Advanced Optimization Attempts

Tested additional optimizations for the fused kernel:

### 1. Interleaved S Storage (ZIP Transpose)

**Concept**: Store S in interleaved format so P loads can use consecutive ld1rw addresses.

```asm
// Transpose S from row-major to interleaved using ZIP
zip1 z0.s, z24.s, z25.s   // [r0c0,r1c0,r0c1,r1c1,...]
zip2 z1.s, z24.s, z25.s
zip1 z2.s, z26.s, z27.s
zip2 z3.s, z26.s, z27.s
zip1 z4.d, z0.d, z2.d     // Final interleave
zip2 z5.d, z0.d, z2.d

// Then P loads use consecutive offsets from single base:
ld1rw {z4.s}, p1/z, [x0]
ld1rw {z5.s}, p1/z, [x0, #4]
ld1rw {z6.s}, p1/z, [x0, #8]
ld1rw {z7.s}, p1/z, [x0, #12]
```

**Result**: 595 cycles, 57.2 B/cy, 44.7% efficiency (+0.5% vs baseline)

**Analysis**: The ZIP overhead (~8 instructions) nearly negates the ld1rw addressing improvement.
The real bottleneck is **ld1b + ld1rw port contention**, not ld1rw addressing alone.

### 2. Sector Cache Streaming Hints

**Concept**: Use `PRFM pldl1strm` for K/V data to mark them as streaming/non-temporal.

```asm
// Prefetch K as streaming data
prfm pldl1strm, [x4, #1024]
prfm pldl1strm, [x4, #1280]

// Regular loads
ld1b {z0.b}, p0/z, [x4]
...
```

**Result**: 624 cycles, 54.6 B/cy, 42.6% efficiency (-1.6% vs baseline)

**Analysis**: **Streaming hints hurt performance** because:
- Working set (33KB) fits in L1 (64KB)
- `pldl1strm` hints cause early eviction or cache bypass
- Data should stay L1-resident for this tile size

**When to use streaming hints**:
- Only when working set exceeds L1 (>64KB)
- When data is accessed exactly once and shouldn't pollute cache

### Optimization Comparison Summary

| Kernel | Cycles | Load BW | Efficiency |
|--------|--------|---------|------------|
| baseline (row-major S) | 602 | 56.6 B/cy | 44.2% |
| interleaved_s (ZIP + consec ld1rw) | 595 | 57.2 B/cy | 44.7% |
| sector_cache (PRFM streaming) | 624 | 54.6 B/cy | 42.6% |
| **Theoretical** | 266 | 128 B/cy | 100% |

### 3. DC ZVA (Zero Cache Line) for Stores

**Hypothesis**: Store bottleneck from read-for-ownership (RFO) when cache line not present.
DC ZVA zeros a 256-byte cache line without reading, potentially avoiding RFO overhead.

**Results**: DC ZVA **hurts** performance significantly on A64FX:

| Test | Baseline | With DC ZVA | Slowdown |
|------|----------|-------------|----------|
| warm store (256B) | 8 cy | 222 cy | **27x slower** |
| multi-cacheline (4KB) | 92 cy | 3352 cy | **36x slower** |
| Q@K^T phase | 279 cy | 468 cy | 1.7x slower |
| Full fused attention | 598 cy | 1954 cy | 3.3x slower |

**Analysis**: DC ZVA is counterproductive because:
1. Cache lines are already L1-resident (warm cache)
2. A64FX write-allocate mechanism is efficient - no need to explicitly zero
3. DC ZVA may introduce synchronization overhead

**Conclusion**: **Do NOT use DC ZVA** for L1-resident stores. The store path is already efficient.

### Key Conclusions

1. **~45-50% is the practical ceiling** for mixed ld1b + ld1rw access patterns
2. **Port contention is the fundamental limit**, not addressing mode or cache policy
3. **Interleaved S provides marginal benefit** (~1%) due to ZIP overhead
4. **Streaming hints hurt for L1-resident data** - use only when spilling to L2
5. **ld1rw optimization (0.51 cy/load) is already maximally effective** given port constraints
6. **DC ZVA hurts performance** - A64FX handles write-allocate efficiently

### Recommendations

For L1-resident fused attention (working set < 64KB):
1. Use optimized ld1rw addressing (pre-computed bases + immediate offsets)
2. **Do NOT use streaming prefetch hints**
3. Row-major S storage is acceptable (interleaving overhead not worth it)
4. Focus optimization effort on **compute overlap**, not memory access

For L2-spilling workloads (working set > 64KB):
1. Consider streaming hints for K/V (single-use data)
2. Keep S/Q/O in L1 with keep hints
3. Use tiling to minimize L2 traffic

## SDOT Dependency Analysis

### Problem: SDOT Latency Chains

Original implementation had severe SDOT-to-SDOT dependencies:

```asm
// BAD: Each SDOT depends on previous (9-cycle latency chain)
sdot z16.s, z0.b, z4.b  // row0 += K[0] * Q[0]
sdot z16.s, z1.b, z4.b  // STALL: depends on z16 (9 cycles)
sdot z16.s, z2.b, z4.b  // STALL: depends on z16 (9 cycles)
sdot z16.s, z3.b, z4.b  // STALL: depends on z16 (9 cycles)
```

Result: 2060 cycles, **0.5 SDOT/cycle** (25% of peak)

### Solution: Separate Accumulators

Use 16 independent accumulators (4 rows × 4 K-vectors):

```asm
// GOOD: Each SDOT to different accumulator (no dependencies)
sdot z16.s, z0.b, z4.b  // row0 += K[0] * Q[0]
sdot z17.s, z1.b, z4.b  // row0 += K[1] * Q[0] (different acc!)
sdot z18.s, z2.b, z4.b  // row0 += K[2] * Q[0]
sdot z19.s, z3.b, z4.b  // row0 += K[3] * Q[0]

// Merge at end: z16 += z17 + z18 + z19
```

### Results

| Version | Cycles | SDOT/cy | Efficiency |
|---------|--------|---------|------------|
| Original (dependencies) | 2060 | 0.5 | 25% |
| No-dep sequential | 676 | 1.5 | 75% |
| No-dep interleaved | **618** | **1.7** | **85%** |
| 4x unroll | 635 | 1.6 | 80% |
| Theoretical | 512 | 2.0 | 100% |

### Key Insights

1. **Breaking dependency chains gives 3x improvement** (2060→618 cycles)
2. **Interleaving loads with SDOT further improves** scheduling
3. **~85% of theoretical SDOT throughput** achieved with proper register allocation
4. **The kernel is compute-bound**, not memory-bound (at 27 B/cy load, well below 128 B/cy peak)

### Memory vs Compute Analysis

For Q@K^T [4,256] @ [256,64]:
- Memory: 17KB load at 128 B/cy = **136 cycles** (load-bound)
- Compute: 1024 SDOT at 2/cy = **512 cycles** (compute-bound)
- Achieved: **618 cycles** (compute-bound, 85% efficiency)

The kernel is **~4x compute-bound** relative to memory. Previous load-only benchmarks showed 50% memory efficiency, but with actual compute the memory subsystem has spare capacity.

## Files

- `bench_pure_load.c` - Pure load throughput tests
- `bench_ld1rw_opt.c` - ld1rw optimization comparison
- `bench_qkt_only.c` - Q@K^T baseline
- `bench_qkt_optimized.c` - Optimized Q@K^T
- `bench_qkt_deep_unroll.c` - Deep unroll variants
- `bench_l2_pattern.c` - L1 vs L2 cache tests
- `bench_fused_int32.c` - Fused INT32 attention (no quantization)
- `bench_fused_optimized.c` - Interleaved S + sector cache tests
- `bench_zfill.c` - DC ZVA (zero cache line) store tests
- `bench_interleaved_sdot.c` - SDOT with dependencies (baseline)
- `bench_sdot_nodep.c` - SDOT with 16 independent accumulators
- `bench_sdot_unroll4.c` - 4x unrolled SDOT version
- `bench_tile_groups.c` - Tile groups analysis for L2 latency hiding

## Tile Groups for L2 Latency Hiding

### Problem: Single Tile Underutilizes FPUs

With a single 4-row tile group (MR=4), we have 16 SDOTs per d_group iteration.
The load pipeline requires time to fetch K/Q data, but there aren't enough
independent SDOTs to keep both FPUs busy while waiting.

### Experiment: Single vs Two Tile Groups

| Configuration | Accumulators | SDOTs/d_group | K loads shared |
|---------------|--------------|---------------|----------------|
| Single tile (MR=4) | 16 | 16 | No |
| Two tiles (MR=8) | 32 | 32 | Yes |

Two tiles share K loads but have separate Q loads, doubling compute per K fetch.

### Results

| Configuration | D | Cycles | SDOT/cy | Efficiency |
|---------------|---|--------|---------|------------|
| Pure SDOT (no memory) | - | 633 | 1.62 | 81% |
| **Single tile** | 64 | 723 | **0.35** | **18%** |
| **Single tile** | 256 | 2493 | **0.41** | **21%** |
| **Two tiles** | 64 | 333 | **1.54** | **77%** |
| **Two tiles** | 256 | 1201 | **1.71** | **86%** |
| **Two tiles** | 1024 | 4764 | **1.72** | **86%** |

### Analysis

**4x improvement** from single to two tile groups:
- Single tile: 0.35-0.42 SDOT/cy (memory pipeline starves compute)
- Two tiles: 1.54-1.74 SDOT/cy (matches standalone SDOT efficiency)

**Why two tiles work**:
1. K loads are reused for both tile groups (256 bytes loaded once, used twice)
2. 32 independent SDOTs per iteration vs 16
3. Sufficient compute to hide load latency (~4 cycles per K load)

**L2 latency hiding**:
- D=1024 spills to L2 (74KB working set > 64KB L1)
- Two tiles still achieve 1.72 SDOT/cy
- 32 independent SDOTs hide ~32 cycle L2 latency effectively

### Recommended Tile Strategy

For peak SDOT throughput (2/cycle theoretical):

| Scenario | Minimum Tiles | SDOTs per d_group |
|----------|---------------|-------------------|
| L1 resident | 2 (MR=8) | 32 |
| L2 latency hiding | 2+ (MR=8+) | 32+ |
| HBM2 latency hiding | 4+ (MR=16+) | 64+ |

**Key insight**: The kernel efficiency is determined by compute-to-load ratio,
not raw memory bandwidth. With shared K loads, two tile groups achieve
**near-peak SDOT throughput (86% of theoretical)**
