# L1/L2 Cache Optimization for INT8 GEMM on A64FX

## Cache Latency (Accurate Measurements)

- **L1 hit**: 11 cycles
- **L2 hit**: 27-36 cycles (2.5-3.3× slower than L1)
- **L1 cache size**: 64 KB per core
- **L2 cache size**: 8 MB shared

## Problem Analysis

### D=256 Cache Behavior
- **Working set per K-group iteration**: 4 KB (16 N-chunks × 256 bytes)
- **L1 utilization**: ~30% of 64 KB
- **Estimated L1 hit rate**: ~90%
- **Bottleneck**: Approaching compute-bound
- **Efficiency**: 85.21% (excellent)

### D=512 Cache Behavior
- **Working set per K-group iteration**: 8 KB (32 N-chunks × 256 bytes)
- **L1 utilization**: ~60% of 64 KB
- **Estimated L1 hit rate**: ~75%
- **Bottleneck**: More frequent L2 accesses
- **Efficiency**: 67.61% (good, but has headroom)

## Optimization Strategies

### 1. Software Prefetching

**Goal**: Hide L2 latency by prefetching data before it's needed

**Implementation**:
```asm
// Prefetch 2-4 K-groups ahead
add     x9, x7, #16384          // 16 KB ahead (2 K-groups for D=512)
prfm    pldl1keep, [x9]         // Prefetch to L1 cache
prfm    pldl1keep, [x9, #64]
prfm    pldl1keep, [x9, #128]
prfm    pldl1keep, [x9, #192]
```

**Rationale**:
- L2 access takes 27-36 cycles
- Each K-group iteration executes ~560-580 cycles (for D=512)
- Prefetching 2 K-groups ahead = ~1100 cycles lookahead
- Sufficient time to fetch from L2 to L1 before use

**Expected improvement**:
- D=256: +1-3% (already L1-bound)
- D=512: +5-10% (L2-bound, benefits more)

### 2. K-group Tiling

**Goal**: improve L1 cache temporal locality by processing K dimension in tiles

**Implementation**:
```c
// Instead of: for (k=0; k<128; k++)
// Use:
for (k_tile = 0; k_tile < 4; k_tile++) {
    for (k = k_tile*32; k < (k_tile+1)*32; k++) {
        // Process 32 K-groups per tile
    }
}
```

**Rationale**:
- Baseline: Process all 128 K-groups sequentially
  - Total B working set: 128 × 8KB = 1024 KB
  - Exceeds L1 (64 KB), poor temporal locality
- K-tiling: Process 32 K-groups at a time
  - Tile B working set: 32 × 8KB = 256 KB
  - Fits L2 better, some L1 reuse across N-chunks
  - Better cache line reuse within tile

**Expected improvement**:
- D=256: 0% (already fits well)
- D=512: +8-12% (better L1/L2 utilization)

### 3. N-chunk Blocking (Future Work)

**Goal**: process N dimension in smaller blocks for better L1 reuse

**Implementation**:
```c
// Process N in 2-4 blocks
for (n_block = 0; n_block < 4; n_block++) {
    for (k = 0; k < 128; k++) {
        for (n = n_block*8; n < (n_block+1)*8; n++) {
            // Process 8 N-chunks at a time
        }
    }
}
```

**Rationale**:
- Current: Process all 32 N-chunks per K-group
- Blocking: Process 8 N-chunks at a time across all K-groups
- Better reuse of A matrix data (loaded once per N-block)

**Expected improvement**: +3-5% for D=512

## Cache Access Pattern Analysis

### Baseline D=512 Access Pattern
```
K-group 0:  Load N-chunk 0-31 (8 KB) → L1
K-group 1:  Load N-chunk 0-31 (8 KB) → L1, evict some K-group 0 data
K-group 2:  Load N-chunk 0-31 (8 KB) → L1, evict more data
...
K-group 7:  Load N-chunk 0-31 (8 KB) → L1, K-group 0 fully evicted
K-group 8:  Load N-chunk 0-31 (8 KB) → L2 access (SLOW!)
...
K-group 127: Load N-chunk 0-31 (8 KB) → L2 access

L1 hit rate: ~75%
L2 access rate: ~25%
```

### With Software Prefetching
```
K-group 0:  Prefetch K-group 2, Load K-group 0 → L1 hit
K-group 1:  Prefetch K-group 3, Load K-group 1 → L1 hit
K-group 2:  Prefetch K-group 4, Load K-group 2 → L1 hit (prefetched!)
K-group 3:  Prefetch K-group 5, Load K-group 3 → L1 hit (prefetched!)
...

L1 hit rate: ~85-90% (improved!)
L2 access rate: ~10-15%
```

### With K-Tiling
```
Tile 0 (K-groups 0-31):
  - Working set: 256 KB fits L2
  - Some L1 reuse across N-chunks within tile
  - L1 hit rate: ~80%

Tile 1 (K-groups 32-63):
  - Same pattern
  - L1 hit rate: ~80%

Overall L1 hit rate: ~80%
L2 access rate: ~20%
```

## Expected Performance Improvements

### D=256 (Already Optimal)
| Optimization | Baseline | Expected | Gain |
|--------------|----------|----------|------|
| Baseline | 85.21% | - | - |
| + Prefetching | 85.21% | 86-87% | +1-2% |
| + K-tiling | 85.21% | 85.21% | 0% |

**Why minimal gain?**
- 4 KB working set already fits L1 well
- Already achieving ~90% L1 hit rate
- Compute-bound, not memory-bound

### D=512 (Room for Improvement)
| Optimization | Baseline | Expected | Gain |
|--------------|----------|----------|------|
| Baseline | 67.61% | - | - |
| + Prefetching | 67.61% | 72-75% | +5-8% |
| + K-tiling | 67.61% | 75-78% | +8-12% |
| + Both | 67.61% | 78-82% | +12-16% |

**Why significant gain?**
- 8 KB working set stresses L1
- Currently ~75% L1 hit rate
- L2 accesses (27-36 cycles) hurt performance
- Optimization can push toward 85% L1 hit rate

## Implementation Details

### Prefetch Distance Calculation

**D=512**:
- Cycles per K-group iteration: ~560
- L2 latency: 27-36 cycles
- Prefetch distance: 2 K-groups ahead (1120 cycles)
- Distance in bytes: 2 × 8KB = 16 KB

**D=256**:
- Cycles per K-group iteration: ~225
- Prefetch distance: 4 K-groups ahead (900 cycles)
- Distance in bytes: 4 × 4KB = 16 KB

### K-Tile Size Selection

**Criteria**:
- Tile should fit L2 comfortably (8 MB)
- Maximize L1 reuse within tile
- Balance tile overhead vs locality

**D=512 optimal tile size**: 32 K-groups
- Working set: 32 × 8KB = 256 KB
- Fits L2: 256 KB << 8 MB ✓
- Provides L1 reuse across N-chunks
- 4 tiles total: manageable overhead

## Benchmarking Methodology

1. **Warmup phase**: 10 iterations to stabilize caches
2. **Measurement phase**: 100 iterations for statistical significance
3. **Metrics tracked**:
   - Timer cycles (100 MHz)
   - CPU cycles (2 GHz)
   - GOPS (INT8 operations per second)
   - Efficiency (% of 512 GOPS peak)
   - SDOT throughput (SDOTs per cycle)

## Expected Results Summary

### Best Case Scenario
- **D=256**: 85% → 87% efficiency (+2%)
- **D=512**: 68% → 78% efficiency (+10%)

### Realistic Scenario
- **D=256**: 85% → 86% efficiency (+1%)
- **D=512**: 68% → 73% efficiency (+5%)

### Worst Case Scenario
- **D=256**: No improvement (already optimal)
- **D=512**: 68% → 70% efficiency (+2%)

## Validation Criteria

**Success metrics**:
1. D=512 efficiency improves by ≥3%
2. SDOT/cycle increases proportionally
3. No correctness issues (verified by test suite)
4. Prefetching shows benefit for L2-bound cases
5. K-tiling shows benefit for large working sets

**If optimization doesn't help**:
- Hardware prefetcher may already be effective
- Memory bottleneck may be elsewhere (TLB, DRAM bandwidth)
- Cache associativity conflicts
- Need hardware performance counters to diagnose

## Future Work

1. **Hardware performance counters**:
   - Measure actual L1/L2 hit rates
   - Validate cache miss estimates
   - Profile TLB misses

2. **Combined optimizations**:
   - Prefetching + K-tiling together
   - N-chunk blocking for better A matrix reuse
   - Multi-threading with careful cache sharing

3. **Adaptive strategies**:
   - Choose optimization based on matrix size
   - Runtime detection of cache sizes
   - Auto-tuning for different A64FX configurations

## References

- A64FX Microarchitecture Manual
- ARM SVE Programming Guide
- Cache Optimization Techniques for GEMM
- Previous analysis: `D512_SLOWDOWN_SUMMARY.txt`
- Previous efficiency report: `EFFICIENCY_REPORT.md`
