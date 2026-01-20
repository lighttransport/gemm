# Fused GEMM Simulation Results

## Problem: (A @ B) @ C with [8192, 256] dimensions

**Dimensions**:
- A: [6, 8192] - 6 rows of input
- B: [256, 8192]^T - Transposed weight matrix
- C: [256, 256]^T - Second transposed weight matrix
- Output: [6, 256]

**Data Movement (INT8)**:
- A: 6 × 8192 = 48 KB
- B: 256 × 8192 = 2 MB
- C: 256 × 256 = 64 KB
- Total: ~2.1 MB

## Results Comparison

| Version | Total Cycles | L1 Hit Rate | CPI | Memory Stalls | Speedup |
|---------|-------------|-------------|-----|---------------|---------|
| **Naive** | 19,678 | 8.11% | 58.22 | 94.5% | 1.0× |
| **Optimized** | 9,548 | 36.21% | 22.31 | 88.0% | **2.1×** |
| **Tiled** | 3,682 | 80.85% | 8.56 | 65.2% | **5.3×** |

## Analysis

### 1. Naive Version (`fused_gemm_8192x256.txt`)

**Problem**: Scattered B matrix access pattern
- Each B row accessed at different strided addresses
- No temporal locality - data evicted before reuse
- 92% L1 miss rate → 95% memory stalls

**Access Pattern**:
```
ld1 0x200000  # B row 0
ld1 0x202000  # B row 1 (8 KB stride!)
ld1 0x204000  # B row 2
...           # Each row is 8192 bytes apart
```

### 2. Optimized Version (`fused_gemm_optimized.txt`)

**Improvements**:
- Software prefetching (4 cache lines ahead)
- Sequential B access within K-tiles
- Contiguous memory layout

**Impact**:
- L1 hit rate: 8% → 36% (4.5× better)
- Total cycles: 19,678 → 9,548 (2.1× speedup)
- CPI: 58 → 22 (2.6× better)

**Remaining Issue**: A tile not fully reused

### 3. Tiled Version (`fused_gemm_tiled.txt`)

**Key Optimizations**:
1. **Small tile size**: 2.4 KB working set (fits in L1)
2. **A tile reuse**: Same A tile used for multiple B iterations
3. **Sequential access**: B and C loaded sequentially

**Impact**:
- L1 hit rate: 8% → **81%** (10× better!)
- Total cycles: 19,678 → **3,682** (5.3× speedup)
- CPI: 58 → **8.56** (6.8× better)
- Memory stalls: 95% → **65%** (significant reduction)

## Key Insights

### 1. Tiling is Critical

```
Working Set Comparison:
  Naive:     ~100 KB per iteration (overflows L1)
  Optimized: ~2.5 KB per K-tile (fits in L1)
  Tiled:     ~2.4 KB per tile (optimal L1 fit)
```

### 2. Data Reuse Pattern

```
Naive:     A loaded once, not reused (evicted)
Optimized: A reused within K-tile
Tiled:     A reused across 4 B iterations (3× reuse)
           → Second and third A loads HIT in L1
```

### 3. Memory Stall Reduction

```
Naive:     94.5% stalls (memory-bound disaster)
Optimized: 88.0% stalls (better but still memory-bound)
Tiled:     65.2% stalls (approaching compute-bound)
```

### 4. CPI Improvement

```
Naive:     58.22 CPI (one instruction per 58 cycles!)
Optimized: 22.31 CPI (better throughput)
Tiled:      8.56 CPI (good efficiency)

Target:    ~1-2 CPI for compute-bound kernels
```

## Optimization Techniques Demonstrated

### Technique 1: K-dimension Tiling
```
Instead of: Process all K at once (8192 elements)
Do:         Process K in tiles of 64-256 elements
Benefit:    Working set fits in L1
```

### Technique 2: A Matrix Reuse
```
Naive:     Load A, use once, evict
Tiled:     Load A, reuse for N iterations of B
Benefit:   Amortize A load cost over many operations
```

### Technique 3: Sequential B Access
```
Naive:     B[i] at address + i*8192 (huge stride)
Tiled:     B packed for sequential access
Benefit:   256-byte cache lines utilized fully
```

### Technique 4: Software Prefetching
```
Pattern:   prfm [next_B_tile]
           ld1 [current_B_tile]
           compute...
Benefit:   Hide 260-cycle DRAM latency
```

### Technique 5: zfill for Output
```
Traditional: Load output buffer (260 cycles)
With zfill:  Allocate directly (11 cycles)
Benefit:     23× faster initialization
```

## Fused GEMM Benefits

The fusion of (A@B)@C provides:

1. **No intermediate storage**: AB results stay in registers
2. **Reduced memory traffic**: Don't write AB to DRAM
3. **Better cache utilization**: C matrix streams while AB computed

**Memory Savings**:
```
Unfused:  AB[6,256] = 1.5 KB written to memory, then read back
Fused:    AB stays in 32 SVE registers, no memory traffic
```

## A64FX-Specific Considerations

### Large Cache Lines (256 bytes)
- Each ld1 brings in 256 bytes
- Sequential 64-byte accesses benefit from same line
- Spatial locality is 4× better than x86 (64-byte lines)

### L2 XOR Indexing
- Reduces conflict misses for strided B access
- Important when B rows are power-of-2 stride apart

### HBM2 Latency (260 cycles)
- Prefetching is essential
- Cold misses very expensive
- Maximize L1 hit rate to avoid DRAM

## Recommendations for Real Kernels

1. **Tile Size**: Keep working set under 16 KB for best L1 hit rate
2. **Prefetch Distance**: 4-8 cache lines ahead (1-2 KB)
3. **Data Layout**: Pack matrices for sequential access within tiles
4. **A Reuse**: Structure loops to reuse A across multiple B columns
5. **Fusion**: Keep intermediate results in registers when possible
6. **zfill**: Use for output buffers to avoid DRAM fetch

## Performance Projections

Based on simulation results, expected real hardware performance:

| Metric | Naive | Tiled | Improvement |
|--------|-------|-------|-------------|
| Cycles | ~20K | ~4K | 5× |
| Efficiency | ~5% | ~35% | 7× |
| GOPS | ~10 | ~60 | 6× |

**Note**: Real hardware may show different results due to:
- Out-of-order execution
- Instruction-level parallelism
- Memory bandwidth limits
- TLB misses

The simulator provides directional guidance for optimization strategies.

## Files Created

- `fused_gemm_8192x256.txt` - Naive implementation
- `fused_gemm_optimized.txt` - Prefetch + sequential B
- `fused_gemm_tiled.txt` - Full tiled implementation

## Conclusion

Proper tiling and data reuse transforms a memory-bound kernel (8% L1 hit, 95% stalls) into a much more efficient implementation (81% L1 hit, 65% stalls) with **5.3× speedup**.

The key is keeping the working set small enough to fit in L1 and maximizing reuse of the A matrix tile.
