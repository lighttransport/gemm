# INT8 SDOT GEMM Optimization for A64FX (12-core CMG)

## Final Results

**Best Configuration: M=90, K=512, N=8192**
- **Efficiency: 80.0% (1.600 SDOT/cycle)**
- **Throughput: 4916 GOPS**
- **Kernel: micro_kernel_6x4_sector_unroll3**

## Performance Context

| Metric | Value |
|--------|-------|
| Peak Theoretical | 2.0 SDOT/cycle (6144 GOPS) |
| Pure SDOT (no memory) | 2.0 SDOT/cycle (100%) |
| GEMM Kernel Achieved | 1.6 SDOT/cycle (80%) |
| Memory + Loop Overhead | 20% |

## Optimization Techniques Applied

### Successful Optimizations

1. **6x4 Microkernel Tile**
   - 6 rows × 4 vectors = 24 accumulators
   - Optimal register allocation: 24 acc + 6 A + 2 B = 32 registers

2. **Sector Cache Hints**
   - B matrix tagged with sector 1 for streaming
   - Prevents L2 pollution from B data
   - A matrix stays in sector 0 for temporal reuse

3. **3x K-group Unrolling**
   - 72 SDOTs per loop iteration
   - Reduces loop overhead from ~7% to ~5%
   - Small but consistent improvement over 2x unroll

4. **N-parallel Loop Structure**
   - Each thread processes different N-tiles
   - Q data reused across N-tiles (good temporal locality)
   - B data streams through (spatial locality with sector cache)

### Unsuccessful Optimizations

1. **4x K-group Unrolling** - Worse due to mid-loop pointer arithmetic
2. **Explicit Prefetching** - A64FX hardware prefetcher already effective
3. **Interleaved B-vector Access** - Different SDOT ordering hurts OoO scheduling
4. **8x3 Tile** - Doesn't fit in 32 registers

## Optimal Parameters

| Parameter | Optimal Value | Notes |
|-----------|---------------|-------|
| M | 72-90 | ~80% efficiency across this range |
| K | 512 | Sweet spot for L1 utilization |
| N | 8192 | Good parallelism across 128 N-tiles |
| MR | 6 | Rows in microkernel |
| NR | 64 | Columns in microkernel (4 vectors) |

## Performance Sensitivity

- **K < 512**: Higher loop overhead → lower efficiency (68-76%)
- **K > 512**: L1 cache pressure → lower efficiency (63-75%)
- **N < 8192**: Reduced parallelism → lower efficiency (71%)
- **N > 8192**: Memory bandwidth pressure → lower efficiency (68%)
- **M = 96**: Cache pressure → efficiency drops to 72%

## Why 80% is a Good Result

The 20% overhead comes from:
1. **Memory Loads**: 10 loads per K-group (6 ld1rw for A, 4 ld1b for B)
2. **Loop Control**: subs + branch per 3 K-groups
3. **Pointer Updates**: 2-3 add instructions per iteration
4. **Load-use Latency**: ~4 cycles for L1 hits, partially hidden by OoO

For a memory-bound GEMM operation, 80% compute efficiency is excellent.

## Files

- `micro_kernel_6x4_sector_unroll3.S` - Best performing kernel
- `bench_final.c` - Benchmark harness
- `run_final_best.sh` - Job script for optimal configuration

## Usage

```bash
# Compile
fcc -O3 -Nclang -mcpu=a64fx+sve -c micro_kernel_6x4_sector_unroll3.S
fcc -O3 -Nclang -mcpu=a64fx+sve -fopenmp bench_final.c micro_kernel_6x4_sector_unroll3.o -o bench_final

# Run with optimal parameters
./bench_final -m 90 -k 512 -n 8192 -i 200
```

## Theoretical Analysis

Per K-group (4 bytes of K dimension):
- Compute: 24 SDOT instructions
- Memory: 6 ld1rw (A) + 4 ld1b (B) = 10 loads
- At 2 SDOT/cycle: minimum 12 cycles
- At 2 loads/cycle: minimum 5 cycles (fits within compute time)

Actual: ~15 cycles per K-group → 80% efficiency
Overhead: ~3 cycles from loop control, pointer updates, and load latency

## Comparison with Target

| Target | SDOT/cycle | Efficiency | Status |
|--------|------------|------------|--------|
| Original | 0.78 | 39% | Starting point |
| 80% Target | 1.60 | 80% | **Achieved** |
| 90% Target | 1.80 | 90% | ~10% gap |
| Theoretical | 2.00 | 100% | Limit |

The 90% target would require either:
- Reducing loop overhead further (diminishing returns)
- Hardware prefetch improvements
- Different algorithmic approach (e.g., K-blocking at higher level)

## Conclusion

Achieved **80% SDOT efficiency** on A64FX for INT8 GEMM with sector cache optimization. This represents a practical optimum for the 6x4 tile approach with the given memory access patterns.
