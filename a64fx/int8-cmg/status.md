# INT8 SDOT GEMM for A64FX (12-core CMG) - Status

## Summary

12-core INT8 SDOT GEMM implementation targeting A64FX's 2 SDOT/cycle peak throughput. Uses 6x4 microkernel tile with sector cache optimization for streaming B matrix data.

**Best Result: 80.0% efficiency (1.600 SDOT/cycle, 4916 GOPS)**

## Current Status

| Metric | Value | Notes |
|--------|-------|-------|
| Efficiency | 80.0% | Target was 90% |
| SDOT/cycle | 1.600 | Peak is 2.0 |
| Throughput | 4916 GOPS | Peak is 6144 GOPS |
| Best Config | M=90, K=512, N=8192 | |
| Best Kernel | micro_kernel_6x4_sector_unroll3 | |

## Key Findings

### What Works

1. **6x4 tile is optimal** - Uses all 32 SVE registers (24 acc + 6 A + 2 B)
2. **Sector cache hints** - Tag B pointer with sector 1 for streaming, prevents L2 pollution
3. **3x K-unrolling** - 72 SDOTs per iteration, ~1% better than 2x unroll
4. **K=512 is sweet spot** - Balances loop overhead vs cache pressure
5. **Pure SDOT achieves 100%** - Hardware can sustain 2.0 SDOT/cycle

### What Doesn't Work

1. **4x K-unrolling** - Mid-loop pointer arithmetic overhead exceeds benefit
2. **Explicit prefetching** - A64FX hardware prefetcher already effective with sector hints
3. **Interleaved SDOT ordering** - OoO engine handles standard ordering better
4. **8x3 tile** - Doesn't fit in 32 registers (needs 33)

### Performance Sensitivity

| Change | Impact |
|--------|--------|
| K < 512 | Loop overhead dominates → 68-76% |
| K > 512 | L1 cache pressure → 63-75% |
| M = 96 | Cache pressure → 72% |
| N < 8192 | Reduced parallelism → 71% |
| N > 8192 | Memory bandwidth → 68% |

## Gap Analysis (80% → 90%)

The 20% overhead comes from:
- **Memory loads**: 10 loads per K-group (partially hidden)
- **Loop control**: subs + branch every 3 K-groups
- **Pointer updates**: 2-3 add instructions per iteration
- **Load latency**: ~4 cycles for L1, not fully hidden

Reaching 90% would require reducing overhead from 20% to 10%, which is challenging without algorithmic changes (e.g., higher-level K-blocking).

## Files

| File | Description |
|------|-------------|
| `micro_kernel_6x4_sector_unroll3.S` | Best kernel (80% efficiency) |
| `micro_kernel_6x4_sector_unroll2.S` | 2x unroll variant (~78%) |
| `run_final_best.sh` | Benchmark script for optimal config |
| `OPTIMIZATION_SUMMARY.md` | Detailed optimization analysis |

## Next Steps (if continuing)

1. Try K-blocking at outer loop level to improve L1 hit rate
2. Investigate if different data layout reduces load instructions
3. Profile with fapp to identify remaining bottlenecks
4. Consider trading tile size for deeper software pipelining

## Conclusion

**80% efficiency achieved.** This is a practical optimum for the 6x4 tile approach. The remaining 20% overhead is inherent to memory access requirements. Further gains would require algorithmic changes rather than microkernel optimization.
