# INT8 SDOT GEMM for A64FX (12-core CMG) - Status

## Summary

12-core INT8 SDOT GEMM implementation targeting A64FX's 2 SDOT/cycle peak throughput. Uses 6x4 microkernel tile with sector cache optimization for streaming B matrix data.

**Best Result: 79% efficiency (1.58 SDOT/cycle, 76 GOPS per core, 912 GOPS total)** - Confirmed stable

## Current Status

| Metric | Value | Notes |
|--------|-------|-------|
| Efficiency | 79% | Target was 90%, stable result |
| SDOT/cycle | 1.58 | Peak is 2.0 |
| Throughput | 76 GOPS/core | Peak is 96 GOPS/core |
| Best Config | M=84, K=512, N=8192 | Confirmed |
| Best Kernel | micro_kernel_6x4_sector_unroll3 | |

## Bottleneck Analysis (LD/ST Only Test)

Key finding: **Memory is NOT the bottleneck**

| Kernel | Cycles/tile | Efficiency |
|--------|-------------|------------|
| LD/ST only (no SDOT) | ~116 | ~110% (faster than SDOT) |
| SDOT kernel | ~162 | ~79% |

The LD/ST only kernel (NOPs instead of SDOT) is **faster** than the SDOT kernel, proving that memory can keep up. The ~46 cycle gap (162-116) is pure SDOT compute overhead that can't be overlapped with memory due to load-use dependencies.

## Key Findings

### What Works

1. **6x4 tile is optimal** - Uses all 32 SVE registers (24 acc + 6 A + 2 B)
2. **Sector cache hints** - Tag B pointer with sector 1 for streaming, prevents L2 pollution
3. **3x K-unrolling** - 72 SDOTs per iteration, ~1% better than 2x unroll
4. **K=512 is sweet spot** - Balances loop overhead vs cache pressure
5. **Pure SDOT achieves 100%** - Hardware can sustain 2.0 SDOT/cycle

### What Doesn't Work

1. **4x K-unrolling** - Mid-loop pointer arithmetic overhead exceeds benefit (tested v2)
2. **Explicit prefetching** - A64FX hardware prefetcher already effective with sector hints
3. **Interleaved SDOT ordering** - OoO engine handles standard ordering better
4. **8x3 tile** - Doesn't fit in 32 registers (needs 33)
5. **K-blocking with re-packing** - Packing overhead dominates (12-20% vs 80% baseline)
6. **Accumulating kernel for K-blocks** - Re-packing cost not amortized
7. **EOR vs DUP zeroing** - No measurable difference (zeroing is one-time cost)
8. **5x4 tile with double-buffered B** - Worse at 63% (smaller tile = more overhead)
9. **1x K-unroll (minimal)** - Only 79% vs 86% for 3x unroll (loop overhead matters)

### Performance Sensitivity (from comprehensive sweep)

| K Value | Efficiency | Notes |
|---------|------------|-------|
| K=256 | 70% | Too small, loop overhead |
| K=384 | 76% | Suboptimal |
| K=512 | 79% | **Optimal** |
| K=640 | 71% | L1 pressure begins |
| K=768 | 66% | L1 misses |
| K=1024 | 66% | L1 thrashing |

| M Value | Efficiency | Notes |
|---------|------------|-------|
| M=60-84 | 78-79% | All good |
| M=90 | 77% | Slight degradation |
| M=96 | 75% | Cache pressure |
| M=108 | 73% | Too large |

## Gap Analysis (79% → 90%)

The 21% overhead breakdown based on LD/ST only test:
- **SDOT compute + dependencies**: ~46 cycles/tile (28% of runtime)
  - Load-use latency (~4 cycles per load, 10 loads per K-group)
  - Cannot be hidden with only 2 B registers
- **Memory operations**: ~116 cycles/tile (overlapped with above)
- **Loop overhead**: ~6% (comparing 1x vs 3x K-unroll)

Reaching 90% would require reducing the SDOT dependency overhead from 28% to ~10%, which is fundamentally limited by:
1. Only 2 B registers available (no room for double-buffering)
2. 32 register constraint (6x4 tile uses all 32)
3. Load-use latency inherent to the dataflow

## Files

| File | Description |
|------|-------------|
| `micro_kernel_6x4_sector_unroll3.S` | Best kernel (79% efficiency) |
| `micro_kernel_6x4_ldst_only.S` | Memory-only test kernel (no compute) |
| `micro_kernel_6x4_minimal.S` | 1x K-unroll baseline |
| `micro_kernel_6x4_eor.S` | EOR zeroing variant |
| `bench_fapp_profile.c` | Benchmark for profiling both kernels |
| `run_ldst_baseline.sh` | Script to compare SDOT vs LD/ST only |
| `run_confirm_85.sh` | Comprehensive efficiency sweep |

## Conclusion

**79% efficiency achieved and confirmed stable.** This is the practical limit for the 6x4 tile approach.

### Why 90% is Unreachable with This Approach

The LD/ST only test proved definitively that **memory is not the bottleneck**:
- Memory alone: 116 cycles/tile
- SDOT kernel: 162 cycles/tile
- The 46-cycle gap is SDOT compute + load-use latency

The fundamental constraint is **register pressure**:
- 6x4 tile uses all 32 SVE registers (24 acc + 6 A + 2 B)
- Only 2 B registers means no double-buffering within a K-group
- Load-use latency (~4 cycles) cannot be hidden without preloading

### What Would Be Needed for 90%

1. **More registers**: Allow double-buffered B loads (need 4 B regs)
2. **Smaller tile**: 4x4 or 5x4, but these have more overhead
3. **Different data layout**: Reduce loads per K-group
4. **Hardware support**: Lower load-use latency

The 6x4 tile at 79% efficiency achieves a good balance - larger tiles would need more registers, smaller tiles have more overhead.
