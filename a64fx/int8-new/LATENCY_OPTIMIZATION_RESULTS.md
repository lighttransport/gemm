# Latency-Optimized Kernel Benchmark Results

**Date:** 2026-01-17
**Platform:** A64FX @ 2GHz, dual FPU (eco_state=0)
**Test:** Same-address L1D-resident data (pure compute)

## Background

SVE load latency on A64FX is 11 cycles. This benchmark compares two strategies for hiding this latency:

1. **6x4 OoO**: Exploit Out-of-Order execution and register renaming. After issuing SDOT using a register, immediately load a new value - OoO preserves the old value for in-flight instructions.

2. **5x4 Double-Buffer**: Use smaller tile (5x4 = 20 accumulators) to free registers for explicit double-buffering of A matrix loads.

## Results

### 6x4 Kernels (24 SDOT/iter, ideal 12 cycles)

| Kernel | GIOPS | Cycles/iter | Efficiency |
|--------|-------|-------------|------------|
| Baseline | 491.5 | 12.50 | 96.0% |
| Interleaved | 512.0 | 12.00 | 100.0% |
| **OoO (new)** | **512.0** | **12.00** | **100.0%** |

### 5x4 Double-Buffer (40 SDOT/iter, ideal 20 cycles)

| Kernel | GIOPS | Cycles/iter | Efficiency |
|--------|-------|-------------|------------|
| DblBuf (new) | 476.3 | 21.50 | 93.0% |

### Normalized to Peak (512 GIOPS)

- 6x4 Baseline: 96.0%
- 6x4 Interleaved: 100.0%
- 6x4 OoO: 100.0%
- 5x4 DblBuf: 93.0%

## Analysis

### Why 6x4 OoO Wins

1. **OoO handles latency hiding automatically**: A64FX's Out-of-Order execution with register renaming preserves old register values for in-flight SDOT instructions, even when new loads target the same architectural register.

2. **Higher compute density**: 6x4 tile has 24 accumulators vs 20 for 5x4, providing more compute per iteration.

3. **No double-buffering overhead**: The 5x4 approach requires extra instructions to manage two A buffers, adding ~1.5 cycles overhead per iteration.

### SDOT Throughput Comparison

- 6x4 OoO: 24 SDOT / 12.00 cycles = **2.00 SDOT/cycle** (peak)
- 5x4 DblBuf: 40 SDOT / 21.50 cycles = **1.86 SDOT/cycle** (93% of peak)

## Conclusion

**6x4 with OoO-exploiting instruction scheduling is optimal** for A64FX INT8 SDOT kernels:

- Achieves 100% peak throughput (512 GIOPS)
- Simpler implementation (no explicit double-buffering)
- Maximum compute density with all 32 SVE registers utilized
- OoO execution naturally hides the 11-cycle SVE load latency

The explicit double-buffering strategy (5x4) is counterproductive - it sacrifices compute density for a latency-hiding mechanism that OoO already provides for free.

## Files

- `kernel_6x4_ooo.S` - 6x4 kernel exploiting OoO register renaming
- `kernel_5x4_dblbuf.S` - 5x4 kernel with explicit double-buffering
- `bench_latency.c` - Benchmark comparing all variants
- `run_latency.sh` - Job script with eco_state=0 for dual FPU
