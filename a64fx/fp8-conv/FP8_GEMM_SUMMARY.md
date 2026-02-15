# FP8 GEMM with FP32 Accumulation on A64FX

## Overview

This implements FP8 GEMM on Fujitsu A64FX using:
- FP8 (E4M3) input matrices
- FP32 accumulation
- LUT-based FP8→FP32 conversion via SVE gather

## Implementation Details

### Kernel Flow
1. Load FP8 A values (3 elements per K, scalar)
2. Convert FP8→FP32 via scalar LUT lookup
3. Broadcast A values to FP32 vectors
4. Load FP8 B values (64 elements = 4×16 per K)
5. Convert FP8→FP32 via SVE gather from LUT
6. FP32 FMLA accumulation (12 ops: 3 rows × 4 vectors)

### Tile Configuration
- MR = 3 (rows of A per tile)
- NR = 4 (FP32 vectors of B per tile)
- N_tile = 64 elements (4 × 16 FP32)
- 12 FP32 accumulator registers

### SVE Instructions Used
- `svld1_gather_u32offset_u32`: Gather 16 FP32 values from LUT
- `svmla_f32_x`: FP32 fused multiply-add
- `svunpklo_u16/u32`: Zero-extend FP8 indices to 32-bit

## Performance Results

### Test Configuration
- M = 384, N = 64, K = 512
- FLOPs = 25.17 MFLOPs
- A64FX @ 2.0 GHz (single core)

### Measurements
| Metric | Value |
|--------|-------|
| Cycles per call | 4.4M |
| GFLOPS | 11.5 |
| Peak FP32 efficiency | 9.0% |
| Cycles per K iteration | 66.8 |

### Bottleneck Analysis

The performance is limited by gather throughput:
- Each K iteration requires 4 SVE gathers (16 elements each)
- Gather latency: ~11 cycles per operation
- Gather throughput: limited by L1 access patterns

Per K iteration breakdown:
```
Operation                   Count    Latency
-----------------------------------------------
Scalar LUT lookups (A)      3        ~3 cycles
SVE gather (B)              4        ~11 cycles each
FP32 FMLA                   12       9 cycles (pipelined)
Unpack + offset calc        4        ~2 cycles each
-----------------------------------------------
Theoretical minimum:        ~44 cycles (gather limited)
Observed:                   ~67 cycles
```

### Comparison with Native FP16/FP32 GEMM

| Data Type | Peak Efficiency | Notes |
|-----------|-----------------|-------|
| INT8 (SDOT) | 94% | Uses SDOT instructions |
| FP16 (FMLA) | 90%+ | Native FP16 operations |
| FP32 (FMLA) | 85%+ | Native FP32 operations |
| FP8→FP32 (gather) | 9% | Gather-limited |

The FP8 GEMM with LUT conversion is significantly slower due to:
1. No native FP8 arithmetic on A64FX
2. Gather operations for each FP8→FP32 conversion
3. Limited gather throughput (1-2 per cycle vs 2 FMLA per cycle)

## Potential Optimizations

1. **Wider tiles**: Increase NR to process more B elements per gather
2. **Software pipelining**: Overlap gather with FMLA from previous K
3. **K-unrolling**: Unroll K loop to hide gather latency
4. **Pre-conversion**: Convert FP8→FP32 before GEMM if matrices are reused

## Files

- `fp8_gemm.h`: Header with declarations
- `fp8_gemm.c`: Micro-kernel implementation
- `bench_fp8_gemm.c`: Benchmark program
- `fp8_convert.c`: FP8 LUT and conversion functions

## Build and Run

```bash
make bench_fp8_gemm
./bench_fp8_gemm
# Or with custom M, K:
./bench_fp8_gemm 768 1024
```
