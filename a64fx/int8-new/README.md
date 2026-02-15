# INT8 GEMM with SVE SDOT for A64FX

Optimized int8 matrix multiplication using A64FX SVE `sdot` instructions.

## Overview

Implements `C[M×N] = A[M×K] × B[N×K]^T` with:
- **Input**: int8 matrices A and B
- **Output**: int32 matrix C
- **K dimension**: Fixed at 256 (reduction axis)
- **Two microkernel variants**:
  - **5×4**: MR=5 rows, NR=64 columns (20 accumulators, 3 spare registers)
  - **6×4**: MR=6 rows, NR=64 columns (24 accumulators, tighter register pressure)

## Key Features

- **No horizontal add**: SVE `sdot` accumulates directly into int32 lanes
- **Optimized memory layout**: K-major packing for B, row-major for A
- **Full vector output**: Stores all 16 lanes per accumulator (5×64 or 6×64 int32 values)
- **Edge case handling**: Scalar fallback for incomplete tiles

## Files

- `kernel_5x4.S` - 5×4 microkernel assembly
- `kernel_6x4.S` - 6×4 microkernel assembly (TODO)
- `gemm_pack.c/h` - Matrix packing functions
- `gemm_driver.c/h` - Macro-kernel drivers (M/N loops)
- `bench_gemm.c` - Benchmark and correctness tests
- `Makefile` - Build system

## Building

Requires Fujitsu FCC compiler (fccpx) or GCC with SVE support.

```bash
# Build 5×4 kernel benchmark
make 5x4

# Build 6×4 kernel benchmark (after implementing kernel_6x4.S)
make 6x4

# Clean build artifacts
make clean
```

## Running

```bash
# Run 5×4 tests and benchmarks
./bench_5x4

# Run 6×4 tests and benchmarks
./bench_6x4
```

## Implementation Details

### SDOT Instruction

The A64FX SVE `sdot` instruction groups 4 consecutive int8 elements and accumulates into int32:
```
sdot z0.s, z1.b, z2.b
  => z0[i] += z1[4i:4i+3] · z2[4i:4i+3] for i in [0, 15]
```

- No horizontal reduction needed (unlike SMMLA)
- Direct accumulation into 16 int32 lanes per vector
- K must be divisible by 4 (satisfied with K=256)

### Memory Layout

**A Matrix Packing** (row-major):
- `Apack[m][k] = A[m0+m][k]` for m in [0, Mr)
- Loaded with `ld1rqb`: replicates 16 bytes across 64-byte vector
- Each row's 4 int8 values broadcast to all 16 lanes

**B Matrix Packing** (K-major):
- `Bpack[k][lane] = B[n0+lane][k]` for k in [0, K), lane in [0, 63]
- Loaded with `ld1b`: full 64-byte vector load
- Sequential access in K-loop

**C Matrix Output** (row-major):
- `C[m][n]` for m in [0, Mr), n in [0, Nr=64)
- Stored with `st1w`: 4 vectors per row (64 int32 values)
- Each accumulator stores all 16 lanes (no SADDV reduction)

### Register Allocation

**5×4 Kernel**:
- z0-z19: 20 accumulators (5 rows × 4 vectors)
- z20-z24: A row buffers (5 vectors)
- z25-z28: B column buffers (4 vectors)
- z29-z31: Spare (for future pipelining/prefetch)

**6×4 Kernel**:
- z0-z23: 24 accumulators (6 rows × 4 vectors)
- z24-z29: A row buffers (6 vectors)
- z30-z31: B column buffers (reloaded twice per iteration)
- Higher compute density but no spare registers

### Performance Target

A64FX single core INT8 SDOT peak: **~8 GOPS**
- 2 FPU pipelines × 2 ops/cycle × 2.0 GHz
- Target efficiency: 70-90% (5.6-7.2 GOPS)

Expected:
- **5×4**: 70-85% efficiency (more registers for pipelining)
- **6×4**: 75-90% efficiency (higher compute density)

## Correctness Tests

Tests validate against naive triple-loop reference:
- Small sizes (M=10, N=128)
- Edge cases (M % Mr != 0, N % Nr != 0)
- Multiple tiles (M=100, N=512)

All tests must pass before performance benchmarks.

## Future Optimizations

1. **K-loop unrolling** (2× or 4×) - reduce loop overhead
2. **Software pipelining** - load next iteration while computing
3. **Prefetch tuning** - hide memory latency
4. **OpenMP parallelization** - multi-threaded scaling

## References

- Existing implementations: `a64fx/int8-gemm/int8_sdot_kernel.S`, `gqa_micro_kernel_6x4.S`
- Design document: `a64fx/int8-new/plan.md`
- Reference FP64 kernel: `a64fx/ref/dgemm.kernel.s`
