# INT8 SDOT GEMM Optimization: 90%+ Efficiency Achieved

## Summary

We achieved **sustained 90%+ SDOT efficiency** on A64FX 12-core CMG through micro-blocking optimization.

### Results

| Metric | Value |
|--------|-------|
| Peak Theoretical | 24.0 SDOT/cycle (12 cores × 2 FPUs) |
| Achieved | 22.7 SDOT/cycle average (94.9%) |
| Minimum | 92.1% |
| Maximum | 96.5% |
| Runs >= 90% | 100/100 (100%) |

### Optimal Configuration

```
Total GEMM: M=672, N=640, K=512
Micro-block: uM=48, uN=256
Working set: A=24KB, B=128KB, Total=152KB

Kernel: micro_kernel_6x4_ooo_v5.S
  - 6×4 tile (6 M-rows × 64 N-cols)
  - 2x K-unroll with out-of-order scheduling
  - Sector cache hint for B (L2 streaming)
  - EOR for fast register zeroing
```

## The Key Insight: Micro-Blocking

### Problem with Flat Loop (~82% efficiency)
```c
for (mt = 0; mt < M_tiles; mt++) {
    for (nt = 0; nt < N_tiles; nt++) {
        kernel(A[mt], B[nt], C[mt,nt], K);
    }
}
```
- Large working set (656KB) exceeds L2 budget per core
- B tiles evicted before reuse

### Solution: Micro-Blocking (~95% efficiency)
```c
for (mb = 0; mb < num_M_blocks; mb++) {
    for (nb = 0; nb < num_N_blocks; nb++) {
        // Process 8×4 = 32 micro-tiles
        for (nt = 0; nt < 4; nt++) {          // N-inner
            for (mt = 0; mt < 8; mt++) {       // M-outer
                kernel(A[mb*8+mt], B[nb*4+nt], C[..], K);
            }
        }
    }
}
```
- Micro-block working set (152KB) fits in L2
- Each B tile reused 8× across M tiles
- N-inner loop maximizes B cache hits

## Performance Comparison

| Approach | Efficiency | SDOT/cyc |
|----------|------------|----------|
| Single-core L1-resident | 94.4% | 1.89 |
| 12-core flat loop | 82.1% | 19.7 |
| 12-core micro-blocking | 94.9% | 22.7 |

## Optimization Path

1. **Kernel optimization** (achieved 94.4% single-core ceiling)
   - 6×4 tile = 24 accumulators (z0-z23)
   - 2x K-unroll for reduced loop overhead
   - Out-of-order instruction scheduling
   - EOR for fast zeroing (2x faster than MOV)

2. **Memory optimization** (closed gap from 82% to 95%)
   - Per-core allocation (avoids contention)
   - Micro-blocking (L2-sized working set)
   - N-inner loop (B reuse)

## Files

- `micro_kernel_6x4_ooo_v5.S`: Best kernel implementation
- `run_90_verified.sh`: Verification benchmark
- `bench_90_verified.c`: C wrapper with micro-blocking

## Usage

```bash
# Compile
fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_6x4_ooo_v5.S -o kernel.o
fcc -O3 -Nclang -mcpu=a64fx+sve -fopenmp your_code.c kernel.o -o your_program

# Run with 12 threads
export OMP_NUM_THREADS=12
./your_program
```
