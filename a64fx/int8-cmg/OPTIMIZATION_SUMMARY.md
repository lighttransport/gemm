# INT8 SDOT GEMM Optimization Summary for A64FX 12-Core CMG

## Final Results

| Configuration | SDOT/cycle | Efficiency | Notes |
|--------------|------------|------------|-------|
| Single-core L1-resident | 1.89 | 94.5% | Upper bound |
| **12-core M=504 N=768/core** | 21.7 | **90.6%** | Best: +12.8% |
| 12-core M=336 N=768/core | 21.5 | 89.5% | |
| 12-core Independent (small) | 21.1 | 87.9% | |
| 12-core N-batch=all | 20.5 | 85.4% | |
| 12-core M-inner | 18.7 | 77.8% | Baseline |
| 12-core N-outer | 18.4 | 76.8% | Original |

## Optimization Path

### What Worked

1. **Independent GEMMs (87.9%, +10.1%)**
   - Each core processes its own independent A, B, C matrices
   - Zero synchronization overhead
   - Each core: M=84, N=682, K=512

2. **N-batch loop ordering (85.4%, +7.6%)**
   - Keep A tile in L1 while streaming all assigned B tiles
   - Process all N-tiles for one M-tile before moving to next M
   - A is small (3KB/tile), B is large (32KB/tile)
   - Maximizes A reuse, lets B stream efficiently

### What Didn't Help

1. **Sector cache hint on A vs B**: <0.5% difference
   - A already fits in L2 easily
   
2. **K×4 unroll**: -2% (worse)
   - More instructions per iteration
   - Loop overhead not the bottleneck
   
3. **Interleaved B layout**: -24% (much worse)
   - Breaks L1 locality within B tile
   - Small strides don't help HW prefetch

4. **Explicit SW prefetch**: -0.5%
   - HW prefetcher already doing good job

5. **Dual-stream with prefetch**: -0.6%
   - Overhead outweighs benefit

## Bandwidth Analysis

| Metric | Value |
|--------|-------|
| L2 bandwidth available | 42.6 B/cycle/core |
| L2 bandwidth used | 18.2 B/cycle/core |
| **Utilization** | **42%** |

Bandwidth is NOT the bottleneck. The remaining gap (85% → 95%) is due to:
1. Multi-core coordination overhead
2. Instruction issue rate limits
3. Memory latency not fully hidden

## Optimal Configuration (90.6%)

```c
// Independent GEMM per core, A in L1, stream B
// Per-core: M=504, N=768, K=512
// Total: M=504, N=9216 (768×12), K=512
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    // Each core has its own A, B, C
    for (int64_t mt = 0; mt < M_tiles; mt++) {
        const int8_t* App = Ap[tid] + mt * Ap_tile_size;
        for (int64_t nt = 0; nt < N_tiles; nt++) {
            const int8_t* Bpp = Bp[tid] + nt * Bp_tile_size;
            kernel(App, Bpp, Cp, K);
        }
    }
}
```

## Optimal Sizes

| Parameter | Value | Notes |
|-----------|-------|-------|
| M per core | 504 | 84 M-tiles × MR=6 |
| N per core | 768 | 12 N-tiles × NR=64 |
| K | 512 | Divisible by 4 |
| A per core | 252 KB | Fits in L2 |
| B per core | 384 KB | Fits in L2 (~667KB/core) |
| Total N | 9216 | 768 × 12 cores |

## Tile Sizes

- MR = 6, NR = 64
- K = 512
- A_tile = 3KB, B_tile = 32KB
- Working set per kernel: ~37KB (fits in 64KB L1)

## Key Insights

1. **A in L1, stream B** - small A (3KB/tile), large B (32KB/tile)
2. **Independent GEMMs per core** - zero sync overhead
3. **B per core ≤ 384KB** - must fit in L2 share (~667KB/core)
4. **N divisible by 192** - LCM(NR=64, cores=12)
5. **90.6% achieved** with M=504, N=768/core
6. **94.5% is L1-resident upper bound** (4% gap = memory latency)
