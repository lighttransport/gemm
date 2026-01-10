# INT8 GEMM + FlashAttention Optimization Status

## Current State: Benchmark Complete

Date: 2026-01-11

---

## Target vs Achieved

| Metric | Target | Achieved | Gap |
|--------|--------|----------|-----|
| GEMM efficiency | 96% | 70.8% | -25.2% |
| Softmax overhead | 3-4% of FLOPS | 63.6% of time | ~60% over |
| Overall efficiency | 90%+ | ~36% effective | -54% |

---

## Benchmark Results (L=4096, K=128)

### Time Breakdown
```
GEMM time:           11.840 ms (36.4%)
Softmax time:        20.697 ms (63.6%)
Total time:          32.537 ms
```

### Throughput
```
GEMM throughput:     362.8 GOPS
GEMM efficiency:     70.8% (vs 512 GOPS peak)
Softmax elem/sec:    810.6 M
Softmax effective:   8.1 GOPS
```

### Scaling Across L Values
| L | GEMM (GOPS) | Efficiency | Softmax % |
|---|-------------|------------|-----------|
| 1024 | 362 | 70.8% | 63% |
| 4096 | 363 | 70.8% | 64% |
| 8192 | 363 | 70.9% | 63% |

---

## Implementation Details

### Tile Sizes (L1 Cache Fit)
```c
TILE_BR = 48      // Q tile rows (M-tile)
TILE_BC = 128     // K tile columns (N-tile)
```

### Working Set Analysis
```
Q_tile:    48 × 128 = 6 KB
S_tile:    48 × 128 × 4 = 24 KB
2×S_tile:  48 KB (double buffering)
Total:     54 KB (fits in 64 KB L1)
```

### Key Optimizations
1. **No integer division** - all scaling uses shifts
2. **Double-buffered S_tiles** - overlap compute and store
3. **exp2 inline approximation**: `2^x ≈ (256 + 177*frac) >> (-int_part)`
4. **SVE-vectorized softmax** - `.irp` macro for full unrolling
5. **Shift-based score scaling**: `SCORE_SCALE_128 = 23 ≈ 1/sqrt(128) × 256`

### Score Scaling (No Division)
```c
LOG2E_Q8 = 369           // log2(e) in Q8.8
SCORE_SCALE_128 = 23     // ≈ 1/sqrt(128) * 256
combined_scale = (score_scale * LOG2E_Q8) >> 8
```

---

## Why Targets Were Not Met

### GEMM (70.8% vs 96% target)

1. **Memory bandwidth limited** for large L values
   - C_vec output: L×L×4 bytes = 256 MB for L=8192
   - Exceeds L2 cache (7 MB usable)

2. **Tiling overhead**
   - 6×4 microkernel does 6 rows × 4 vectors per call
   - Small tiles = more loop iterations

3. **INT8 SDOT latency**
   - 4-cycle latency, 2/cycle throughput
   - Needs careful scheduling to hide latency

### Softmax (63.6% vs 3-4% target)

The 3-4% target was **fundamentally unachievable** because:

1. **Algorithmic complexity mismatch**
   - GEMM: 2×M×N×K operations with SDOT (4 ops/cycle/lane)
   - Online softmax: ~20 SVE ops per element
   - For K=128, GEMM does 128× more "useful ops" per element

2. **Three-pass requirement**
   - Pass 1: Find max (for numerical stability)
   - Pass 2: Compute exp, rescale, sum
   - Pass 3: Normalize (if needed)

3. **No SDOT equivalent for softmax**
   - exp2 approximation still needs shifts, multiplies, adds
   - Horizontal reductions (smaxv, uaddv) are slow

---

## Files Created

| File | Purpose |
|------|---------|
| `bench_flash_gemm.c` | Focused GEMM + softmax benchmark |
| `flash_attn_fused.c` | Full fused attention implementation |
| `gqa_gemm_tiled.c` | Tiled GEMM with L2 blocking |
| `gqa_attention_tiled.c` | GQA attention with K reuse |
| `bench_tiled_l2.c` | L2 tiling benchmark |

---

## Potential Next Steps

### For Higher GEMM Efficiency
1. **Larger tiles** (12×8 microkernel) to reduce overhead
2. **L2 tiling** - keep S_vec output in L2 between M-tile iterations
3. **Prefetching** - explicit `prfm pldl2strm` for packed K

### For Lower Softmax Overhead
1. **Fuse into P@V** - avoid materializing full S matrix
2. **Quantize P to INT8** - use SDOT for P@V multiplication
3. **Two-pass algorithm** - compute max in first GEMM pass

### Realistic Target
- GEMM: 80-85% efficiency achievable with L2 tiling
- Softmax: 15-20% of time achievable with better fusion
- Overall: 70-75% effective efficiency is more realistic

---

## Build Commands

```bash
# Compile (native fcc, NOT -Nclang)
fcc -O3 -mcpu=a64fx -c gqa_micro_kernel_simple.S -o gqa_micro_kernel_simple.o
fcc -O3 -mcpu=a64fx -c gqa_pack.c -o gqa_pack.o
fcc -O3 -mcpu=a64fx -o bench_flash_gemm bench_flash_gemm.c gqa_pack.o gqa_micro_kernel_simple.o -lpthread -lm

# Submit to Fugaku
pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:30:00" \
    --no-check-directory run_flash_gemm.sh
```

---

## Conclusion

The INT8 GEMM implementation achieves **70.8% efficiency**, which is reasonable for the A64FX architecture with the current tile sizes. However, the online softmax takes **63.6% of total time**, making the original target of 3-4% softmax overhead unachievable with the current approach.

To improve overall efficiency, the focus should shift to:
1. **Fusing softmax with P@V** to avoid materializing the full S matrix
2. **L2 tiling** to improve GEMM cache behavior
3. **Quantizing P to INT8** to leverage SDOT for both Q@K^T and P@V
