# Two-Pass Fused exp2 + GEMM Analysis

## User's Proposal

**Pass 1**: Fused exp2 + GEMM(partial V) + store exp2 to L1 → ~70% efficiency
**Pass 2**: Load cached exp2 + GEMM(remaining V) → ~80% efficiency
**Combined**: ~75% efficiency

## Actual Results

### M=4, Nc=64, D=128 (2 V tiles)

| Approach | Cycles | GFLOPS | % Peak |
|----------|--------|--------|--------|
| Pass1 (K-blocked fused) | 1371 | 47.8 | 37.3% |
| Pass2 (load P + GEMM) | 647 | 101.3 | **79.1%** |
| Combined two-pass | 2152 | 60.9 | 47.6% |
| **Baseline (separate)** | 1765 | 74.3 | **58.0%** |
| Pure GEMM | 1287 | 101.9 | 79.6% |

### Analysis

**Pass2 achieves 79% (as expected)** - P is in L1 cache, broadcast loads are efficient.

**Pass1 only achieves 37% (not 70%)** - The bottleneck is:

1. **Per-K exp2 uses only 4 elements** (one per row)
   - FEXPA processes 16 elements, but we only use 4 (25% utilization)
   - K-blocking helps but requires DUP to extract elements (expensive)

2. **Instruction count per K-block of 16:**
   ```
   exp2: 4 rows × 14 instructions = 56 instructions
   DUP:  16 K × 4 DUP = 64 instructions (to broadcast exp2)
   GEMM: 16 K × 20 (load+FMLA) = 320 instructions
   Total: 440 instructions / 16 K = 27.5 instructions per K
   ```
   vs pure GEMM: ~24 instructions per K

3. **DUP z.s, z.s[n] latency** - Each extraction adds ~1 cycle dependency

## Why Baseline Wins

The baseline approach:
```
exp2_rows(S, P, ...)     // Vectorized: 16 elements at once
gemm_fp32_4x4(P, V1, O1) // P in L1, broadcast loads
gemm_fp32_4x4(P, V2, O2) // P still in L1 (reused)
```

Benefits:
1. **exp2 fully vectorized**: Processes 16 elements per FEXPA instruction
2. **P naturally stays in L1**: 4×64×4 = 1KB (L1 = 64KB)
3. **GEMM runs at full efficiency**: No exp2 interleaving

## Achievable Performance

| Config | Baseline | Pure GEMM | exp2 Overhead |
|--------|----------|-----------|---------------|
| M=4, Nc=64, D=128 | 58% | 80% | 47% of GEMM |
| M=4, Nc=128, D=128 | 34% | 45% | 58% of GEMM |

**Conclusion**: The baseline two-pass (vectorized exp2 + GEMM) is optimal.
The fused approach cannot achieve 70% due to FEXPA underutilization and DUP overhead.

## Recommendation

Use the simple two-pass approach:
```c
// Pass 1: Vectorized exp2 (16 elements at a time)
exp2_rows(S, P, M, Nc, scale, max_val, ld_s, ld_p);

// Pass 2+: GEMM with P from L1 cache
for (int tile = 0; tile < n_tiles; tile++) {
    gemm_fp32_4x4(P, V + tile*D_tile, O + tile*D_tile, ...);
}
```

Expected efficiency: **55-60% of peak** for typical attention tile sizes.

To achieve higher efficiency would require:
- Larger M (more rows) to better utilize FEXPA
- Hardware support for faster exp2
- Different algorithm (e.g., FlashAttention tiling to keep data in L1)
