# FP16 and FP8→FP16 GEMM on A64FX

## Summary

This document compares three GEMM approaches on A64FX:
1. **Pure FP32 GEMM** (baseline)
2. **FP16→FP32 GEMM** (FP16 input with FP32 accumulation)
3. **FP8→FP16→FP32 GEMM** (FP8 stored as FP16 intermediate)

## Configuration

- **Processor**: A64FX (Fugaku)
- **Peak FP32**: 128 GFLOPS
- **Test Matrix**: M=384, K=512, N=48 (18.9M FLOPs)
- **Tile**: 8×3 (MR=8, NR=3)
- **Ideal**: 14,745 ticks/GEMM

## Results Summary

| Approach | Ticks | Efficiency | Notes |
|----------|-------|------------|-------|
| Pure FP32 | 13,948 | **105%** | Baseline (data pre-converted) |
| FP16→FP32 (pre-packed) | 23,950 | **61%** | FCVT overhead in kernel |
| FP16→FP32 (convert per-panel) | 41,286 | **35%** | Includes packing cost |
| **FP8→FP32** | **19,638** | **75%** | SVE direct conversion |
| FP8→FP16→FP32 | 53,388 | **27%** | Slow conversion + slow kernel |

## Key Findings

### 1. FP16→FP32 Kernel Performance

The `micro_kernel_fp16fp32_8x3` kernel uses `ld1rh` + `fcvt` to convert FP16 to FP32 on-the-fly:

```assembly
ld1rh {z27.s}, p0/z, [x0, #0]   // Load FP16, broadcast to FP32 positions
fcvt z27.s, p0/m, z27.h         // Convert FP16 → FP32
fmla z0.s, p0/m, z27.s, z24.s   // Accumulate
```

**Performance**: 23,950 ticks (61% efficiency)
- FCVT adds ~10,000 ticks overhead (71% of kernel time)
- **1.7× slower than pure FP32 kernel**

### 2. FP16→FP32 Conversion Performance

| Method | Ticks | Cycles/elem |
|--------|-------|-------------|
| Scalar (flat array) | 2,780 | 0.28 |
| SVE (flat array) | 2,773 | 0.28 |
| Scalar (with packing) | 17,820 | 1.81 |
| **SVE (with packing)** | **4,731** | **0.48** |

**SVE is 3.8× faster** for packed conversion due to vectorized loads.

### 3. FP8→FP16 Conversion Performance

| Method | Ticks | Cycles/elem |
|--------|-------|-------------|
| FP8→FP16 (scalar, flat) | 6,244 | 0.64 |
| FP8→FP16 (packed) | 29,435 | 2.99 |
| **FP8→FP32 (SVE direct, packed)** | **5,455** | **0.55** |

**Finding**: FP8→FP32 is **5.4× faster** than FP8→FP16 with packing.

**Why FP8→FP16 is slow:**
- LUT gather for FP16 requires complex indexing
- Packing fp16_t[K][M] layout has poor cache behavior
- FP16 LUT (512 bytes) less efficient than FP32 direct bitwise ops

### 4. Full GEMM Comparison

```
FP8 → FP32 → FP32 GEMM:     19,638 ticks (75% eff) ✓ BEST
FP8 → FP16 → FP16→FP32 GEMM: 53,388 ticks (27% eff) ✗ SLOW
```

The FP8→FP16 path is **2.7× slower** due to:
1. Slower conversion (29,435 vs 5,455 ticks)
2. Slower kernel (23,919 vs 13,927 ticks)

## Storage Comparison

| Format | A Matrix Size | B Matrix Size | Total |
|--------|---------------|---------------|-------|
| FP8 | 192 KB | 24 KB | 216 KB |
| FP16 | 384 KB | 48 KB | 432 KB |
| FP32 | 768 KB | 96 KB | 864 KB |

**FP16 saves 50% storage vs FP32**, but this doesn't offset the performance loss.

## Optimization Attempts

### Tested: FP16→FP32 Kernel Optimization
- Tried different tile sizes
- Tried K-unrolling to hide FCVT latency
- **Result**: FCVT overhead is inherent, cannot be hidden

### Tested: FP8→FP16 Conversion Optimization
- Tried SVE gather (complex, slow for FP16)
- Tried direct bitwise (FP16 format more complex than FP8→FP32)
- **Result**: FP8→FP32 direct bitwise is simpler and faster

## Recommendations

### For FP16 Input Data
```
Scenario: A and B are stored in FP16

Option 1: Convert to FP32, use FP32 kernel
  - Best performance: 105% efficiency (pure GEMM)
  - Conversion cost: 4,731 ticks (34% of GEMM)
  - Total: ~78% efficiency with reuse

Option 2: Use FP16→FP32 kernel directly
  - Performance: 61% efficiency
  - No pre-conversion needed
  - Good for one-time use

Recommendation: Pre-convert to FP32 for best performance
```

### For FP8 Input Data
```
Best path: FP8 → FP32 (direct) → FP32 GEMM

DO NOT use: FP8 → FP16 → FP16→FP32 GEMM
  - 2.7× slower
  - More complex
  - No benefit from FP16 intermediate
```

## Technical Details

### Why FP8→FP32 is Fast

Direct bitwise conversion without LUT:
```c
// FP8 E4M3: 1 sign + 4 exp (bias=7) + 3 mant
// FP32:     1 sign + 8 exp (bias=127) + 23 mant

sign = (fp8 & 0x80) << 24
exp  = ((fp8 & 0x78) + (120 << 3)) << 20
mant = (fp8 & 0x7) << 20
result = sign | exp | mant
```

**Advantages:**
- Simple ALU operations (1-2 cycle latency)
- No memory access (no LUT gather)
- Vectorizes well with SVE

### Why FP8→FP16 is Slow

FP16 format is more complex:
```
FP16: 1 sign + 5 exp (bias=15) + 10 mant
```

Problems:
- Need to handle exp conversion: (fp8_exp - 7) + 15
- Need to handle denormals differently
- Simpler to use LUT, but gather is expensive
- Direct bitwise not as efficient as FP8→FP32

## Conclusion

**For A64FX:**

1. **FP32 kernel is optimal** (105% efficiency)
2. **FP16→FP32 kernel acceptable** (61% efficiency) if input is FP16
3. **FP8→FP32 path recommended** (75% efficiency single-use)
4. **FP8→FP16 path NOT recommended** (27% efficiency)

**Key Insight**: A64FX lacks native FP16 compute. FCVT in the kernel adds 71% overhead. Pre-converting to FP32 and using the FP32 kernel is always faster than using the FP16→FP32 kernel.

For FP8 data, direct FP8→FP32 conversion is simpler and faster than FP8→FP16→FP32.

## Files

```
fp16-gemm/
├── bench_fp16_gemm.c          # FP16→FP32 kernel comparison
├── bench_fp16_opt.c           # SVE conversion optimization
├── bench_fp8_fp16.c           # FP8→FP16 path analysis
└── FP16_FP8_GEMM_SUMMARY.md   # This document
```

## References

- FP32 8×3 kernel: `../fp8-conv/fp8_kernel_asm.S`
- FP16→FP32 8×3 kernel: `../int8-cmg/micro_kernel_fp16fp32_8x3.S`
- FP8→FP32 optimization: `../fp8-conv/FP8_GEMM_OPTIMIZATION.md`
