# FP8 GEMM Optimization on A64FX

## Summary

This document summarizes the optimization work for FP8 (E4M3 format) GEMM on Fujitsu A64FX processor, targeting 90%+ efficiency of peak FP32 performance.

## Configuration

- **Processor**: A64FX (Fugaku)
- **Clock**: 2.0 GHz
- **Peak FP32**: 128 GFLOPS (2 FMA pipes × 16 elements × 2 ops × 2 GHz)
- **SVE Vector Length**: 512 bits (16 × FP32)
- **Tile Size**: 8×3 (8 rows × 3 SVE vectors = 48 columns)
- **Test Matrix**: M=384, K=512, N=48 (18.9M FLOPs)

## Final Results

| Scenario | Ticks | Cycles | Efficiency |
|----------|-------|--------|------------|
| Pure FP32 GEMM | 15,766 | 315,320 | **93%** |
| Single-use FP8 (SVE direct) | 24,023 | 480,460 | **61%** |
| FP8 with reuse=3 | 18,466 | 369,320 | **79%** |
| FP8 with reuse=14+ | ~16,500 | ~330,000 | **90%+** |

## Optimization Journey

### Initial State (27% efficiency)
- Scalar LUT-based FP8→FP32 conversion
- 3.76 cycles/element conversion cost
- Conversion overhead: 2.36× GEMM time

### Final State (61% single-use, 93% pure GEMM)
- SVE direct bitwise conversion (no LUT)
- 0.82 cycles/element conversion cost
- Conversion overhead: 0.5× GEMM time

## Key Optimizations

### 1. SVE Direct Bitwise Conversion

Replaced LUT gather with direct bitwise operations:

```c
// FP8 E4M3: 1 sign + 4 exp (bias=7) + 3 mantissa
// FP32:     1 sign + 8 exp (bias=127) + 23 mantissa

void convert_sve_direct(const uint8_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();  // 16 for A64FX
    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);
        svuint32_t fp8 = svld1ub_u32(pg, src + i);

        // sign: (fp8 & 0x80) << 24
        svuint32_t sign = svlsl_x(pg, svand_x(pg, fp8, svdup_u32(0x80)), 24);

        // exp: ((fp8 & 0x78) + (120 << 3)) << 20
        svuint32_t exp = svlsl_x(pg,
            svadd_x(pg, svand_x(pg, fp8, svdup_u32(0x78)), svdup_u32(120 << 3)), 20);

        // mant: (fp8 & 0x7) << 20
        svuint32_t mant = svlsl_x(pg, svand_x(pg, fp8, svdup_u32(0x7)), 20);

        svuint32_t result = svorr_x(pg, sign, svorr_x(pg, exp, mant));

        // Handle zero case
        svbool_t is_zero = svcmpeq(pg, svand_x(pg, fp8, svdup_u32(0x78)), svdup_u32(0));
        result = svsel(is_zero, sign, result);

        svst1(pg, dst + i, svreinterpret_f32(result));
    }
}
```

**Why this is faster:**
- Eliminates 11-cycle gather latency from LUT lookup
- Uses simple ALU operations (1-2 cycle latency)
- Processes 16 elements per iteration

### 2. FP32 8×3 Micro-kernel

The kernel achieves 93% of peak with:
- 24 accumulators (z0-z23)
- K-unrolled by 2 for pipelining
- `ld1rw` for fast broadcast loads (vs slow `mov z.s, z.s[i]`)
- Prefetch hints for A and B

### 3. Conversion Approaches Tested

| Approach | Ticks/iter | Cycles/elem | Notes |
|----------|------------|-------------|-------|
| Scalar (original) | 36,977 | 3.76 | Baseline |
| Scalar (unrolled) | 22,021 | 2.24 | 1.7× faster |
| SVE gather (LUT) | 7,420 | 0.76 | 5× faster |
| SVE direct (bitwise) | 7,320 | 0.74 | Fastest, no LUT |

### 4. Rejected Approaches

**Fused Kernel (FP8 gather inside GEMM loop):**
- Tested `mov z.s, z.s[i]` broadcast: too slow (1.7× slower than FP32 kernel)
- Tested K-unrolling to hide gather latency: still slower
- Result: Pre-conversion beats fused approach

**FP16 Intermediate Path:**
- FP8→FP16→FP32 with FCVT in kernel
- A64FX lacks FP16 FMLA, so FCVT adds overhead
- Result: FP16 path achieves only 55% vs 93% for pure FP32

## Practical Recommendations

### For Flash Attention
```
Q, K, V: [seq_len, d_head] in FP8

1. Pre-convert Q, K, V from FP8 to FP32 (one-time cost)
2. Run attention computation in FP32:
   - scores = Q @ K^T (GEMM)
   - softmax(scores)
   - output = scores @ V (GEMM)

With reuse ~3 (Q,K,V each used twice), expect ~79% efficiency
```

### For Batch Inference
```
If same A matrix used across multiple B matrices:
- Pre-convert A once
- Run multiple GEMMs with pre-converted A
- With reuse >= 14, achieve 90%+ efficiency
```

### For Single-Use Workloads
```
Use SVE direct conversion + FP32 GEMM
- 61% efficiency (2.3× better than naive scalar approach)
- Consider if FP8 storage savings justify the 39% efficiency loss
```

## File Structure

```
fp8-conv/
├── bench_final_opt.c      # Final benchmark with SVE direct conversion
├── bench_summary2.c       # Summary benchmark (integer-only output)
├── bench_direct_conv.c    # Direct conversion comparison
├── fp8_kernel_asm.S       # FP32 8×3 micro-kernel
├── fp8_to_fp32_lut[]      # 256-entry LUT (for reference)
└── FP8_GEMM_OPTIMIZATION.md  # This document
```

## Conclusion

FP8 GEMM on A64FX achieves:
- **93% efficiency** for pure FP32 kernel (near-optimal)
- **61% efficiency** for single-use FP8 with SVE direct conversion
- **79-90%+ efficiency** when FP8 data can be reused multiple times

The main bottleneck is FP8→FP32 conversion. SVE direct bitwise conversion is the fastest approach, eliminating LUT gather latency. For workloads like Flash Attention where converted data is reused, FP8 GEMM is practical and efficient on A64FX.
