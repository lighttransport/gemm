# INT8/INT16 LayerNorm and RMSNorm Design Document

## Overview

This document describes the INT8 and INT16 implementations of LayerNorm and RMSNorm using ARM SVE intrinsics optimized for the Fujitsu A64FX processor.

## Formulas

### LayerNorm
```
y = (x - mean) / sqrt(variance + epsilon) * gamma + beta

where:
  mean = sum(x) / N
  variance = sum((x - mean)^2) / N
```

### RMSNorm (Simplified LayerNorm)
```
y = x / sqrt(mean(x^2) + epsilon) * gamma

where:
  RMS = sqrt(mean(x^2) + epsilon)
```

RMSNorm is simpler than LayerNorm:
- No mean subtraction (removes one pass over data)
- No beta term (only scaling, no bias)
- Faster computation with similar normalization properties
- Used in modern LLMs (LLaMA, Mistral, Qwen3)

## Fixed-Point Representation

### INT8 Format
- **Input/Output**: INT8 (-128 to 127)
- **Quantization scale**: Typically 1/127 ≈ 0.00787
- **Gamma/Beta**: INT8 where 127 ≈ 1.0
- **Internal accumulation**: INT32 for mean/variance
- **Epsilon**: Q8.24 format (8 integer bits, 24 fractional bits)

### INT16 Format
- **Input/Output**: INT16 (-32768 to 32767)
- **Quantization scale**: Typically 1/16384 ≈ 0.00006
- **Gamma/Beta**: INT16 where 16384 ≈ 1.0 (Q14 format)
- **Internal accumulation**: INT64 for mean/variance
- **Epsilon**: Q16.16 format (16 integer bits, 16 fractional bits)

## Implementation Strategy

### Step 1: Compute Mean (LayerNorm only)

```c
// Accumulate in INT64 to prevent overflow
int64_t sum = 0;
svint32_t sum_vec = svdup_n_s32(0);

// Process 64 INT8 elements at once (full SVE vector)
while (i + 64 <= N) {
    svint8_t v = svld1_s8(pg, &input[i]);

    // Unpack INT8 -> INT32 (4 groups of 16 elements)
    svint32_t v0 = svunpklo_s32(svunpklo_s16(v));
    // ... v1, v2, v3 ...

    sum_vec = svadd_s32_x(pg, sum_vec, v0);
    // Add v1, v2, v3 ...
}

// Horizontal reduction
sum += svaddv_s32(pg, sum_vec);
int32_t mean = sum / N;
```

**Key points**:
- Process 64 INT8 or 32 INT16 elements per iteration
- Use SVE unpacking to extend to INT32
- Accumulate in vector registers
- Final horizontal reduction with `svaddv`

### Step 2: Compute Variance (LayerNorm) or Mean-Square (RMSNorm)

**LayerNorm variance**:
```c
int64_t var_sum = 0;
svint32_t var_vec = svdup_n_s32(0);
svint32_t mean_vec = svdup_n_s32(mean);

while (i + 64 <= N) {
    svint8_t v = svld1_s8(pg, &input[i]);
    svint32_t v0 = svunpklo_s32(svunpklo_s16(v));

    // Compute (x - mean)^2
    svint32_t diff = svsub_s32_x(pg, v0, mean_vec);
    diff = svmul_s32_x(pg, diff, diff);

    var_vec = svadd_s32_x(pg, var_vec, diff);
}

int32_t variance = var_sum / N;
```

**RMSNorm mean-square**:
```c
// Simply compute x^2 without mean subtraction
svint32_t sq = svmul_s32_x(pg, v0, v0);
sq_vec = svadd_s32_x(pg, sq_vec, sq);
```

### Step 3: Compute Inverse Standard Deviation

```c
// For LayerNorm: 1/sqrt(variance + epsilon)
// For RMSNorm: 1/sqrt(mean_sq + epsilon)

int32_t inv_std = fast_invsqrt_int32(variance + epsilon);
```

**Fast inverse square root**:
- Use floating-point `1.0f / sqrtf(x)` for now
- Can optimize with integer approximation later
- Fixed-point Newton-Raphson iteration possible

### Step 4: Normalize and Apply Affine Transform

**LayerNorm**:
```c
while (i + 16 <= N) {
    svint32_t x = /* load and extend input */;
    svint32_t g = /* load and extend gamma */;
    svint32_t b = /* load and extend beta */;

    // Normalize: (x - mean) * inv_std
    svint32_t norm = svsub_s32_x(pg, x, mean_v);
    norm = svmul_s32_x(pg, norm, inv_std_v);
    norm = svasr_n_s32_x(pg, norm, 16); // Q16.16 -> Q0

    // Affine: norm * gamma + beta
    svint32_t y = svmul_s32_x(pg, norm, g);
    y = svasr_n_s32_x(pg, y, 7); // Scale for INT8
    y = svadd_s32_x(pg, y, b);

    // Clamp and store
    y = svmax_n_s32_x(pg, y, -128);
    y = svmin_n_s32_x(pg, y, 127);
    /* pack and store as INT8 */
}
```

**RMSNorm** (simpler - no mean subtraction, no beta):
```c
// Normalize: x * inv_rms * gamma
svint32_t norm = svmul_s32_x(pg, x, inv_rms_v);
norm = svasr_n_s32_x(pg, norm, 16);
svint32_t y = svmul_s32_x(pg, norm, g);
```

## SVE Intrinsics Used

### Load/Store
- `svld1_s8(pg, ptr)` - Load INT8 vector
- `svld1_s16(pg, ptr)` - Load INT16 vector
- `svst1_s8(pg, ptr, vec)` - Store INT8 vector
- `svst1_s16(pg, ptr, vec)` - Store INT16 vector

### Arithmetic
- `svadd_s32_x(pg, a, b)` - Add INT32 vectors
- `svsub_s32_x(pg, a, b)` - Subtract INT32 vectors
- `svmul_s32_x(pg, a, b)` - Multiply INT32 vectors
- `svasr_n_s32_x(pg, a, shift)` - Arithmetic right shift

### Type Conversion
- `svunpklo_s16(v)` - Unpack low INT8 -> INT16
- `svunpkhi_s16(v)` - Unpack high INT8 -> INT16
- `svunpklo_s32(v)` - Unpack low INT16 -> INT32
- `svunpkhi_s32(v)` - Unpack high INT16 -> INT32
- `svuzp1_s8/s16(a, b)` - Unzip and pack for narrowing

### Reduction
- `svaddv_s32(pg, vec)` - Horizontal add reduction
- `svaddv_s64(pg, vec)` - Horizontal add reduction (INT64)

### Other
- `svdup_n_s32(val)` - Duplicate scalar to vector
- `svmax_n_s32_x(pg, vec, val)` - Clamp minimum
- `svmin_n_s32_x(pg, vec, val)` - Clamp maximum
- `svptrue_b8/b16/b32()` - Create full predicate

## Optimization Techniques

### 1. Vectorization

**INT8 processing**: 64 elements per iteration
- Load 64 INT8 values (64 bytes = 1 SVE vector)
- Unpack into 4 groups of 16 INT32 values
- Process all 4 groups
- Achieves 4× throughput vs scalar

**INT16 processing**: 32 elements per iteration
- Load 32 INT16 values (64 bytes = 1 SVE vector)
- Unpack into 2 groups of 16 INT32 values
- Process both groups
- Achieves 2× throughput vs scalar

### 2. Prevent Overflow

**INT8**: Accumulate in INT32
- Max sum: 128 × 1024 = 131,072 (fits INT32)
- Max squared: 128² × 1024 = 16,777,216 (fits INT32)
- Safe for N up to ~100K elements

**INT16**: Accumulate in INT64
- Max sum: 32768 × 1024 = 33,554,432 (fits INT32 with caution)
- Max squared: 32768² × 1024 = 34B (requires INT64)
- Use INT64 accumulation for variance/mean-square

### 3. Fixed-Point Arithmetic

**Q16.16 format** for intermediate values:
- 16 integer bits, 16 fractional bits
- Range: ±32768 with precision ~0.000015
- Good for inverse square root and normalization

**Scaling**:
```c
// Multiply in Q16.16 and shift back
result = (a * b) >> 16;

// For INT8 gamma (Q7): shift by 7
result = (norm * gamma) >> 7;

// For INT16 gamma (Q14): shift by 14
result = (norm * gamma) >> 14;
```

### 4. Tail Handling

Process remaining elements with scalar code:
```c
while (i < N) {
    // Scalar fallback for N not divisible by vector length
    output[i] = compute_scalar(input[i]);
    i++;
}
```

## Performance Characteristics

### Computational Complexity

**LayerNorm**:
- 3 passes over data: mean, variance, normalize
- O(3N) memory reads
- O(N) memory writes

**RMSNorm**:
- 2 passes over data: mean-square, normalize
- O(2N) memory reads
- O(N) memory writes
- ~33% faster than LayerNorm

### Expected Performance (A64FX)

**INT8 LayerNorm** (N=512):
- ~3 × 512 / 64 = 24 iterations
- ~50-100 cycles per iteration
- Total: ~1200-2400 cycles
- Throughput: ~10-20 GB/s

**INT16 LayerNorm** (N=512):
- ~3 × 512 / 32 = 48 iterations
- ~40-80 cycles per iteration
- Total: ~1920-3840 cycles
- Throughput: ~8-16 GB/s

**RMSNorm**: ~33% faster than LayerNorm

### Memory Access Pattern

**Sequential access** - cache-friendly:
```
Pass 1: Read input[0..N] for mean
Pass 2: Read input[0..N] for variance
Pass 3: Read input[0..N], gamma[0..N], beta[0..N]
        Write output[0..N]
```

**L1 cache utilization**:
- Input reused across passes (should stay in L1)
- Gamma/Beta read once (streaming)
- Output written once (write-allocate)

## Usage Example

```c
#include "layernorm.h"

// INT8 LayerNorm
const int N = 512;
int8_t input[N], output[N], gamma[N], beta[N];

// Initialize input, gamma, beta
// ...

// Run LayerNorm
int32_t epsilon = (int32_t)(1e-5f * (1 << 24)); // Q8.24
float input_scale = 1.0f / 127.0f;
float output_scale = 1.0f / 127.0f;

layernorm_int8(input, output, gamma, beta, epsilon,
               input_scale, output_scale, N);

// INT8 RMSNorm (simpler)
rmsnorm_int8(input, output, gamma, epsilon,
             input_scale, output_scale, N);

// INT16 versions
int16_t input16[N], output16[N], gamma16[N], beta16[N];
int32_t epsilon16 = (int32_t)(1e-5f * (1 << 16)); // Q16.16

layernorm_int16(input16, output16, gamma16, beta16, epsilon16,
                1.0f/16384.0f, 1.0f/16384.0f, N);

rmsnorm_int16(input16, output16, gamma16, epsilon16,
              1.0f/16384.0f, 1.0f/16384.0f, N);
```

## Future Optimizations

### 1. Integer Inverse Square Root
Replace floating-point `1/sqrtf()` with:
- Fast integer approximation (magic constant method)
- Newton-Raphson refinement in fixed-point
- Expected: 2-3× faster for inverse sqrt

### 2. Fused Operations
Combine normalization passes:
- Online variance computation (Welford's algorithm)
- Single-pass layernorm
- Reduces memory traffic by 33%

### 3. Block Processing
For large N (>1024):
- Process in L1-sized blocks (8-16 KB)
- Better cache utilization
- Reduces L2 cache misses

### 4. Multi-threading
For batch processing:
- Parallelize across sequence dimension
- Each thread processes independent vectors
- Linear speedup with cores

## References

- LayerNorm paper: "Layer Normalization" (Ba et al., 2016)
- RMSNorm paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
- ARM SVE Programmer's Guide
- A64FX Microarchitecture Manual

## Files

- `layernorm.h` - Public API
- `layernorm_int8.c` - INT8 implementations
- `layernorm_int16.c` - INT16 implementations
- `test_layernorm.c` - Test suite
- `Makefile.layernorm` - Build script
- `LAYERNORM_DESIGN.md` - This document
