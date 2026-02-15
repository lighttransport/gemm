# INT8 FFN Activation Implementation for A64FX SVE

## Overview

This implementation provides high-performance activation functions for INT8 quantized LLM inference on A64FX. Key innovations include:

1. **Pure INT32 activation approximations** - no float conversions needed
2. **Piecewise linear sigmoid/SiLU/GELU** - avoids expensive division operations
3. **Stochastic rounding** with Philox RNG for unbiased quantization

## Performance Results (A64FX @ 2.0 GHz)

### SiLU Performance (1M elements)

| Method | Time (ms) | GOPS | vs FP32 |
|--------|-----------|------|---------|
| FP32 SiLU | 3.21 | 0.33 | 1.0x |
| INT32 SiLU (division) | 7.50 | 0.14 | 0.4x (slower!) |
| **INT32 SiLU fast** | 1.28 | 0.82 | **2.5x faster** |
| **Hard Swish** | 1.17 | 0.90 | **2.7x faster** |

### GELU Performance (1M elements)

| Method | Time (ms) | GOPS | vs FP32 |
|--------|-----------|------|---------|
| FP32 GELU | 3.96 | 0.26 | 1.0x |
| INT32 GELU (division) | 10.19 | 0.10 | 0.4x (slower!) |
| **INT32 GELU fast** | 2.15 | 0.49 | **1.9x faster** |

## Accuracy Results

### Sigmoid Approximation

| Method | Max Error | Avg Error |
|--------|-----------|-----------|
| Rational (div) | 0.0823 | 0.0597 |
| **Fast (no div)** | **0.0506** | **0.0139** |
| Hard sigmoid | 0.0692 | 0.0216 |

### SiLU Approximation

| Method | Max Error | Avg Error |
|--------|-----------|-----------|
| FP32 rational | 0.4418 | 0.2654 |
| INT32 div | 0.4418 | 0.2654 |
| **INT32 fast** | **0.1236** | **0.0380** |
| Hard swish | 0.1420 | 0.0462 |

### GELU Approximation

| Method | Max Error | Avg Error |
|--------|-----------|-----------|
| FP32 (sig div) | 0.2737 | 0.2071 |
| INT32 div | 0.4445 | 0.2421 |
| **INT32 fast** | **0.0824** | **0.0116** |

## Key Insight

The "fast" piecewise linear approximation is both **faster AND more accurate** than the rational approximation with division! This is because:

1. **No division** - A64FX integer division is very slow (~40 cycles)
2. **Better approximation** - Piecewise linear is closer to true sigmoid in the important range [-4, 4]
3. **Simpler code** - Just shifts, adds, and multiplies

## Implementation

### Fast Sigmoid (Q16 fixed-point)

```c
static inline int32_t sigmoid_q16_fast(int32_t x) {
    if (x >= 262144) return 65536;  // x >= 4 -> 1.0
    if (x <= -262144) return 0;      // x <= -4 -> 0.0

    int32_t abs_x = x >= 0 ? x : -x;
    if (abs_x <= 65536) {
        // |x| <= 1: 0.25*x + 0.5
        return 32768 + (x >> 2);
    } else {
        // 1 < |x| <= 4: piecewise linear
        int32_t linear = (int32_t)(((int64_t)x * 5461) >> 16);
        int32_t offset = x >= 0 ? 10923 : -10923;
        return 32768 + linear + offset;
    }
}
```

### Fast SiLU

```c
static inline int32_t silu_q16_fast(int32_t x) {
    int32_t sig = sigmoid_q16_fast(x);
    return (int32_t)(((int64_t)x * sig) >> 16);
}
```

## Files

- `activation_int32.h` - Core activation functions
- `ffn_int8_v4.h` - Complete FFN pipeline with activation paths
- `bench_activation.c` - Benchmark suite
- `silu_fast.h` - SVE-optimized FP32 SiLU
- `philox.h` - Philox RNG for stochastic rounding

## Usage Recommendations

1. **Use INT32 fast** for maximum performance and accuracy
2. **Use Hard Swish** when slightly lower accuracy is acceptable (fastest option)
3. **Use stochastic rounding** for quantization to preserve gradient information
4. **Avoid division-based** approximations (slower than FP32!)

## Stochastic Rounding

Stochastic rounding eliminates systematic bias in quantization:

- Deterministic: 0.5 always rounds to 1 (biased)
- Stochastic: 0.5 rounds to 0 or 1 with 50% probability (unbiased)

Test results (100,000 samples of 0.5):
- Deterministic sum: 100000 (100% bias)
- Stochastic sum: 50282 (expected ~50000)

## Integration with INT8 GEMM

The activation functions are designed to work seamlessly with INT8 SDOT GEMM:

```
INT8 GEMM (gate) ─┐
                  ├─> SiLU/GELU ─> Quantize ─> INT8 GEMM (down)
INT8 GEMM (up) ───┘
```

All operations stay in INT32 domain, avoiding expensive float conversions.
