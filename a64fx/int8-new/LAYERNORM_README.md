# INT8/INT16 LayerNorm and RMSNorm Implementation

## Overview

This directory contains INT8 and INT16 implementations of LayerNorm and RMSNorm using ARM SVE intrinsics, optimized for the Fujitsu A64FX processor.

## Files

- **layernorm.h** - Public API header
- **layernorm_int8.c** - INT8 LayerNorm and RMSNorm implementations
- **layernorm_int16.c** - INT16 LayerNorm and RMSNorm implementations
- **test_layernorm.c** - Comprehensive test suite
- **Makefile.layernorm** - Build configuration
- **LAYERNORM_DESIGN.md** - Detailed design documentation

## API

### INT8 Functions

```c
// LayerNorm with affine parameters (gamma, beta)
void layernorm_int8(
    const int8_t* input,    // [N] input tensor
    int8_t* output,         // [N] output tensor
    const int8_t* gamma,    // [N] scale weights
    const int8_t* beta,     // [N] bias
    int32_t epsilon,        // Stability constant (Q8.24)
    float input_scale,      // Input quantization scale
    float output_scale,     // Output quantization scale
    size_t N                // Dimension
);

// LayerNorm without affine (simplified)
void layernorm_int8_noaffine(
    const int8_t* input,
    int8_t* output,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N
);

// RMSNorm with gamma
void rmsnorm_int8(
    const int8_t* input,    // [N] input tensor
    int8_t* output,         // [N] output tensor
    const int8_t* gamma,    // [N] scale weights
    int32_t epsilon,        // Stability constant (Q8.24)
    float input_scale,
    float output_scale,
    size_t N
);

// RMSNorm without gamma (simplified)
void rmsnorm_int8_noaffine(
    const int8_t* input,
    int8_t* output,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N
);
```

### INT16 Functions

Similar API with `int16_t` types and Q16.16 fixed-point format.

## Usage Example

```c
#include "layernorm.h"

// Prepare data
const size_t N = 512;
int8_t input[N], output[N], gamma[N], beta[N];

// Initialize input, gamma (scale), beta (bias)
for (size_t i = 0; i < N; i++) {
    input[i] = /* your INT8 data */;
    gamma[i] = 127; // ~1.0 scale
    beta[i] = 0;     // No bias
}

// Set epsilon (small constant for numerical stability)
int32_t epsilon = (int32_t)(1e-5f * (1 << 24)); // Q8.24 format

// Run LayerNorm
layernorm_int8(input, output, gamma, beta, epsilon,
               1.0f/127.0f, 1.0f/127.0f, N);

// Or run simpler RMSNorm (no mean subtraction, no beta)
rmsnorm_int8(input, output, gamma, epsilon,
             1.0f/127.0f, 1.0f/127.0f, N);
```

## Implementation Highlights

### Vectorization Strategy

- **INT8**: Process 64 elements per SVE iteration
- **INT16**: Process 32 elements per SVE iteration
- Accumulation in INT32/INT64 to prevent overflow
- SVE unpacking: INT8 → INT16 → INT32 → computation

### Fixed-Point Arithmetic

**INT8**:
- Input/output range: -128 to 127
- Gamma/beta: INT8 where 127 ≈ 1.0
- Internal: INT32 accumulation
- Epsilon: Q8.24 format

**INT16**:
- Input/output range: -32768 to 32767
- Gamma/beta: INT16 where 16384 ≈ 1.0 (Q14)
- Internal: INT64 accumulation
- Epsilon: Q16.16 format

### SVE Intrinsics Used

- `svld1_s8/s16` - Load vectors
- `svst1_s8/s16` - Store vectors
- `svunpklo/hi_s16/s32/s64` - Type extension
- `svadd/sub/mul_s32/s64_x` - Arithmetic
- `svaddv_s32/s64` - Horizontal reduction
- `svmax/min_n_s32_x` - Clamping
- `svasr_n_s32_x` - Fixed-point scaling

## Algorithm

### LayerNorm
```
1. Compute mean: μ = sum(x) / N
2. Compute variance: σ² = sum((x - μ)²) / N
3. Normalize: y = (x - μ) / sqrt(σ² + ε) * γ + β
```

### RMSNorm (Simpler)
```
1. Compute RMS: rms = sqrt(sum(x²) / N + ε)
2. Normalize: y = x / rms * γ
```

RMSNorm is ~33% faster (2 passes vs 3 passes).

## Performance

### Expected Performance (A64FX, N=512)

**INT8 LayerNorm**:
- ~1200-2400 cycles
- ~10-20 GB/s throughput

**INT8 RMSNorm**:
- ~800-1600 cycles (33% faster)
- ~15-25 GB/s throughput

**INT16**:
- ~2× slower than INT8 (fewer elements per iteration)

## Building

```bash
# Build test suite
make -f Makefile.layernorm

# Run tests
./test_layernorm

# Or use the run script (for batch job)
pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:05:00" \
    --no-check-directory run_layernorm_test.sh
```

## Status

⚠️ **Note**: Implementation is complete but has runtime issues that need debugging:
- Basic SVE operations verified working
- Full LayerNorm/RMSNorm logic needs refinement
- Memory access patterns may need adjustment
- Predicate usage may need review

The code compiles successfully and demonstrates the full algorithmic approach using SVE intrinsics. Additional debugging needed to resolve runtime issues.

## References

- LayerNorm paper: "Layer Normalization" (Ba et al., 2016)
- RMSNorm paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
- ARM SVE Programmer's Guide
- A64FX Microarchitecture Manual

## Future Work

1. Debug runtime issues with current implementation
2. Add integer inverse square root (replace floating-point)
3. Implement Welford's online variance algorithm (single-pass)
4. Add block processing for large N (>1024)
5. Benchmark and optimize performance
6. Add multi-threading support for batch processing

## License

This code is for research and educational purposes.
