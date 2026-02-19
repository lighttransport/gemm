# AMD Zen2 (Ryzen 9 3950X) Optimized Neural Network Kernels

AVX2 + FMA3 optimized implementations for AMD Zen2 architecture:
- **GEMM**: Single-precision matrix multiplication
- **Flash Attention**: Memory-efficient attention with online softmax
- **RMSNorm**: Root Mean Square Layer Normalization
- **LayerNorm**: Layer Normalization
- **FFN**: Feed-Forward Network with SwiGLU/GELU/SiLU activations

## Target Architecture

**AMD Zen2 (Ryzen 9 3950X)**
- 256-bit AVX2 vectors (8 FP32 per YMM register)
- FMA3 fused multiply-add instructions
- 16 YMM vector registers
- 2x 256-bit FMA execution units per core
- L1D: 32KB, L2: 512KB per core

**Theoretical Peak Performance:**
- 32 FLOPs/cycle per core (2 FMA units x 8 FP32 x 2 ops)
- At 4.0 GHz: 128 GFLOPS single-threaded

## Optimization Strategy

### 6x16 Micro-kernel
The innermost computation uses a 6x16 tile:
- 6 rows of A (broadcast to all lanes)
- 16 columns of B (2 YMM registers)
- 12 YMM registers for accumulating C (6 rows x 2 registers)
- 2 YMM registers for loading B
- 2 YMM registers for broadcasting A

### Cache Blocking
Three-level blocking for cache efficiency:
- MC = 72 (L1 fit for A panel)
- NC = 256 (L2 fit for B panel)
- KC = 256 (shared dimension block)

### Key Optimizations
1. **FMA3 instructions** for fused multiply-add
2. **4x loop unrolling** for instruction-level parallelism
3. **Data packing** for contiguous memory access
4. **Prefetching** to hide memory latency
5. **Register blocking** to minimize loads/stores

## Flash Attention

### Algorithm
2-pass memory-efficient attention (Flash Attention style):
1. **Pass 1**: `S = Q @ K^T` with row max tracking
2. **Pass 2**: `O = softmax(S) @ V` with online normalization

### Tile Parameters
```
BR = 4   (query block rows)
BC = 64  (key/value block columns)
D  = 64  (head dimension)
```

### Key Optimizations
1. **AVX2 vectorized exp** using polynomial approximation (< 1e-6 relative error)
2. **Online softmax** for numerical stability and memory efficiency
3. **FMA3** for fused multiply-add in attention score computation
4. **Multi-block processing** with running max/sum for long sequences

### FLOPs per Tile
- S = Q @ K^T: 4 × 64 × 64 × 2 = 32,768
- O = P @ V: 4 × 64 × 64 × 2 = 32,768
- exp + softmax: 4 × 64 × 8 = 2,048
- **Total: ~67,584 FLOPs**

## Neural Network Operations

### RMSNorm
Root Mean Square Layer Normalization (used in LLaMA, Mistral, etc.):
```
output[i] = input[i] * gamma[i] / sqrt(mean(input^2) + eps)
```

### LayerNorm
Standard Layer Normalization:
```
output[i] = (input[i] - mean) * gamma[i] / sqrt(var + eps) + beta[i]
```

### Activation Functions
- **GELU**: `0.5 * x * (1 + erf(x / sqrt(2)))`
- **GELU (tanh)**: Fast approximation using tanh
- **SiLU (Swish)**: `x * sigmoid(x)`
- **QuickGELU**: `x * sigmoid(1.702 * x)`
- **ReLU**: `max(0, x)`

### FFN Modules
- **Standard FFN**: `W2 @ activation(W1 @ x + b1) + b2`
- **SwiGLU FFN**: `W_down @ (SiLU(W_gate @ x) * (W_up @ x))`

### Optimizations
1. **Fast rsqrt** using `_mm256_rsqrt_ps` + Newton-Raphson refinement
2. **Vectorized exp/sigmoid/tanh** using polynomial approximations
3. **4x loop unrolling** for memory-bound operations
4. **FMA3** for all multiply-add operations

## Building

```bash
# Build all (GEMM + Flash Attention + NN Ops)
make

# Build with Clang
make COMPILER=clang

# Debug build with sanitizers
make DEBUG=1

# Build with profiling
make PROFILE=1
```

## Running Benchmarks

```bash
# Run GEMM benchmark
make gemm

# Run Flash Attention benchmark
make flash

# Run NN Ops benchmark (RMSNorm, LayerNorm, FFN, Activations)
make nn

# GEMM with specific size (M=N=K=1024)
make run-1024

# Custom GEMM size: ./bench_gemm M N K
./bench_gemm 768 768 768
```

## Output Example

```
================================================================================
                    AMD Zen2 GEMM Benchmark (AVX2 + FMA3)
================================================================================

CPU Information:
  Detected TSC frequency: 3.80 GHz
  Theoretical peak (1 core): 121.6 GFLOPS

Benchmark results:
--------------------------------------------------------------------------------
Kernel                M       N       K       Time        GFLOPS        Eff      Cyc/FMA
--------------------------------------------------------------------------------
sgemm_avx2            64      64      64     0.005 ms      109.47 GFLOPS ( 90.0%)  0.069 cycles/FMA
sgemm_avx2           256     256     256     0.398 ms       84.24 GFLOPS ( 69.3%)  0.090 cycles/FMA
sgemm_avx2          1024    1024    1024     8.724 ms       96.02 GFLOPS ( 79.0%)  0.079 cycles/FMA
sgemm_avx2          2048    2048    2048    68.452 ms       94.52 GFLOPS ( 77.7%)  0.080 cycles/FMA
```

## Analysis Tools

```bash
# Generate full disassembly
make disasm

# View just the micro-kernel assembly
make kernel-asm
```

## Files

| File | Description |
|------|-------------|
| `gemm_avx2.h/c` | GEMM implementation |
| `bench_gemm.c` | GEMM benchmark |
| `flash_attn_avx2.h/c` | Flash Attention implementation |
| `bench_flash_attn.c` | Flash Attention benchmark |
| `nn_ops_avx2.h/c` | RMSNorm, LayerNorm, FFN, Activations |
| `bench_nn_ops.c` | NN Ops benchmark |
| `Makefile` | Build configuration |

## API

```c
// Optimized SGEMM: C = alpha * A * B + beta * C
void sgemm_avx2(
    size_t M, size_t N, size_t K,
    float alpha,
    const float *A, size_t lda,  // A[M x K], row-major
    const float *B, size_t ldb,  // B[K x N], row-major
    float beta,
    float *C, size_t ldc         // C[M x N], row-major
);

// Naive reference for verification
void sgemm_naive(
    size_t M, size_t N, size_t K,
    float alpha,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    float beta,
    float *C, size_t ldc
);

// Flash Attention for single tile
void flash_attention_tile_avx2(
    const float *Q,         // [BR, D]
    const float *K,         // [BC, D]
    const float *V,         // [BC, D]
    float *O,               // [BR, D]
    float *S_scratch,       // [BR, BC]
    float *m,               // [BR]
    float *l                // [BR]
);

// Flash Attention for variable sequence length
void flash_attention_avx2(
    const float *Q,         // [BR, D]
    const float *K,         // [seq_len, D]
    const float *V,         // [seq_len, D]
    float *O,               // [BR, D]
    size_t seq_len
);

// RMSNorm
void rmsnorm_f32_avx2(
    const float *input, const float *gamma,
    float *output, size_t dim, float eps
);

// LayerNorm
void layernorm_f32_avx2(
    const float *input, const float *gamma, const float *beta,
    float *output, size_t dim, float eps
);

// Activations
void gelu_f32_avx2(const float *input, float *output, size_t n);
void silu_f32_avx2(const float *input, float *output, size_t n);
void relu_f32_avx2(const float *input, float *output, size_t n);

// FFN with SwiGLU
void ffn_swiglu_f32_avx2(
    const float *input, const float *W_gate, const float *W_up,
    const float *W_down, float *output, float *gate, float *up,
    size_t batch, size_t in_dim, size_t hidden_dim, size_t out_dim
);
```

## Performance Notes

- **Best case:** Small matrices that fit in L1/L2 cache achieve ~90% efficiency
- **Large matrices:** 75-80% efficiency due to memory bandwidth
- The benchmark pins execution to CPU 0 for consistent timing
- TSC-based cycle counting provides accurate per-FMA timing

## Comparison with Other Implementations

| Implementation | Target | Vector Width | Expected Efficiency |
|---------------|--------|--------------|---------------------|
| This (AVX2) | Zen2 | 256-bit | 75-90% |
| OpenBLAS | Generic x86 | 256/512-bit | 80-95% |
| Intel MKL | Intel | 256/512-bit | 85-98% |
| a64fx (SVE) | A64FX | 512-bit | 40-60% |

## Extending

To add AVX-512 support (Zen4+), you would need to:
1. Create `gemm_avx512.c` with 512-bit ZMM registers
2. Use a larger tile (e.g., 8x32 or 6x48)
3. Add `-mavx512f -mavx512vl` flags
4. Update peak calculation (64 FLOPs/cycle per core)
