# D=256 INT8 FFN with Qwen3-next SwiGLU Configuration

## Overview

Extends the 6-row split loading optimization to D=256 head dimension and implements Qwen3-next style SwiGLU (Swish-Gated Linear Unit) activation for modern LLM architectures.

## D=256 GEMM Kernel

**File:** `kernel_ffn_6row_gemm_d256.S`

### Configuration
- Input dimension: D = 256
- FFN expansion: D_ff = 1024 (4x)
- Output dimension: 256
- K groups: 64 (256/4)
- N chunks: 16 (1024/64)

### Register Allocation
Same as D=512 variant:
- z0-z23: 24 accumulators (6 rows × 4 N-vectors)
- z24-z27: 4 B (weight) vectors
- z28-z31: A (input) vectors for rows 0-3
- z28-z29: Reused for rows 4-5

### Performance Characteristics
- 24 SDOTs per 10 loads = 2.40 SDOT/load ratio
- 64 K-group iterations (vs 128 for D=512)
- Faster per-call execution due to fewer iterations
- Row stride: 4096 bytes (1024 elements × 4 bytes)

### Verification
- Test: `test_d256_kernel.c`
- Result: PASSED ✓ - All 6,144 elements correct (6 rows × 1024 columns)
- Each output element = 256 (sum of 256 products of 1×1)

## Qwen3-next SwiGLU Configuration

**Files:** `ffn_qwen3.h`, `ffn_qwen3.c`

### SwiGLU Activation

SwiGLU is a gated activation function combining SiLU (Swish) with gating:

```
FFN(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
```

Where:
- `SiLU(x) = x * sigmoid(x)` (Swish activation)
- `⊙` is element-wise multiplication
- Three weight matrices: W_gate, W_up, W_down

### Computational Flow

1. **Gate path**: `gate = input @ W_gate` → apply SiLU
2. **Up path**: `up = input @ W_up` (no activation or identity)
3. **Gating**: `hidden = SiLU(gate) * up` (element-wise)
4. **Down projection**: `output = hidden @ W_down`

### Qwen3-next Model Configurations

| Config | D_model | D_ff | Expansion | Model Size |
|--------|---------|------|-----------|------------|
| D=256  | 256     | 1024 | 4.0x      | Test/Small |
| D=512  | 512     | 2048 | 4.0x      | Test/Medium |
| Qwen3-0.5B | 896 | 4864 | 5.43x | 0.5B params |
| Qwen3-1.5B | 1536 | 8960 | 5.83x | 1.5B params |
| Qwen3-3B+ | 2048 | 10240 | 5.0x | 3B+ params |

### Implementation

**Optimized variants:**
- `qwen3_ffn_forward_d256()` - Uses `kernel_ffn_6row_gemm_d256`
- `qwen3_ffn_forward_d512()` - Uses `kernel_ffn_6row_gemm_d512`

**Generic variant:**
- `qwen3_ffn_forward_generic()` - Reference implementation for other sizes

### API Example

```c
#include "ffn_qwen3.h"

// Query configuration
const qwen3_ffn_config* config = qwen3_get_config(QWEN3_CONFIG_256);
printf("%s: D=%d, D_ff=%d, %.2fx expansion\n",
       config->name, config->D_model, config->D_ff, config->expansion);

// Run SwiGLU FFN
const int M = 60;  // Batch size (must be divisible by 6)
int8_t* input = ...;     // [M, 256]
int8_t* W_gate = ...;    // [256, 1024]
int8_t* W_up = ...;      // [256, 1024]
int8_t* W_down = ...;    // [1024, 256]
int32_t* output = ...;   // [M, 256]

qwen3_ffn_forward_d256(input, W_gate, W_up, W_down, output, M);
```

## Performance Comparison: D=256 vs D=512

### Compute Complexity

**D=256 (4x expansion):**
- Gate GEMM: 2 × M × 256 × 1024 FLOPs
- Up GEMM: 2 × M × 256 × 1024 FLOPs
- Down GEMM: 2 × M × 1024 × 256 FLOPs
- Total: 3 × 2 × M × 256 × 1024 = 1,572,864M FLOPs

**D=512 (4x expansion):**
- Total: 3 × 2 × M × 512 × 2048 = 6,291,456M FLOPs
- **4x more compute than D=256**

### Memory Traffic

**D=256:**
- Weights: (256×1024 + 256×1024 + 1024×256) = 786,432 bytes
- Activations per sample: 256 + 1024 + 256 = 1,536 bytes

**D=512:**
- Weights: (512×2048 + 512×2048 + 2048×512) = 3,145,728 bytes (4x)
- Activations per sample: 512 + 2048 + 512 = 3,072 bytes (2x)

### Kernel Iterations

| Dimension | K-groups | N-chunks | Total calls per 6 rows |
|-----------|----------|----------|------------------------|
| D=256     | 64       | 16       | 64 × 16 = 1,024        |
| D=512     | 128      | 32       | 128 × 32 = 4,096       |

D=512 requires **4x more kernel calls** per batch.

### Expected Efficiency

Both variants should achieve similar efficiency (~50-60%) because:
- Same SDOT/load ratio (2.40)
- Both optimized for 11-cycle latency
- Same register allocation strategy
- D=256 has better cache locality (smaller working set)
- D=512 has more opportunities for latency hiding (more iterations)

## Advantages of D=256

1. **Faster execution**: 4x fewer FLOPs and kernel calls
2. **Better cache utilization**: 4x smaller weight matrices
3. **Lower memory bandwidth**: Reduced data movement
4. **Suitable for**:
   - Edge devices with limited compute
   - Faster inference for small models
   - Multi-head attention with more heads, smaller D per head

## Advantages of D=512

1. **Higher model capacity**: Larger hidden dimension
2. **Better for large models**: Standard for 7B+ parameter models
3. **More compute-bound**: Better GPU/accelerator utilization
4. **Suitable for**:
   - Large language models
   - High-accuracy applications
   - Server-side inference

## SwiGLU vs Other Activations

### Comparison

| Activation | Formula | Compute Cost | Use Case |
|------------|---------|--------------|----------|
| ReLU | max(0, x) | Very low | Classic CNNs |
| GELU | 0.5x(1+erf(x/√2)) | Medium | BERT-style models |
| SiLU/Swish | x·sigmoid(x) | Medium | EfficientNet |
| **SwiGLU** | (SiLU(x@W₁) ⊙ (x@W₂)) @ W₃ | High | Modern LLMs |
| Squared ReLU | max(0,x)² | Low | Our custom |

### Why SwiGLU?

1. **Gating mechanism**: Allows network to control information flow
2. **Smooth gradients**: Better than ReLU for deep networks
3. **State-of-the-art**: Used in LLaMA, PaLM, Qwen3, etc.
4. **Empirically better**: Consistently outperforms in LLM training

## File Structure

```
a64fx/int8-new/
├── kernel_ffn_6row_gemm_d256.S      # D=256 optimized kernel
├── kernel_ffn_6row_gemm.S           # D=512 optimized kernel (previously created)
├── ffn_qwen3.h                      # Qwen3 FFN API
├── ffn_qwen3.c                      # Qwen3 FFN implementation
├── test_d256_kernel.c               # D=256 kernel test
├── test_qwen3_ffn.c                 # Qwen3 FFN test
├── bench_qwen3_ffn.c                # Qwen3 FFN benchmark
├── run_test_d256_kernel.sh          # Test script
├── run_test_qwen3.sh                # Qwen3 test script
├── run_bench_qwen3.sh               # Qwen3 benchmark script
└── D256_QWEN3_FFN.md                # This documentation
```

## Future Work

1. **Fix memory management**: Debug Qwen3 FFN test allocation issues
2. **Benchmark performance**: Run full D=256 vs D=512 comparison
3. **Other dimensions**: Add D=128, D=384, D=768 variants
4. **Fused kernels**: Combine GEMM + SiLU activation
5. **INT16 accumulation**: Higher precision intermediate values
6. **Multi-batch optimization**: Better utilize 6-row processing for larger batches

## References

- Shazeer, N. (2020). "GLU Variants Improve Transformer"
- Qwen3 technical report: Model architecture with SwiGLU
- A64FX optimization: 11-cycle L1 load latency, 2 SDOT/cycle
- Previous work: D512_6ROW_KERNEL.md, INT8_FFN_6ROW.md
