# INT8 Feed-Forward Network (FFN) with 6-row Split Loading for A64FX

## Overview

This implements optimized INT8 FFN using the 6-row split loading technique discovered for fused attention, achieving better compute density to hide A64FX's 11-cycle load latency.

## Architecture

### Feed-Forward Network Structure

**Basic FFN:**
```
Input [M, D_in]
  → W1 @ [D_in, D_ff] → Activation
  → W2 @ [D_ff, D_out]
  → Output [M, D_out]
```

**Gated FFN (SwiGLU-style):**
```
Input [M, D_in]
  ├→ W_gate @ [D_in, D_ff] → gate_activation
  └→ W_up @ [D_in, D_ff] → up_activation
       → gate * up → W_down @ [D_ff, D_out] → Output [M, D_out]
```

### Dimensions Tested

- D_in = 512 (hidden dimension)
- D_ff = 2048 (4x expansion)
- D_out = 512
- Batch size M: divisible by 6

## 6-row Split Loading GEMM Kernel

Located in: `kernel_ffn_6row_gemm.S`

### Key Innovation: Split Q Loading

Based on the attention kernel optimization, processes 6 rows simultaneously:

**Register Allocation:**
- z0-z23: 24 accumulators (6 rows × 4 N-vectors)
- z24-z27: 4 B (weight) vectors
- z28-z31: A (input) vectors for rows 0-3
- z28-z29: Reused for rows 4-5

**Split Loading Pattern:**
```asm
// Load B[k, :, 0:3] - 4 weight vectors
ld1b    z24.b, p0/z, [x6]
ld1b    z25.b, p0/z, [x6, #1, MUL VL]
ld1b    z26.b, p0/z, [x6, #2, MUL VL]
ld1b    z27.b, p0/z, [x6, #3, MUL VL]

// Phase 1: Load A[0:3, k] and compute
ld1rw   {z28.s}, p1/z, [x7]         // Row 0
ld1rw   {z29.s}, p1/z, [x8]         // Row 1
ld1rw   {z30.s}, p1/z, [x8, #512]   // Row 2
ld1rw   {z31.s}, p1/z, [x8, #1024]  // Row 3

// 16 SDOTs for rows 0-3
sdot    z0.s, z24.b, z28.b
sdot    z1.s, z25.b, z28.b
...

// Phase 2: Reuse z28-z29 for A[4:5, k]
ld1rw   {z28.s}, p1/z, [x8, #1536]  // Row 4
ld1rw   {z29.s}, p1/z, [x8, #2048]  // Row 5

// 8 SDOTs for rows 4-5
sdot    z16.s, z24.b, z28.b
sdot    z17.s, z25.b, z28.b
...
```

### Performance Characteristics

**Compute Density:**
- 24 SDOTs per 10 loads = 2.40 SDOT/load
- Improvement over 5-row: 20 SDOTs / 9 loads = 2.22 SDOT/load
- Theoretical speedup: ~8%

**Latency Hiding:**
- 11-cycle load latency requires ~22 SDOTs at 2 SDOT/cycle
- 6-row with 24 SDOTs comes closer to hiding full latency

**Verification:**
- Test: `test_ffn_kernel.c`
- Result: PASSED - All 12,288 elements (6 × 2048) correct
- Each output element = 512 (sum of 512 products of 1×1)

## Activation Functions

### 1. Squared ReLU

```
y = max(0, x)²
```

**Implementation:**
- Zero negative values: `smax z.s, z.s, z_zero`
- Square: `x * x` with scaling
- Scale down by shift amount

**Use case:** More selective than ReLU, emphasizes large activations

### 2. Shift-GELU

Approximate GELU with shift:
```
GELU(x) ≈ x * sigmoid(1.702 * x)
Simplified: x * max(0, min(1, 0.5 + 0.25*x))
```

**Implementation:**
```c
sig_approx = (1 << 23) + (x >> 2);  // 0.5 + 0.25*x in Q8.24
sig_approx = clamp(sig_approx, 0, 1 << 24);
result = (x * sig_approx) >> 24;
```

**Use case:** Smooth non-linearity similar to GELU

### 3. SiLU (Swish)

```
SiLU(x) = x * sigmoid(x)
```

Similar implementation to Shift-GELU without the shift.

## Memory Layout

### Weight Packing for Efficient GEMM

**Original:** `W[K, N]` (row-major)

**Packed:** `W_packed[K/4, N, 4]` with 256-byte stride

This layout optimizes for:
- Contiguous 64-byte loads for B vectors
- Cache-friendly access pattern
- Aligns with SDOT instruction (dot product of 4 elements)

## File Structure

- `kernel_ffn_6row_gemm.S` - 6-row GEMM kernel with split loading
- `ffn_int8_6row.h` - FFN API definitions
- `ffn_int8_6row.c` - FFN implementation (basic + gated variants)
- `activation.c` - Activation functions (Squared ReLU, Shift-GELU, SiLU)
- `test_ffn_kernel.c` - Kernel correctness test
- `bench_ffn_6row.c` - Full FFN benchmark
- `run_test_ffn_kernel.sh` - Test script
- `run_ffn_6row.sh` - Benchmark script

## Usage Example

```c
// Basic FFN
const int M = 60;  // Batch size (must be divisible by 6)
int8_t* input = ...; // [M, 512]
int8_t* W1 = ...;    // [512, 2048]
int8_t* W2 = ...;    // [2048, 512]
int32_t* output = ...;

ffn_int8_forward_d512(input, W1, W2, output, M, ACT_SQUARED_RELU);
```

```c
// Gated FFN (SwiGLU-style)
ffn_int8_gated_d512(
    input, W_gate, W_up, W_down, output, M,
    ACT_SILU,          // Gate activation
    ACT_SQUARED_RELU   // Up activation
);
```

## Performance Analysis

### Theoretical Peak

- A64FX peak: 128 GOPS at 2 GHz (2 FPU × 2 SDOT/cycle × 16 lanes × 2 GHz)
- Target: >50% efficiency for compute-bound workloads

### Expected Performance

**6-row vs 5-row comparison:**
- Cycles per SDOT: 6-row should be ~6% better
- Memory bandwidth: Same (both saturate L1 cache bandwidth)
- Overall: ~7-8% speedup expected (matches attention kernel results)

### Optimization for 11-cycle Latency

The split loading technique is specifically tuned for A64FX's 11-cycle L1 load latency:

1. **Issue 4 loads** (B vectors) - 11 cycles each, overlapped
2. **Issue 4 loads** (A rows 0-3) - broadcast loads, fast
3. **Compute 16 SDOTs** - while loads complete
4. **Issue 2 loads** (A rows 4-5) - reusing registers
5. **Compute 8 SDOTs** - final computation

Total: 10 loads, 24 SDOTs = 2.4 SDOT/load ratio

## Future Optimizations

1. **SVE-optimized activations**: Rewrite activations in SVE assembly for better performance
2. **Fused kernels**: Fuse GEMM + activation into single kernel
3. **INT16 intermediate**: Use INT16 for better precision with minimal overhead
4. **8-row variant**: Experiment with 8 rows for even better compute density
5. **Prefetching**: Add software prefetch for next tile

## References

- Similar technique used in `kernel_fused_d512_6row.S` for attention
- D512_6ROW_KERNEL.md - Detailed analysis of 6-row attention kernel
- A64FX optimization guide: 11-cycle L1 load latency, 2 SDOT/cycle throughput
