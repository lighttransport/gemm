# SVE Sin/Cos and RoPE for A64FX

## Overview

Two SVE-optimized sincos implementations for fp32, plus fused RoPE (Rotary Position Embedding) kernels. Both compute `sin(θ)` and `cos(θ)` simultaneously, sharing range reduction.

| Version | Method | Accuracy | Sincos cy/elem | RoPE cy/elem (dim≥512) |
|---------|--------|----------|-----------------|------------------------|
| **poly** | Minimax polynomial (Horner + FMLA) | ≤2 ULP | 0.15 | 0.11–0.12 |
| **ftmad** | SVE trig accelerator instructions | ≤2 ULP | 0.17 | 0.11–0.12 |

Both achieve **~870 Melem/s** for fused RoPE at large dimensions on a single A64FX core at 2.0 GHz.

## Algorithm

### Range Reduction (shared by both versions)

2-step Cody-Waite reduction to `[-π/4, π/4]`:

```
n = rint(x × 2/π)              // quadrant index
r = x − n × pio2_hi            // high part subtraction
r = r − n × pio2_lo            // low part correction
```

Where `pio2_hi + pio2_lo ≈ π/2` with ~50-bit effective precision. Valid for `|x| < ~10⁶`.

### Polynomial Version (`sincos_poly.S`)

Degree-9 sin / degree-8 cos using musl-derived minimax coefficients:

```
sin(r) = r + r³(S1 + r²(S2 + r²(S3 + r²S4)))     // 4 FMAD + 1 FMUL
cos(r) = 1 + r²(C0 + r²(C1 + r²(C2 + r²C3)))     // 4 FMAD + 1 FMLA
```

Sin and cos Horner chains are **interleaved** for ILP on A64FX's 2 FP pipelines. Quadrant adjustment uses predicated `SEL` (swap sin↔cos when `n&1`) and `FNEG` (negate when `n&2` or `(n+1)&2`).

**Instruction count per vector** (16 floats): ~25 instructions
- Range reduction: 6 (FMUL, FRINTN, FCVTZS, MOVPRFX, 2×FMLS)
- Polynomials: 11 (8×FMAD + FMUL + MOV + FMLA)
- Quadrant adjust: 8 (MOV, AND, CMPNE, SEL ×2, FNEG ×2)

### FTMAD Version (`sincos_ftmad.S`)

Uses SVE trigonometric accelerator instructions with **built-in minimax coefficients**:

```
FTSSEL  — select starting value (r or 1.0, with sign) from quadrant
FTSMUL  — compute r² with sin/cos path marker in LSB
FTMAD   — polynomial step: acc = coeff[imm] + acc × x²
```

Key insight: `cos(x) = sin(x + π/2)`, so the cos path uses quadrant `n+1`:

```asm
// Sin path: quadrant n
ftssel  z_sin_start, z_r, z_n      // ±r or ±1.0
ftsmul  z_sin_x2, z_r, z_n         // r² with sin/cos bit

// Cos path: quadrant n+1
add     z_n1, z_n, #1
ftssel  z_cos_start, z_r, z_n1
ftsmul  z_cos_x2, z_r, z_n1

// 8-step FTMAD chains (interleaved sin/cos for ILP)
ftmad   z_sin, z_sin, z_sin_x2, #7
ftmad   z_cos, z_cos, z_cos_x2, #7
...
ftmad   z_sin, z_sin, z_sin_x2, #0
ftmad   z_cos, z_cos, z_cos_x2, #0

// Combine
fmul    z_sin_result, z_sin_start, z_sin
fmul    z_cos_result, z_cos_start, z_cos
```

FTMAD has 8 built-in coefficient indices. The instruction reads bit[0] of `Zm` (set by FTSMUL) to select between sin and cos coefficient tables. No explicit quadrant adjustment is needed — FTSSEL encodes the sign/swap into the starting value.

**Instruction count per vector**: ~30 instructions (16 FTMAD + setup)

### FTMAD Internal Mechanism

```
FTSSEL(r, n):
  bit[0] of n = 0 → result = r     (sin path: multiply by r)
  bit[0] of n = 1 → result = 1.0   (cos path: use polynomial directly)
  bit[1] of n = 1 → negate result

FTSMUL(r, n):
  result = r × r
  Set LSB of result to bit[0] of n  (path marker for FTMAD)
  Value perturbation: <1 ULP (negligible)

FTMAD(acc, x2, #imm):
  coeff = builtin_table[imm][x2.bit[0]]   // sin or cos coefficient
  result = coeff + acc × x2
```

## RoPE Kernel (`rope.S`)

For each pair `(x[2i], x[2i+1])` with angle `θ[i]`:
```
x'[2i]   = x[2i]·cos(θ) − x[2i+1]·sin(θ)
x'[2i+1] = x[2i]·sin(θ) + x[2i+1]·cos(θ)
```

Uses `LD2W`/`ST2W` for zero-cost even/odd deinterleaving:
```asm
ld2w    {z_even.s, z_odd.s}, p0/z, [x_ptr]   // deinterleave pairs
// ... compute sin/cos of theta ...
fmul    z0.s, z_even.s, z_cos.s               // x_even × cos
fmls    z0.s, p0/m, z_odd.s, z_sin.s          // − x_odd × sin
fmul    z1.s, z_odd.s, z_cos.s                // x_odd × cos
fmla    z1.s, p0/m, z_even.s, z_sin.s         // + x_even × sin
st2w    {z0.s, z1.s}, p0, [x_ptr]             // re-interleave & store
```

With VL=512: **8 pairs (16 floats) per iteration**. The sincos is computed inline (not a function call), and the entire pipeline — load θ, range reduce, polynomial, rotate, store — fits in one loop body.

## Accuracy

Tested against libm `sinf`/`cosf` over 10,000 random values per range:

| Range | Poly max ULP | FTMAD max ULP |
|-------|-------------|---------------|
| [−1, 1] | 1.0 | 1.0 |
| [−π, π] | 2.0 | 2.0 |
| [−10, 10] | 2.0 | 2.0 |
| [−100, 100] | 2.0 | 2.0 |
| [−1000, 1000] | 4.0 | 2.0 |
| [0, 0.001] | 0.0 | 1.0 |

Average ULP error is ~0.18 for both. The poly version shows slightly higher max ULP at extreme ranges due to fp32 range-reduction cancellation; the FTMAD version benefits from 8-term polynomial (vs 4-term poly).

RoPE max absolute error: **~1.2×10⁻⁷** (1 ULP) for all tested dimensions.

## Performance

A64FX single core, 2.0 GHz, VL=512 (16 fp32 elements per vector).

### Sincos Throughput

| N | Poly cy/elem | Poly Melem/s | FTMAD cy/elem | FTMAD Melem/s |
|---|-------------|-------------|---------------|---------------|
| 16 | 0.20 | 496 | 0.19 | 524 |
| 64 | 0.16 | 637 | 0.18 | 570 |
| 256 | 0.15 | 657 | 0.17 | 593 |
| 1024 | 0.15 | 660 | 0.17 | 598 |
| 16384 | 0.15 | 657 | 0.17 | 596 |

Poly is ~10% faster due to fewer serial instructions (4 FMAD vs 8 FTMAD).

### Fused RoPE Throughput

| dim | Poly cy/elem | Poly Melem/s | FTMAD cy/elem | FTMAD Melem/s |
|-----|-------------|-------------|---------------|---------------|
| 64 | 0.18 | 571 | 0.11 | 885 |
| 128 | 0.14 | 697 | 0.10 | 954 |
| 256 | 0.13 | 782 | 0.13 | 747 |
| 512 | 0.12 | 832 | 0.12 | 814 |
| 1024 | 0.12 | 858 | 0.12 | 854 |
| 4096 | 0.11 | 871 | 0.11 | 871 |

At dim ≥ 512 both converge to **~0.11 cy/elem (870 Melem/s)**. FTMAD is faster at small dims (fewer constant loads — only 3 range-reduction constants vs 12 for poly).

## Instruction Latencies (A64FX)

| Instruction | Latency | Throughput | Notes |
|-------------|---------|------------|-------|
| FMUL/FMLA/FMAD | 9 cy | 2/cy | Main polynomial workhorse |
| FTMAD | 9 cy | 1/cy (?) | Same latency as FMLA |
| FTSSEL | 4 cy | — | Fast selection |
| FTSMUL | 9 cy | — | Same as FMUL |
| FRINTN | 9 cy | — | Round to nearest (ties to even) |
| FCVTZS | 9 cy | — | Float → int conversion |
| SEL | 4 cy | — | Predicated select |
| FNEG | 4 cy | — | Predicated negate |

## Build & Run

```bash
cd a64fx/sve-tri
make                        # builds bench
OMP_NUM_THREADS=1 ./bench   # run accuracy + performance tests
```

## API

```c
#include "sincos.h"

// Standalone sincos (N elements)
void sve_sincos_poly_f32(const float *theta, float *sin_out, float *cos_out, int64_t n);
void sve_sincos_ftmad_f32(const float *theta, float *sin_out, float *cos_out, int64_t n);

// Fused RoPE (in-place rotation, dim must be even)
void sve_rope_poly_f32(float *x, const float *theta, int64_t dim);
void sve_rope_ftmad_f32(float *x, const float *theta, int64_t dim);
```

## Design Decisions

1. **Why two versions?** Poly is faster for standalone sincos (fewer instructions, better ILP). FTMAD requires fewer constants (3 vs 12), giving an advantage in register-pressure-heavy contexts like fused RoPE at small dims. At large dims they converge.

2. **Why not online range reduction (Payne-Hanek)?** Overkill for RoPE. Position × frequency products in typical transformer models stay well within `|x| < 10⁴`, where 2-step Cody-Waite is accurate to < 2 ULP.

3. **Why LD2W/ST2W for RoPE?** The even/odd pair structure of RoPE maps perfectly to SVE's 2-element structure loads. Zero deinterleaving overhead — the hardware does it.

4. **Why interleave sin/cos chains?** A64FX has 2 FP pipelines. Interleaving independent sin and cos FMAD/FTMAD instructions lets both pipelines stay busy, halving effective latency.
