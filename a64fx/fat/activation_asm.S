// activation_asm.S
// Hand-optimized SVE assembly for activation functions
// GELU (tanh approx), QuickGELU, SiLU
// Targeting A64FX with 512-bit SVE (VL=64 bytes)

.arch armv8.2-a+sve

//=============================================================================
// Constants (loaded via literal pool)
//=============================================================================

.section .rodata
.align 4

// exp() constants - FP32
const_log2e_f32:      .float 1.4426950408889634
const_ln2_hi_f32:     .float 0.6931471805599453
const_exp_clamp_lo:   .float -88.0
const_exp_clamp_hi:   .float 88.0

// GELU tanh approx constants
const_sqrt_2_pi_f32:  .float 0.7978845608028654
const_gelu_coef_f32:  .float 0.044715

// QuickGELU constant
const_quickgelu_alpha:.float 1.702

// Common constants - FP32
const_one_f32:        .float 1.0
const_two_f32:        .float 2.0
const_half_f32:       .float 0.5
const_sixth_f32:      .float 0.16666666666666666
const_24th_f32:       .float 0.041666666666666664

// FP64 constants
.align 8
const_log2e_f64:      .double 1.4426950408889634
const_ln2_hi_f64:     .double 0.6931471805599453
const_exp_clamp_lo64: .double -708.0
const_exp_clamp_hi64: .double 708.0
const_sqrt_2_pi_f64:  .double 0.7978845608028654
const_gelu_coef_f64:  .double 0.044715
const_quickgelu_alpha64: .double 1.702
const_one_f64:        .double 1.0
const_two_f64:        .double 2.0
const_half_f64:       .double 0.5
const_sixth_f64:      .double 0.16666666666666666
const_24th_f64:       .double 0.041666666666666664

.text

//=============================================================================
// SiLU FP32: x * sigmoid(x) = x / (1 + exp(-x))
// void silu_f32_asm(const float* input, float* output, size_t n)
// x0 = input, x1 = output, x2 = n
//=============================================================================
.global silu_f32_asm
.type silu_f32_asm, %function
silu_f32_asm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    cbz     x2, .Lsilu_f32_done

    // Initialize predicate before using
    ptrue   p0.s

    // Load constants into registers
    adrp    x10, const_log2e_f32
    add     x10, x10, :lo12:const_log2e_f32
    ld1rw   {z16.s}, p0/z, [x10]              // log2e

    adrp    x10, const_ln2_hi_f32
    add     x10, x10, :lo12:const_ln2_hi_f32
    ld1rw   {z17.s}, p0/z, [x10]              // ln2

    adrp    x10, const_one_f32
    add     x10, x10, :lo12:const_one_f32
    ld1rw   {z18.s}, p0/z, [x10]              // 1.0

    adrp    x10, const_exp_clamp_lo
    add     x10, x10, :lo12:const_exp_clamp_lo
    ld1rw   {z19.s}, p0/z, [x10]              // -88.0

    adrp    x10, const_exp_clamp_hi
    add     x10, x10, :lo12:const_exp_clamp_hi
    ld1rw   {z20.s}, p0/z, [x10]              // 88.0

    adrp    x10, const_half_f32
    add     x10, x10, :lo12:const_half_f32
    ld1rw   {z21.s}, p0/z, [x10]              // 0.5

    cntw    x3                                 // VL in words
    lsl     x4, x3, #2                         // VL bytes

    mov     x5, x0                             // input ptr
    mov     x6, x1                             // output ptr
    mov     x7, x2                             // count

.Lsilu_f32_loop:
    whilelt p1.s, xzr, x7
    b.none  .Lsilu_f32_done

    // Load x
    ld1w    {z0.s}, p1/z, [x5]

    // Compute sigmoid(x) using: sigmoid(x) = 1/(1+exp(-x))
    // For numerical stability, use:
    // x >= 0: sigmoid = 1/(1+exp(-x))
    // x < 0:  sigmoid = exp(x)/(1+exp(x))

    // Get |x| and -|x|
    fabs    z1.s, p1/m, z0.s                   // |x|
    fneg    z1.s, p1/m, z1.s                   // -|x|

    // Clamp for exp
    fmax    z1.s, p1/m, z1.s, z19.s
    fmin    z1.s, p1/m, z1.s, z20.s

    // exp(-|x|) via range reduction: exp(y) = 2^n * exp(r)
    // n = round(y * log2e), r = y - n * ln2
    fmul    z2.s, z1.s, z16.s                  // y * log2e
    frintn  z3.s, p1/m, z2.s                   // n = round(...)

    // r = y - n * ln2
    fmul    z4.s, z3.s, z17.s                  // n * ln2
    fsub    z2.s, z1.s, z4.s                   // r = y - n*ln2

    // Polynomial: exp(r) ≈ 1 + r + r²/2 + r³/6
    fmul    z4.s, z2.s, z2.s                   // r²
    fmul    z5.s, z4.s, z2.s                   // r³

    // p = 1 + r
    fadd    z6.s, z18.s, z2.s

    // p += r²/2
    fmla    z6.s, p1/m, z4.s, z21.s            // z6 += r² * 0.5

    // p += r³/6 (load 1/6 from memory via constant pool)
    adrp    x11, const_sixth_f32
    add     x11, x11, :lo12:const_sixth_f32
    ld1rw   {z7.s}, p1/z, [x11]
    fmla    z6.s, p1/m, z5.s, z7.s             // z6 += r³ * (1/6)

    // Scale by 2^n: convert n to integer, add bias, shift to exponent
    fcvtzs  z3.s, p1/m, z3.s                   // n as int32
    add     z3.s, z3.s, #127                   // add IEEE754 bias
    // Clamp to valid exponent range [1, 254] to avoid inf/denorm
    mov     z25.s, #1
    smax    z3.s, p1/m, z3.s, z25.s
    mov     z25.s, #254
    smin    z3.s, p1/m, z3.s, z25.s
    lsl     z3.s, z3.s, #23                    // shift to exponent position
    // z3 now contains bit pattern for 2^n (use unsigned multiply to reinterpret)

    // exp(-|x|) = p * 2^n - use fscale to avoid reinterpret issues
    // Alternative: reinterpret z3 as float for multiply
    fmul    z6.s, z6.s, z3.s                   // exp(-|x|)

    // denom = 1 + exp(-|x|)
    fadd    z7.s, z6.s, z18.s

    // 1/denom using frecpe + 2 NR iterations
    frecpe  z8.s, z7.s
    frecps  z9.s, z7.s, z8.s
    fmul    z8.s, z8.s, z9.s
    frecps  z9.s, z7.s, z8.s
    fmul    z8.s, z8.s, z9.s                   // z8 = 1/(1+exp(-|x|))

    // For x >= 0: sigmoid = z8
    // For x < 0:  sigmoid = exp(-|x|) * z8
    fcmge   p2.s, p1/z, z0.s, #0.0             // p2 = (x >= 0)
    fmul    z9.s, z6.s, z8.s                   // exp(-|x|)/(1+exp(-|x|))

    // Select: sigmoid = (x >= 0) ? z8 : z9
    sel     z8.s, p2, z8.s, z9.s               // sigmoid(x)

    // SiLU = x * sigmoid(x)
    fmul    z8.s, z0.s, z8.s

    // Store result
    st1w    {z8.s}, p1, [x6]

    // Advance pointers
    add     x5, x5, x4
    add     x6, x6, x4
    sub     x7, x7, x3
    b       .Lsilu_f32_loop

.Lsilu_f32_done:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.size silu_f32_asm, .-silu_f32_asm


//=============================================================================
// QuickGELU FP32: x * sigmoid(1.702 * x)
// void quickgelu_f32_asm(const float* input, float* output, size_t n)
//=============================================================================
.global quickgelu_f32_asm
.type quickgelu_f32_asm, %function
quickgelu_f32_asm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    cbz     x2, .Lquickgelu_f32_done

    // Initialize predicate before using
    ptrue   p0.s

    // Load constants
    adrp    x10, const_log2e_f32
    add     x10, x10, :lo12:const_log2e_f32
    ld1rw   {z16.s}, p0/z, [x10]

    adrp    x10, const_ln2_hi_f32
    add     x10, x10, :lo12:const_ln2_hi_f32
    ld1rw   {z17.s}, p0/z, [x10]

    adrp    x10, const_one_f32
    add     x10, x10, :lo12:const_one_f32
    ld1rw   {z18.s}, p0/z, [x10]

    adrp    x10, const_exp_clamp_lo
    add     x10, x10, :lo12:const_exp_clamp_lo
    ld1rw   {z19.s}, p0/z, [x10]

    adrp    x10, const_exp_clamp_hi
    add     x10, x10, :lo12:const_exp_clamp_hi
    ld1rw   {z20.s}, p0/z, [x10]

    adrp    x10, const_half_f32
    add     x10, x10, :lo12:const_half_f32
    ld1rw   {z21.s}, p0/z, [x10]

    adrp    x10, const_quickgelu_alpha
    add     x10, x10, :lo12:const_quickgelu_alpha
    ld1rw   {z22.s}, p0/z, [x10]              // 1.702

    cntw    x3
    lsl     x4, x3, #2

    mov     x5, x0
    mov     x6, x1
    mov     x7, x2

.Lquickgelu_f32_loop:
    whilelt p1.s, xzr, x7
    b.none  .Lquickgelu_f32_done

    // Load x
    ld1w    {z0.s}, p1/z, [x5]

    // scaled_x = 1.702 * x
    fmul    z1.s, z0.s, z22.s

    // sigmoid(scaled_x)
    fabs    z2.s, p1/m, z1.s
    fneg    z2.s, p1/m, z2.s

    fmax    z2.s, p1/m, z2.s, z19.s
    fmin    z2.s, p1/m, z2.s, z20.s

    fmul    z3.s, z2.s, z16.s
    frintn  z4.s, p1/m, z3.s

    fmul    z5.s, z4.s, z17.s
    fsub    z3.s, z2.s, z5.s

    fmul    z5.s, z3.s, z3.s
    fmul    z6.s, z5.s, z3.s

    fadd    z7.s, z18.s, z3.s
    fmla    z7.s, p1/m, z5.s, z21.s
    adrp    x11, const_sixth_f32
    add     x11, x11, :lo12:const_sixth_f32
    ld1rw   {z8.s}, p1/z, [x11]
    fmla    z7.s, p1/m, z6.s, z8.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, #127
    // Clamp to valid exponent range [1, 254]
    mov     z25.s, #1
    smax    z4.s, p1/m, z4.s, z25.s
    mov     z25.s, #254
    smin    z4.s, p1/m, z4.s, z25.s
    lsl     z4.s, z4.s, #23
    fmul    z7.s, z7.s, z4.s                   // exp(-|scaled_x|)

    fadd    z8.s, z7.s, z18.s
    frecpe  z9.s, z8.s
    frecps  z10.s, z8.s, z9.s
    fmul    z9.s, z9.s, z10.s
    frecps  z10.s, z8.s, z9.s
    fmul    z9.s, z9.s, z10.s

    fcmge   p2.s, p1/z, z1.s, #0.0
    fmul    z10.s, z7.s, z9.s
    sel     z9.s, p2, z9.s, z10.s              // sigmoid(scaled_x)

    // QuickGELU = x * sigmoid(1.702*x)
    fmul    z9.s, z0.s, z9.s

    st1w    {z9.s}, p1, [x6]

    add     x5, x5, x4
    add     x6, x6, x4
    sub     x7, x7, x3
    b       .Lquickgelu_f32_loop

.Lquickgelu_f32_done:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.size quickgelu_f32_asm, .-quickgelu_f32_asm


//=============================================================================
// GELU tanh approx FP32: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// void gelu_tanh_f32_asm(const float* input, float* output, size_t n)
//=============================================================================
.global gelu_tanh_f32_asm
.type gelu_tanh_f32_asm, %function
gelu_tanh_f32_asm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    cbz     x2, .Lgelu_tanh_f32_done

    // Initialize predicate before using
    ptrue   p0.s

    // Load constants
    adrp    x10, const_log2e_f32
    add     x10, x10, :lo12:const_log2e_f32
    ld1rw   {z16.s}, p0/z, [x10]

    adrp    x10, const_ln2_hi_f32
    add     x10, x10, :lo12:const_ln2_hi_f32
    ld1rw   {z17.s}, p0/z, [x10]

    adrp    x10, const_one_f32
    add     x10, x10, :lo12:const_one_f32
    ld1rw   {z18.s}, p0/z, [x10]

    adrp    x10, const_exp_clamp_lo
    add     x10, x10, :lo12:const_exp_clamp_lo
    ld1rw   {z19.s}, p0/z, [x10]

    adrp    x10, const_exp_clamp_hi
    add     x10, x10, :lo12:const_exp_clamp_hi
    ld1rw   {z20.s}, p0/z, [x10]

    adrp    x10, const_half_f32
    add     x10, x10, :lo12:const_half_f32
    ld1rw   {z21.s}, p0/z, [x10]              // 0.5

    adrp    x10, const_two_f32
    add     x10, x10, :lo12:const_two_f32
    ld1rw   {z22.s}, p0/z, [x10]              // 2.0

    adrp    x10, const_sqrt_2_pi_f32
    add     x10, x10, :lo12:const_sqrt_2_pi_f32
    ld1rw   {z23.s}, p0/z, [x10]              // sqrt(2/pi)

    adrp    x10, const_gelu_coef_f32
    add     x10, x10, :lo12:const_gelu_coef_f32
    ld1rw   {z24.s}, p0/z, [x10]              // 0.044715

    cntw    x3
    lsl     x4, x3, #2

    mov     x5, x0
    mov     x6, x1
    mov     x7, x2

.Lgelu_tanh_f32_loop:
    whilelt p1.s, xzr, x7
    b.none  .Lgelu_tanh_f32_done

    // Load x
    ld1w    {z0.s}, p1/z, [x5]

    // inner = sqrt(2/pi) * (x + 0.044715*x^3)
    fmul    z1.s, z0.s, z0.s                   // x^2
    fmul    z2.s, z1.s, z0.s                   // x^3
    fmul    z3.s, z2.s, z24.s                  // 0.044715*x^3
    fadd    z3.s, z0.s, z3.s                   // x + 0.044715*x^3
    fmul    z3.s, z3.s, z23.s                  // inner

    // tanh(inner) = 2*sigmoid(2*inner) - 1
    fmul    z4.s, z3.s, z22.s                  // 2*inner

    // sigmoid(2*inner)
    fabs    z5.s, p1/m, z4.s
    fneg    z5.s, p1/m, z5.s

    fmax    z5.s, p1/m, z5.s, z19.s
    fmin    z5.s, p1/m, z5.s, z20.s

    fmul    z6.s, z5.s, z16.s
    frintn  z7.s, p1/m, z6.s

    fmul    z8.s, z7.s, z17.s
    fsub    z6.s, z5.s, z8.s

    fmul    z8.s, z6.s, z6.s
    fmul    z9.s, z8.s, z6.s

    fadd    z10.s, z18.s, z6.s
    fmla    z10.s, p1/m, z8.s, z21.s
    adrp    x11, const_sixth_f32
    add     x11, x11, :lo12:const_sixth_f32
    ld1rw   {z11.s}, p1/z, [x11]
    fmla    z10.s, p1/m, z9.s, z11.s

    fcvtzs  z7.s, p1/m, z7.s
    add     z7.s, z7.s, #127
    // Clamp to valid exponent range [1, 254]
    mov     z25.s, #1
    smax    z7.s, p1/m, z7.s, z25.s
    mov     z25.s, #254
    smin    z7.s, p1/m, z7.s, z25.s
    lsl     z7.s, z7.s, #23
    fmul    z10.s, z10.s, z7.s                 // exp(-|2*inner|)

    fadd    z11.s, z10.s, z18.s
    frecpe  z12.s, z11.s
    frecps  z13.s, z11.s, z12.s
    fmul    z12.s, z12.s, z13.s
    frecps  z13.s, z11.s, z12.s
    fmul    z12.s, z12.s, z13.s

    fcmge   p2.s, p1/z, z4.s, #0.0
    fmul    z13.s, z10.s, z12.s
    sel     z12.s, p2, z12.s, z13.s            // sigmoid(2*inner)

    // tanh = 2*sigmoid - 1
    fmul    z12.s, z12.s, z22.s                // 2*sigmoid
    fsub    z12.s, z12.s, z18.s                // tanh

    // GELU = 0.5 * x * (1 + tanh)
    fadd    z12.s, z12.s, z18.s                // 1 + tanh
    fmul    z12.s, z0.s, z12.s                 // x * (1 + tanh)
    fmul    z12.s, z12.s, z21.s                // 0.5 * ...

    st1w    {z12.s}, p1, [x6]

    add     x5, x5, x4
    add     x6, x6, x4
    sub     x7, x7, x3
    b       .Lgelu_tanh_f32_loop

.Lgelu_tanh_f32_done:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.size gelu_tanh_f32_asm, .-gelu_tanh_f32_asm


//=============================================================================
// SiLU FP64: x * sigmoid(x)
// void silu_f64_asm(const double* input, double* output, size_t n)
//=============================================================================
.global silu_f64_asm
.type silu_f64_asm, %function
silu_f64_asm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    cbz     x2, .Lsilu_f64_done

    // Initialize predicate before using
    ptrue   p0.d

    // Load FP64 constants
    adrp    x10, const_log2e_f64
    add     x10, x10, :lo12:const_log2e_f64
    ld1rd   {z16.d}, p0/z, [x10]

    adrp    x10, const_ln2_hi_f64
    add     x10, x10, :lo12:const_ln2_hi_f64
    ld1rd   {z17.d}, p0/z, [x10]

    adrp    x10, const_one_f64
    add     x10, x10, :lo12:const_one_f64
    ld1rd   {z18.d}, p0/z, [x10]

    adrp    x10, const_exp_clamp_lo64
    add     x10, x10, :lo12:const_exp_clamp_lo64
    ld1rd   {z19.d}, p0/z, [x10]

    adrp    x10, const_exp_clamp_hi64
    add     x10, x10, :lo12:const_exp_clamp_hi64
    ld1rd   {z20.d}, p0/z, [x10]

    adrp    x10, const_half_f64
    add     x10, x10, :lo12:const_half_f64
    ld1rd   {z21.d}, p0/z, [x10]

    adrp    x10, const_sixth_f64
    add     x10, x10, :lo12:const_sixth_f64
    ld1rd   {z22.d}, p0/z, [x10]              // 1/6

    adrp    x10, const_24th_f64
    add     x10, x10, :lo12:const_24th_f64
    ld1rd   {z23.d}, p0/z, [x10]              // 1/24

    cntd    x3
    lsl     x4, x3, #3                         // VL bytes for FP64

    mov     x5, x0
    mov     x6, x1
    mov     x7, x2

.Lsilu_f64_loop:
    whilelt p1.d, xzr, x7
    b.none  .Lsilu_f64_done

    ld1d    {z0.d}, p1/z, [x5]

    // sigmoid(x)
    fabs    z1.d, p1/m, z0.d
    fneg    z1.d, p1/m, z1.d

    fmax    z1.d, p1/m, z1.d, z19.d
    fmin    z1.d, p1/m, z1.d, z20.d

    fmul    z2.d, z1.d, z16.d
    frintn  z3.d, p1/m, z2.d

    fmul    z4.d, z3.d, z17.d
    fsub    z2.d, z1.d, z4.d

    // Higher order polynomial for FP64
    fmul    z4.d, z2.d, z2.d                   // r^2
    fmul    z5.d, z4.d, z2.d                   // r^3
    fmul    z6.d, z4.d, z4.d                   // r^4

    fadd    z7.d, z18.d, z2.d                  // 1 + r
    fmla    z7.d, p1/m, z4.d, z21.d            // + r^2/2
    fmla    z7.d, p1/m, z5.d, z22.d            // + r^3/6
    fmla    z7.d, p1/m, z6.d, z23.d            // + r^4/24

    fcvtzs  z3.d, p1/m, z3.d
    mov     x8, #1023
    dup     z8.d, x8
    add     z3.d, z3.d, z8.d
    // Clamp to valid exponent range [1, 2046]
    mov     x8, #1
    dup     z24.d, x8
    smax    z3.d, p1/m, z3.d, z24.d
    mov     x8, #2046
    dup     z24.d, x8
    smin    z3.d, p1/m, z3.d, z24.d
    lsl     z3.d, z3.d, #52
    fmul    z7.d, z7.d, z3.d                   // exp(-|x|)

    fadd    z8.d, z7.d, z18.d
    frecpe  z9.d, z8.d
    frecps  z10.d, z8.d, z9.d
    fmul    z9.d, z9.d, z10.d
    frecps  z10.d, z8.d, z9.d
    fmul    z9.d, z9.d, z10.d
    frecps  z10.d, z8.d, z9.d
    fmul    z9.d, z9.d, z10.d                  // 3 NR iterations for FP64

    fcmge   p2.d, p1/z, z0.d, #0.0
    fmul    z10.d, z7.d, z9.d
    sel     z9.d, p2, z9.d, z10.d

    fmul    z9.d, z0.d, z9.d                   // x * sigmoid(x)

    st1d    {z9.d}, p1, [x6]

    add     x5, x5, x4
    add     x6, x6, x4
    sub     x7, x7, x3
    b       .Lsilu_f64_loop

.Lsilu_f64_done:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.size silu_f64_asm, .-silu_f64_asm


//=============================================================================
// QuickGELU FP64: x * sigmoid(1.702 * x)
// void quickgelu_f64_asm(const double* input, double* output, size_t n)
//=============================================================================
.global quickgelu_f64_asm
.type quickgelu_f64_asm, %function
quickgelu_f64_asm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    cbz     x2, .Lquickgelu_f64_done

    // Initialize predicate before using
    ptrue   p0.d

    adrp    x10, const_log2e_f64
    add     x10, x10, :lo12:const_log2e_f64
    ld1rd   {z16.d}, p0/z, [x10]

    adrp    x10, const_ln2_hi_f64
    add     x10, x10, :lo12:const_ln2_hi_f64
    ld1rd   {z17.d}, p0/z, [x10]

    adrp    x10, const_one_f64
    add     x10, x10, :lo12:const_one_f64
    ld1rd   {z18.d}, p0/z, [x10]

    adrp    x10, const_exp_clamp_lo64
    add     x10, x10, :lo12:const_exp_clamp_lo64
    ld1rd   {z19.d}, p0/z, [x10]

    adrp    x10, const_exp_clamp_hi64
    add     x10, x10, :lo12:const_exp_clamp_hi64
    ld1rd   {z20.d}, p0/z, [x10]

    adrp    x10, const_half_f64
    add     x10, x10, :lo12:const_half_f64
    ld1rd   {z21.d}, p0/z, [x10]

    adrp    x10, const_sixth_f64
    add     x10, x10, :lo12:const_sixth_f64
    ld1rd   {z22.d}, p0/z, [x10]

    adrp    x10, const_24th_f64
    add     x10, x10, :lo12:const_24th_f64
    ld1rd   {z23.d}, p0/z, [x10]

    adrp    x10, const_quickgelu_alpha64
    add     x10, x10, :lo12:const_quickgelu_alpha64
    ld1rd   {z24.d}, p0/z, [x10]

    cntd    x3
    lsl     x4, x3, #3

    mov     x5, x0
    mov     x6, x1
    mov     x7, x2

.Lquickgelu_f64_loop:
    whilelt p1.d, xzr, x7
    b.none  .Lquickgelu_f64_done

    ld1d    {z0.d}, p1/z, [x5]

    fmul    z1.d, z0.d, z24.d                  // 1.702 * x

    fabs    z2.d, p1/m, z1.d
    fneg    z2.d, p1/m, z2.d

    fmax    z2.d, p1/m, z2.d, z19.d
    fmin    z2.d, p1/m, z2.d, z20.d

    fmul    z3.d, z2.d, z16.d
    frintn  z4.d, p1/m, z3.d

    fmul    z5.d, z4.d, z17.d
    fsub    z3.d, z2.d, z5.d

    fmul    z5.d, z3.d, z3.d
    fmul    z6.d, z5.d, z3.d
    fmul    z7.d, z5.d, z5.d

    fadd    z8.d, z18.d, z3.d
    fmla    z8.d, p1/m, z5.d, z21.d
    fmla    z8.d, p1/m, z6.d, z22.d
    fmla    z8.d, p1/m, z7.d, z23.d

    fcvtzs  z4.d, p1/m, z4.d
    mov     x8, #1023
    dup     z9.d, x8
    add     z4.d, z4.d, z9.d
    // Clamp to valid exponent range [1, 2046]
    mov     x8, #1
    dup     z25.d, x8
    smax    z4.d, p1/m, z4.d, z25.d
    mov     x8, #2046
    dup     z25.d, x8
    smin    z4.d, p1/m, z4.d, z25.d
    lsl     z4.d, z4.d, #52
    fmul    z8.d, z8.d, z4.d

    fadd    z9.d, z8.d, z18.d
    frecpe  z10.d, z9.d
    frecps  z11.d, z9.d, z10.d
    fmul    z10.d, z10.d, z11.d
    frecps  z11.d, z9.d, z10.d
    fmul    z10.d, z10.d, z11.d
    frecps  z11.d, z9.d, z10.d
    fmul    z10.d, z10.d, z11.d

    fcmge   p2.d, p1/z, z1.d, #0.0
    fmul    z11.d, z8.d, z10.d
    sel     z10.d, p2, z10.d, z11.d

    fmul    z10.d, z0.d, z10.d

    st1d    {z10.d}, p1, [x6]

    add     x5, x5, x4
    add     x6, x6, x4
    sub     x7, x7, x3
    b       .Lquickgelu_f64_loop

.Lquickgelu_f64_done:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.size quickgelu_f64_asm, .-quickgelu_f64_asm


//=============================================================================
// GELU tanh approx FP64
// void gelu_tanh_f64_asm(const double* input, double* output, size_t n)
//=============================================================================
.global gelu_tanh_f64_asm
.type gelu_tanh_f64_asm, %function
gelu_tanh_f64_asm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    cbz     x2, .Lgelu_tanh_f64_done

    // Initialize predicate before using
    ptrue   p0.d

    adrp    x10, const_log2e_f64
    add     x10, x10, :lo12:const_log2e_f64
    ld1rd   {z16.d}, p0/z, [x10]

    adrp    x10, const_ln2_hi_f64
    add     x10, x10, :lo12:const_ln2_hi_f64
    ld1rd   {z17.d}, p0/z, [x10]

    adrp    x10, const_one_f64
    add     x10, x10, :lo12:const_one_f64
    ld1rd   {z18.d}, p0/z, [x10]

    adrp    x10, const_exp_clamp_lo64
    add     x10, x10, :lo12:const_exp_clamp_lo64
    ld1rd   {z19.d}, p0/z, [x10]

    adrp    x10, const_exp_clamp_hi64
    add     x10, x10, :lo12:const_exp_clamp_hi64
    ld1rd   {z20.d}, p0/z, [x10]

    adrp    x10, const_half_f64
    add     x10, x10, :lo12:const_half_f64
    ld1rd   {z21.d}, p0/z, [x10]

    adrp    x10, const_two_f64
    add     x10, x10, :lo12:const_two_f64
    ld1rd   {z22.d}, p0/z, [x10]

    adrp    x10, const_sixth_f64
    add     x10, x10, :lo12:const_sixth_f64
    ld1rd   {z23.d}, p0/z, [x10]

    adrp    x10, const_24th_f64
    add     x10, x10, :lo12:const_24th_f64
    ld1rd   {z24.d}, p0/z, [x10]

    adrp    x10, const_sqrt_2_pi_f64
    add     x10, x10, :lo12:const_sqrt_2_pi_f64
    ld1rd   {z25.d}, p0/z, [x10]

    adrp    x10, const_gelu_coef_f64
    add     x10, x10, :lo12:const_gelu_coef_f64
    ld1rd   {z26.d}, p0/z, [x10]

    cntd    x3
    lsl     x4, x3, #3

    mov     x5, x0
    mov     x6, x1
    mov     x7, x2

.Lgelu_tanh_f64_loop:
    whilelt p1.d, xzr, x7
    b.none  .Lgelu_tanh_f64_done

    ld1d    {z0.d}, p1/z, [x5]

    // inner = sqrt(2/pi) * (x + 0.044715*x^3)
    fmul    z1.d, z0.d, z0.d                   // x^2
    fmul    z2.d, z1.d, z0.d                   // x^3
    fmul    z3.d, z2.d, z26.d                  // 0.044715*x^3
    fadd    z3.d, z0.d, z3.d
    fmul    z3.d, z3.d, z25.d                  // inner

    // tanh via 2*sigmoid(2*inner) - 1
    fmul    z4.d, z3.d, z22.d                  // 2*inner

    fabs    z5.d, p1/m, z4.d
    fneg    z5.d, p1/m, z5.d

    fmax    z5.d, p1/m, z5.d, z19.d
    fmin    z5.d, p1/m, z5.d, z20.d

    fmul    z6.d, z5.d, z16.d
    frintn  z7.d, p1/m, z6.d

    fmul    z8.d, z7.d, z17.d
    fsub    z6.d, z5.d, z8.d

    fmul    z8.d, z6.d, z6.d
    fmul    z9.d, z8.d, z6.d
    fmul    z10.d, z8.d, z8.d

    fadd    z11.d, z18.d, z6.d
    fmla    z11.d, p1/m, z8.d, z21.d
    fmla    z11.d, p1/m, z9.d, z23.d
    fmla    z11.d, p1/m, z10.d, z24.d

    fcvtzs  z7.d, p1/m, z7.d
    mov     x8, #1023
    dup     z12.d, x8
    add     z7.d, z7.d, z12.d
    // Clamp to valid exponent range [1, 2046]
    mov     x8, #1
    dup     z27.d, x8
    smax    z7.d, p1/m, z7.d, z27.d
    mov     x8, #2046
    dup     z27.d, x8
    smin    z7.d, p1/m, z7.d, z27.d
    lsl     z7.d, z7.d, #52
    fmul    z11.d, z11.d, z7.d                 // exp(-|2*inner|)

    fadd    z12.d, z11.d, z18.d
    frecpe  z13.d, z12.d
    frecps  z14.d, z12.d, z13.d
    fmul    z13.d, z13.d, z14.d
    frecps  z14.d, z12.d, z13.d
    fmul    z13.d, z13.d, z14.d
    frecps  z14.d, z12.d, z13.d
    fmul    z13.d, z13.d, z14.d

    fcmge   p2.d, p1/z, z4.d, #0.0
    fmul    z14.d, z11.d, z13.d
    sel     z13.d, p2, z13.d, z14.d            // sigmoid(2*inner)

    // tanh = 2*sigmoid - 1
    fmul    z13.d, z13.d, z22.d
    fsub    z13.d, z13.d, z18.d

    // GELU = 0.5 * x * (1 + tanh)
    fadd    z13.d, z13.d, z18.d
    fmul    z13.d, z0.d, z13.d
    fmul    z13.d, z13.d, z21.d

    st1d    {z13.d}, p1, [x6]

    add     x5, x5, x4
    add     x6, x6, x4
    sub     x7, x7, x3
    b       .Lgelu_tanh_f64_loop

.Lgelu_tanh_f64_done:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.size gelu_tanh_f64_asm, .-gelu_tanh_f64_asm


//=============================================================================
// SwiGLU FP32: output = x * gate * sigmoid(gate) = x * SiLU(gate)
// void swiglu_f32_asm(const float* x, const float* gate, float* output, size_t n)
// x0 = x, x1 = gate, x2 = output, x3 = n
//=============================================================================
.global swiglu_f32_asm
.type swiglu_f32_asm, %function
swiglu_f32_asm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    cbz     x3, .Lswiglu_f32_done

    // Initialize predicate before using
    ptrue   p0.s

    // Load constants
    adrp    x10, const_log2e_f32
    add     x10, x10, :lo12:const_log2e_f32
    ld1rw   {z16.s}, p0/z, [x10]

    adrp    x10, const_ln2_hi_f32
    add     x10, x10, :lo12:const_ln2_hi_f32
    ld1rw   {z17.s}, p0/z, [x10]

    adrp    x10, const_one_f32
    add     x10, x10, :lo12:const_one_f32
    ld1rw   {z18.s}, p0/z, [x10]

    adrp    x10, const_exp_clamp_lo
    add     x10, x10, :lo12:const_exp_clamp_lo
    ld1rw   {z19.s}, p0/z, [x10]

    adrp    x10, const_exp_clamp_hi
    add     x10, x10, :lo12:const_exp_clamp_hi
    ld1rw   {z20.s}, p0/z, [x10]

    adrp    x10, const_half_f32
    add     x10, x10, :lo12:const_half_f32
    ld1rw   {z21.s}, p0/z, [x10]

    cntw    x4                                 // VL in words
    lsl     x5, x4, #2                         // VL bytes

    mov     x6, x0                             // x ptr
    mov     x7, x1                             // gate ptr
    mov     x8, x2                             // output ptr
    mov     x9, x3                             // count

.Lswiglu_f32_loop:
    whilelt p1.s, xzr, x9
    b.none  .Lswiglu_f32_done

    // Load x and gate
    ld1w    {z0.s}, p1/z, [x6]                 // x
    ld1w    {z1.s}, p1/z, [x7]                 // gate

    // Compute sigmoid(gate)
    fabs    z2.s, p1/m, z1.s
    fneg    z2.s, p1/m, z2.s

    fmax    z2.s, p1/m, z2.s, z19.s
    fmin    z2.s, p1/m, z2.s, z20.s

    fmul    z3.s, z2.s, z16.s
    frintn  z4.s, p1/m, z3.s

    fmul    z5.s, z4.s, z17.s
    fsub    z3.s, z2.s, z5.s

    fmul    z5.s, z3.s, z3.s
    fmul    z6.s, z5.s, z3.s

    fadd    z7.s, z18.s, z3.s
    fmla    z7.s, p1/m, z5.s, z21.s
    adrp    x11, const_sixth_f32
    add     x11, x11, :lo12:const_sixth_f32
    ld1rw   {z8.s}, p1/z, [x11]
    fmla    z7.s, p1/m, z6.s, z8.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, #127
    // Clamp to valid exponent range [1, 254]
    mov     z25.s, #1
    smax    z4.s, p1/m, z4.s, z25.s
    mov     z25.s, #254
    smin    z4.s, p1/m, z4.s, z25.s
    lsl     z4.s, z4.s, #23
    fmul    z7.s, z7.s, z4.s                   // exp(-|gate|)

    fadd    z8.s, z7.s, z18.s
    frecpe  z9.s, z8.s
    frecps  z10.s, z8.s, z9.s
    fmul    z9.s, z9.s, z10.s
    frecps  z10.s, z8.s, z9.s
    fmul    z9.s, z9.s, z10.s

    fcmge   p2.s, p1/z, z1.s, #0.0
    fmul    z10.s, z7.s, z9.s
    sel     z9.s, p2, z9.s, z10.s              // sigmoid(gate)

    // SiLU(gate) = gate * sigmoid(gate)
    fmul    z9.s, z1.s, z9.s

    // output = x * SiLU(gate)
    fmul    z9.s, z0.s, z9.s

    st1w    {z9.s}, p1, [x8]

    add     x6, x6, x5
    add     x7, x7, x5
    add     x8, x8, x5
    sub     x9, x9, x4
    b       .Lswiglu_f32_loop

.Lswiglu_f32_done:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.size swiglu_f32_asm, .-swiglu_f32_asm


//=============================================================================
// SwiGLU FP64: output = x * gate * sigmoid(gate)
// void swiglu_f64_asm(const double* x, const double* gate, double* output, size_t n)
//=============================================================================
.global swiglu_f64_asm
.type swiglu_f64_asm, %function
swiglu_f64_asm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]

    cbz     x3, .Lswiglu_f64_done

    // Initialize predicate before using
    ptrue   p0.d

    adrp    x10, const_log2e_f64
    add     x10, x10, :lo12:const_log2e_f64
    ld1rd   {z16.d}, p0/z, [x10]

    adrp    x10, const_ln2_hi_f64
    add     x10, x10, :lo12:const_ln2_hi_f64
    ld1rd   {z17.d}, p0/z, [x10]

    adrp    x10, const_one_f64
    add     x10, x10, :lo12:const_one_f64
    ld1rd   {z18.d}, p0/z, [x10]

    adrp    x10, const_exp_clamp_lo64
    add     x10, x10, :lo12:const_exp_clamp_lo64
    ld1rd   {z19.d}, p0/z, [x10]

    adrp    x10, const_exp_clamp_hi64
    add     x10, x10, :lo12:const_exp_clamp_hi64
    ld1rd   {z20.d}, p0/z, [x10]

    adrp    x10, const_half_f64
    add     x10, x10, :lo12:const_half_f64
    ld1rd   {z21.d}, p0/z, [x10]

    adrp    x10, const_sixth_f64
    add     x10, x10, :lo12:const_sixth_f64
    ld1rd   {z22.d}, p0/z, [x10]

    adrp    x10, const_24th_f64
    add     x10, x10, :lo12:const_24th_f64
    ld1rd   {z23.d}, p0/z, [x10]

    cntd    x4
    lsl     x5, x4, #3

    mov     x6, x0
    mov     x7, x1
    mov     x8, x2
    mov     x9, x3

.Lswiglu_f64_loop:
    whilelt p1.d, xzr, x9
    b.none  .Lswiglu_f64_done

    ld1d    {z0.d}, p1/z, [x6]                 // x
    ld1d    {z1.d}, p1/z, [x7]                 // gate

    // sigmoid(gate)
    fabs    z2.d, p1/m, z1.d
    fneg    z2.d, p1/m, z2.d

    fmax    z2.d, p1/m, z2.d, z19.d
    fmin    z2.d, p1/m, z2.d, z20.d

    fmul    z3.d, z2.d, z16.d
    frintn  z4.d, p1/m, z3.d

    fmul    z5.d, z4.d, z17.d
    fsub    z3.d, z2.d, z5.d

    fmul    z5.d, z3.d, z3.d
    fmul    z6.d, z5.d, z3.d
    fmul    z7.d, z5.d, z5.d

    fadd    z8.d, z18.d, z3.d
    fmla    z8.d, p1/m, z5.d, z21.d
    fmla    z8.d, p1/m, z6.d, z22.d
    fmla    z8.d, p1/m, z7.d, z23.d

    fcvtzs  z4.d, p1/m, z4.d
    mov     x10, #1023
    dup     z9.d, x10
    add     z4.d, z4.d, z9.d
    // Clamp to valid exponent range [1, 2046]
    mov     x10, #1
    dup     z24.d, x10
    smax    z4.d, p1/m, z4.d, z24.d
    mov     x10, #2046
    dup     z24.d, x10
    smin    z4.d, p1/m, z4.d, z24.d
    lsl     z4.d, z4.d, #52
    fmul    z8.d, z8.d, z4.d                   // exp(-|gate|)

    fadd    z9.d, z8.d, z18.d
    frecpe  z10.d, z9.d
    frecps  z11.d, z9.d, z10.d
    fmul    z10.d, z10.d, z11.d
    frecps  z11.d, z9.d, z10.d
    fmul    z10.d, z10.d, z11.d
    frecps  z11.d, z9.d, z10.d
    fmul    z10.d, z10.d, z11.d

    fcmge   p2.d, p1/z, z1.d, #0.0
    fmul    z11.d, z8.d, z10.d
    sel     z10.d, p2, z10.d, z11.d            // sigmoid(gate)

    // SiLU(gate) = gate * sigmoid(gate)
    fmul    z10.d, z1.d, z10.d

    // output = x * SiLU(gate)
    fmul    z10.d, z0.d, z10.d

    st1d    {z10.d}, p1, [x8]

    add     x6, x6, x5
    add     x7, x7, x5
    add     x8, x8, x5
    sub     x9, x9, x4
    b       .Lswiglu_f64_loop

.Lswiglu_f64_done:
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.size swiglu_f64_asm, .-swiglu_f64_asm
