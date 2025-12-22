// layernorm_asm.S
// Hand-optimized SVE assembly for LayerNorm and RMSNorm
// Uses rsqrte + Newton-Raphson for sqrt, recpe + Newton-Raphson for division
// Targeting A64FX with 512-bit SVE (VL=64 bytes)

.arch armv8.2-a+sve

//=============================================================================
// RMSNorm FP32 - Assembly Implementation
// void rmsnorm_f32_asm(const float* input, const float* gamma,
//                      float* output, size_t dim, float eps)
// x0 = input, x1 = gamma, x2 = output, x3 = dim, s0 = eps
//=============================================================================
.global rmsnorm_f32_asm
.type rmsnorm_f32_asm, %function
rmsnorm_f32_asm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Save eps (s0 is part of z0 which we'll zero)
    fmov    s20, s0                 // s20 = eps (preserved)

    // Get vector length in words
    cntw    x4                      // x4 = VL in words (16 for A64FX)
    ptrue   p0.s                    // All-true predicate

    // Compute VL bytes stride for FP32: x10 = VL * 4
    lsl     x10, x4, #2

    // Step 1: Compute sum of squares with 4 accumulators
    fmov    z0.s, #0.0
    fmov    z1.s, #0.0
    fmov    z2.s, #0.0
    fmov    z3.s, #0.0

    mov     x5, x0                  // x5 = input ptr
    mov     x6, x3                  // x6 = remaining count
    lsl     x7, x4, #2              // x7 = VL*4 elements for 4x unroll

    // Main loop: 4x unrolled
.Lrmsnorm_f32_sum_loop4:
    cmp     x6, x7
    b.lt    .Lrmsnorm_f32_sum_remainder

    ld1w    {z4.s}, p0/z, [x5]
    add     x5, x5, x10
    ld1w    {z5.s}, p0/z, [x5]
    add     x5, x5, x10
    ld1w    {z6.s}, p0/z, [x5]
    add     x5, x5, x10
    ld1w    {z7.s}, p0/z, [x5]
    add     x5, x5, x10

    fmla    z0.s, p0/m, z4.s, z4.s
    fmla    z1.s, p0/m, z5.s, z5.s
    fmla    z2.s, p0/m, z6.s, z6.s
    fmla    z3.s, p0/m, z7.s, z7.s

    sub     x6, x6, x7
    b       .Lrmsnorm_f32_sum_loop4

.Lrmsnorm_f32_sum_remainder:
    cbz     x6, .Lrmsnorm_f32_sum_done

.Lrmsnorm_f32_sum_loop1:
    whilelt p1.s, xzr, x6
    ld1w    {z4.s}, p1/z, [x5]
    fmla    z0.s, p1/m, z4.s, z4.s
    add     x5, x5, x10             // increment by VL*4 bytes
    subs    x6, x6, x4              // decrement by VL elements
    b.gt    .Lrmsnorm_f32_sum_loop1

.Lrmsnorm_f32_sum_done:
    // Combine accumulators
    fadd    z0.s, p0/m, z0.s, z1.s
    fadd    z2.s, p0/m, z2.s, z3.s
    fadd    z0.s, p0/m, z0.s, z2.s

    // Horizontal sum
    faddv   s4, p0, z0.s            // s4 = total sum of squares

    // Step 2: Compute inv_std = rsqrt(mean_sq + eps)
    ucvtf   s5, w3                  // s5 = (float)dim
    fdiv    s4, s4, s5              // s4 = mean_sq = sum_sq / dim
    fadd    s4, s4, s20             // s4 = variance + eps (use saved eps)

    // rsqrte + 2 Newton-Raphson iterations
    frsqrte s5, s4
    fmul    s6, s5, s5
    frsqrts s6, s4, s6
    fmul    s5, s5, s6

    fmul    s6, s5, s5
    frsqrts s6, s4, s6
    fmul    s5, s5, s6              // s5 = inv_std

    // Broadcast inv_std to vector - use index syntax
    dup     z8.s, z5.s[0]           // z8 = inv_std (broadcast z5 lane 0 to all lanes)

    // Step 3: Normalize and scale
    mov     x5, x0                  // input ptr
    mov     x6, x1                  // gamma ptr
    mov     x7, x2                  // output ptr
    mov     x8, x3                  // count
    lsl     x9, x4, #2              // 4x unroll element count

.Lrmsnorm_f32_norm_loop4:
    cmp     x8, x9
    b.lt    .Lrmsnorm_f32_norm_remainder

    // Load input and gamma (4 vectors each)
    ld1w    {z0.s}, p0/z, [x5]
    ld1w    {z4.s}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    ld1w    {z1.s}, p0/z, [x5]
    ld1w    {z5.s}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    ld1w    {z2.s}, p0/z, [x5]
    ld1w    {z6.s}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    ld1w    {z3.s}, p0/z, [x5]
    ld1w    {z7.s}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    // y = x * inv_std * gamma
    fmul    z0.s, z0.s, z8.s
    fmul    z1.s, z1.s, z8.s
    fmul    z2.s, z2.s, z8.s
    fmul    z3.s, z3.s, z8.s

    fmul    z0.s, z0.s, z4.s
    fmul    z1.s, z1.s, z5.s
    fmul    z2.s, z2.s, z6.s
    fmul    z3.s, z3.s, z7.s

    // Store output
    st1w    {z0.s}, p0, [x7]
    add     x7, x7, x10
    st1w    {z1.s}, p0, [x7]
    add     x7, x7, x10
    st1w    {z2.s}, p0, [x7]
    add     x7, x7, x10
    st1w    {z3.s}, p0, [x7]
    add     x7, x7, x10

    sub     x8, x8, x9
    b       .Lrmsnorm_f32_norm_loop4

.Lrmsnorm_f32_norm_remainder:
    cbz     x8, .Lrmsnorm_f32_done

.Lrmsnorm_f32_norm_loop1:
    whilelt p1.s, xzr, x8
    ld1w    {z0.s}, p1/z, [x5]
    ld1w    {z4.s}, p1/z, [x6]
    fmul    z0.s, z0.s, z8.s
    fmul    z0.s, z0.s, z4.s
    st1w    {z0.s}, p1, [x7]
    add     x5, x5, x10
    add     x6, x6, x10
    add     x7, x7, x10
    subs    x8, x8, x4
    b.gt    .Lrmsnorm_f32_norm_loop1

.Lrmsnorm_f32_done:
    ldp     x29, x30, [sp], #16
    ret
.size rmsnorm_f32_asm, .-rmsnorm_f32_asm

//=============================================================================
// RMSNorm FP64 - Assembly Implementation
// void rmsnorm_f64_asm(const double* input, const double* gamma,
//                      double* output, size_t dim, double eps)
// x0 = input, x1 = gamma, x2 = output, x3 = dim, d0 = eps
//=============================================================================
.global rmsnorm_f64_asm
.type rmsnorm_f64_asm, %function
rmsnorm_f64_asm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Save eps (d0 is part of z0 which we'll zero)
    fmov    d20, d0                 // d20 = eps (preserved)

    cntd    x4                      // VL in doublewords (8 for A64FX)
    ptrue   p0.d

    // VL bytes stride for FP64: x10 = VL * 8
    lsl     x10, x4, #3

    // Sum of squares with 4 accumulators
    fmov    z0.d, #0.0
    fmov    z1.d, #0.0
    fmov    z2.d, #0.0
    fmov    z3.d, #0.0

    mov     x5, x0
    mov     x6, x3
    lsl     x7, x4, #2              // 4x unroll element count

.Lrmsnorm_f64_sum_loop4:
    cmp     x6, x7
    b.lt    .Lrmsnorm_f64_sum_remainder

    ld1d    {z4.d}, p0/z, [x5]
    add     x5, x5, x10
    ld1d    {z5.d}, p0/z, [x5]
    add     x5, x5, x10
    ld1d    {z6.d}, p0/z, [x5]
    add     x5, x5, x10
    ld1d    {z7.d}, p0/z, [x5]
    add     x5, x5, x10

    fmla    z0.d, p0/m, z4.d, z4.d
    fmla    z1.d, p0/m, z5.d, z5.d
    fmla    z2.d, p0/m, z6.d, z6.d
    fmla    z3.d, p0/m, z7.d, z7.d

    sub     x6, x6, x7
    b       .Lrmsnorm_f64_sum_loop4

.Lrmsnorm_f64_sum_remainder:
    cbz     x6, .Lrmsnorm_f64_sum_done

.Lrmsnorm_f64_sum_loop1:
    whilelt p1.d, xzr, x6
    ld1d    {z4.d}, p1/z, [x5]
    fmla    z0.d, p1/m, z4.d, z4.d
    add     x5, x5, x10
    subs    x6, x6, x4
    b.gt    .Lrmsnorm_f64_sum_loop1

.Lrmsnorm_f64_sum_done:
    fadd    z0.d, p0/m, z0.d, z1.d
    fadd    z2.d, p0/m, z2.d, z3.d
    fadd    z0.d, p0/m, z0.d, z2.d
    faddv   d4, p0, z0.d

    // mean_sq = sum_sq / dim
    ucvtf   d5, x3
    fdiv    d4, d4, d5
    fadd    d4, d4, d20             // d4 = variance + eps (use saved eps)

    // rsqrte + 3 NR for FP64 precision
    frsqrte d5, d4
    fmul    d6, d5, d5
    frsqrts d6, d4, d6
    fmul    d5, d5, d6

    fmul    d6, d5, d5
    frsqrts d6, d4, d6
    fmul    d5, d5, d6

    fmul    d6, d5, d5
    frsqrts d6, d4, d6
    fmul    d5, d5, d6              // d5 = inv_std

    dup     z8.d, z5.d[0]

    // Normalize and scale
    mov     x5, x0
    mov     x6, x1
    mov     x7, x2
    mov     x8, x3
    lsl     x9, x4, #2

.Lrmsnorm_f64_norm_loop4:
    cmp     x8, x9
    b.lt    .Lrmsnorm_f64_norm_remainder

    ld1d    {z0.d}, p0/z, [x5]
    ld1d    {z4.d}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    ld1d    {z1.d}, p0/z, [x5]
    ld1d    {z5.d}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    ld1d    {z2.d}, p0/z, [x5]
    ld1d    {z6.d}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    ld1d    {z3.d}, p0/z, [x5]
    ld1d    {z7.d}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    fmul    z0.d, z0.d, z8.d
    fmul    z1.d, z1.d, z8.d
    fmul    z2.d, z2.d, z8.d
    fmul    z3.d, z3.d, z8.d

    fmul    z0.d, z0.d, z4.d
    fmul    z1.d, z1.d, z5.d
    fmul    z2.d, z2.d, z6.d
    fmul    z3.d, z3.d, z7.d

    st1d    {z0.d}, p0, [x7]
    add     x7, x7, x10
    st1d    {z1.d}, p0, [x7]
    add     x7, x7, x10
    st1d    {z2.d}, p0, [x7]
    add     x7, x7, x10
    st1d    {z3.d}, p0, [x7]
    add     x7, x7, x10

    sub     x8, x8, x9
    b       .Lrmsnorm_f64_norm_loop4

.Lrmsnorm_f64_norm_remainder:
    cbz     x8, .Lrmsnorm_f64_done

.Lrmsnorm_f64_norm_loop1:
    whilelt p1.d, xzr, x8
    ld1d    {z0.d}, p1/z, [x5]
    ld1d    {z4.d}, p1/z, [x6]
    fmul    z0.d, z0.d, z8.d
    fmul    z0.d, z0.d, z4.d
    st1d    {z0.d}, p1, [x7]
    add     x5, x5, x10
    add     x6, x6, x10
    add     x7, x7, x10
    subs    x8, x8, x4
    b.gt    .Lrmsnorm_f64_norm_loop1

.Lrmsnorm_f64_done:
    ldp     x29, x30, [sp], #16
    ret
.size rmsnorm_f64_asm, .-rmsnorm_f64_asm

//=============================================================================
// LayerNorm FP32 - Assembly Implementation
// void layernorm_f32_asm(const float* input, const float* gamma,
//                        const float* beta, float* output, size_t dim, float eps)
// x0 = input, x1 = gamma, x2 = beta, x3 = output, x4 = dim, s0 = eps
//=============================================================================
.global layernorm_f32_asm
.type layernorm_f32_asm, %function
layernorm_f32_asm:
    stp     x29, x30, [sp, #-48]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    mov     x29, sp

    // Save eps (s0 is part of z0 which we'll zero)
    fmov    s20, s0                 // s20 = eps (preserved)

    // Save parameters
    mov     x19, x0                 // input
    mov     x20, x1                 // gamma
    mov     x21, x2                 // beta
    mov     x22, x3                 // output
    // x4 = dim, s0 = eps

    cntw    x5                      // VL in words
    ptrue   p0.s

    // VL bytes stride
    lsl     x10, x5, #2

    //=== Pass 1: Compute mean ===
    fmov    z0.s, #0.0
    fmov    z1.s, #0.0
    fmov    z2.s, #0.0
    fmov    z3.s, #0.0

    mov     x6, x0
    mov     x7, x4
    lsl     x8, x5, #2              // 4x unroll count

.Llayernorm_f32_mean_loop4:
    cmp     x7, x8
    b.lt    .Llayernorm_f32_mean_remainder

    ld1w    {z4.s}, p0/z, [x6]
    add     x6, x6, x10
    ld1w    {z5.s}, p0/z, [x6]
    add     x6, x6, x10
    ld1w    {z6.s}, p0/z, [x6]
    add     x6, x6, x10
    ld1w    {z7.s}, p0/z, [x6]
    add     x6, x6, x10

    fadd    z0.s, p0/m, z0.s, z4.s
    fadd    z1.s, p0/m, z1.s, z5.s
    fadd    z2.s, p0/m, z2.s, z6.s
    fadd    z3.s, p0/m, z3.s, z7.s

    sub     x7, x7, x8
    b       .Llayernorm_f32_mean_loop4

.Llayernorm_f32_mean_remainder:
    cbz     x7, .Llayernorm_f32_mean_done

.Llayernorm_f32_mean_loop1:
    whilelt p1.s, xzr, x7
    ld1w    {z4.s}, p1/z, [x6]
    fadd    z0.s, p1/m, z0.s, z4.s
    add     x6, x6, x10
    subs    x7, x7, x5
    b.gt    .Llayernorm_f32_mean_loop1

.Llayernorm_f32_mean_done:
    fadd    z0.s, p0/m, z0.s, z1.s
    fadd    z2.s, p0/m, z2.s, z3.s
    fadd    z0.s, p0/m, z0.s, z2.s
    faddv   s16, p0, z0.s           // s16 = sum

    ucvtf   s17, w4                 // s17 = (float)dim
    fdiv    s16, s16, s17           // s16 = mean
    dup     z16.s, z16.s[0]         // z16 = mean (broadcast to all lanes)

    //=== Pass 2: Compute variance ===
    fmov    z0.s, #0.0
    fmov    z1.s, #0.0
    fmov    z2.s, #0.0
    fmov    z3.s, #0.0

    mov     x6, x19                 // input
    mov     x7, x4                  // dim

.Llayernorm_f32_var_loop4:
    cmp     x7, x8
    b.lt    .Llayernorm_f32_var_remainder

    ld1w    {z4.s}, p0/z, [x6]
    add     x6, x6, x10
    ld1w    {z5.s}, p0/z, [x6]
    add     x6, x6, x10
    ld1w    {z6.s}, p0/z, [x6]
    add     x6, x6, x10
    ld1w    {z7.s}, p0/z, [x6]
    add     x6, x6, x10

    fsub    z4.s, z4.s, z16.s
    fsub    z5.s, z5.s, z16.s
    fsub    z6.s, z6.s, z16.s
    fsub    z7.s, z7.s, z16.s

    fmla    z0.s, p0/m, z4.s, z4.s
    fmla    z1.s, p0/m, z5.s, z5.s
    fmla    z2.s, p0/m, z6.s, z6.s
    fmla    z3.s, p0/m, z7.s, z7.s

    sub     x7, x7, x8
    b       .Llayernorm_f32_var_loop4

.Llayernorm_f32_var_remainder:
    cbz     x7, .Llayernorm_f32_var_done

.Llayernorm_f32_var_loop1:
    whilelt p1.s, xzr, x7
    ld1w    {z4.s}, p1/z, [x6]
    fsub    z4.s, z4.s, z16.s
    fmla    z0.s, p1/m, z4.s, z4.s
    add     x6, x6, x10
    subs    x7, x7, x5
    b.gt    .Llayernorm_f32_var_loop1

.Llayernorm_f32_var_done:
    fadd    z0.s, p0/m, z0.s, z1.s
    fadd    z2.s, p0/m, z2.s, z3.s
    fadd    z0.s, p0/m, z0.s, z2.s
    faddv   s17, p0, z0.s           // s17 = sum_sq

    ucvtf   s18, w4
    fdiv    s17, s17, s18           // s17 = variance
    fadd    s17, s17, s20           // s17 = variance + eps (use saved eps)

    // rsqrte + 2 NR
    frsqrte s18, s17
    fmul    s19, s18, s18
    frsqrts s19, s17, s19
    fmul    s18, s18, s19

    fmul    s19, s18, s18
    frsqrts s19, s17, s19
    fmul    s18, s18, s19           // s18 = inv_std

    dup     z17.s, z18.s[0]         // z17 = inv_std (broadcast to all lanes)

    //=== Pass 3: Normalize, scale, shift ===
    mov     x6, x19                 // input
    mov     x7, x20                 // gamma
    mov     x9, x21                 // beta
    mov     x11, x22                // output
    mov     x12, x4                 // dim

.Llayernorm_f32_norm_loop4:
    cmp     x12, x8
    b.lt    .Llayernorm_f32_norm_remainder

    // Load input (4 vectors)
    ld1w    {z0.s}, p0/z, [x6]
    ld1w    {z1.s}, p0/z, [x6, #1, mul vl]
    ld1w    {z2.s}, p0/z, [x6, #2, mul vl]
    ld1w    {z3.s}, p0/z, [x6, #3, mul vl]
    add     x6, x6, x8, lsl #2

    // Load gamma
    ld1w    {z4.s}, p0/z, [x7]
    ld1w    {z5.s}, p0/z, [x7, #1, mul vl]
    ld1w    {z6.s}, p0/z, [x7, #2, mul vl]
    ld1w    {z7.s}, p0/z, [x7, #3, mul vl]
    add     x7, x7, x8, lsl #2

    // Load beta
    ld1w    {z8.s}, p0/z, [x9]
    ld1w    {z9.s}, p0/z, [x9, #1, mul vl]
    ld1w    {z10.s}, p0/z, [x9, #2, mul vl]
    ld1w    {z11.s}, p0/z, [x9, #3, mul vl]
    add     x9, x9, x8, lsl #2

    // (x - mean) * inv_std
    fsub    z0.s, z0.s, z16.s
    fsub    z1.s, z1.s, z16.s
    fsub    z2.s, z2.s, z16.s
    fsub    z3.s, z3.s, z16.s

    fmul    z0.s, z0.s, z17.s
    fmul    z1.s, z1.s, z17.s
    fmul    z2.s, z2.s, z17.s
    fmul    z3.s, z3.s, z17.s

    // * gamma + beta
    fmla    z8.s, p0/m, z0.s, z4.s
    fmla    z9.s, p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s

    // Store
    st1w    {z8.s}, p0, [x11]
    st1w    {z9.s}, p0, [x11, #1, mul vl]
    st1w    {z10.s}, p0, [x11, #2, mul vl]
    st1w    {z11.s}, p0, [x11, #3, mul vl]
    add     x11, x11, x8, lsl #2

    sub     x12, x12, x8
    b       .Llayernorm_f32_norm_loop4

.Llayernorm_f32_norm_remainder:
    cbz     x12, .Llayernorm_f32_done

.Llayernorm_f32_norm_loop1:
    whilelt p1.s, xzr, x12
    ld1w    {z0.s}, p1/z, [x6]
    ld1w    {z4.s}, p1/z, [x7]
    ld1w    {z8.s}, p1/z, [x9]

    fsub    z0.s, z0.s, z16.s
    fmul    z0.s, z0.s, z17.s
    fmla    z8.s, p1/m, z0.s, z4.s
    st1w    {z8.s}, p1, [x11]

    add     x6, x6, x10
    add     x7, x7, x10
    add     x9, x9, x10
    add     x11, x11, x10
    subs    x12, x12, x5
    b.gt    .Llayernorm_f32_norm_loop1

.Llayernorm_f32_done:
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.size layernorm_f32_asm, .-layernorm_f32_asm

//=============================================================================
// LayerNorm FP64 - Assembly Implementation
// void layernorm_f64_asm(const double* input, const double* gamma,
//                        const double* beta, double* output, size_t dim, double eps)
// x0 = input, x1 = gamma, x2 = beta, x3 = output, x4 = dim, d0 = eps
//=============================================================================
.global layernorm_f64_asm
.type layernorm_f64_asm, %function
layernorm_f64_asm:
    stp     x29, x30, [sp, #-48]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    mov     x29, sp

    // Save eps (d0 is part of z0 which we'll zero)
    fmov    d20, d0                 // d20 = eps (preserved)

    mov     x19, x0                 // input
    mov     x20, x1                 // gamma
    mov     x21, x2                 // beta
    mov     x22, x3                 // output
    // x4 = dim, d0 = eps

    cntd    x5                      // VL in doublewords
    ptrue   p0.d

    // VL bytes stride
    lsl     x10, x5, #3

    //=== Pass 1: Compute mean ===
    fmov    z0.d, #0.0
    fmov    z1.d, #0.0
    fmov    z2.d, #0.0
    fmov    z3.d, #0.0

    mov     x6, x0
    mov     x7, x4
    lsl     x8, x5, #2

.Llayernorm_f64_mean_loop4:
    cmp     x7, x8
    b.lt    .Llayernorm_f64_mean_remainder

    ld1d    {z4.d}, p0/z, [x6]
    add     x6, x6, x10
    ld1d    {z5.d}, p0/z, [x6]
    add     x6, x6, x10
    ld1d    {z6.d}, p0/z, [x6]
    add     x6, x6, x10
    ld1d    {z7.d}, p0/z, [x6]
    add     x6, x6, x10

    fadd    z0.d, p0/m, z0.d, z4.d
    fadd    z1.d, p0/m, z1.d, z5.d
    fadd    z2.d, p0/m, z2.d, z6.d
    fadd    z3.d, p0/m, z3.d, z7.d

    sub     x7, x7, x8
    b       .Llayernorm_f64_mean_loop4

.Llayernorm_f64_mean_remainder:
    cbz     x7, .Llayernorm_f64_mean_done

.Llayernorm_f64_mean_loop1:
    whilelt p1.d, xzr, x7
    ld1d    {z4.d}, p1/z, [x6]
    fadd    z0.d, p1/m, z0.d, z4.d
    add     x6, x6, x10
    subs    x7, x7, x5
    b.gt    .Llayernorm_f64_mean_loop1

.Llayernorm_f64_mean_done:
    fadd    z0.d, p0/m, z0.d, z1.d
    fadd    z2.d, p0/m, z2.d, z3.d
    fadd    z0.d, p0/m, z0.d, z2.d
    faddv   d16, p0, z0.d

    ucvtf   d17, x4
    fdiv    d16, d16, d17           // d16 = mean
    dup     z16.d, z16.d[0]

    //=== Pass 2: Compute variance ===
    fmov    z0.d, #0.0
    fmov    z1.d, #0.0
    fmov    z2.d, #0.0
    fmov    z3.d, #0.0

    mov     x6, x19
    mov     x7, x4

.Llayernorm_f64_var_loop4:
    cmp     x7, x8
    b.lt    .Llayernorm_f64_var_remainder

    ld1d    {z4.d}, p0/z, [x6]
    add     x6, x6, x10
    ld1d    {z5.d}, p0/z, [x6]
    add     x6, x6, x10
    ld1d    {z6.d}, p0/z, [x6]
    add     x6, x6, x10
    ld1d    {z7.d}, p0/z, [x6]
    add     x6, x6, x10

    fsub    z4.d, z4.d, z16.d
    fsub    z5.d, z5.d, z16.d
    fsub    z6.d, z6.d, z16.d
    fsub    z7.d, z7.d, z16.d

    fmla    z0.d, p0/m, z4.d, z4.d
    fmla    z1.d, p0/m, z5.d, z5.d
    fmla    z2.d, p0/m, z6.d, z6.d
    fmla    z3.d, p0/m, z7.d, z7.d

    sub     x7, x7, x8
    b       .Llayernorm_f64_var_loop4

.Llayernorm_f64_var_remainder:
    cbz     x7, .Llayernorm_f64_var_done

.Llayernorm_f64_var_loop1:
    whilelt p1.d, xzr, x7
    ld1d    {z4.d}, p1/z, [x6]
    fsub    z4.d, z4.d, z16.d
    fmla    z0.d, p1/m, z4.d, z4.d
    add     x6, x6, x10
    subs    x7, x7, x5
    b.gt    .Llayernorm_f64_var_loop1

.Llayernorm_f64_var_done:
    fadd    z0.d, p0/m, z0.d, z1.d
    fadd    z2.d, p0/m, z2.d, z3.d
    fadd    z0.d, p0/m, z0.d, z2.d
    faddv   d17, p0, z0.d

    ucvtf   d18, x4
    fdiv    d17, d17, d18           // d17 = variance
    fadd    d17, d17, d20           // + eps (use saved eps)

    // rsqrte + 3 NR for FP64
    frsqrte d18, d17
    fmul    d19, d18, d18
    frsqrts d19, d17, d19
    fmul    d18, d18, d19

    fmul    d19, d18, d18
    frsqrts d19, d17, d19
    fmul    d18, d18, d19

    fmul    d19, d18, d18
    frsqrts d19, d17, d19
    fmul    d18, d18, d19           // d18 = inv_std

    dup     z17.d, z18.d[0]

    //=== Pass 3: Normalize, scale, shift ===
    mov     x6, x19                 // input
    mov     x7, x20                 // gamma
    mov     x9, x21                 // beta
    mov     x11, x22                // output
    mov     x12, x4                 // dim

.Llayernorm_f64_norm_loop4:
    cmp     x12, x8
    b.lt    .Llayernorm_f64_norm_remainder

    ld1d    {z0.d}, p0/z, [x6]
    ld1d    {z1.d}, p0/z, [x6, #1, mul vl]
    ld1d    {z2.d}, p0/z, [x6, #2, mul vl]
    ld1d    {z3.d}, p0/z, [x6, #3, mul vl]
    add     x6, x6, x8, lsl #3

    ld1d    {z4.d}, p0/z, [x7]
    ld1d    {z5.d}, p0/z, [x7, #1, mul vl]
    ld1d    {z6.d}, p0/z, [x7, #2, mul vl]
    ld1d    {z7.d}, p0/z, [x7, #3, mul vl]
    add     x7, x7, x8, lsl #3

    ld1d    {z8.d}, p0/z, [x9]
    ld1d    {z9.d}, p0/z, [x9, #1, mul vl]
    ld1d    {z10.d}, p0/z, [x9, #2, mul vl]
    ld1d    {z11.d}, p0/z, [x9, #3, mul vl]
    add     x9, x9, x8, lsl #3

    fsub    z0.d, z0.d, z16.d
    fsub    z1.d, z1.d, z16.d
    fsub    z2.d, z2.d, z16.d
    fsub    z3.d, z3.d, z16.d

    fmul    z0.d, z0.d, z17.d
    fmul    z1.d, z1.d, z17.d
    fmul    z2.d, z2.d, z17.d
    fmul    z3.d, z3.d, z17.d

    fmla    z8.d, p0/m, z0.d, z4.d
    fmla    z9.d, p0/m, z1.d, z5.d
    fmla    z10.d, p0/m, z2.d, z6.d
    fmla    z11.d, p0/m, z3.d, z7.d

    st1d    {z8.d}, p0, [x11]
    st1d    {z9.d}, p0, [x11, #1, mul vl]
    st1d    {z10.d}, p0, [x11, #2, mul vl]
    st1d    {z11.d}, p0, [x11, #3, mul vl]
    add     x11, x11, x8, lsl #3

    sub     x12, x12, x8
    b       .Llayernorm_f64_norm_loop4

.Llayernorm_f64_norm_remainder:
    cbz     x12, .Llayernorm_f64_done

.Llayernorm_f64_norm_loop1:
    whilelt p1.d, xzr, x12
    ld1d    {z0.d}, p1/z, [x6]
    ld1d    {z4.d}, p1/z, [x7]
    ld1d    {z8.d}, p1/z, [x9]

    fsub    z0.d, z0.d, z16.d
    fmul    z0.d, z0.d, z17.d
    fmla    z8.d, p1/m, z0.d, z4.d
    st1d    {z8.d}, p1, [x11]

    add     x6, x6, x10
    add     x7, x7, x10
    add     x9, x9, x10
    add     x11, x11, x10
    subs    x12, x12, x5
    b.gt    .Llayernorm_f64_norm_loop1

.Llayernorm_f64_done:
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.size layernorm_f64_asm, .-layernorm_f64_asm

//=============================================================================
// RMSNorm FP32 with fast-math (1 NR iteration only)
// void rmsnorm_f32_fast_asm(const float* input, const float* gamma,
//                           float* output, size_t dim, float eps)
//=============================================================================
.global rmsnorm_f32_fast_asm
.type rmsnorm_f32_fast_asm, %function
rmsnorm_f32_fast_asm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Save eps (s0 is part of z0 which we'll zero)
    fmov    s20, s0                 // s20 = eps (preserved)

    cntw    x4
    ptrue   p0.s
    lsl     x10, x4, #2             // VL bytes stride

    // Sum of squares
    fmov    z0.s, #0.0
    fmov    z1.s, #0.0
    fmov    z2.s, #0.0
    fmov    z3.s, #0.0

    mov     x5, x0
    mov     x6, x3
    lsl     x7, x4, #2

.Lrmsnorm_f32_fast_sum_loop4:
    cmp     x6, x7
    b.lt    .Lrmsnorm_f32_fast_sum_rem

    ld1w    {z4.s}, p0/z, [x5]
    add     x5, x5, x10
    ld1w    {z5.s}, p0/z, [x5]
    add     x5, x5, x10
    ld1w    {z6.s}, p0/z, [x5]
    add     x5, x5, x10
    ld1w    {z7.s}, p0/z, [x5]
    add     x5, x5, x10

    fmla    z0.s, p0/m, z4.s, z4.s
    fmla    z1.s, p0/m, z5.s, z5.s
    fmla    z2.s, p0/m, z6.s, z6.s
    fmla    z3.s, p0/m, z7.s, z7.s

    sub     x6, x6, x7
    b       .Lrmsnorm_f32_fast_sum_loop4

.Lrmsnorm_f32_fast_sum_rem:
    cbz     x6, .Lrmsnorm_f32_fast_sum_done

.Lrmsnorm_f32_fast_sum_loop1:
    whilelt p1.s, xzr, x6
    ld1w    {z4.s}, p1/z, [x5]
    fmla    z0.s, p1/m, z4.s, z4.s
    add     x5, x5, x10
    subs    x6, x6, x4
    b.gt    .Lrmsnorm_f32_fast_sum_loop1

.Lrmsnorm_f32_fast_sum_done:
    fadd    z0.s, p0/m, z0.s, z1.s
    fadd    z2.s, p0/m, z2.s, z3.s
    fadd    z0.s, p0/m, z0.s, z2.s
    faddv   s4, p0, z0.s

    // Fast inverse: recpe for 1/dim (1 NR)
    ucvtf   s5, w3
    frecpe  s6, s5
    frecps  s7, s5, s6
    fmul    s6, s6, s7

    fmul    s4, s4, s6              // mean_sq
    fadd    s4, s4, s20             // + eps (use saved value)

    // rsqrte + 1 NR only (fast)
    frsqrte s5, s4
    fmul    s6, s5, s5
    frsqrts s6, s4, s6
    fmul    s5, s5, s6              // inv_std (1 NR only)

    dup     z8.s, z5.s[0]

    // Normalize
    mov     x5, x0
    mov     x6, x1
    mov     x7, x2
    mov     x8, x3
    lsl     x9, x4, #2

.Lrmsnorm_f32_fast_norm_loop4:
    cmp     x8, x9
    b.lt    .Lrmsnorm_f32_fast_norm_rem

    ld1w    {z0.s}, p0/z, [x5]
    ld1w    {z4.s}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    ld1w    {z1.s}, p0/z, [x5]
    ld1w    {z5.s}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    ld1w    {z2.s}, p0/z, [x5]
    ld1w    {z6.s}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    ld1w    {z3.s}, p0/z, [x5]
    ld1w    {z7.s}, p0/z, [x6]
    add     x5, x5, x10
    add     x6, x6, x10

    fmul    z0.s, z0.s, z8.s
    fmul    z1.s, z1.s, z8.s
    fmul    z2.s, z2.s, z8.s
    fmul    z3.s, z3.s, z8.s

    fmul    z0.s, z0.s, z4.s
    fmul    z1.s, z1.s, z5.s
    fmul    z2.s, z2.s, z6.s
    fmul    z3.s, z3.s, z7.s

    st1w    {z0.s}, p0, [x7]
    add     x7, x7, x10
    st1w    {z1.s}, p0, [x7]
    add     x7, x7, x10
    st1w    {z2.s}, p0, [x7]
    add     x7, x7, x10
    st1w    {z3.s}, p0, [x7]
    add     x7, x7, x10

    sub     x8, x8, x9
    b       .Lrmsnorm_f32_fast_norm_loop4

.Lrmsnorm_f32_fast_norm_rem:
    cbz     x8, .Lrmsnorm_f32_fast_done

.Lrmsnorm_f32_fast_norm_loop1:
    whilelt p1.s, xzr, x8
    ld1w    {z0.s}, p1/z, [x5]
    ld1w    {z4.s}, p1/z, [x6]
    fmul    z0.s, z0.s, z8.s
    fmul    z0.s, z0.s, z4.s
    st1w    {z0.s}, p1, [x7]
    add     x5, x5, x10
    add     x6, x6, x10
    add     x7, x7, x10
    subs    x8, x8, x4
    b.gt    .Lrmsnorm_f32_fast_norm_loop1

.Lrmsnorm_f32_fast_done:
    ldp     x29, x30, [sp], #16
    ret
.size rmsnorm_f32_fast_asm, .-rmsnorm_f32_fast_asm
