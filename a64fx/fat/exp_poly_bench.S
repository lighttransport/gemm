// exp_poly_bench.S
// SVE exp() implementations with varying polynomial terms (1-5)
// Tests accuracy vs performance tradeoff
// Target: A64FX with 512-bit SVE
// Optimized: Load constants from L1 cache instead of fmov/fdup (FLA port limited)

    .arch armv8.2-a+sve
    .text

// Macro to save callee-saved SVE registers z8-z15
.macro SAVE_SVE_REGS
    sub     sp, sp, #512
    str     z8, [sp, #0, mul vl]
    str     z9, [sp, #1, mul vl]
    str     z10, [sp, #2, mul vl]
    str     z11, [sp, #3, mul vl]
    str     z12, [sp, #4, mul vl]
    str     z13, [sp, #5, mul vl]
    str     z14, [sp, #6, mul vl]
    str     z15, [sp, #7, mul vl]
.endm

// Macro to restore callee-saved SVE registers z8-z15
.macro RESTORE_SVE_REGS
    ldr     z8, [sp, #0, mul vl]
    ldr     z9, [sp, #1, mul vl]
    ldr     z10, [sp, #2, mul vl]
    ldr     z11, [sp, #3, mul vl]
    ldr     z12, [sp, #4, mul vl]
    ldr     z13, [sp, #5, mul vl]
    ldr     z14, [sp, #6, mul vl]
    ldr     z15, [sp, #7, mul vl]
    add     sp, sp, #512
.endm

// ============================================
// FP32 exp with 1-term polynomial: exp(r) ≈ 1 + r
// Very fast but low accuracy
// ============================================
    .align 6
    .global exp_f32_poly1
    .type exp_f32_poly1, %function
exp_f32_poly1:
    SAVE_SVE_REGS
    ptrue   p0.s

    // Load constants from L1 cache
    adrp    x10, .Lconst_f32
    add     x10, x10, :lo12:.Lconst_f32
    ld1rw   z28.s, p0/z, [x10]              // 1/ln(2)
    ld1rw   z29.s, p0/z, [x10, #4]          // ln(2)
    ld1rw   z30.s, p0/z, [x10, #8]          // 1.0

    cntw    x3
    lsl     x4, x3, #2                      // 4 vectors
    mov     x5, x2

.Lloop_f32_p1:
    cmp     x5, x4
    b.lt    .Ltail_f32_p1

    // Load 4 vectors
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // n = round(x / ln2)
    fmul    z4.s, z0.s, z28.s
    fmul    z5.s, z1.s, z28.s
    fmul    z6.s, z2.s, z28.s
    fmul    z7.s, z3.s, z28.s

    frintn  z4.s, p0/m, z4.s
    frintn  z5.s, p0/m, z5.s
    frintn  z6.s, p0/m, z6.s
    frintn  z7.s, p0/m, z7.s

    // r = x - n * ln2
    fmls    z0.s, p0/m, z4.s, z29.s
    fmls    z1.s, p0/m, z5.s, z29.s
    fmls    z2.s, p0/m, z6.s, z29.s
    fmls    z3.s, p0/m, z7.s, z29.s

    // 2^n via IEEE754 exponent manipulation
    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    add     z5.s, z5.s, z31.s
    add     z6.s, z6.s, z31.s
    add     z7.s, z7.s, z31.s

    lsl     z4.s, z4.s, #23
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // 1-term: p = 1 + r
    fadd    z8.s, z30.s, z0.s
    fadd    z9.s, z30.s, z1.s
    fadd    z10.s, z30.s, z2.s
    fadd    z11.s, z30.s, z3.s

    // result = 2^n * p (z4-z7 reinterpreted as float)
    fmul    z8.s, z8.s, z4.s
    fmul    z9.s, z9.s, z5.s
    fmul    z10.s, z10.s, z6.s
    fmul    z11.s, z11.s, z7.s

    st1w    z8.s, p0, [x1]
    st1w    z9.s, p0, [x1, #1, mul vl]
    st1w    z10.s, p0, [x1, #2, mul vl]
    st1w    z11.s, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x5, x5, x4
    b       .Lloop_f32_p1

.Ltail_f32_p1:
    cbz     x5, .Ldone_f32_p1
    whilelt p1.s, xzr, x5
    ld1w    z0.s, p1/z, [x0]
    fmul    z4.s, z0.s, z28.s
    frintn  z4.s, p0/m, z4.s
    fmls    z0.s, p0/m, z4.s, z29.s
    fcvtzs  z4.s, p0/m, z4.s
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    lsl     z4.s, z4.s, #23
    fadd    z8.s, z30.s, z0.s
    fmul    z8.s, z8.s, z4.s
    st1w    z8.s, p1, [x1]
    add     x0, x0, x3, lsl #2
    add     x1, x1, x3, lsl #2
    sub     x5, x5, x3
    b       .Ltail_f32_p1

.Ldone_f32_p1:
    RESTORE_SVE_REGS
    ret
    .size exp_f32_poly1, .-exp_f32_poly1


// ============================================
// FP32 exp with 2-term polynomial: exp(r) ≈ 1 + r(1 + r/2)
// ============================================
    .align 6
    .global exp_f32_poly2
    .type exp_f32_poly2, %function
exp_f32_poly2:
    SAVE_SVE_REGS
    ptrue   p0.s

    adrp    x10, .Lconst_f32
    add     x10, x10, :lo12:.Lconst_f32
    ld1rw   z28.s, p0/z, [x10]              // 1/ln(2)
    ld1rw   z29.s, p0/z, [x10, #4]          // ln(2)
    ld1rw   z30.s, p0/z, [x10, #8]          // 1.0
    ld1rw   z27.s, p0/z, [x10, #12]         // 0.5

    cntw    x3
    lsl     x4, x3, #2
    mov     x5, x2

.Lloop_f32_p2:
    cmp     x5, x4
    b.lt    .Ltail_f32_p2

    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    fmul    z4.s, z0.s, z28.s
    fmul    z5.s, z1.s, z28.s
    fmul    z6.s, z2.s, z28.s
    fmul    z7.s, z3.s, z28.s

    frintn  z4.s, p0/m, z4.s
    frintn  z5.s, p0/m, z5.s
    frintn  z6.s, p0/m, z6.s
    frintn  z7.s, p0/m, z7.s

    fmls   z0.s, p0/m, z4.s, z29.s
    fmls   z1.s, p0/m, z5.s, z29.s
    fmls   z2.s, p0/m, z6.s, z29.s
    fmls   z3.s, p0/m, z7.s, z29.s

    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    add     z5.s, z5.s, z31.s
    add     z6.s, z6.s, z31.s
    add     z7.s, z7.s, z31.s

    lsl     z4.s, z4.s, #23
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // 2-term Horner: p = 1 + r*(1 + r*0.5) = 1 + r + r^2/2
    // Start: p = 0.5*r + 1
    movprfx z8, z30
    fmla    z8.s, p0/m, z27.s, z0.s
    movprfx z9, z30
    fmla    z9.s, p0/m, z27.s, z1.s
    movprfx z10, z30
    fmla    z10.s, p0/m, z27.s, z2.s
    movprfx z11, z30
    fmla    z11.s, p0/m, z27.s, z3.s

    // p = p*r + 1
    fmad    z8.s, p0/m, z0.s, z30.s
    fmad    z9.s, p0/m, z1.s, z30.s
    fmad    z10.s, p0/m, z2.s, z30.s
    fmad    z11.s, p0/m, z3.s, z30.s

    fmul    z8.s, z8.s, z4.s
    fmul    z9.s, z9.s, z5.s
    fmul    z10.s, z10.s, z6.s
    fmul    z11.s, z11.s, z7.s

    st1w    z8.s, p0, [x1]
    st1w    z9.s, p0, [x1, #1, mul vl]
    st1w    z10.s, p0, [x1, #2, mul vl]
    st1w    z11.s, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x5, x5, x4
    b       .Lloop_f32_p2

.Ltail_f32_p2:
    cbz     x5, .Ldone_f32_p2
    whilelt p1.s, xzr, x5
    ld1w    z0.s, p1/z, [x0]
    fmul    z4.s, z0.s, z28.s
    frintn  z4.s, p0/m, z4.s
    fmls    z0.s, p0/m, z4.s, z29.s
    fcvtzs  z4.s, p0/m, z4.s
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    lsl     z4.s, z4.s, #23
    movprfx z8, z30
    fmla    z8.s, p0/m, z27.s, z0.s
    fmad    z8.s, p0/m, z0.s, z30.s
    fmul    z8.s, z8.s, z4.s
    st1w    z8.s, p1, [x1]
    add     x0, x0, x3, lsl #2
    add     x1, x1, x3, lsl #2
    sub     x5, x5, x3
    b       .Ltail_f32_p2

.Ldone_f32_p2:
    RESTORE_SVE_REGS
    ret
    .size exp_f32_poly2, .-exp_f32_poly2


// ============================================
// FP32 exp with 3-term polynomial: exp(r) ≈ 1 + r(1 + r(0.5 + r/6))
// ============================================
    .align 6
    .global exp_f32_poly3
    .type exp_f32_poly3, %function
exp_f32_poly3:
    SAVE_SVE_REGS
    ptrue   p0.s

    adrp    x10, .Lconst_f32
    add     x10, x10, :lo12:.Lconst_f32
    ld1rw   z28.s, p0/z, [x10]              // 1/ln(2)
    ld1rw   z29.s, p0/z, [x10, #4]          // ln(2)
    ld1rw   z30.s, p0/z, [x10, #8]          // 1.0
    ld1rw   z27.s, p0/z, [x10, #12]         // 0.5
    ld1rw   z26.s, p0/z, [x10, #16]         // 1/6

    cntw    x3
    lsl     x4, x3, #2
    mov     x5, x2

.Lloop_f32_p3:
    cmp     x5, x4
    b.lt    .Ltail_f32_p3

    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    fmul    z4.s, z0.s, z28.s
    fmul    z5.s, z1.s, z28.s
    fmul    z6.s, z2.s, z28.s
    fmul    z7.s, z3.s, z28.s

    frintn  z4.s, p0/m, z4.s
    frintn  z5.s, p0/m, z5.s
    frintn  z6.s, p0/m, z6.s
    frintn  z7.s, p0/m, z7.s

    fmls   z0.s, p0/m, z4.s, z29.s
    fmls   z1.s, p0/m, z5.s, z29.s
    fmls   z2.s, p0/m, z6.s, z29.s
    fmls   z3.s, p0/m, z7.s, z29.s

    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    add     z5.s, z5.s, z31.s
    add     z6.s, z6.s, z31.s
    add     z7.s, z7.s, z31.s

    lsl     z4.s, z4.s, #23
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // 3-term Horner: p = 1 + r*(1 + r*(0.5 + r/6))
    // Start: p = (1/6)*r + 0.5
    movprfx z8, z27
    fmla    z8.s, p0/m, z26.s, z0.s
    movprfx z9, z27
    fmla    z9.s, p0/m, z26.s, z1.s
    movprfx z10, z27
    fmla    z10.s, p0/m, z26.s, z2.s
    movprfx z11, z27
    fmla    z11.s, p0/m, z26.s, z3.s

    // p = p*r + 1
    fmad    z8.s, p0/m, z0.s, z30.s
    fmad    z9.s, p0/m, z1.s, z30.s
    fmad    z10.s, p0/m, z2.s, z30.s
    fmad    z11.s, p0/m, z3.s, z30.s

    // p = p*r + 1
    fmad    z8.s, p0/m, z0.s, z30.s
    fmad    z9.s, p0/m, z1.s, z30.s
    fmad    z10.s, p0/m, z2.s, z30.s
    fmad    z11.s, p0/m, z3.s, z30.s

    fmul    z8.s, z8.s, z4.s
    fmul    z9.s, z9.s, z5.s
    fmul    z10.s, z10.s, z6.s
    fmul    z11.s, z11.s, z7.s

    st1w    z8.s, p0, [x1]
    st1w    z9.s, p0, [x1, #1, mul vl]
    st1w    z10.s, p0, [x1, #2, mul vl]
    st1w    z11.s, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x5, x5, x4
    b       .Lloop_f32_p3

.Ltail_f32_p3:
    cbz     x5, .Ldone_f32_p3
    whilelt p1.s, xzr, x5
    ld1w    z0.s, p1/z, [x0]
    fmul    z4.s, z0.s, z28.s
    frintn  z4.s, p0/m, z4.s
    fmls    z0.s, p0/m, z4.s, z29.s
    fcvtzs  z4.s, p0/m, z4.s
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    lsl     z4.s, z4.s, #23
    movprfx z8, z27
    fmla    z8.s, p0/m, z26.s, z0.s
    fmad    z8.s, p0/m, z0.s, z30.s
    fmad    z8.s, p0/m, z0.s, z30.s
    fmul    z8.s, z8.s, z4.s
    st1w    z8.s, p1, [x1]
    add     x0, x0, x3, lsl #2
    add     x1, x1, x3, lsl #2
    sub     x5, x5, x3
    b       .Ltail_f32_p3

.Ldone_f32_p3:
    RESTORE_SVE_REGS
    ret
    .size exp_f32_poly3, .-exp_f32_poly3


// ============================================
// FP32 exp with 4-term polynomial: exp(r) ≈ 1 + r(1 + r(0.5 + r(1/6 + r/24)))
// ============================================
    .align 6
    .global exp_f32_poly4
    .type exp_f32_poly4, %function
exp_f32_poly4:
    SAVE_SVE_REGS
    ptrue   p0.s

    adrp    x10, .Lconst_f32
    add     x10, x10, :lo12:.Lconst_f32
    ld1rw   z28.s, p0/z, [x10]              // 1/ln(2)
    ld1rw   z29.s, p0/z, [x10, #4]          // ln(2)
    ld1rw   z30.s, p0/z, [x10, #8]          // 1.0
    ld1rw   z27.s, p0/z, [x10, #12]         // 0.5
    ld1rw   z26.s, p0/z, [x10, #16]         // 1/6
    ld1rw   z25.s, p0/z, [x10, #20]         // 1/24

    cntw    x3
    lsl     x4, x3, #2
    mov     x5, x2

.Lloop_f32_p4:
    cmp     x5, x4
    b.lt    .Ltail_f32_p4

    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    fmul    z4.s, z0.s, z28.s
    fmul    z5.s, z1.s, z28.s
    fmul    z6.s, z2.s, z28.s
    fmul    z7.s, z3.s, z28.s

    frintn  z4.s, p0/m, z4.s
    frintn  z5.s, p0/m, z5.s
    frintn  z6.s, p0/m, z6.s
    frintn  z7.s, p0/m, z7.s

    fmls   z0.s, p0/m, z4.s, z29.s
    fmls   z1.s, p0/m, z5.s, z29.s
    fmls   z2.s, p0/m, z6.s, z29.s
    fmls   z3.s, p0/m, z7.s, z29.s

    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    add     z5.s, z5.s, z31.s
    add     z6.s, z6.s, z31.s
    add     z7.s, z7.s, z31.s

    lsl     z4.s, z4.s, #23
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // 4-term Horner: p = 1 + r*(1 + r*(0.5 + r*(1/6 + r/24)))
    // Start: p = (1/24)*r + 1/6
    movprfx z8, z26
    fmla    z8.s, p0/m, z25.s, z0.s
    movprfx z9, z26
    fmla    z9.s, p0/m, z25.s, z1.s
    movprfx z10, z26
    fmla    z10.s, p0/m, z25.s, z2.s
    movprfx z11, z26
    fmla    z11.s, p0/m, z25.s, z3.s

    // p = p*r + 0.5
    fmad    z8.s, p0/m, z0.s, z27.s
    fmad    z9.s, p0/m, z1.s, z27.s
    fmad    z10.s, p0/m, z2.s, z27.s
    fmad    z11.s, p0/m, z3.s, z27.s

    // p = p*r + 1
    fmad    z8.s, p0/m, z0.s, z30.s
    fmad    z9.s, p0/m, z1.s, z30.s
    fmad    z10.s, p0/m, z2.s, z30.s
    fmad    z11.s, p0/m, z3.s, z30.s

    // p = p*r + 1
    fmad    z8.s, p0/m, z0.s, z30.s
    fmad    z9.s, p0/m, z1.s, z30.s
    fmad    z10.s, p0/m, z2.s, z30.s
    fmad    z11.s, p0/m, z3.s, z30.s

    fmul    z8.s, z8.s, z4.s
    fmul    z9.s, z9.s, z5.s
    fmul    z10.s, z10.s, z6.s
    fmul    z11.s, z11.s, z7.s

    st1w    z8.s, p0, [x1]
    st1w    z9.s, p0, [x1, #1, mul vl]
    st1w    z10.s, p0, [x1, #2, mul vl]
    st1w    z11.s, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x5, x5, x4
    b       .Lloop_f32_p4

.Ltail_f32_p4:
    cbz     x5, .Ldone_f32_p4
    whilelt p1.s, xzr, x5
    ld1w    z0.s, p1/z, [x0]
    fmul    z4.s, z0.s, z28.s
    frintn  z4.s, p0/m, z4.s
    fmls    z0.s, p0/m, z4.s, z29.s
    fcvtzs  z4.s, p0/m, z4.s
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    lsl     z4.s, z4.s, #23
    movprfx z8, z26
    fmla    z8.s, p0/m, z25.s, z0.s
    fmad    z8.s, p0/m, z0.s, z27.s
    fmad    z8.s, p0/m, z0.s, z30.s
    fmad    z8.s, p0/m, z0.s, z30.s
    fmul    z8.s, z8.s, z4.s
    st1w    z8.s, p1, [x1]
    add     x0, x0, x3, lsl #2
    add     x1, x1, x3, lsl #2
    sub     x5, x5, x3
    b       .Ltail_f32_p4

.Ldone_f32_p4:
    RESTORE_SVE_REGS
    ret
    .size exp_f32_poly4, .-exp_f32_poly4


// ============================================
// FP32 exp with 5-term polynomial: exp(r) ≈ 1 + r(1 + r(0.5 + r(1/6 + r(1/24 + r/120))))
// ============================================
    .align 6
    .global exp_f32_poly5
    .type exp_f32_poly5, %function
exp_f32_poly5:
    SAVE_SVE_REGS
    ptrue   p0.s

    adrp    x10, .Lconst_f32
    add     x10, x10, :lo12:.Lconst_f32
    ld1rw   z28.s, p0/z, [x10]              // 1/ln(2)
    ld1rw   z29.s, p0/z, [x10, #4]          // ln(2)
    ld1rw   z30.s, p0/z, [x10, #8]          // 1.0
    ld1rw   z27.s, p0/z, [x10, #12]         // 0.5
    ld1rw   z26.s, p0/z, [x10, #16]         // 1/6
    ld1rw   z25.s, p0/z, [x10, #20]         // 1/24
    ld1rw   z24.s, p0/z, [x10, #24]         // 1/120

    cntw    x3
    lsl     x4, x3, #2
    mov     x5, x2

.Lloop_f32_p5:
    cmp     x5, x4
    b.lt    .Ltail_f32_p5

    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    fmul    z4.s, z0.s, z28.s
    fmul    z5.s, z1.s, z28.s
    fmul    z6.s, z2.s, z28.s
    fmul    z7.s, z3.s, z28.s

    frintn  z4.s, p0/m, z4.s
    frintn  z5.s, p0/m, z5.s
    frintn  z6.s, p0/m, z6.s
    frintn  z7.s, p0/m, z7.s

    fmls   z0.s, p0/m, z4.s, z29.s
    fmls   z1.s, p0/m, z5.s, z29.s
    fmls   z2.s, p0/m, z6.s, z29.s
    fmls   z3.s, p0/m, z7.s, z29.s

    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    add     z5.s, z5.s, z31.s
    add     z6.s, z6.s, z31.s
    add     z7.s, z7.s, z31.s

    lsl     z4.s, z4.s, #23
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // 5-term Horner: p = 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120))))
    // Start: p = (1/120)*r + 1/24
    movprfx z8, z25
    fmla    z8.s, p0/m, z24.s, z0.s
    movprfx z9, z25
    fmla    z9.s, p0/m, z24.s, z1.s
    movprfx z10, z25
    fmla    z10.s, p0/m, z24.s, z2.s
    movprfx z11, z25
    fmla    z11.s, p0/m, z24.s, z3.s

    // p = p*r + 1/6
    fmad    z8.s, p0/m, z0.s, z26.s
    fmad    z9.s, p0/m, z1.s, z26.s
    fmad    z10.s, p0/m, z2.s, z26.s
    fmad    z11.s, p0/m, z3.s, z26.s

    // p = p*r + 0.5
    fmad    z8.s, p0/m, z0.s, z27.s
    fmad    z9.s, p0/m, z1.s, z27.s
    fmad    z10.s, p0/m, z2.s, z27.s
    fmad    z11.s, p0/m, z3.s, z27.s

    // p = p*r + 1
    fmad    z8.s, p0/m, z0.s, z30.s
    fmad    z9.s, p0/m, z1.s, z30.s
    fmad    z10.s, p0/m, z2.s, z30.s
    fmad    z11.s, p0/m, z3.s, z30.s

    // p = p*r + 1
    fmad    z8.s, p0/m, z0.s, z30.s
    fmad    z9.s, p0/m, z1.s, z30.s
    fmad    z10.s, p0/m, z2.s, z30.s
    fmad    z11.s, p0/m, z3.s, z30.s

    fmul    z8.s, z8.s, z4.s
    fmul    z9.s, z9.s, z5.s
    fmul    z10.s, z10.s, z6.s
    fmul    z11.s, z11.s, z7.s

    st1w    z8.s, p0, [x1]
    st1w    z9.s, p0, [x1, #1, mul vl]
    st1w    z10.s, p0, [x1, #2, mul vl]
    st1w    z11.s, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x5, x5, x4
    b       .Lloop_f32_p5

.Ltail_f32_p5:
    cbz     x5, .Ldone_f32_p5
    whilelt p1.s, xzr, x5
    ld1w    z0.s, p1/z, [x0]
    fmul    z4.s, z0.s, z28.s
    frintn  z4.s, p0/m, z4.s
    fmls    z0.s, p0/m, z4.s, z29.s
    fcvtzs  z4.s, p0/m, z4.s
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    lsl     z4.s, z4.s, #23
    movprfx z8, z25
    fmla    z8.s, p0/m, z24.s, z0.s
    fmad    z8.s, p0/m, z0.s, z26.s
    fmad    z8.s, p0/m, z0.s, z27.s
    fmad    z8.s, p0/m, z0.s, z30.s
    fmad    z8.s, p0/m, z0.s, z30.s
    fmul    z8.s, z8.s, z4.s
    st1w    z8.s, p1, [x1]
    add     x0, x0, x3, lsl #2
    add     x1, x1, x3, lsl #2
    sub     x5, x5, x3
    b       .Ltail_f32_p5

.Ldone_f32_p5:
    RESTORE_SVE_REGS
    ret
    .size exp_f32_poly5, .-exp_f32_poly5


// ============================================
// FP64 exp with 1-term polynomial
// ============================================
    .align 6
    .global exp_f64_poly1
    .type exp_f64_poly1, %function
exp_f64_poly1:
    SAVE_SVE_REGS
    ptrue   p0.d

    adrp    x10, .Lconst_f64
    add     x10, x10, :lo12:.Lconst_f64
    ld1rd   z28.d, p0/z, [x10]              // 1/ln(2)
    ld1rd   z29.d, p0/z, [x10, #8]          // ln(2)
    ld1rd   z30.d, p0/z, [x10, #16]         // 1.0

    cntd    x3
    lsl     x4, x3, #2
    mov     x5, x2

.Lloop_f64_p1:
    cmp     x5, x4
    b.lt    .Ltail_f64_p1

    ld1d    z0.d, p0/z, [x0]
    ld1d    z1.d, p0/z, [x0, #1, mul vl]
    ld1d    z2.d, p0/z, [x0, #2, mul vl]
    ld1d    z3.d, p0/z, [x0, #3, mul vl]

    fmul    z4.d, z0.d, z28.d
    fmul    z5.d, z1.d, z28.d
    fmul    z6.d, z2.d, z28.d
    fmul    z7.d, z3.d, z28.d

    frintn  z4.d, p0/m, z4.d
    frintn  z5.d, p0/m, z5.d
    frintn  z6.d, p0/m, z6.d
    frintn  z7.d, p0/m, z7.d

    fmls   z0.d, p0/m, z4.d, z29.d
    fmls   z1.d, p0/m, z5.d, z29.d
    fmls   z2.d, p0/m, z6.d, z29.d
    fmls   z3.d, p0/m, z7.d, z29.d

    fcvtzs  z4.d, p0/m, z4.d
    fcvtzs  z5.d, p0/m, z5.d
    fcvtzs  z6.d, p0/m, z6.d
    fcvtzs  z7.d, p0/m, z7.d

    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    add     z5.d, z5.d, z31.d
    add     z6.d, z6.d, z31.d
    add     z7.d, z7.d, z31.d

    lsl     z4.d, z4.d, #52
    lsl     z5.d, z5.d, #52
    lsl     z6.d, z6.d, #52
    lsl     z7.d, z7.d, #52

    // 1-term: p = 1 + r
    fadd    z8.d, z30.d, z0.d
    fadd    z9.d, z30.d, z1.d
    fadd    z10.d, z30.d, z2.d
    fadd    z11.d, z30.d, z3.d

    fmul    z8.d, z8.d, z4.d
    fmul    z9.d, z9.d, z5.d
    fmul    z10.d, z10.d, z6.d
    fmul    z11.d, z11.d, z7.d

    st1d    z8.d, p0, [x1]
    st1d    z9.d, p0, [x1, #1, mul vl]
    st1d    z10.d, p0, [x1, #2, mul vl]
    st1d    z11.d, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #3
    add     x1, x1, x4, lsl #3
    sub     x5, x5, x4
    b       .Lloop_f64_p1

.Ltail_f64_p1:
    cbz     x5, .Ldone_f64_p1
    whilelt p1.d, xzr, x5
    ld1d    z0.d, p1/z, [x0]
    fmul    z4.d, z0.d, z28.d
    frintn  z4.d, p0/m, z4.d
    fmls    z0.d, p0/m, z4.d, z29.d
    fcvtzs  z4.d, p0/m, z4.d
    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    lsl     z4.d, z4.d, #52
    fadd    z8.d, z30.d, z0.d
    fmul    z8.d, z8.d, z4.d
    st1d    z8.d, p1, [x1]
    add     x0, x0, x3, lsl #3
    add     x1, x1, x3, lsl #3
    sub     x5, x5, x3
    b       .Ltail_f64_p1

.Ldone_f64_p1:
    RESTORE_SVE_REGS
    ret
    .size exp_f64_poly1, .-exp_f64_poly1


// ============================================
// FP64 exp with 2-term polynomial
// ============================================
    .align 6
    .global exp_f64_poly2
    .type exp_f64_poly2, %function
exp_f64_poly2:
    SAVE_SVE_REGS
    ptrue   p0.d

    adrp    x10, .Lconst_f64
    add     x10, x10, :lo12:.Lconst_f64
    ld1rd   z28.d, p0/z, [x10]              // 1/ln(2)
    ld1rd   z29.d, p0/z, [x10, #8]          // ln(2)
    ld1rd   z30.d, p0/z, [x10, #16]         // 1.0
    ld1rd   z27.d, p0/z, [x10, #24]         // 0.5

    cntd    x3
    lsl     x4, x3, #2
    mov     x5, x2

.Lloop_f64_p2:
    cmp     x5, x4
    b.lt    .Ltail_f64_p2

    ld1d    z0.d, p0/z, [x0]
    ld1d    z1.d, p0/z, [x0, #1, mul vl]
    ld1d    z2.d, p0/z, [x0, #2, mul vl]
    ld1d    z3.d, p0/z, [x0, #3, mul vl]

    fmul    z4.d, z0.d, z28.d
    fmul    z5.d, z1.d, z28.d
    fmul    z6.d, z2.d, z28.d
    fmul    z7.d, z3.d, z28.d

    frintn  z4.d, p0/m, z4.d
    frintn  z5.d, p0/m, z5.d
    frintn  z6.d, p0/m, z6.d
    frintn  z7.d, p0/m, z7.d

    fmls   z0.d, p0/m, z4.d, z29.d
    fmls   z1.d, p0/m, z5.d, z29.d
    fmls   z2.d, p0/m, z6.d, z29.d
    fmls   z3.d, p0/m, z7.d, z29.d

    fcvtzs  z4.d, p0/m, z4.d
    fcvtzs  z5.d, p0/m, z5.d
    fcvtzs  z6.d, p0/m, z6.d
    fcvtzs  z7.d, p0/m, z7.d

    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    add     z5.d, z5.d, z31.d
    add     z6.d, z6.d, z31.d
    add     z7.d, z7.d, z31.d

    lsl     z4.d, z4.d, #52
    lsl     z5.d, z5.d, #52
    lsl     z6.d, z6.d, #52
    lsl     z7.d, z7.d, #52

    // 2-term
    movprfx z8, z30
    fmla    z8.d, p0/m, z27.d, z0.d
    movprfx z9, z30
    fmla    z9.d, p0/m, z27.d, z1.d
    movprfx z10, z30
    fmla    z10.d, p0/m, z27.d, z2.d
    movprfx z11, z30
    fmla    z11.d, p0/m, z27.d, z3.d

    fmad    z8.d, p0/m, z0.d, z30.d
    fmad    z9.d, p0/m, z1.d, z30.d
    fmad    z10.d, p0/m, z2.d, z30.d
    fmad    z11.d, p0/m, z3.d, z30.d

    fmul    z8.d, z8.d, z4.d
    fmul    z9.d, z9.d, z5.d
    fmul    z10.d, z10.d, z6.d
    fmul    z11.d, z11.d, z7.d

    st1d    z8.d, p0, [x1]
    st1d    z9.d, p0, [x1, #1, mul vl]
    st1d    z10.d, p0, [x1, #2, mul vl]
    st1d    z11.d, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #3
    add     x1, x1, x4, lsl #3
    sub     x5, x5, x4
    b       .Lloop_f64_p2

.Ltail_f64_p2:
    cbz     x5, .Ldone_f64_p2
    whilelt p1.d, xzr, x5
    ld1d    z0.d, p1/z, [x0]
    fmul    z4.d, z0.d, z28.d
    frintn  z4.d, p0/m, z4.d
    fmls    z0.d, p0/m, z4.d, z29.d
    fcvtzs  z4.d, p0/m, z4.d
    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    lsl     z4.d, z4.d, #52
    movprfx z8, z30
    fmla    z8.d, p0/m, z27.d, z0.d
    fmad    z8.d, p0/m, z0.d, z30.d
    fmul    z8.d, z8.d, z4.d
    st1d    z8.d, p1, [x1]
    add     x0, x0, x3, lsl #3
    add     x1, x1, x3, lsl #3
    sub     x5, x5, x3
    b       .Ltail_f64_p2

.Ldone_f64_p2:
    RESTORE_SVE_REGS
    ret
    .size exp_f64_poly2, .-exp_f64_poly2


// ============================================
// FP64 exp with 3-term polynomial
// ============================================
    .align 6
    .global exp_f64_poly3
    .type exp_f64_poly3, %function
exp_f64_poly3:
    SAVE_SVE_REGS
    ptrue   p0.d

    adrp    x10, .Lconst_f64
    add     x10, x10, :lo12:.Lconst_f64
    ld1rd   z28.d, p0/z, [x10]              // 1/ln(2)
    ld1rd   z29.d, p0/z, [x10, #8]          // ln(2)
    ld1rd   z30.d, p0/z, [x10, #16]         // 1.0
    ld1rd   z27.d, p0/z, [x10, #24]         // 0.5
    ld1rd   z26.d, p0/z, [x10, #32]         // 1/6

    cntd    x3
    lsl     x4, x3, #2
    mov     x5, x2

.Lloop_f64_p3:
    cmp     x5, x4
    b.lt    .Ltail_f64_p3

    ld1d    z0.d, p0/z, [x0]
    ld1d    z1.d, p0/z, [x0, #1, mul vl]
    ld1d    z2.d, p0/z, [x0, #2, mul vl]
    ld1d    z3.d, p0/z, [x0, #3, mul vl]

    fmul    z4.d, z0.d, z28.d
    fmul    z5.d, z1.d, z28.d
    fmul    z6.d, z2.d, z28.d
    fmul    z7.d, z3.d, z28.d

    frintn  z4.d, p0/m, z4.d
    frintn  z5.d, p0/m, z5.d
    frintn  z6.d, p0/m, z6.d
    frintn  z7.d, p0/m, z7.d

    fmls   z0.d, p0/m, z4.d, z29.d
    fmls   z1.d, p0/m, z5.d, z29.d
    fmls   z2.d, p0/m, z6.d, z29.d
    fmls   z3.d, p0/m, z7.d, z29.d

    fcvtzs  z4.d, p0/m, z4.d
    fcvtzs  z5.d, p0/m, z5.d
    fcvtzs  z6.d, p0/m, z6.d
    fcvtzs  z7.d, p0/m, z7.d

    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    add     z5.d, z5.d, z31.d
    add     z6.d, z6.d, z31.d
    add     z7.d, z7.d, z31.d

    lsl     z4.d, z4.d, #52
    lsl     z5.d, z5.d, #52
    lsl     z6.d, z6.d, #52
    lsl     z7.d, z7.d, #52

    // 3-term
    movprfx z8, z27
    fmla    z8.d, p0/m, z26.d, z0.d
    movprfx z9, z27
    fmla    z9.d, p0/m, z26.d, z1.d
    movprfx z10, z27
    fmla    z10.d, p0/m, z26.d, z2.d
    movprfx z11, z27
    fmla    z11.d, p0/m, z26.d, z3.d

    fmad    z8.d, p0/m, z0.d, z30.d
    fmad    z9.d, p0/m, z1.d, z30.d
    fmad    z10.d, p0/m, z2.d, z30.d
    fmad    z11.d, p0/m, z3.d, z30.d

    fmad    z8.d, p0/m, z0.d, z30.d
    fmad    z9.d, p0/m, z1.d, z30.d
    fmad    z10.d, p0/m, z2.d, z30.d
    fmad    z11.d, p0/m, z3.d, z30.d

    fmul    z8.d, z8.d, z4.d
    fmul    z9.d, z9.d, z5.d
    fmul    z10.d, z10.d, z6.d
    fmul    z11.d, z11.d, z7.d

    st1d    z8.d, p0, [x1]
    st1d    z9.d, p0, [x1, #1, mul vl]
    st1d    z10.d, p0, [x1, #2, mul vl]
    st1d    z11.d, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #3
    add     x1, x1, x4, lsl #3
    sub     x5, x5, x4
    b       .Lloop_f64_p3

.Ltail_f64_p3:
    cbz     x5, .Ldone_f64_p3
    whilelt p1.d, xzr, x5
    ld1d    z0.d, p1/z, [x0]
    fmul    z4.d, z0.d, z28.d
    frintn  z4.d, p0/m, z4.d
    fmls    z0.d, p0/m, z4.d, z29.d
    fcvtzs  z4.d, p0/m, z4.d
    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    lsl     z4.d, z4.d, #52
    movprfx z8, z27
    fmla    z8.d, p0/m, z26.d, z0.d
    fmad    z8.d, p0/m, z0.d, z30.d
    fmad    z8.d, p0/m, z0.d, z30.d
    fmul    z8.d, z8.d, z4.d
    st1d    z8.d, p1, [x1]
    add     x0, x0, x3, lsl #3
    add     x1, x1, x3, lsl #3
    sub     x5, x5, x3
    b       .Ltail_f64_p3

.Ldone_f64_p3:
    RESTORE_SVE_REGS
    ret
    .size exp_f64_poly3, .-exp_f64_poly3


// ============================================
// FP64 exp with 4-term polynomial
// ============================================
    .align 6
    .global exp_f64_poly4
    .type exp_f64_poly4, %function
exp_f64_poly4:
    SAVE_SVE_REGS
    ptrue   p0.d

    adrp    x10, .Lconst_f64
    add     x10, x10, :lo12:.Lconst_f64
    ld1rd   z28.d, p0/z, [x10]              // 1/ln(2)
    ld1rd   z29.d, p0/z, [x10, #8]          // ln(2)
    ld1rd   z30.d, p0/z, [x10, #16]         // 1.0
    ld1rd   z27.d, p0/z, [x10, #24]         // 0.5
    ld1rd   z26.d, p0/z, [x10, #32]         // 1/6
    ld1rd   z25.d, p0/z, [x10, #40]         // 1/24

    cntd    x3
    lsl     x4, x3, #2
    mov     x5, x2

.Lloop_f64_p4:
    cmp     x5, x4
    b.lt    .Ltail_f64_p4

    ld1d    z0.d, p0/z, [x0]
    ld1d    z1.d, p0/z, [x0, #1, mul vl]
    ld1d    z2.d, p0/z, [x0, #2, mul vl]
    ld1d    z3.d, p0/z, [x0, #3, mul vl]

    fmul    z4.d, z0.d, z28.d
    fmul    z5.d, z1.d, z28.d
    fmul    z6.d, z2.d, z28.d
    fmul    z7.d, z3.d, z28.d

    frintn  z4.d, p0/m, z4.d
    frintn  z5.d, p0/m, z5.d
    frintn  z6.d, p0/m, z6.d
    frintn  z7.d, p0/m, z7.d

    fmls   z0.d, p0/m, z4.d, z29.d
    fmls   z1.d, p0/m, z5.d, z29.d
    fmls   z2.d, p0/m, z6.d, z29.d
    fmls   z3.d, p0/m, z7.d, z29.d

    fcvtzs  z4.d, p0/m, z4.d
    fcvtzs  z5.d, p0/m, z5.d
    fcvtzs  z6.d, p0/m, z6.d
    fcvtzs  z7.d, p0/m, z7.d

    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    add     z5.d, z5.d, z31.d
    add     z6.d, z6.d, z31.d
    add     z7.d, z7.d, z31.d

    lsl     z4.d, z4.d, #52
    lsl     z5.d, z5.d, #52
    lsl     z6.d, z6.d, #52
    lsl     z7.d, z7.d, #52

    // 4-term
    movprfx z8, z26
    fmla    z8.d, p0/m, z25.d, z0.d
    movprfx z9, z26
    fmla    z9.d, p0/m, z25.d, z1.d
    movprfx z10, z26
    fmla    z10.d, p0/m, z25.d, z2.d
    movprfx z11, z26
    fmla    z11.d, p0/m, z25.d, z3.d

    fmad    z8.d, p0/m, z0.d, z27.d
    fmad    z9.d, p0/m, z1.d, z27.d
    fmad    z10.d, p0/m, z2.d, z27.d
    fmad    z11.d, p0/m, z3.d, z27.d

    fmad    z8.d, p0/m, z0.d, z30.d
    fmad    z9.d, p0/m, z1.d, z30.d
    fmad    z10.d, p0/m, z2.d, z30.d
    fmad    z11.d, p0/m, z3.d, z30.d

    fmad    z8.d, p0/m, z0.d, z30.d
    fmad    z9.d, p0/m, z1.d, z30.d
    fmad    z10.d, p0/m, z2.d, z30.d
    fmad    z11.d, p0/m, z3.d, z30.d

    fmul    z8.d, z8.d, z4.d
    fmul    z9.d, z9.d, z5.d
    fmul    z10.d, z10.d, z6.d
    fmul    z11.d, z11.d, z7.d

    st1d    z8.d, p0, [x1]
    st1d    z9.d, p0, [x1, #1, mul vl]
    st1d    z10.d, p0, [x1, #2, mul vl]
    st1d    z11.d, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #3
    add     x1, x1, x4, lsl #3
    sub     x5, x5, x4
    b       .Lloop_f64_p4

.Ltail_f64_p4:
    cbz     x5, .Ldone_f64_p4
    whilelt p1.d, xzr, x5
    ld1d    z0.d, p1/z, [x0]
    fmul    z4.d, z0.d, z28.d
    frintn  z4.d, p0/m, z4.d
    fmls    z0.d, p0/m, z4.d, z29.d
    fcvtzs  z4.d, p0/m, z4.d
    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    lsl     z4.d, z4.d, #52
    movprfx z8, z26
    fmla    z8.d, p0/m, z25.d, z0.d
    fmad    z8.d, p0/m, z0.d, z27.d
    fmad    z8.d, p0/m, z0.d, z30.d
    fmad    z8.d, p0/m, z0.d, z30.d
    fmul    z8.d, z8.d, z4.d
    st1d    z8.d, p1, [x1]
    add     x0, x0, x3, lsl #3
    add     x1, x1, x3, lsl #3
    sub     x5, x5, x3
    b       .Ltail_f64_p4

.Ldone_f64_p4:
    RESTORE_SVE_REGS
    ret
    .size exp_f64_poly4, .-exp_f64_poly4


// ============================================
// FP64 exp with 5-term polynomial
// ============================================
    .align 6
    .global exp_f64_poly5
    .type exp_f64_poly5, %function
exp_f64_poly5:
    SAVE_SVE_REGS
    ptrue   p0.d

    adrp    x10, .Lconst_f64
    add     x10, x10, :lo12:.Lconst_f64
    ld1rd   z28.d, p0/z, [x10]              // 1/ln(2)
    ld1rd   z29.d, p0/z, [x10, #8]          // ln(2)
    ld1rd   z30.d, p0/z, [x10, #16]         // 1.0
    ld1rd   z27.d, p0/z, [x10, #24]         // 0.5
    ld1rd   z26.d, p0/z, [x10, #32]         // 1/6
    ld1rd   z25.d, p0/z, [x10, #40]         // 1/24
    ld1rd   z24.d, p0/z, [x10, #48]         // 1/120

    cntd    x3
    lsl     x4, x3, #2
    mov     x5, x2

.Lloop_f64_p5:
    cmp     x5, x4
    b.lt    .Ltail_f64_p5

    ld1d    z0.d, p0/z, [x0]
    ld1d    z1.d, p0/z, [x0, #1, mul vl]
    ld1d    z2.d, p0/z, [x0, #2, mul vl]
    ld1d    z3.d, p0/z, [x0, #3, mul vl]

    fmul    z4.d, z0.d, z28.d
    fmul    z5.d, z1.d, z28.d
    fmul    z6.d, z2.d, z28.d
    fmul    z7.d, z3.d, z28.d

    frintn  z4.d, p0/m, z4.d
    frintn  z5.d, p0/m, z5.d
    frintn  z6.d, p0/m, z6.d
    frintn  z7.d, p0/m, z7.d

    fmls   z0.d, p0/m, z4.d, z29.d
    fmls   z1.d, p0/m, z5.d, z29.d
    fmls   z2.d, p0/m, z6.d, z29.d
    fmls   z3.d, p0/m, z7.d, z29.d

    fcvtzs  z4.d, p0/m, z4.d
    fcvtzs  z5.d, p0/m, z5.d
    fcvtzs  z6.d, p0/m, z6.d
    fcvtzs  z7.d, p0/m, z7.d

    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    add     z5.d, z5.d, z31.d
    add     z6.d, z6.d, z31.d
    add     z7.d, z7.d, z31.d

    lsl     z4.d, z4.d, #52
    lsl     z5.d, z5.d, #52
    lsl     z6.d, z6.d, #52
    lsl     z7.d, z7.d, #52

    // 5-term
    movprfx z8, z25
    fmla    z8.d, p0/m, z24.d, z0.d
    movprfx z9, z25
    fmla    z9.d, p0/m, z24.d, z1.d
    movprfx z10, z25
    fmla    z10.d, p0/m, z24.d, z2.d
    movprfx z11, z25
    fmla    z11.d, p0/m, z24.d, z3.d

    fmad    z8.d, p0/m, z0.d, z26.d
    fmad    z9.d, p0/m, z1.d, z26.d
    fmad    z10.d, p0/m, z2.d, z26.d
    fmad    z11.d, p0/m, z3.d, z26.d

    fmad    z8.d, p0/m, z0.d, z27.d
    fmad    z9.d, p0/m, z1.d, z27.d
    fmad    z10.d, p0/m, z2.d, z27.d
    fmad    z11.d, p0/m, z3.d, z27.d

    fmad    z8.d, p0/m, z0.d, z30.d
    fmad    z9.d, p0/m, z1.d, z30.d
    fmad    z10.d, p0/m, z2.d, z30.d
    fmad    z11.d, p0/m, z3.d, z30.d

    fmad    z8.d, p0/m, z0.d, z30.d
    fmad    z9.d, p0/m, z1.d, z30.d
    fmad    z10.d, p0/m, z2.d, z30.d
    fmad    z11.d, p0/m, z3.d, z30.d

    fmul    z8.d, z8.d, z4.d
    fmul    z9.d, z9.d, z5.d
    fmul    z10.d, z10.d, z6.d
    fmul    z11.d, z11.d, z7.d

    st1d    z8.d, p0, [x1]
    st1d    z9.d, p0, [x1, #1, mul vl]
    st1d    z10.d, p0, [x1, #2, mul vl]
    st1d    z11.d, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #3
    add     x1, x1, x4, lsl #3
    sub     x5, x5, x4
    b       .Lloop_f64_p5

.Ltail_f64_p5:
    cbz     x5, .Ldone_f64_p5
    whilelt p1.d, xzr, x5
    ld1d    z0.d, p1/z, [x0]
    fmul    z4.d, z0.d, z28.d
    frintn  z4.d, p0/m, z4.d
    fmls    z0.d, p0/m, z4.d, z29.d
    fcvtzs  z4.d, p0/m, z4.d
    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    lsl     z4.d, z4.d, #52
    movprfx z8, z25
    fmla    z8.d, p0/m, z24.d, z0.d
    fmad    z8.d, p0/m, z0.d, z26.d
    fmad    z8.d, p0/m, z0.d, z27.d
    fmad    z8.d, p0/m, z0.d, z30.d
    fmad    z8.d, p0/m, z0.d, z30.d
    fmul    z8.d, z8.d, z4.d
    st1d    z8.d, p1, [x1]
    add     x0, x0, x3, lsl #3
    add     x1, x1, x3, lsl #3
    sub     x5, x5, x3
    b       .Ltail_f64_p5

.Ldone_f64_p5:
    RESTORE_SVE_REGS
    ret
    .size exp_f64_poly5, .-exp_f64_poly5


// ============================================
// Data section - constants loaded from L1
// ============================================
    .section .rodata
    .align 6

// FP32 constants
.Lconst_f32:
    .word   0x3fb8aa3b      // 1/ln(2) = 1.4426950408889634
    .word   0x3f317218      // ln(2) = 0.6931471805599453
    .word   0x3f800000      // 1.0
    .word   0x3f000000      // 0.5
    .word   0x3e2aaaab      // 1/6 = 0.16666666666666666
    .word   0x3d2aaaab      // 1/24 = 0.041666666666666664
    .word   0x3c088889      // 1/120 = 0.008333333333333333

// FP64 constants
    .align 3
.Lconst_f64:
    .quad   0x3ff71547652b82fe      // 1/ln(2) = 1.4426950408889634
    .quad   0x3fe62e42fefa39ef      // ln(2) = 0.6931471805599453
    .quad   0x3ff0000000000000      // 1.0
    .quad   0x3fe0000000000000      // 0.5
    .quad   0x3fc5555555555555      // 1/6
    .quad   0x3fa5555555555555      // 1/24
    .quad   0x3f81111111111111      // 1/120
