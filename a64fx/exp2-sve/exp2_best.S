/*
 * Best performing exp2 + GEMM kernel
 *
 * Combines:
 * 1. 4K unrolling for better ILP
 * 2. DUP broadcast (faster than LD1RW due to no store/load overhead)
 * 3. No P buffer needed
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*
 * exp2_best_4x4 - Best performing version
 */
    .global exp2_best_4x4
    .type exp2_best_4x4, %function
exp2_best_4x4:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.s

    /* Setup constants */
    mov     z20.s, s0                   // scale
    mov     z21.s, s1                   // max_val
    mov     w8, #0x42800000
    mov     z18.s, w8                   // 64.0f
    mov     w8, #127
    mov     z19.s, w8                   // bias

    /* Zero accumulators */
    fmov    z0.s, #0
    fmov    z1.s, #0
    fmov    z2.s, #0
    fmov    z3.s, #0
    fmov    z4.s, #0
    fmov    z5.s, #0
    fmov    z6.s, #0
    fmov    z7.s, #0
    fmov    z8.s, #0
    fmov    z9.s, #0
    fmov    z10.s, #0
    fmov    z11.s, #0
    fmov    z12.s, #0
    fmov    z13.s, #0
    fmov    z14.s, #0
    fmov    z15.s, #0

    cbz     x4, .Lbest_store

    /* S row pointers */
    lsl     x5, x5, #2                  // ld_s in bytes
    mov     x8, x0
    add     x9, x0, x5
    add     x10, x9, x5
    add     x11, x10, x5

    lsr     x17, x4, #2                 // loop counter = Nc / 4

.Lbest_loop:
    /* === Load 4 columns of S (16 values) === */
    /* Column k+0 */
    ldr     w12, [x8], #4
    fmov    s22, w12
    ldr     w12, [x9], #4
    mov     v22.s[1], w12
    ldr     w12, [x10], #4
    mov     v22.s[2], w12
    ldr     w12, [x11], #4
    mov     v22.s[3], w12

    /* Column k+1 */
    ldr     w12, [x8], #4
    fmov    s23, w12
    ldr     w12, [x9], #4
    mov     v23.s[1], w12
    ldr     w12, [x10], #4
    mov     v23.s[2], w12
    ldr     w12, [x11], #4
    mov     v23.s[3], w12

    /* Column k+2 */
    ldr     w12, [x8], #4
    fmov    s24, w12
    ldr     w12, [x9], #4
    mov     v24.s[1], w12
    ldr     w12, [x10], #4
    mov     v24.s[2], w12
    ldr     w12, [x11], #4
    mov     v24.s[3], w12

    /* Column k+3 */
    ldr     w12, [x8], #4
    fmov    s25, w12
    ldr     w12, [x9], #4
    mov     v25.s[1], w12
    ldr     w12, [x10], #4
    mov     v25.s[2], w12
    ldr     w12, [x11], #4
    mov     v25.s[3], w12

    /* === exp2 for all 4 columns (z22-z25) === */
    /* Pipeline: do all scvtf, then all fmul, etc. for better ILP */
    scvtf   z22.s, p0/m, z22.s
    scvtf   z23.s, p0/m, z23.s
    scvtf   z24.s, p0/m, z24.s
    scvtf   z25.s, p0/m, z25.s

    fmul    z22.s, z22.s, z20.s
    fmul    z23.s, z23.s, z20.s
    fmul    z24.s, z24.s, z20.s
    fmul    z25.s, z25.s, z20.s

    fsub    z22.s, z22.s, z21.s
    fsub    z23.s, z23.s, z21.s
    fsub    z24.s, z24.s, z21.s
    fsub    z25.s, z25.s, z21.s

    /* Floor and fractional part */
    frintm  z26.s, p0/m, z22.s          // N0
    frintm  z27.s, p0/m, z23.s          // N1
    frintm  z28.s, p0/m, z24.s          // N2
    frintm  z29.s, p0/m, z25.s          // N3

    fsub    z22.s, z22.s, z26.s         // f0
    fsub    z23.s, z23.s, z27.s         // f1
    fsub    z24.s, z24.s, z28.s         // f2
    fsub    z25.s, z25.s, z29.s         // f3

    fmul    z22.s, z22.s, z18.s         // f0 * 64
    fmul    z23.s, z23.s, z18.s         // f1 * 64
    fmul    z24.s, z24.s, z18.s         // f2 * 64
    fmul    z25.s, z25.s, z18.s         // f3 * 64

    fcvtzs  z22.s, p0/m, z22.s          // m0
    fcvtzs  z23.s, p0/m, z23.s          // m1
    fcvtzs  z24.s, p0/m, z24.s          // m2
    fcvtzs  z25.s, p0/m, z25.s          // m3

    fcvtzs  z26.s, p0/m, z26.s          // N0 as int
    fcvtzs  z27.s, p0/m, z27.s          // N1 as int
    fcvtzs  z28.s, p0/m, z28.s          // N2 as int
    fcvtzs  z29.s, p0/m, z29.s          // N3 as int

    add     z26.s, z26.s, z19.s         // N0 + 127
    add     z27.s, z27.s, z19.s         // N1 + 127
    add     z28.s, z28.s, z19.s         // N2 + 127
    add     z29.s, z29.s, z19.s         // N3 + 127

    lsl     z26.s, z26.s, #6            // (N0+127) << 6
    lsl     z27.s, z27.s, #6
    lsl     z28.s, z28.s, #6
    lsl     z29.s, z29.s, #6

    orr     z22.s, z26.s, z22.s         // combine
    orr     z23.s, z27.s, z23.s
    orr     z24.s, z28.s, z24.s
    orr     z25.s, z29.s, z25.s

    fexpa   z22.s, z22.s                // exp2 for k+0
    fexpa   z23.s, z23.s                // exp2 for k+1
    fexpa   z24.s, z24.s                // exp2 for k+2
    fexpa   z25.s, z25.s                // exp2 for k+3

    /* === K+0 GEMM === */
    dup     z26.s, z22.s[0]             // P[0,k+0]
    dup     z27.s, z22.s[1]             // P[1,k+0]
    dup     z28.s, z22.s[2]             // P[2,k+0]
    dup     z29.s, z22.s[3]             // P[3,k+0]

    ld1w    {z30.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z16.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z17.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

    fmla    z0.s, p0/m, z26.s, z30.s
    fmla    z1.s, p0/m, z26.s, z31.s
    fmla    z2.s, p0/m, z26.s, z16.s
    fmla    z3.s, p0/m, z26.s, z17.s
    fmla    z4.s, p0/m, z27.s, z30.s
    fmla    z5.s, p0/m, z27.s, z31.s
    fmla    z6.s, p0/m, z27.s, z16.s
    fmla    z7.s, p0/m, z27.s, z17.s
    fmla    z8.s, p0/m, z28.s, z30.s
    fmla    z9.s, p0/m, z28.s, z31.s
    fmla    z10.s, p0/m, z28.s, z16.s
    fmla    z11.s, p0/m, z28.s, z17.s
    fmla    z12.s, p0/m, z29.s, z30.s
    fmla    z13.s, p0/m, z29.s, z31.s
    fmla    z14.s, p0/m, z29.s, z16.s
    fmla    z15.s, p0/m, z29.s, z17.s

    /* === K+1 GEMM === */
    dup     z26.s, z23.s[0]
    dup     z27.s, z23.s[1]
    dup     z28.s, z23.s[2]
    dup     z29.s, z23.s[3]

    ld1w    {z30.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z16.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z17.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

    fmla    z0.s, p0/m, z26.s, z30.s
    fmla    z1.s, p0/m, z26.s, z31.s
    fmla    z2.s, p0/m, z26.s, z16.s
    fmla    z3.s, p0/m, z26.s, z17.s
    fmla    z4.s, p0/m, z27.s, z30.s
    fmla    z5.s, p0/m, z27.s, z31.s
    fmla    z6.s, p0/m, z27.s, z16.s
    fmla    z7.s, p0/m, z27.s, z17.s
    fmla    z8.s, p0/m, z28.s, z30.s
    fmla    z9.s, p0/m, z28.s, z31.s
    fmla    z10.s, p0/m, z28.s, z16.s
    fmla    z11.s, p0/m, z28.s, z17.s
    fmla    z12.s, p0/m, z29.s, z30.s
    fmla    z13.s, p0/m, z29.s, z31.s
    fmla    z14.s, p0/m, z29.s, z16.s
    fmla    z15.s, p0/m, z29.s, z17.s

    /* === K+2 GEMM === */
    dup     z26.s, z24.s[0]
    dup     z27.s, z24.s[1]
    dup     z28.s, z24.s[2]
    dup     z29.s, z24.s[3]

    ld1w    {z30.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z16.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z17.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

    fmla    z0.s, p0/m, z26.s, z30.s
    fmla    z1.s, p0/m, z26.s, z31.s
    fmla    z2.s, p0/m, z26.s, z16.s
    fmla    z3.s, p0/m, z26.s, z17.s
    fmla    z4.s, p0/m, z27.s, z30.s
    fmla    z5.s, p0/m, z27.s, z31.s
    fmla    z6.s, p0/m, z27.s, z16.s
    fmla    z7.s, p0/m, z27.s, z17.s
    fmla    z8.s, p0/m, z28.s, z30.s
    fmla    z9.s, p0/m, z28.s, z31.s
    fmla    z10.s, p0/m, z28.s, z16.s
    fmla    z11.s, p0/m, z28.s, z17.s
    fmla    z12.s, p0/m, z29.s, z30.s
    fmla    z13.s, p0/m, z29.s, z31.s
    fmla    z14.s, p0/m, z29.s, z16.s
    fmla    z15.s, p0/m, z29.s, z17.s

    /* === K+3 GEMM === */
    dup     z26.s, z25.s[0]
    dup     z27.s, z25.s[1]
    dup     z28.s, z25.s[2]
    dup     z29.s, z25.s[3]

    ld1w    {z30.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z16.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z17.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

    fmla    z0.s, p0/m, z26.s, z30.s
    fmla    z1.s, p0/m, z26.s, z31.s
    fmla    z2.s, p0/m, z26.s, z16.s
    fmla    z3.s, p0/m, z26.s, z17.s
    fmla    z4.s, p0/m, z27.s, z30.s
    fmla    z5.s, p0/m, z27.s, z31.s
    fmla    z6.s, p0/m, z27.s, z16.s
    fmla    z7.s, p0/m, z27.s, z17.s
    fmla    z8.s, p0/m, z28.s, z30.s
    fmla    z9.s, p0/m, z28.s, z31.s
    fmla    z10.s, p0/m, z28.s, z16.s
    fmla    z11.s, p0/m, z28.s, z17.s
    fmla    z12.s, p0/m, z29.s, z30.s
    fmla    z13.s, p0/m, z29.s, z31.s
    fmla    z14.s, p0/m, z29.s, z16.s
    fmla    z15.s, p0/m, z29.s, z17.s

    subs    x17, x17, #1
    b.gt    .Lbest_loop

.Lbest_store:
    mov     x8, x2
    st1w    {z0.s}, p0, [x8, #0, mul vl]
    st1w    {z1.s}, p0, [x8, #1, mul vl]
    st1w    {z2.s}, p0, [x8, #2, mul vl]
    st1w    {z3.s}, p0, [x8, #3, mul vl]
    add     x8, x8, x7
    st1w    {z4.s}, p0, [x8, #0, mul vl]
    st1w    {z5.s}, p0, [x8, #1, mul vl]
    st1w    {z6.s}, p0, [x8, #2, mul vl]
    st1w    {z7.s}, p0, [x8, #3, mul vl]
    add     x8, x8, x7
    st1w    {z8.s}, p0, [x8, #0, mul vl]
    st1w    {z9.s}, p0, [x8, #1, mul vl]
    st1w    {z10.s}, p0, [x8, #2, mul vl]
    st1w    {z11.s}, p0, [x8, #3, mul vl]
    add     x8, x8, x7
    st1w    {z12.s}, p0, [x8, #0, mul vl]
    st1w    {z13.s}, p0, [x8, #1, mul vl]
    st1w    {z14.s}, p0, [x8, #2, mul vl]
    st1w    {z15.s}, p0, [x8, #3, mul vl]

    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret

    .size exp2_best_4x4, .-exp2_best_4x4
