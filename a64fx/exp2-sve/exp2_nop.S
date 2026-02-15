/*
 * exp2 + GEMM with no P buffer - keep exp2 results in registers
 *
 * Uses DUP to broadcast exp2 results directly from NEON registers
 * This eliminates the store-load overhead but uses FLA pipe for DUP
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*
 * exp2_nop_4x4 - No P buffer version
 *
 * void exp2_nop_4x4(
 *     const int32_t* S,   // x0: [4][Nc] scores
 *     const float* V,     // x1: [Nc][64] values
 *     float* O,           // x2: [4][64] output
 *     float* P,           // x3: [4][Nc] exp2 buffer (optional, can be NULL)
 *     int Nc,             // x4: sequence length
 *     float scale,        // s0
 *     float max_val,      // s1
 *     int ld_s,           // x5: S stride in elements
 *     int ld_v,           // x6: V stride in bytes
 *     int ld_o            // x7: O stride in bytes
 * );
 */
    .global exp2_nop_4x4
    .type exp2_nop_4x4, %function
exp2_nop_4x4:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.s

    /* Setup constants */
    mov     z20.s, s0                   // scale
    mov     z21.s, s1                   // max_val
    mov     w8, #0x42800000
    mov     z22.s, w8                   // 64.0f
    mov     w8, #127
    mov     z23.s, w8                   // bias

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

    cbz     x4, .Lnop_store

    /* S row pointers */
    lsl     x5, x5, #2                  // ld_s in bytes
    mov     x8, x0
    add     x9, x0, x5
    add     x10, x9, x5
    add     x11, x10, x5

    mov     x17, x4                     // loop counter

.Lnop_loop:
    /* Load S[row, k] */
    ldr     w18, [x8], #4
    fmov    s24, w18
    ldr     w18, [x9], #4
    mov     v24.s[1], w18
    ldr     w18, [x10], #4
    mov     v24.s[2], w18
    ldr     w18, [x11], #4
    mov     v24.s[3], w18

    /* exp2 computation */
    scvtf   z24.s, p0/m, z24.s
    fmul    z24.s, z24.s, z20.s
    fsub    z24.s, z24.s, z21.s
    frintm  z25.s, p0/m, z24.s
    fsub    z26.s, z24.s, z25.s
    fmul    z26.s, z26.s, z22.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z25.s, p0/m, z25.s
    add     z25.s, z25.s, z23.s
    lsl     z25.s, z25.s, #6
    orr     z24.s, z25.s, z26.s
    fexpa   z24.s, z24.s

    /* DUP broadcast from SVE vector lanes (runs on FLA pipe, but no store/load) */
    /* z24[0:3] has exp2 results, broadcast each to full vector */
    dup     z25.s, z24.s[0]             // P[0,k] broadcast
    dup     z26.s, z24.s[1]             // P[1,k] broadcast
    dup     z27.s, z24.s[2]             // P[2,k] broadcast
    dup     z28.s, z24.s[3]             // P[3,k] broadcast

    /* Load V[k, :] */
    ld1w    {z29.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z16.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

    /* 16 FMLAs */
    fmla    z0.s, p0/m, z25.s, z29.s
    fmla    z1.s, p0/m, z25.s, z30.s
    fmla    z2.s, p0/m, z25.s, z31.s
    fmla    z3.s, p0/m, z25.s, z16.s
    fmla    z4.s, p0/m, z26.s, z29.s
    fmla    z5.s, p0/m, z26.s, z30.s
    fmla    z6.s, p0/m, z26.s, z31.s
    fmla    z7.s, p0/m, z26.s, z16.s
    fmla    z8.s, p0/m, z27.s, z29.s
    fmla    z9.s, p0/m, z27.s, z30.s
    fmla    z10.s, p0/m, z27.s, z31.s
    fmla    z11.s, p0/m, z27.s, z16.s
    fmla    z12.s, p0/m, z28.s, z29.s
    fmla    z13.s, p0/m, z28.s, z30.s
    fmla    z14.s, p0/m, z28.s, z31.s
    fmla    z15.s, p0/m, z28.s, z16.s

    subs    x17, x17, #1
    b.gt    .Lnop_loop

.Lnop_store:
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

    .size exp2_nop_4x4, .-exp2_nop_4x4
