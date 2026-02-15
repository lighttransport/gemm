/*
 * Simple fused exp2 + GEMM with exp2 store
 *
 * Per-K iteration: exp2(S[*,k]) + store P + GEMM
 * The store overlaps with FMLA on A64FX (separate store pipe)
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*============================================================================
 * exp2_gemm_store - exp2 + GEMM + store exp2 (simple per-K version)
 *
 * void exp2_gemm_store(
 *     const int32_t* S,   // x0: [4][Nc]
 *     const float* V,     // x1: [Nc][64]
 *     float* O,           // x2: [4][64]
 *     float* P,           // x3: [4][Nc] exp2 output
 *     int Nc,             // x4
 *     float scale,        // s0
 *     float max_val,      // s1
 *     int ld_s,           // x5: S stride in elements
 *     int ld_v,           // x6: V stride in bytes
 *     int ld_o            // x7: O stride in bytes
 * );
 *============================================================================*/

    .global exp2_gemm_store
    .type exp2_gemm_store, %function
exp2_gemm_store:
    stp     d8, d9, [sp, #-96]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #64]
    stp     x21, x22, [sp, #80]

    ptrue   p0.s

    /* Save parameters */
    mov     x19, x3                     // P buffer
    mov     x20, x4                     // Nc
    fmov    s16, s0                     // scale
    fmov    s17, s1                     // max
    mov     z20.s, s16
    mov     z21.s, s17

    /* exp2 constants */
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

    cbz     x20, .Lsimple_store

    /* S row pointers */
    lsl     x5, x5, #2
    mov     x21, x0
    add     x22, x0, x5
    add     x8, x22, x5
    add     x9, x8, x5

    /* P row pointers (stride = Nc * 4) */
    mov     x10, x19
    lsl     x11, x20, #2                // Nc * 4 bytes
    add     x12, x10, x11
    add     x13, x12, x11
    add     x14, x13, x11

.Lsimple_loop:
    /* Load S[row, k] for all 4 rows */
    ldr     w15, [x21]
    ldr     w16, [x22]
    ldr     w17, [x8]
    ldr     w18, [x9]

    /* Pack into NEON vector for vectorized exp2 */
    fmov    s24, w15
    mov     v24.s[1], w16
    mov     v24.s[2], w17
    mov     v24.s[3], w18

    /* Convert to float and scale */
    scvtf   v24.4s, v24.4s
    dup     v25.4s, v20.s[0]            // scale
    dup     v26.4s, v21.s[0]            // max
    fmul    v24.4s, v24.4s, v25.4s
    fsub    v24.4s, v24.4s, v26.4s

    /* exp2 using SVE FEXPA (on z24) */
    frintm  z28.s, p0/m, z24.s
    fsub    z25.s, z24.s, z28.s
    fmul    z25.s, z25.s, z22.s
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z28.s, p0/m, z28.s
    add     z28.s, z28.s, z23.s
    lsl     z28.s, z28.s, #6
    orr     z24.s, z28.s, z25.s
    fexpa   z24.s, z24.s

    /* Store exp2 values to P (4 scalars) */
    /* This overlaps with subsequent FMLA on A64FX */
    str     s24, [x10]
    mov     w15, v24.s[1]
    str     w15, [x12]
    mov     w15, v24.s[2]
    str     w15, [x13]
    mov     w15, v24.s[3]
    str     w15, [x14]

    /* Broadcast exp2 values for FMLA */
    dup     z25.s, z24.s[1]
    dup     z26.s, z24.s[2]
    dup     z27.s, z24.s[3]
    dup     z24.s, z24.s[0]

    /* Load V[k, :] */
    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]

    /* 16 FMLAs */
    fmla    z0.s, p0/m, z24.s, z28.s
    fmla    z1.s, p0/m, z24.s, z29.s
    fmla    z2.s, p0/m, z24.s, z30.s
    fmla    z3.s, p0/m, z24.s, z31.s
    fmla    z4.s, p0/m, z25.s, z28.s
    fmla    z5.s, p0/m, z25.s, z29.s
    fmla    z6.s, p0/m, z25.s, z30.s
    fmla    z7.s, p0/m, z25.s, z31.s
    fmla    z8.s, p0/m, z26.s, z28.s
    fmla    z9.s, p0/m, z26.s, z29.s
    fmla    z10.s, p0/m, z26.s, z30.s
    fmla    z11.s, p0/m, z26.s, z31.s
    fmla    z12.s, p0/m, z27.s, z28.s
    fmla    z13.s, p0/m, z27.s, z29.s
    fmla    z14.s, p0/m, z27.s, z30.s
    fmla    z15.s, p0/m, z27.s, z31.s

    /* Advance pointers */
    add     x21, x21, #4
    add     x22, x22, #4
    add     x8, x8, #4
    add     x9, x9, #4
    add     x10, x10, #4
    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x1, x1, x6

    subs    x20, x20, #1
    b.gt    .Lsimple_loop

.Lsimple_store:
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

    ldp     x21, x22, [sp, #80]
    ldp     x19, x20, [sp, #64]
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #96
    ret

    .size exp2_gemm_store, .-exp2_gemm_store
