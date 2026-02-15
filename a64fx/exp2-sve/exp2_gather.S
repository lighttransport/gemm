/*
 * Optimized exp2 + GEMM using SVE gather load for S values
 *
 * Key optimization: Use LD1W with gather to load 4 S values at once
 * instead of 4 scalar loads
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*
 * exp2_gather_4x4 - Gather-optimized fused exp2 + GEMM
 *
 * void exp2_gather_4x4(
 *     const int32_t* S,   // x0: [4][Nc] scores (row-major)
 *     const float* V,     // x1: [Nc][64] values
 *     float* O,           // x2: [4][64] output
 *     float* P,           // x3: [4][Nc] exp2 buffer
 *     int Nc,             // x4: sequence length
 *     float scale,        // s0
 *     float max_val,      // s1
 *     int ld_s,           // x5: S stride in elements
 *     int ld_v,           // x6: V stride in bytes
 *     int ld_o            // x7: O stride in bytes
 * );
 */
    .global exp2_gather_4x4
    .type exp2_gather_4x4, %function
exp2_gather_4x4:
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

    cbz     x4, .Lgather_store

    /* Setup gather offset vector: [0, ld_s, 2*ld_s, 3*ld_s] in bytes */
    /* z16 will hold byte offsets for gathering 4 rows */
    lsl     x5, x5, #2                  // ld_s * 4 bytes
    mov     x8, #0
    index   z16.s, #0, w5               // z16 = [0, ld_s*4, 2*ld_s*4, 3*ld_s*4]

    /* Create predicate for first 4 elements only */
    mov     w9, #0xF                    // bits for 4 elements
    whilelo p1.s, xzr, x9               // p1 = first 4 elements active

    /* P row pointers (stride = Nc * 4 bytes) */
    lsl     x12, x4, #2                 // Nc * 4
    mov     x13, x3                     // P row 0
    add     x14, x3, x12                // P row 1
    add     x15, x14, x12               // P row 2
    add     x16, x15, x12               // P row 3

    mov     x17, x4                     // loop counter
    mov     x8, x0                      // S base pointer

.Lgather_loop:
    /* Gather load S[0:3, k] using z16 offsets */
    /* ld1w with scalar + vector offsets: S[base + offset[i]] */
    ld1w    {z24.s}, p1/z, [x8, z16.s, uxtw]  // Gather 4 S values
    add     x8, x8, #4                  // S += 1 (next k column)

    /* Convert to float and compute x = S * scale - max */
    scvtf   z24.s, p0/m, z24.s
    fmul    z24.s, z24.s, z20.s
    fsub    z24.s, z24.s, z21.s

    /* FEXPA-based exp2 */
    frintm  z25.s, p0/m, z24.s
    fsub    z26.s, z24.s, z25.s
    fmul    z26.s, z26.s, z22.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z25.s, p0/m, z25.s
    add     z25.s, z25.s, z23.s
    lsl     z25.s, z25.s, #6
    orr     z24.s, z25.s, z26.s
    fexpa   z24.s, z24.s

    /* Scatter store exp2 to P (or use scalar stores) */
    /* For simplicity, use scalar stores to P buffer */
    str     s24, [x13], #4
    mov     w9, v24.s[1]
    str     w9, [x14], #4
    mov     w9, v24.s[2]
    str     w9, [x15], #4
    mov     w9, v24.s[3]
    str     w9, [x16], #4

    /* LD1RW broadcasts */
    sub     x9, x13, #4
    ld1rw   {z24.s}, p0/z, [x9]
    sub     x9, x14, #4
    ld1rw   {z25.s}, p0/z, [x9]
    sub     x9, x15, #4
    ld1rw   {z26.s}, p0/z, [x9]
    sub     x9, x16, #4
    ld1rw   {z27.s}, p0/z, [x9]

    /* Load V[k, :] */
    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

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

    subs    x17, x17, #1
    b.gt    .Lgather_loop

.Lgather_store:
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

    .size exp2_gather_4x4, .-exp2_gather_4x4
