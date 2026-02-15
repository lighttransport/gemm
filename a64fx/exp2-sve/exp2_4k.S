/*
 * exp2 + GEMM processing 4 K columns at once
 *
 * Key insight: Process 4 K iterations simultaneously to utilize
 * full 16-element SVE vector for exp2:
 *   - 4 M rows × 4 K columns = 16 values = 1 full SVE vector
 *
 * Layout:
 *   z24[0:3]   = S[0:3, k+0] (column k+0, all 4 rows)
 *   z24[4:7]   = S[0:3, k+1] (column k+1, all 4 rows)
 *   z24[8:11]  = S[0:3, k+2] (column k+2, all 4 rows)
 *   z24[12:15] = S[0:3, k+3] (column k+3, all 4 rows)
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*
 * exp2_4k_4x4 - Process 4 K columns at once for full vector utilization
 *
 * void exp2_4k_4x4(
 *     const int32_t* S,   // x0: [4][Nc] scores (row-major)
 *     const float* V,     // x1: [Nc][64] values
 *     float* O,           // x2: [4][64] output
 *     float* P,           // x3: [4][Nc] exp2 buffer
 *     int Nc,             // x4: sequence length (must be multiple of 4)
 *     float scale,        // s0
 *     float max_val,      // s1
 *     int ld_s,           // x5: S stride in elements
 *     int ld_v,           // x6: V stride in bytes
 *     int ld_o            // x7: O stride in bytes
 * );
 */
    .global exp2_4k_4x4
    .type exp2_4k_4x4, %function
exp2_4k_4x4:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.s

    /* Setup constants */
    mov     z20.s, s0                   // scale (broadcast)
    mov     z21.s, s1                   // max_val (broadcast)
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

    cbz     x4, .L4k_store

    /* S row pointers */
    lsl     x5, x5, #2                  // ld_s in bytes
    mov     x8, x0                      // S row 0
    add     x9, x0, x5                  // S row 1
    add     x10, x9, x5                 // S row 2
    add     x11, x10, x5                // S row 3

    /* P row pointers */
    lsl     x12, x4, #2                 // Nc * 4
    mov     x13, x3                     // P row 0
    add     x14, x3, x12                // P row 1
    add     x15, x14, x12               // P row 2
    add     x16, x15, x12               // P row 3

    lsr     x17, x4, #2                 // loop counter = Nc / 4

.L4k_loop:
    /* Load 4 K columns × 4 rows = 16 S values */
    /* Pack into z24: [S00,S10,S20,S30, S01,S11,S21,S31, S02,S12,S22,S32, S03,S13,S23,S33] */
    /* Column k+0 */
    ldr     w18, [x8], #4               // S[0, k+0]
    fmov    s24, w18
    ldr     w18, [x9], #4               // S[1, k+0]
    mov     v24.s[1], w18
    ldr     w18, [x10], #4              // S[2, k+0]
    mov     v24.s[2], w18
    ldr     w18, [x11], #4              // S[3, k+0]
    mov     v24.s[3], w18

    /* Column k+1 */
    ldr     w18, [x8], #4
    mov     z25.s, w18
    ldr     w18, [x9], #4
    mov     v25.s[1], w18
    ldr     w18, [x10], #4
    mov     v25.s[2], w18
    ldr     w18, [x11], #4
    mov     v25.s[3], w18

    /* Column k+2 */
    ldr     w18, [x8], #4
    mov     z26.s, w18
    ldr     w18, [x9], #4
    mov     v26.s[1], w18
    ldr     w18, [x10], #4
    mov     v26.s[2], w18
    ldr     w18, [x11], #4
    mov     v26.s[3], w18

    /* Column k+3 */
    ldr     w18, [x8], #4
    mov     z27.s, w18
    ldr     w18, [x9], #4
    mov     v27.s[1], w18
    ldr     w18, [x10], #4
    mov     v27.s[2], w18
    ldr     w18, [x11], #4
    mov     v27.s[3], w18

    /* Interleave into z24: use ZIP to create proper layout */
    /* z24 = [S00,S10,S20,S30], z25 = [S01,S11,S21,S31] */
    /* z26 = [S02,S12,S22,S32], z27 = [S03,S13,S23,S33] */
    /* Need: z24 = [S00,S10,S20,S30, S01,S11,S21,S31, S02,S12,S22,S32, S03,S13,S23,S33] */
    /* But we can just keep them as 4 separate 128-bit chunks */

    /* exp2 for column k+0 */
    scvtf   z24.s, p0/m, z24.s
    fmul    z24.s, z24.s, z20.s
    fsub    z24.s, z24.s, z21.s
    frintm  z28.s, p0/m, z24.s
    fsub    z29.s, z24.s, z28.s
    fmul    z29.s, z29.s, z22.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z28.s, p0/m, z28.s
    add     z28.s, z28.s, z23.s
    lsl     z28.s, z28.s, #6
    orr     z24.s, z28.s, z29.s
    fexpa   z24.s, z24.s

    /* exp2 for column k+1 */
    scvtf   z25.s, p0/m, z25.s
    fmul    z25.s, z25.s, z20.s
    fsub    z25.s, z25.s, z21.s
    frintm  z28.s, p0/m, z25.s
    fsub    z29.s, z25.s, z28.s
    fmul    z29.s, z29.s, z22.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z28.s, p0/m, z28.s
    add     z28.s, z28.s, z23.s
    lsl     z28.s, z28.s, #6
    orr     z25.s, z28.s, z29.s
    fexpa   z25.s, z25.s

    /* exp2 for column k+2 */
    scvtf   z26.s, p0/m, z26.s
    fmul    z26.s, z26.s, z20.s
    fsub    z26.s, z26.s, z21.s
    frintm  z28.s, p0/m, z26.s
    fsub    z29.s, z26.s, z28.s
    fmul    z29.s, z29.s, z22.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z28.s, p0/m, z28.s
    add     z28.s, z28.s, z23.s
    lsl     z28.s, z28.s, #6
    orr     z26.s, z28.s, z29.s
    fexpa   z26.s, z26.s

    /* exp2 for column k+3 */
    scvtf   z27.s, p0/m, z27.s
    fmul    z27.s, z27.s, z20.s
    fsub    z27.s, z27.s, z21.s
    frintm  z28.s, p0/m, z27.s
    fsub    z29.s, z27.s, z28.s
    fmul    z29.s, z29.s, z22.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z28.s, p0/m, z28.s
    add     z28.s, z28.s, z23.s
    lsl     z28.s, z28.s, #6
    orr     z27.s, z28.s, z29.s
    fexpa   z27.s, z27.s

    /* Store P values - 4 columns worth */
    /* z24 = exp2 for k+0, z25 = k+1, z26 = k+2, z27 = k+3 */
    /* Each zN has 4 values: [row0, row1, row2, row3] */

    /* Store column k+0 */
    str     s24, [x13]
    mov     w18, v24.s[1]
    str     w18, [x14]
    mov     w18, v24.s[2]
    str     w18, [x15]
    mov     w18, v24.s[3]
    str     w18, [x16]

    /* Store column k+1 */
    str     s25, [x13, #4]
    mov     w18, v25.s[1]
    str     w18, [x14, #4]
    mov     w18, v25.s[2]
    str     w18, [x15, #4]
    mov     w18, v25.s[3]
    str     w18, [x16, #4]

    /* Store column k+2 */
    str     s26, [x13, #8]
    mov     w18, v26.s[1]
    str     w18, [x14, #8]
    mov     w18, v26.s[2]
    str     w18, [x15, #8]
    mov     w18, v26.s[3]
    str     w18, [x16, #8]

    /* Store column k+3 */
    str     s27, [x13, #12]
    mov     w18, v27.s[1]
    str     w18, [x14, #12]
    mov     w18, v27.s[2]
    str     w18, [x15, #12]
    mov     w18, v27.s[3]
    str     w18, [x16, #12]

    /* ================ K+0 GEMM ================ */
    ld1rw   {z24.s}, p0/z, [x13]
    ld1rw   {z25.s}, p0/z, [x14]
    ld1rw   {z26.s}, p0/z, [x15]
    ld1rw   {z27.s}, p0/z, [x16]

    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

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

    /* ================ K+1 GEMM ================ */
    add     x18, x13, #4
    ld1rw   {z24.s}, p0/z, [x18]
    add     x18, x14, #4
    ld1rw   {z25.s}, p0/z, [x18]
    add     x18, x15, #4
    ld1rw   {z26.s}, p0/z, [x18]
    add     x18, x16, #4
    ld1rw   {z27.s}, p0/z, [x18]

    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

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

    /* ================ K+2 GEMM ================ */
    add     x18, x13, #8
    ld1rw   {z24.s}, p0/z, [x18]
    add     x18, x14, #8
    ld1rw   {z25.s}, p0/z, [x18]
    add     x18, x15, #8
    ld1rw   {z26.s}, p0/z, [x18]
    add     x18, x16, #8
    ld1rw   {z27.s}, p0/z, [x18]

    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

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

    /* ================ K+3 GEMM ================ */
    add     x18, x13, #12
    ld1rw   {z24.s}, p0/z, [x18]
    add     x18, x14, #12
    ld1rw   {z25.s}, p0/z, [x18]
    add     x18, x15, #12
    ld1rw   {z26.s}, p0/z, [x18]
    add     x18, x16, #12
    ld1rw   {z27.s}, p0/z, [x18]

    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

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

    /* Advance P pointers by 4 columns */
    add     x13, x13, #16
    add     x14, x14, #16
    add     x15, x15, #16
    add     x16, x16, #16

    subs    x17, x17, #1
    b.gt    .L4k_loop

.L4k_store:
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

    .size exp2_4k_4x4, .-exp2_4k_4x4
