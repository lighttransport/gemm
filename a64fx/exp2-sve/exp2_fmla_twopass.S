/*
 * Two-Pass Fused exp2 + GEMM
 *
 * Pass 1: exp2 + GEMM + store exp2 to cache
 *   - Compute exp2(S) for K block
 *   - Do GEMM with V
 *   - Store exp2 results to P buffer (stays in L1/L2)
 *
 * Pass 2: Load cached exp2 + GEMM
 *   - Load P from cache (L1 hit)
 *   - Do GEMM with next V tile
 *
 * Expected efficiency:
 *   Pass 1: ~70% (exp2 + GEMM fused, store overlaps with compute)
 *   Pass 2: ~80% (pure GEMM with L1 loads)
 *   Combined: ~75%
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*============================================================================
 * exp2_gemm_pass1 - Fused exp2 + GEMM + store exp2
 *
 * void exp2_gemm_pass1(
 *     const int32_t* S,   // x0: [4][Nc] attention scores
 *     const float* V,     // x1: [Nc][64] value matrix
 *     float* O,           // x2: [4][64] output
 *     float* P,           // x3: [4][Nc] exp2 output (cache buffer)
 *     int Nc,             // x4: K dimension
 *     float scale,        // s0
 *     float max_val,      // s1
 *     int ld_s,           // x5: S stride in elements
 *     int ld_v,           // x6: V stride in bytes
 *     int ld_o            // x7: O stride in bytes
 * );
 *
 * Key optimization: Store exp2 results to P buffer while doing GEMM
 * The store can overlap with FMLA execution on A64FX
 *============================================================================*/

    .global exp2_gemm_pass1
    .type exp2_gemm_pass1, %function
exp2_gemm_pass1:
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
    mov     z20.s, s0                   // scale
    mov     z21.s, s1                   // max

    /* exp2 constants */
    mov     w8, #0x42800000             // 64.0f
    mov     z22.s, w8
    mov     w8, #127
    mov     z23.s, w8

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

    cbz     x20, .Lp1_store

    /* Compute row pointers for S */
    lsl     x5, x5, #2                  // ld_s in bytes
    mov     x21, x0                     // S row 0
    add     x22, x0, x5                 // S row 1
    add     x8, x22, x5                 // S row 2
    add     x9, x8, x5                  // S row 3

    /* P row pointers (same stride as Nc) */
    mov     x10, x19                    // P row 0
    lsl     x11, x20, #2                // Nc * 4 bytes
    add     x12, x10, x11               // P row 1
    add     x13, x12, x11               // P row 2
    add     x14, x13, x11               // P row 3

/*============================================================================
 * Main loop: K-blocked, process 16 K values at a time
 * For each K block:
 *   1. Load S[row, k:k+16], compute exp2, store to P
 *   2. Do 16 iterations of GEMM using the computed exp2
 *============================================================================*/
    lsr     x20, x20, #4                // Nc / 16

.Lp1_kblock:
    /* Load S and compute exp2 for all 4 rows (16 K values each) */
    /* Row 0 */
    ld1w    {z16.s}, p0/z, [x21]
    scvtf   z16.s, p0/m, z16.s
    fmul    z16.s, z16.s, z20.s
    fsub    z16.s, z16.s, z21.s
    frintm  z24.s, p0/m, z16.s
    fsub    z16.s, z16.s, z24.s
    fmul    z16.s, z16.s, z22.s
    fcvtzs  z16.s, p0/m, z16.s
    fcvtzs  z24.s, p0/m, z24.s
    add     z24.s, z24.s, z23.s
    lsl     z24.s, z24.s, #6
    orr     z16.s, z24.s, z16.s
    fexpa   z16.s, z16.s
    st1w    {z16.s}, p0, [x10]          // Store exp2 to P

    /* Row 1 */
    ld1w    {z17.s}, p0/z, [x22]
    scvtf   z17.s, p0/m, z17.s
    fmul    z17.s, z17.s, z20.s
    fsub    z17.s, z17.s, z21.s
    frintm  z24.s, p0/m, z17.s
    fsub    z17.s, z17.s, z24.s
    fmul    z17.s, z17.s, z22.s
    fcvtzs  z17.s, p0/m, z17.s
    fcvtzs  z24.s, p0/m, z24.s
    add     z24.s, z24.s, z23.s
    lsl     z24.s, z24.s, #6
    orr     z17.s, z24.s, z17.s
    fexpa   z17.s, z17.s
    st1w    {z17.s}, p0, [x12]

    /* Row 2 */
    ld1w    {z18.s}, p0/z, [x8]
    scvtf   z18.s, p0/m, z18.s
    fmul    z18.s, z18.s, z20.s
    fsub    z18.s, z18.s, z21.s
    frintm  z24.s, p0/m, z18.s
    fsub    z18.s, z18.s, z24.s
    fmul    z18.s, z18.s, z22.s
    fcvtzs  z18.s, p0/m, z18.s
    fcvtzs  z24.s, p0/m, z24.s
    add     z24.s, z24.s, z23.s
    lsl     z24.s, z24.s, #6
    orr     z18.s, z24.s, z18.s
    fexpa   z18.s, z18.s
    st1w    {z18.s}, p0, [x13]

    /* Row 3 */
    ld1w    {z19.s}, p0/z, [x9]
    scvtf   z19.s, p0/m, z19.s
    fmul    z19.s, z19.s, z20.s
    fsub    z19.s, z19.s, z21.s
    frintm  z24.s, p0/m, z19.s
    fsub    z19.s, z19.s, z24.s
    fmul    z19.s, z19.s, z22.s
    fcvtzs  z19.s, p0/m, z19.s
    fcvtzs  z24.s, p0/m, z24.s
    add     z24.s, z24.s, z23.s
    lsl     z24.s, z24.s, #6
    orr     z19.s, z24.s, z19.s
    fexpa   z19.s, z19.s
    st1w    {z19.s}, p0, [x14]

    /* Now z16-z19 have exp2 values, do 16 K iterations of GEMM */
    /* Unroll all 16 iterations */

    .irp idx, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    dup     z24.s, z16.s[\idx]
    dup     z25.s, z17.s[\idx]
    dup     z26.s, z18.s[\idx]
    dup     z27.s, z19.s[\idx]
    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]
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
    add     x1, x1, x6
    .endr

    /* Advance pointers */
    add     x21, x21, #64
    add     x22, x22, #64
    add     x8, x8, #64
    add     x9, x9, #64
    add     x10, x10, #64
    add     x12, x12, #64
    add     x13, x13, #64
    add     x14, x14, #64

    subs    x20, x20, #1
    b.gt    .Lp1_kblock

.Lp1_store:
    /* Store output */
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

    .size exp2_gemm_pass1, .-exp2_gemm_pass1


/*============================================================================
 * gemm_pass2 - Load cached exp2 + GEMM
 *
 * void gemm_pass2(
 *     const float* P,     // x0: [4][Nc] cached exp2 (in L1/L2)
 *     const float* V,     // x1: [Nc][64] value matrix
 *     float* O,           // x2: [4][64] output
 *     int Nc,             // x3: K dimension
 *     int ld_p,           // x4: P stride in bytes
 *     int ld_v,           // x5: V stride in bytes
 *     int ld_o            // x6: O stride in bytes
 * );
 *
 * This is essentially the same as gemm_fp32_4x4 but optimized for
 * when P is already in L1 cache from Pass 1.
 *============================================================================*/

    .global gemm_pass2
    .type gemm_pass2, %function
gemm_pass2:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.s

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

    cbz     x3, .Lp2_store

    /* P row pointers */
    mov     x7, x0                      // P row 0
    add     x8, x0, x4                  // P row 1
    add     x9, x8, x4                  // P row 2
    add     x10, x9, x4                 // P row 3

.Lp2_loop:
    /* Load P[row, k] - broadcast from L1 cache */
    ld1rw   {z16.s}, p0/z, [x7]
    ld1rw   {z17.s}, p0/z, [x8]
    ld1rw   {z18.s}, p0/z, [x9]
    ld1rw   {z19.s}, p0/z, [x10]

    /* Load V[k, :] */
    ld1w    {z20.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z21.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z22.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z23.s}, p0/z, [x1, #3, mul vl]

    /* 16 FMLAs */
    fmla    z0.s, p0/m, z16.s, z20.s
    fmla    z1.s, p0/m, z16.s, z21.s
    fmla    z2.s, p0/m, z16.s, z22.s
    fmla    z3.s, p0/m, z16.s, z23.s
    fmla    z4.s, p0/m, z17.s, z20.s
    fmla    z5.s, p0/m, z17.s, z21.s
    fmla    z6.s, p0/m, z17.s, z22.s
    fmla    z7.s, p0/m, z17.s, z23.s
    fmla    z8.s, p0/m, z18.s, z20.s
    fmla    z9.s, p0/m, z18.s, z21.s
    fmla    z10.s, p0/m, z18.s, z22.s
    fmla    z11.s, p0/m, z18.s, z23.s
    fmla    z12.s, p0/m, z19.s, z20.s
    fmla    z13.s, p0/m, z19.s, z21.s
    fmla    z14.s, p0/m, z19.s, z22.s
    fmla    z15.s, p0/m, z19.s, z23.s

    /* Advance */
    add     x7, x7, #4
    add     x8, x8, #4
    add     x9, x9, #4
    add     x10, x10, #4
    add     x1, x1, x5

    subs    x3, x3, #1
    b.gt    .Lp2_loop

.Lp2_store:
    mov     x7, x2
    st1w    {z0.s}, p0, [x7, #0, mul vl]
    st1w    {z1.s}, p0, [x7, #1, mul vl]
    st1w    {z2.s}, p0, [x7, #2, mul vl]
    st1w    {z3.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x6
    st1w    {z4.s}, p0, [x7, #0, mul vl]
    st1w    {z5.s}, p0, [x7, #1, mul vl]
    st1w    {z6.s}, p0, [x7, #2, mul vl]
    st1w    {z7.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x6
    st1w    {z8.s}, p0, [x7, #0, mul vl]
    st1w    {z9.s}, p0, [x7, #1, mul vl]
    st1w    {z10.s}, p0, [x7, #2, mul vl]
    st1w    {z11.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x6
    st1w    {z12.s}, p0, [x7, #0, mul vl]
    st1w    {z13.s}, p0, [x7, #1, mul vl]
    st1w    {z14.s}, p0, [x7, #2, mul vl]
    st1w    {z15.s}, p0, [x7, #3, mul vl]

    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret

    .size gemm_pass2, .-gemm_pass2
