/*
 * K-Blocked Fused exp2 + FMLA GEMM Kernel
 *
 * Key optimization: Process K in blocks of 16
 * - Compute exp2 for 4 rows × 16 K values = 64 elements using 4 FEXPA ops
 * - Keep exp2 results in registers (z16-z19)
 * - Do 16 K iterations of FMLA using DUP to broadcast
 *
 * Target: exp2 overhead < 5% of GEMM time
 *
 * Cost per K_block=16:
 *   exp2: 4 rows × ~10 instr = 40 instructions
 *   FMLA: 16 K × (4 DUP + 4 LD + 16 FMLA) = ~380 instructions
 *   exp2 ratio: 40/420 = 9.5%
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*============================================================================
 * exp2_fmla_kblock16 - K-blocked fused kernel (K block = 16)
 *
 * void exp2_fmla_kblock16(
 *     const int32_t* S,   // x0: [4][Nc] row-major
 *     const float* V,     // x1: [Nc][64] row-major
 *     float* O,           // x2: [4][64] row-major
 *     int Nc,             // x3: must be multiple of 16
 *     float scale,        // s0
 *     float max_val,      // s1
 *     int ld_s,           // x4: S stride in elements
 *     int ld_v,           // x5: V stride in bytes
 *     int ld_o            // x6: O stride in bytes
 * );
 *============================================================================*/

    .global exp2_fmla_kblock16
    .type exp2_fmla_kblock16, %function
exp2_fmla_kblock16:
    stp     d8, d9, [sp, #-96]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #64]
    stp     x21, x22, [sp, #80]

    ptrue   p0.s

    /* Save scale/max to vector registers */
    mov     z20.s, s0                   // scale broadcast
    mov     z21.s, s1                   // max broadcast

    /* Zero 16 FP32 accumulators */
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

    /* Constants for fast exp2 */
    mov     w7, #127
    mov     z22.s, w7                   // bias = 127
    mov     w7, #0x42800000             // 64.0f
    mov     z23.s, w7

    cbz     x3, .Lkb_store

    /* Compute row offsets for S */
    lsl     x4, x4, #2                  // ld_s in bytes
    mov     x19, x0                     // S row 0
    add     x20, x0, x4                 // S row 1
    add     x21, x20, x4                // S row 2
    add     x22, x21, x4                // S row 3

    /* Number of K blocks */
    lsr     x3, x3, #4                  // Nc / 16

/*============================================================================
 * Main loop: process K in blocks of 16
 *============================================================================*/
.Lkb_kblock_loop:
    /*========================================================================
     * Phase 1: Load S[row, k:k+16] and compute exp2 for all 4 rows
     * Results in z16-z19 (one vector per row, 16 elements each)
     *========================================================================*/

    /* Row 0: Load S[0, k:k+16], compute exp2 */
    ld1w    {z16.s}, p0/z, [x19]
    scvtf   z16.s, p0/m, z16.s          // int32 → fp32
    fmul    z16.s, z16.s, z20.s         // * scale
    fsub    z16.s, z16.s, z21.s         // - max

    /* Fast exp2: minimize instructions */
    frintm  z24.s, p0/m, z16.s          // N = floor(x)
    fsub    z16.s, z16.s, z24.s         // f = x - N
    fmul    z16.s, z16.s, z23.s         // f * 64
    fcvtzs  z16.s, p0/m, z16.s          // m = int(f*64)
    fcvtzs  z24.s, p0/m, z24.s          // N as int
    add     z24.s, z24.s, z22.s         // N + 127
    lsl     z24.s, z24.s, #6            // << 6
    orr     z16.s, z24.s, z16.s         // FEXPA input
    fexpa   z16.s, z16.s                // exp2 result for row 0

    /* Row 1 */
    ld1w    {z17.s}, p0/z, [x20]
    scvtf   z17.s, p0/m, z17.s
    fmul    z17.s, z17.s, z20.s
    fsub    z17.s, z17.s, z21.s
    frintm  z24.s, p0/m, z17.s
    fsub    z17.s, z17.s, z24.s
    fmul    z17.s, z17.s, z23.s
    fcvtzs  z17.s, p0/m, z17.s
    fcvtzs  z24.s, p0/m, z24.s
    add     z24.s, z24.s, z22.s
    lsl     z24.s, z24.s, #6
    orr     z17.s, z24.s, z17.s
    fexpa   z17.s, z17.s

    /* Row 2 */
    ld1w    {z18.s}, p0/z, [x21]
    scvtf   z18.s, p0/m, z18.s
    fmul    z18.s, z18.s, z20.s
    fsub    z18.s, z18.s, z21.s
    frintm  z24.s, p0/m, z18.s
    fsub    z18.s, z18.s, z24.s
    fmul    z18.s, z18.s, z23.s
    fcvtzs  z18.s, p0/m, z18.s
    fcvtzs  z24.s, p0/m, z24.s
    add     z24.s, z24.s, z22.s
    lsl     z24.s, z24.s, #6
    orr     z18.s, z24.s, z18.s
    fexpa   z18.s, z18.s

    /* Row 3 */
    ld1w    {z19.s}, p0/z, [x22]
    scvtf   z19.s, p0/m, z19.s
    fmul    z19.s, z19.s, z20.s
    fsub    z19.s, z19.s, z21.s
    frintm  z24.s, p0/m, z19.s
    fsub    z19.s, z19.s, z24.s
    fmul    z19.s, z19.s, z23.s
    fcvtzs  z19.s, p0/m, z19.s
    fcvtzs  z24.s, p0/m, z24.s
    add     z24.s, z24.s, z22.s
    lsl     z24.s, z24.s, #6
    orr     z19.s, z24.s, z19.s
    fexpa   z19.s, z19.s

    /* Now z16-z19 contain exp2 values for 4 rows × 16 K values */

    /*========================================================================
     * Phase 2: 16 K iterations of FMLA
     * For each k_sub in 0..15:
     *   - DUP broadcast p[row, k_sub] from z16-z19
     *   - Load V[k, :]
     *   - 16 FMLAs
     *========================================================================*/

    /* K iteration 0 */
    dup     z24.s, z16.s[0]
    dup     z25.s, z17.s[0]
    dup     z26.s, z18.s[0]
    dup     z27.s, z19.s[0]
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
    add     x1, x1, x5

    /* K iteration 1 */
    dup     z24.s, z16.s[1]
    dup     z25.s, z17.s[1]
    dup     z26.s, z18.s[1]
    dup     z27.s, z19.s[1]
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
    add     x1, x1, x5

    /* K iteration 2 */
    dup     z24.s, z16.s[2]
    dup     z25.s, z17.s[2]
    dup     z26.s, z18.s[2]
    dup     z27.s, z19.s[2]
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
    add     x1, x1, x5

    /* K iteration 3 */
    dup     z24.s, z16.s[3]
    dup     z25.s, z17.s[3]
    dup     z26.s, z18.s[3]
    dup     z27.s, z19.s[3]
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
    add     x1, x1, x5

    /* K iteration 4 */
    dup     z24.s, z16.s[4]
    dup     z25.s, z17.s[4]
    dup     z26.s, z18.s[4]
    dup     z27.s, z19.s[4]
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
    add     x1, x1, x5

    /* K iteration 5 */
    dup     z24.s, z16.s[5]
    dup     z25.s, z17.s[5]
    dup     z26.s, z18.s[5]
    dup     z27.s, z19.s[5]
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
    add     x1, x1, x5

    /* K iteration 6 */
    dup     z24.s, z16.s[6]
    dup     z25.s, z17.s[6]
    dup     z26.s, z18.s[6]
    dup     z27.s, z19.s[6]
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
    add     x1, x1, x5

    /* K iteration 7 */
    dup     z24.s, z16.s[7]
    dup     z25.s, z17.s[7]
    dup     z26.s, z18.s[7]
    dup     z27.s, z19.s[7]
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
    add     x1, x1, x5

    /* K iteration 8 */
    dup     z24.s, z16.s[8]
    dup     z25.s, z17.s[8]
    dup     z26.s, z18.s[8]
    dup     z27.s, z19.s[8]
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
    add     x1, x1, x5

    /* K iteration 9 */
    dup     z24.s, z16.s[9]
    dup     z25.s, z17.s[9]
    dup     z26.s, z18.s[9]
    dup     z27.s, z19.s[9]
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
    add     x1, x1, x5

    /* K iteration 10 */
    dup     z24.s, z16.s[10]
    dup     z25.s, z17.s[10]
    dup     z26.s, z18.s[10]
    dup     z27.s, z19.s[10]
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
    add     x1, x1, x5

    /* K iteration 11 */
    dup     z24.s, z16.s[11]
    dup     z25.s, z17.s[11]
    dup     z26.s, z18.s[11]
    dup     z27.s, z19.s[11]
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
    add     x1, x1, x5

    /* K iteration 12 */
    dup     z24.s, z16.s[12]
    dup     z25.s, z17.s[12]
    dup     z26.s, z18.s[12]
    dup     z27.s, z19.s[12]
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
    add     x1, x1, x5

    /* K iteration 13 */
    dup     z24.s, z16.s[13]
    dup     z25.s, z17.s[13]
    dup     z26.s, z18.s[13]
    dup     z27.s, z19.s[13]
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
    add     x1, x1, x5

    /* K iteration 14 */
    dup     z24.s, z16.s[14]
    dup     z25.s, z17.s[14]
    dup     z26.s, z18.s[14]
    dup     z27.s, z19.s[14]
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
    add     x1, x1, x5

    /* K iteration 15 */
    dup     z24.s, z16.s[15]
    dup     z25.s, z17.s[15]
    dup     z26.s, z18.s[15]
    dup     z27.s, z19.s[15]
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
    add     x1, x1, x5

    /* Advance S pointers by 16 elements */
    add     x19, x19, #64
    add     x20, x20, #64
    add     x21, x21, #64
    add     x22, x22, #64

    subs    x3, x3, #1
    b.gt    .Lkb_kblock_loop

/*============================================================================
 * Store results
 *============================================================================*/
.Lkb_store:
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

    ldp     x21, x22, [sp, #80]
    ldp     x19, x20, [sp, #64]
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #96
    ret

    .size exp2_fmla_kblock16, .-exp2_fmla_kblock16
