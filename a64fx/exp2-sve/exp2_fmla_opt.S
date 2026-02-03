/*
 * Optimized Fused exp2 + FMLA GEMM Kernel for Flash Attention Stage 2
 *
 * Key optimization: Process K in chunks with software pipelining
 * - Load next S values while computing current exp2
 * - Compute next exp2 while doing current FMLAs
 *
 * A64FX Peak Performance:
 *   FP32 FMLA: 2 pipes × 2 GHz × 16 elem × 2 FLOP = 128 GFLOPS
 */

    .arch armv8.2-a+sve
    .text
    .align 4

    .section .rodata
    .align 4
.Lconst_64f_opt:
    .float 64.0

    .text

/*============================================================================
 * exp2_fmla_fp32_opt - Optimized fused kernel with K unrolling
 *
 * Key changes from basic version:
 * 1. Unroll K loop by 4 to reduce loop overhead
 * 2. Precompute all 16 exp2 values (4 rows × 4 K iterations) in parallel
 * 3. Software pipeline loads and computation
 *
 * void exp2_fmla_fp32_opt(
 *     const int32_t* S,   // x0: [4][Nc] attention scores (row-major)
 *     const float* V,     // x1: [Nc][64] value matrix (row-major, 4 SVE vectors)
 *     float* O,           // x2: [4][64] output (row-major)
 *     int Nc,             // x3: inner dimension (must be multiple of 4)
 *     float scale,        // s0: softmax scale
 *     float max_val,      // s1: max for numerical stability
 *     int ld_s,           // x4: leading dim of S in elements
 *     int ld_v,           // x5: leading dim of V in bytes
 *     int ld_o            // x6: leading dim of O in bytes
 * );
 *============================================================================*/

    .global exp2_fmla_fp32_opt
    .type exp2_fmla_fp32_opt, %function
exp2_fmla_fp32_opt:
    stp     d8, d9, [sp, #-96]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #64]
    stp     x21, x22, [sp, #80]

    ptrue   p0.s

    /* Save scale/max */
    fmov    s16, s0                     // scale
    fmov    s17, s1                     // max_val
    mov     z20.s, s16                  // scale broadcast for vectorized exp2
    mov     z21.s, s17                  // max broadcast

    /* Zero 16 FP32 accumulators (4 rows × 4 vectors) */
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

    /* Load constants */
    adrp    x7, .Lconst_64f_opt
    add     x7, x7, :lo12:.Lconst_64f_opt
    ld1rw   {z22.s}, p0/z, [x7]         // 64.0f

    mov     w7, #127
    mov     z23.s, w7                   // bias

    cbz     x3, .Lopt_store

    /* Convert ld_s to bytes and compute row offsets */
    lsl     x4, x4, #2                  // ld_s * 4 bytes
    mov     x19, x0                     // S row 0
    add     x20, x0, x4                 // S row 1
    add     x21, x20, x4                // S row 2
    add     x22, x21, x4                // S row 3

    /* Main loop - process one K at a time but optimize exp2 */
.Lopt_loop:
    /* Load S[0:4, k] into low 4 lanes of z24 using NEON inserts */
    ldr     w8, [x19]                   // S[0, k]
    ldr     w9, [x20]                   // S[1, k]
    ldr     w10, [x21]                  // S[2, k]
    ldr     w11, [x22]                  // S[3, k]

    /* Pack into v24 as int32 */
    fmov    s24, w8
    mov     v24.s[1], w9
    mov     v24.s[2], w10
    mov     v24.s[3], w11

    /* Convert int32 to fp32 (vectorized) */
    scvtf   v24.4s, v24.4s

    /* Apply scale and subtract max (vectorized NEON for low 4 lanes) */
    dup     v25.4s, v20.s[0]            // scale
    dup     v26.4s, v21.s[0]            // max
    fmul    v24.4s, v24.4s, v25.4s
    fsub    v24.4s, v24.4s, v26.4s

    /* exp2 using FEXPA (on z24, only bottom 4 lanes matter) */
    frintm  z28.s, p0/m, z24.s          // N = floor(x)
    fsub    z25.s, z24.s, z28.s         // f = x - N

    fmul    z25.s, z25.s, z22.s         // f * 64
    fcvtzs  z25.s, p0/m, z25.s          // m = int(f * 64)
    fcvtzs  z28.s, p0/m, z28.s          // N as int

    add     z28.s, z28.s, z23.s         // N + 127
    lsl     z28.s, z28.s, #6            // << 6
    orr     z24.s, z28.s, z25.s         // combine
    fexpa   z24.s, z24.s                // exp2 via FEXPA

    /* Broadcast each exp2 result to full vectors */
    dup     z25.s, z24.s[1]             // p1 broadcast
    dup     z26.s, z24.s[2]             // p2 broadcast
    dup     z27.s, z24.s[3]             // p3 broadcast
    dup     z24.s, z24.s[0]             // p0 broadcast

    /* Load V[k, 0:64] - 4 fp32 vectors */
    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]

    /* FMLA: O[i, :] += p[i] * V[k, :] */
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
    add     x19, x19, #4
    add     x20, x20, #4
    add     x21, x21, #4
    add     x22, x22, #4
    add     x1, x1, x5

    subs    x3, x3, #1
    b.gt    .Lopt_loop

.Lopt_store:
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

    .size exp2_fmla_fp32_opt, .-exp2_fmla_fp32_opt


/*============================================================================
 * exp2_fmla_fp32_k4 - K-unrolled fused kernel
 *
 * Process 4 K iterations at once to improve ILP
 * - Compute 16 exp2 values (4 rows × 4 K)
 * - Then do 64 FMLAs
 *
 * This amortizes exp2 overhead over more FMLA work
 *============================================================================*/

    .global exp2_fmla_fp32_k4
    .type exp2_fmla_fp32_k4, %function
exp2_fmla_fp32_k4:
    stp     d8, d9, [sp, #-128]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #64]
    stp     x21, x22, [sp, #80]
    stp     x23, x24, [sp, #96]
    stp     x25, x26, [sp, #112]

    ptrue   p0.s

    /* Save scale/max */
    fmov    s16, s0
    fmov    s17, s1
    mov     z20.s, s16
    mov     z21.s, s17

    /* Zero 16 accumulators */
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

    /* Load constants */
    adrp    x7, .Lconst_64f_opt
    add     x7, x7, :lo12:.Lconst_64f_opt
    ld1rw   {z22.s}, p0/z, [x7]
    mov     w7, #127
    mov     z23.s, w7

    cbz     x3, .Lk4_store

    /* Row pointers */
    lsl     x4, x4, #2
    mov     x19, x0
    add     x20, x0, x4
    add     x21, x20, x4
    add     x22, x21, x4

    /* Check if we can do K/4 iterations */
    lsr     x23, x3, #2                 // K / 4
    and     x24, x3, #3                 // K % 4

    cbz     x23, .Lk4_remainder

.Lk4_loop:
    /*====================================================================
     * Load S[row, k:k+4] for all 4 rows (16 values total)
     * Pack into 4 vectors: each row gets one vector with 4 consecutive K values
     *====================================================================*/

    /* Row 0: S[0, k:k+4] -> z24 low 4 lanes */
    ldp     w8, w9, [x19]
    ldp     w10, w11, [x19, #8]
    fmov    s24, w8
    mov     v24.s[1], w9
    mov     v24.s[2], w10
    mov     v24.s[3], w11

    /* Row 1: S[1, k:k+4] -> z25 low 4 lanes */
    ldp     w8, w9, [x20]
    ldp     w10, w11, [x20, #8]
    fmov    s25, w8
    mov     v25.s[1], w9
    mov     v25.s[2], w10
    mov     v25.s[3], w11

    /* Row 2: S[2, k:k+4] -> z26 low 4 lanes */
    ldp     w8, w9, [x21]
    ldp     w10, w11, [x21, #8]
    fmov    s26, w8
    mov     v26.s[1], w9
    mov     v26.s[2], w10
    mov     v26.s[3], w11

    /* Row 3: S[3, k:k+4] -> z27 low 4 lanes */
    ldp     w8, w9, [x22]
    ldp     w10, w11, [x22, #8]
    fmov    s27, w8
    mov     v27.s[1], w9
    mov     v27.s[2], w10
    mov     v27.s[3], w11

    /* Convert all 16 values to fp32 (4 NEON ops) */
    scvtf   v24.4s, v24.4s
    scvtf   v25.4s, v25.4s
    scvtf   v26.4s, v26.4s
    scvtf   v27.4s, v27.4s

    /* Scale and subtract max (NEON) */
    dup     v28.4s, v20.s[0]            // scale
    dup     v29.4s, v21.s[0]            // max

    fmul    v24.4s, v24.4s, v28.4s
    fmul    v25.4s, v25.4s, v28.4s
    fmul    v26.4s, v26.4s, v28.4s
    fmul    v27.4s, v27.4s, v28.4s

    fsub    v24.4s, v24.4s, v29.4s
    fsub    v25.4s, v25.4s, v29.4s
    fsub    v26.4s, v26.4s, v29.4s
    fsub    v27.4s, v27.4s, v29.4s

    /* Now z24-z27 each have 4 exp2 inputs for rows 0-3 */

    /* exp2 using FEXPA for all 16 values (4 vectors × 4 lanes) */
    /* Row 0 */
    frintm  z28.s, p0/m, z24.s
    fsub    z16.s, z24.s, z28.s
    fmul    z16.s, z16.s, z22.s
    fcvtzs  z16.s, p0/m, z16.s
    fcvtzs  z28.s, p0/m, z28.s
    add     z28.s, z28.s, z23.s
    lsl     z28.s, z28.s, #6
    orr     z24.s, z28.s, z16.s
    fexpa   z24.s, z24.s

    /* Row 1 */
    frintm  z28.s, p0/m, z25.s
    fsub    z16.s, z25.s, z28.s
    fmul    z16.s, z16.s, z22.s
    fcvtzs  z16.s, p0/m, z16.s
    fcvtzs  z28.s, p0/m, z28.s
    add     z28.s, z28.s, z23.s
    lsl     z28.s, z28.s, #6
    orr     z25.s, z28.s, z16.s
    fexpa   z25.s, z25.s

    /* Row 2 */
    frintm  z28.s, p0/m, z26.s
    fsub    z16.s, z26.s, z28.s
    fmul    z16.s, z16.s, z22.s
    fcvtzs  z16.s, p0/m, z16.s
    fcvtzs  z28.s, p0/m, z28.s
    add     z28.s, z28.s, z23.s
    lsl     z28.s, z28.s, #6
    orr     z26.s, z28.s, z16.s
    fexpa   z26.s, z26.s

    /* Row 3 */
    frintm  z28.s, p0/m, z27.s
    fsub    z16.s, z27.s, z28.s
    fmul    z16.s, z16.s, z22.s
    fcvtzs  z16.s, p0/m, z16.s
    fcvtzs  z28.s, p0/m, z28.s
    add     z28.s, z28.s, z23.s
    lsl     z28.s, z28.s, #6
    orr     z27.s, z28.s, z16.s
    fexpa   z27.s, z27.s

    /* Now z24-z27 contain:
     * z24: [p00, p01, p02, p03, ...] (row 0, k=0,1,2,3)
     * z25: [p10, p11, p12, p13, ...] (row 1, k=0,1,2,3)
     * z26: [p20, p21, p22, p23, ...] (row 2, k=0,1,2,3)
     * z27: [p30, p31, p32, p33, ...] (row 3, k=0,1,2,3)
     */

    /* Process K iteration 0 */
    dup     z16.s, z24.s[0]             // p00 broadcast
    dup     z17.s, z25.s[0]             // p10 broadcast
    dup     z18.s, z26.s[0]             // p20 broadcast
    dup     z19.s, z27.s[0]             // p30 broadcast

    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]

    fmla    z0.s, p0/m, z16.s, z28.s
    fmla    z1.s, p0/m, z16.s, z29.s
    fmla    z2.s, p0/m, z16.s, z30.s
    fmla    z3.s, p0/m, z16.s, z31.s
    fmla    z4.s, p0/m, z17.s, z28.s
    fmla    z5.s, p0/m, z17.s, z29.s
    fmla    z6.s, p0/m, z17.s, z30.s
    fmla    z7.s, p0/m, z17.s, z31.s
    fmla    z8.s, p0/m, z18.s, z28.s
    fmla    z9.s, p0/m, z18.s, z29.s
    fmla    z10.s, p0/m, z18.s, z30.s
    fmla    z11.s, p0/m, z18.s, z31.s
    fmla    z12.s, p0/m, z19.s, z28.s
    fmla    z13.s, p0/m, z19.s, z29.s
    fmla    z14.s, p0/m, z19.s, z30.s
    fmla    z15.s, p0/m, z19.s, z31.s

    add     x1, x1, x5

    /* Process K iteration 1 */
    dup     z16.s, z24.s[1]
    dup     z17.s, z25.s[1]
    dup     z18.s, z26.s[1]
    dup     z19.s, z27.s[1]

    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]

    fmla    z0.s, p0/m, z16.s, z28.s
    fmla    z1.s, p0/m, z16.s, z29.s
    fmla    z2.s, p0/m, z16.s, z30.s
    fmla    z3.s, p0/m, z16.s, z31.s
    fmla    z4.s, p0/m, z17.s, z28.s
    fmla    z5.s, p0/m, z17.s, z29.s
    fmla    z6.s, p0/m, z17.s, z30.s
    fmla    z7.s, p0/m, z17.s, z31.s
    fmla    z8.s, p0/m, z18.s, z28.s
    fmla    z9.s, p0/m, z18.s, z29.s
    fmla    z10.s, p0/m, z18.s, z30.s
    fmla    z11.s, p0/m, z18.s, z31.s
    fmla    z12.s, p0/m, z19.s, z28.s
    fmla    z13.s, p0/m, z19.s, z29.s
    fmla    z14.s, p0/m, z19.s, z30.s
    fmla    z15.s, p0/m, z19.s, z31.s

    add     x1, x1, x5

    /* Process K iteration 2 */
    dup     z16.s, z24.s[2]
    dup     z17.s, z25.s[2]
    dup     z18.s, z26.s[2]
    dup     z19.s, z27.s[2]

    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]

    fmla    z0.s, p0/m, z16.s, z28.s
    fmla    z1.s, p0/m, z16.s, z29.s
    fmla    z2.s, p0/m, z16.s, z30.s
    fmla    z3.s, p0/m, z16.s, z31.s
    fmla    z4.s, p0/m, z17.s, z28.s
    fmla    z5.s, p0/m, z17.s, z29.s
    fmla    z6.s, p0/m, z17.s, z30.s
    fmla    z7.s, p0/m, z17.s, z31.s
    fmla    z8.s, p0/m, z18.s, z28.s
    fmla    z9.s, p0/m, z18.s, z29.s
    fmla    z10.s, p0/m, z18.s, z30.s
    fmla    z11.s, p0/m, z18.s, z31.s
    fmla    z12.s, p0/m, z19.s, z28.s
    fmla    z13.s, p0/m, z19.s, z29.s
    fmla    z14.s, p0/m, z19.s, z30.s
    fmla    z15.s, p0/m, z19.s, z31.s

    add     x1, x1, x5

    /* Process K iteration 3 */
    dup     z16.s, z24.s[3]
    dup     z17.s, z25.s[3]
    dup     z18.s, z26.s[3]
    dup     z19.s, z27.s[3]

    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]

    fmla    z0.s, p0/m, z16.s, z28.s
    fmla    z1.s, p0/m, z16.s, z29.s
    fmla    z2.s, p0/m, z16.s, z30.s
    fmla    z3.s, p0/m, z16.s, z31.s
    fmla    z4.s, p0/m, z17.s, z28.s
    fmla    z5.s, p0/m, z17.s, z29.s
    fmla    z6.s, p0/m, z17.s, z30.s
    fmla    z7.s, p0/m, z17.s, z31.s
    fmla    z8.s, p0/m, z18.s, z28.s
    fmla    z9.s, p0/m, z18.s, z29.s
    fmla    z10.s, p0/m, z18.s, z30.s
    fmla    z11.s, p0/m, z18.s, z31.s
    fmla    z12.s, p0/m, z19.s, z28.s
    fmla    z13.s, p0/m, z19.s, z29.s
    fmla    z14.s, p0/m, z19.s, z30.s
    fmla    z15.s, p0/m, z19.s, z31.s

    add     x1, x1, x5

    /* Advance S pointers by 4 */
    add     x19, x19, #16
    add     x20, x20, #16
    add     x21, x21, #16
    add     x22, x22, #16

    subs    x23, x23, #1
    b.gt    .Lk4_loop

.Lk4_remainder:
    cbz     x24, .Lk4_store

    /* Handle remaining K iterations (0-3) */
.Lk4_rem_loop:
    /* Same as single K iteration */
    ldr     w8, [x19]
    ldr     w9, [x20]
    ldr     w10, [x21]
    ldr     w11, [x22]

    fmov    s24, w8
    mov     v24.s[1], w9
    mov     v24.s[2], w10
    mov     v24.s[3], w11

    scvtf   v24.4s, v24.4s
    dup     v28.4s, v20.s[0]
    dup     v29.4s, v21.s[0]
    fmul    v24.4s, v24.4s, v28.4s
    fsub    v24.4s, v24.4s, v29.4s

    frintm  z28.s, p0/m, z24.s
    fsub    z25.s, z24.s, z28.s
    fmul    z25.s, z25.s, z22.s
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z28.s, p0/m, z28.s
    add     z28.s, z28.s, z23.s
    lsl     z28.s, z28.s, #6
    orr     z24.s, z28.s, z25.s
    fexpa   z24.s, z24.s

    dup     z25.s, z24.s[1]
    dup     z26.s, z24.s[2]
    dup     z27.s, z24.s[3]
    dup     z24.s, z24.s[0]

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

    add     x19, x19, #4
    add     x20, x20, #4
    add     x21, x21, #4
    add     x22, x22, #4
    add     x1, x1, x5

    subs    x24, x24, #1
    b.gt    .Lk4_rem_loop

.Lk4_store:
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

    ldp     x25, x26, [sp, #112]
    ldp     x23, x24, [sp, #96]
    ldp     x21, x22, [sp, #80]
    ldp     x19, x20, [sp, #64]
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #128
    ret

    .size exp2_fmla_fp32_k4, .-exp2_fmla_fp32_k4
