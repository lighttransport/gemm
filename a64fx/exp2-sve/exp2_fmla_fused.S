/*
 * Fused exp2 + FMLA GEMM Kernel for Flash Attention Stage 2
 *
 * Computes: O = exp2(S * scale - max) @ V
 *
 * Where:
 *   S: Attention scores from Q @ K^T (int32 input)
 *   V: Value matrix (fp32 or fp16)
 *   O: Output (fp32 or fp16)
 *
 * A64FX Peak Performance:
 *   FP32 FMLA: 2 pipes × 2 GHz × 16 elem × 2 FLOP = 128 GFLOPS
 *   FP16 FMLA: 2 pipes × 2 GHz × 32 elem × 2 FLOP = 256 GFLOPS
 */

    .arch armv8.2-a+sve
    .text
    .align 4

    .section .rodata
    .align 4
.Lconst_64f_fmla:
    .float 64.0

    .text

/*============================================================================
 * exp2_fmla_fp32_4x4
 *
 * Fused exp2 + FP32 FMLA GEMM for one tile
 *
 * void exp2_fmla_fp32_4x4(
 *     const int32_t* S,   // x0: [4][Nc] attention scores (row-major)
 *     const float* V,     // x1: [Nc][64] value matrix (row-major, 4 SVE vectors)
 *     float* O,           // x2: [4][64] output (row-major)
 *     int Nc,             // x3: inner dimension (K for GEMM)
 *     float scale,        // s0: softmax scale
 *     float max_val,      // s1: max for numerical stability
 *     int ld_s,           // x4: leading dim of S in elements
 *     int ld_v,           // x5: leading dim of V in bytes
 *     int ld_o            // x6: leading dim of O in bytes
 * );
 *
 * Algorithm:
 *   for each k in 0..Nc:
 *     p[0:4] = exp2(S[0:4, k] * scale - max)  // 4 scalars
 *     O[0:4, :] += p[0:4] * V[k, :]           // 4x4 FMLA outer product
 *============================================================================*/

    .global exp2_fmla_fp32_4x4
    .type exp2_fmla_fp32_4x4, %function
exp2_fmla_fp32_4x4:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.s

    /* Save scale/max before using z0/z1 */
    fmov    s16, s0                     // scale
    fmov    s17, s1                     // max_val

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
    mov     z20.s, s16                  // scale broadcast
    mov     z21.s, s17                  // max broadcast

    adrp    x7, .Lconst_64f_fmla
    add     x7, x7, :lo12:.Lconst_64f_fmla
    ld1rw   {z22.s}, p0/z, [x7]         // 64.0f

    mov     w7, #127
    mov     z23.s, w7                   // bias

    cbz     x3, .Lfp32_store

    /* Convert ld_s to bytes */
    lsl     x4, x4, #2                  // ld_s * 4 bytes

    /*========================================================================
     * Main loop: for each k, compute exp2 and FMLA
     *========================================================================*/
.Lfp32_loop:
    /* Load S[0:4, k] - 4 int32 scalars, one from each row */
    ldr     w8, [x0]                    // S[0, k]
    ldr     w9, [x0, x4]                // S[1, k]
    add     x7, x0, x4, lsl #1          // S + 2*ld_s
    ldr     w10, [x7]                   // S[2, k]
    ldr     w11, [x7, x4]               // S[3, k]

    /* Convert int32 to fp32 */
    scvtf   s24, w8
    scvtf   s25, w9
    scvtf   s26, w10
    scvtf   s27, w11

    /* Apply scale: p = S * scale */
    fmul    s24, s24, s16
    fmul    s25, s25, s16
    fmul    s26, s26, s16
    fmul    s27, s27, s16

    /* Subtract max: p = p - max */
    fsub    s24, s24, s17
    fsub    s25, s25, s17
    fsub    s26, s26, s17
    fsub    s27, s27, s17

    /* exp2 using vectorized FEXPA
     * Pack 4 scaled floats into v24 (Advanced SIMD, bottom 128 bits of z24)
     * Then apply SVE FEXPA, then broadcast back
     */
    /* s24, s25, s26, s27 contain the 4 scaled values */
    /* Build v24 = [s24, s25, s26, s27] using NEON inserts */
    /* v24.s[0] already has s24 */
    ins     v24.s[1], v25.s[0]
    ins     v24.s[2], v26.s[0]
    ins     v24.s[3], v27.s[0]

    /* Now apply vectorized exp2 on z24 (only bottom 4 lanes matter) */
    frintm  z28.s, p0/m, z24.s          // N = floor(x)
    fsub    z25.s, z24.s, z28.s         // f = x - N

    fmul    z25.s, z25.s, z22.s         // f * 64
    fcvtzs  z25.s, p0/m, z25.s          // m = int(f * 64)
    fcvtzs  z28.s, p0/m, z28.s          // N as int

    add     z28.s, z28.s, z23.s         // N + 127
    lsl     z28.s, z28.s, #6            // << 6
    orr     z24.s, z28.s, z25.s         // combine
    fexpa   z24.s, z24.s                // exp2 via FEXPA

    /* Extract scalars and broadcast to full vectors for FMLA */
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
    /* Row 0 */
    fmla    z0.s, p0/m, z24.s, z28.s
    fmla    z1.s, p0/m, z24.s, z29.s
    fmla    z2.s, p0/m, z24.s, z30.s
    fmla    z3.s, p0/m, z24.s, z31.s

    /* Row 1 */
    fmla    z4.s, p0/m, z25.s, z28.s
    fmla    z5.s, p0/m, z25.s, z29.s
    fmla    z6.s, p0/m, z25.s, z30.s
    fmla    z7.s, p0/m, z25.s, z31.s

    /* Row 2 */
    fmla    z8.s, p0/m, z26.s, z28.s
    fmla    z9.s, p0/m, z26.s, z29.s
    fmla    z10.s, p0/m, z26.s, z30.s
    fmla    z11.s, p0/m, z26.s, z31.s

    /* Row 3 */
    fmla    z12.s, p0/m, z27.s, z28.s
    fmla    z13.s, p0/m, z27.s, z29.s
    fmla    z14.s, p0/m, z27.s, z30.s
    fmla    z15.s, p0/m, z27.s, z31.s

    /* Advance pointers */
    add     x0, x0, #4                  // S += 1 element
    add     x1, x1, x5                  // V += ld_v

    subs    x3, x3, #1
    b.gt    .Lfp32_loop

    /*========================================================================
     * Store results
     *========================================================================*/
.Lfp32_store:
    mov     x7, x2

    /* Row 0 */
    st1w    {z0.s}, p0, [x7, #0, mul vl]
    st1w    {z1.s}, p0, [x7, #1, mul vl]
    st1w    {z2.s}, p0, [x7, #2, mul vl]
    st1w    {z3.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x6

    /* Row 1 */
    st1w    {z4.s}, p0, [x7, #0, mul vl]
    st1w    {z5.s}, p0, [x7, #1, mul vl]
    st1w    {z6.s}, p0, [x7, #2, mul vl]
    st1w    {z7.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x6

    /* Row 2 */
    st1w    {z8.s}, p0, [x7, #0, mul vl]
    st1w    {z9.s}, p0, [x7, #1, mul vl]
    st1w    {z10.s}, p0, [x7, #2, mul vl]
    st1w    {z11.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x6

    /* Row 3 */
    st1w    {z12.s}, p0, [x7, #0, mul vl]
    st1w    {z13.s}, p0, [x7, #1, mul vl]
    st1w    {z14.s}, p0, [x7, #2, mul vl]
    st1w    {z15.s}, p0, [x7, #3, mul vl]

    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

    .size exp2_fmla_fp32_4x4, .-exp2_fmla_fp32_4x4


/*============================================================================
 * exp2_fmla_fp32_vec
 *
 * Vectorized version: processes 16 S values at once (1 SVE vector)
 * Then does 16 × 4 FMLA operations
 *
 * This is more efficient for larger tiles where we can vectorize exp2
 *
 * void exp2_fmla_fp32_vec(
 *     const int32_t* S,   // x0: [16][Nc] attention scores
 *     const float* V,     // x1: [Nc][64] value matrix
 *     float* O,           // x2: [16][64] output
 *     int Nc,             // x3: inner dimension
 *     float scale,        // s0
 *     float max_val,      // s1
 *     int ld_s,           // x4: leading dim of S in elements
 *     int ld_v,           // x5: leading dim of V in bytes
 *     int ld_o            // x6: leading dim of O in bytes
 * );
 *============================================================================*/

    .global exp2_fmla_fp32_vec
    .type exp2_fmla_fp32_vec, %function
exp2_fmla_fp32_vec:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.s

    /* Save parameters */
    fmov    s18, s0                     // scale
    fmov    s19, s1                     // max_val

    /* Zero 4 accumulator vectors (will accumulate 16 rows × 4 cols per row) */
    /* Actually for 16×64 output we need 16×4 = 64 accumulators - too many!
     * Instead, process in 4-row chunks */

    /* For simplicity, this version processes 4 rows at a time with vectorized exp2 */
    /* We'll do 4 passes for 16 rows total */

    mov     x8, #4                      // 4 row chunks
    mov     x9, x0                      // Save S base
    mov     x10, x2                     // Save O base

.Lfp32_vec_row_loop:
    /* Zero accumulators for this 4-row chunk */
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

    /* Setup constants */
    mov     z20.s, s18                  // scale
    mov     z21.s, s19                  // max

    adrp    x7, .Lconst_64f_fmla
    add     x7, x7, :lo12:.Lconst_64f_fmla
    ld1rw   {z22.s}, p0/z, [x7]

    mov     w7, #127
    mov     z23.s, w7

    mov     x11, x3                     // Nc counter
    mov     x12, x1                     // V pointer

    cbz     x11, .Lfp32_vec_store_chunk

    /* Compute row stride in bytes */
    lsl     x13, x4, #2                 // ld_s * 4

.Lfp32_vec_k_loop:
    /* Load S[row:row+4, k] as 4 scalars, convert to vector for exp2 */
    /* Compute addresses first since ld1rw doesn't support register offset */
    /* Use x14, x15, x7 as temps (not x8 which is row counter) */
    add     x14, x0, x13                        // &S[row+1, k]
    add     x15, x0, x13, lsl #1                // &S[row+2, k]
    add     x7, x15, x13                        // &S[row+3, k]

    ld1rw   {z24.s}, p0/z, [x0]                 // S[row+0, k] broadcast
    ld1rw   {z25.s}, p0/z, [x14]                // S[row+1, k] broadcast
    ld1rw   {z26.s}, p0/z, [x15]                // S[row+2, k] broadcast
    ld1rw   {z27.s}, p0/z, [x7]                 // S[row+3, k] broadcast

    /* Convert int32 to fp32 */
    scvtf   z24.s, p0/m, z24.s
    scvtf   z25.s, p0/m, z25.s
    scvtf   z26.s, p0/m, z26.s
    scvtf   z27.s, p0/m, z27.s

    /* Scale */
    fmul    z24.s, z24.s, z20.s
    fmul    z25.s, z25.s, z20.s
    fmul    z26.s, z26.s, z20.s
    fmul    z27.s, z27.s, z20.s

    /* Subtract max */
    fsub    z24.s, z24.s, z21.s
    fsub    z25.s, z25.s, z21.s
    fsub    z26.s, z26.s, z21.s
    fsub    z27.s, z27.s, z21.s

    /* exp2 using FEXPA (vectorized) */
    frintm  z28.s, p0/m, z24.s          // N
    frintm  z29.s, p0/m, z25.s
    frintm  z30.s, p0/m, z26.s
    frintm  z31.s, p0/m, z27.s

    fsub    z24.s, z24.s, z28.s         // f = x - N
    fsub    z25.s, z25.s, z29.s
    fsub    z26.s, z26.s, z30.s
    fsub    z27.s, z27.s, z31.s

    fmul    z24.s, z24.s, z22.s         // f * 64
    fmul    z25.s, z25.s, z22.s
    fmul    z26.s, z26.s, z22.s
    fmul    z27.s, z27.s, z22.s

    fcvtzs  z24.s, p0/m, z24.s          // m
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z27.s, p0/m, z27.s

    fcvtzs  z28.s, p0/m, z28.s          // N as int
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s

    add     z28.s, z28.s, z23.s         // N + 127
    add     z29.s, z29.s, z23.s
    add     z30.s, z30.s, z23.s
    add     z31.s, z31.s, z23.s

    lsl     z28.s, z28.s, #6            // (N + 127) << 6
    lsl     z29.s, z29.s, #6
    lsl     z30.s, z30.s, #6
    lsl     z31.s, z31.s, #6

    orr     z24.s, z28.s, z24.s         // FEXPA input
    orr     z25.s, z29.s, z25.s
    orr     z26.s, z30.s, z26.s
    orr     z27.s, z31.s, z27.s

    fexpa   z24.s, z24.s                // exp2 result (broadcast)
    fexpa   z25.s, z25.s
    fexpa   z26.s, z26.s
    fexpa   z27.s, z27.s

    /* Load V[k, :] - 4 vectors */
    ld1w    {z28.s}, p0/z, [x12, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x12, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x12, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x12, #3, mul vl]

    /* FMLA: O[i,:] += p[i] * V[k,:] */
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

    /* Advance */
    add     x0, x0, #4                  // S += 1 element per row
    add     x12, x12, x5                // V += ld_v

    subs    x11, x11, #1
    b.gt    .Lfp32_vec_k_loop

.Lfp32_vec_store_chunk:
    /* Store this 4-row chunk */
    mov     x7, x10

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
    add     x10, x7, x6                 // Advance O base for next chunk

    /* Advance S base for next 4 rows */
    lsl     x7, x4, #2                  // ld_s in bytes
    lsl     x7, x7, #2                  // * 4 rows
    add     x9, x9, x7
    mov     x0, x9

    subs    x8, x8, #1
    b.gt    .Lfp32_vec_row_loop

    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

    .size exp2_fmla_fp32_vec, .-exp2_fmla_fp32_vec


/*============================================================================
 * exp2_fmla_fp16_4x4
 *
 * FP16 version: exp2 in fp32, convert to fp16, FMLA in fp16
 *
 * void exp2_fmla_fp16_4x4(
 *     const int32_t* S,   // x0: [4][Nc] attention scores
 *     const __fp16* V,    // x1: [Nc][64] value matrix (fp16)
 *     __fp16* O,          // x2: [4][64] output (fp16)
 *     int Nc,             // x3: inner dimension
 *     float scale,        // s0
 *     float max_val,      // s1
 *     int ld_s,           // x4: leading dim of S in elements
 *     int ld_v,           // x5: leading dim of V in bytes
 *     int ld_o            // x6: leading dim of O in bytes
 * );
 *============================================================================*/

    .global exp2_fmla_fp16_4x4
    .type exp2_fmla_fp16_4x4, %function
exp2_fmla_fp16_4x4:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.h                        // FP16 predicate

    /* Save parameters */
    fmov    s18, s0                     // scale (fp32)
    fmov    s19, s1                     // max_val (fp32)

    /* Zero 16 FP16 accumulators (4 rows × 4 vectors of 32 fp16 each) */
    fmov    z0.h, #0
    fmov    z1.h, #0
    fmov    z2.h, #0
    fmov    z3.h, #0
    fmov    z4.h, #0
    fmov    z5.h, #0
    fmov    z6.h, #0
    fmov    z7.h, #0
    fmov    z8.h, #0
    fmov    z9.h, #0
    fmov    z10.h, #0
    fmov    z11.h, #0
    fmov    z12.h, #0
    fmov    z13.h, #0
    fmov    z14.h, #0
    fmov    z15.h, #0

    /* Load constants for exp2 (fp32) */
    ptrue   p1.s
    mov     z20.s, s18                  // scale
    mov     z21.s, s19                  // max

    adrp    x7, .Lconst_64f_fmla
    add     x7, x7, :lo12:.Lconst_64f_fmla
    ld1rw   {z22.s}, p1/z, [x7]

    mov     w7, #127
    mov     z23.s, w7

    cbz     x3, .Lfp16_store

    lsl     x4, x4, #2                  // ld_s in bytes

.Lfp16_loop:
    /* Load S[0:4, k] */
    ldr     w8, [x0]
    ldr     w9, [x0, x4]
    add     x7, x0, x4, lsl #1
    ldr     w10, [x7]
    ldr     w11, [x7, x4]

    /* exp2 in fp32 (same as fp32 version) */
    scvtf   s24, w8
    scvtf   s25, w9
    scvtf   s26, w10
    scvtf   s27, w11

    fmul    s24, s24, s18
    fmul    s25, s25, s18
    fmul    s26, s26, s18
    fmul    s27, s27, s18

    fsub    s24, s24, s19
    fsub    s25, s25, s19
    fsub    s26, s26, s19
    fsub    s27, s27, s19

    frintm  s28, s24
    frintm  s29, s25
    frintm  s30, s26
    frintm  s31, s27

    fsub    s24, s24, s28
    fsub    s25, s25, s29
    fsub    s26, s26, s30
    fsub    s27, s27, s31

    mov     w7, #0x42800000             // 64.0f
    fmov    s17, w7
    fmul    s24, s24, s17
    fmul    s25, s25, s17
    fmul    s26, s26, s17
    fmul    s27, s27, s17

    fcvtzs  w8, s24
    fcvtzs  w9, s25
    fcvtzs  w10, s26
    fcvtzs  w11, s27

    fcvtzs  w12, s28
    fcvtzs  w13, s29
    fcvtzs  w14, s30
    fcvtzs  w15, s31

    add     w12, w12, #127
    add     w13, w13, #127
    add     w14, w14, #127
    add     w15, w15, #127

    lsl     w12, w12, #6
    lsl     w13, w13, #6
    lsl     w14, w14, #6
    lsl     w15, w15, #6

    orr     w12, w12, w8
    orr     w13, w13, w9
    orr     w14, w14, w10
    orr     w15, w15, w11

    fmov    s24, w12
    fmov    s25, w13
    fmov    s26, w14
    fmov    s27, w15

    /* Broadcast and FEXPA */
    mov     z24.s, s24
    mov     z25.s, s25
    mov     z26.s, s26
    mov     z27.s, s27

    fexpa   z24.s, z24.s
    fexpa   z25.s, z25.s
    fexpa   z26.s, z26.s
    fexpa   z27.s, z27.s

    /* Convert exp2 results from fp32 to fp16 and broadcast */
    fcvt    z24.h, p1/m, z24.s          // fp32 -> fp16
    fcvt    z25.h, p1/m, z25.s
    fcvt    z26.h, p1/m, z26.s
    fcvt    z27.h, p1/m, z27.s

    /* Replicate fp16 values to fill vector */
    /* After fcvt, only even lanes have values. Need to broadcast the single fp16 */
    /* Use DUP to broadcast the first fp16 element */
    mov     z24.h, h24
    mov     z25.h, h25
    mov     z26.h, h26
    mov     z27.h, h27

    /* Load V[k, :] - 4 fp16 vectors (32 elements each) */
    ld1h    {z28.h}, p0/z, [x1, #0, mul vl]
    ld1h    {z29.h}, p0/z, [x1, #1, mul vl]
    ld1h    {z30.h}, p0/z, [x1, #2, mul vl]
    ld1h    {z31.h}, p0/z, [x1, #3, mul vl]

    /* FMLA in fp16 */
    fmla    z0.h, p0/m, z24.h, z28.h
    fmla    z1.h, p0/m, z24.h, z29.h
    fmla    z2.h, p0/m, z24.h, z30.h
    fmla    z3.h, p0/m, z24.h, z31.h

    fmla    z4.h, p0/m, z25.h, z28.h
    fmla    z5.h, p0/m, z25.h, z29.h
    fmla    z6.h, p0/m, z25.h, z30.h
    fmla    z7.h, p0/m, z25.h, z31.h

    fmla    z8.h, p0/m, z26.h, z28.h
    fmla    z9.h, p0/m, z26.h, z29.h
    fmla    z10.h, p0/m, z26.h, z30.h
    fmla    z11.h, p0/m, z26.h, z31.h

    fmla    z12.h, p0/m, z27.h, z28.h
    fmla    z13.h, p0/m, z27.h, z29.h
    fmla    z14.h, p0/m, z27.h, z30.h
    fmla    z15.h, p0/m, z27.h, z31.h

    add     x0, x0, #4
    add     x1, x1, x5

    subs    x3, x3, #1
    b.gt    .Lfp16_loop

.Lfp16_store:
    mov     x7, x2

    st1h    {z0.h}, p0, [x7, #0, mul vl]
    st1h    {z1.h}, p0, [x7, #1, mul vl]
    st1h    {z2.h}, p0, [x7, #2, mul vl]
    st1h    {z3.h}, p0, [x7, #3, mul vl]
    add     x7, x7, x6

    st1h    {z4.h}, p0, [x7, #0, mul vl]
    st1h    {z5.h}, p0, [x7, #1, mul vl]
    st1h    {z6.h}, p0, [x7, #2, mul vl]
    st1h    {z7.h}, p0, [x7, #3, mul vl]
    add     x7, x7, x6

    st1h    {z8.h}, p0, [x7, #0, mul vl]
    st1h    {z9.h}, p0, [x7, #1, mul vl]
    st1h    {z10.h}, p0, [x7, #2, mul vl]
    st1h    {z11.h}, p0, [x7, #3, mul vl]
    add     x7, x7, x6

    st1h    {z12.h}, p0, [x7, #0, mul vl]
    st1h    {z13.h}, p0, [x7, #1, mul vl]
    st1h    {z14.h}, p0, [x7, #2, mul vl]
    st1h    {z15.h}, p0, [x7, #3, mul vl]

    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

    .size exp2_fmla_fp16_4x4, .-exp2_fmla_fp16_4x4


/*============================================================================
 * gemm_fp32_4x4 - Pure FP32 GEMM without exp2
 *
 * void gemm_fp32_4x4(
 *     const float* A,    // x0: [4][K] row-major
 *     const float* B,    // x1: [K][64] row-major (4 SVE vectors)
 *     float* O,          // x2: [4][64] row-major
 *     int K,             // x3: inner dimension
 *     int ld_a,          // x4: A leading dim in bytes
 *     int ld_b,          // x5: B leading dim in bytes
 *     int ld_o           // x6: O leading dim in bytes
 * );
 *
 * Optimized for A64FX: software pipelined with 4x unroll potential
 *============================================================================*/

    .global gemm_fp32_4x4
    .type gemm_fp32_4x4, %function
gemm_fp32_4x4:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.s

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

    cbz     x3, .Lgemm_fp32_store

    /* Compute row offsets for A */
    add     x7, x0, x4                  // A row 1
    add     x8, x7, x4                  // A row 2
    add     x9, x8, x4                  // A row 3

.Lgemm_fp32_loop:
    /* Load A[0:4, k] - 4 scalars broadcast */
    ld1rw   {z16.s}, p0/z, [x0]
    ld1rw   {z17.s}, p0/z, [x7]
    ld1rw   {z18.s}, p0/z, [x8]
    ld1rw   {z19.s}, p0/z, [x9]

    /* Load B[k, 0:64] - 4 vectors */
    ld1w    {z20.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z21.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z22.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z23.s}, p0/z, [x1, #3, mul vl]

    /* FMLA: 16 operations */
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

    /* Advance pointers */
    add     x0, x0, #4                  // A col++
    add     x7, x7, #4
    add     x8, x8, #4
    add     x9, x9, #4
    add     x1, x1, x5                  // B row++

    subs    x3, x3, #1
    b.gt    .Lgemm_fp32_loop

.Lgemm_fp32_store:
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

    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

    .size gemm_fp32_4x4, .-gemm_fp32_4x4


/*============================================================================
 * exp2_rows - Compute exp2 for multiple rows using exp2_softmax_fast
 *
 * void exp2_rows(
 *     const int32_t* S,  // x0: [M][Nc] input
 *     float* P,          // x1: [M][Nc] output
 *     int M,             // x2: number of rows
 *     int Nc,            // x3: columns per row
 *     float scale,       // s0
 *     float max_val,     // s1
 *     int ld_s,          // x4: S stride in elements
 *     int ld_p           // x5: P stride in elements
 * );
 *
 * Calls exp2_softmax_fast for each row
 *============================================================================*/

    .global exp2_rows
    .type exp2_rows, %function
exp2_rows:
    /* Save callee-saved and link register */
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8, d9, [sp, #64]

    /* Save parameters */
    mov     x19, x0                     // S base
    mov     x20, x1                     // P base
    mov     x21, x2                     // M (row count)
    mov     x22, x3                     // Nc
    fmov    s8, s0                      // scale
    fmov    s9, s1                      // max_val
    lsl     x23, x4, #2                 // ld_s in bytes
    lsl     x24, x5, #2                 // ld_p in bytes

    cbz     x21, .Lexp2_rows_done

.Lexp2_rows_loop:
    /* Call exp2_softmax_fast(S, P, Nc, scale, max_val) */
    mov     x0, x19                     // S row
    mov     x1, x20                     // P row
    mov     x2, x22                     // n = Nc
    fmov    s0, s8                      // scale
    fmov    s1, s9                      // max_val
    bl      exp2_softmax_fast

    /* Advance row pointers */
    add     x19, x19, x23
    add     x20, x20, x24

    subs    x21, x21, #1
    b.gt    .Lexp2_rows_loop

.Lexp2_rows_done:
    ldp     d8, d9, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret

    .size exp2_rows, .-exp2_rows
