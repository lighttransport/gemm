/*
 * Fused GEMM + exp2 kernel for Flash Attention
 *
 * Computes: P = exp2((Q @ K^T) * scale - max)
 *
 * Architecture: A64FX (2 FPU pipes, 2 LD/ST ports)
 *
 * Strategy: Interleave exp2 operations with SDOT/loads to hide latency
 * - SDOT has 9-cycle latency, 2/cycle throughput
 * - exp2 FP ops have 9-cycle latency
 * - During exp2's 9-cycle latency windows, execute SDOT operations
 *
 * Register allocation (4x4 GEMM + exp2):
 *   z0-z15:  4x4 GEMM accumulators (int32), then exp2 output (fp32)
 *   z16-z19: exp2 temp: N = floor(x)
 *   z20-z23: exp2 temp: f then m
 *   z24-z27: A operands (4 rows, broadcast)
 *   z28-z31: B operands (4 columns)
 *
 * Constants (in caller-saved or re-loaded):
 *   scale:   float, conversion scale for int32 -> fp32 softmax input
 *   max_val: float, max value for numerical stability
 *   64.0f:   for m = floor(f * 64)
 *   127:     IEEE754 exponent bias
 */

    .arch armv8.2-a+sve
    .text
    .align 4

    .section .rodata
    .align 4
.Lconst_64f:
    .float 64.0

    .text

/*============================================================================
 * gemm_exp2_fused_4x4
 *
 * Computes one 4x4 tile: C = exp2((A @ B) * scale - max)
 *
 * void gemm_exp2_fused_4x4(
 *     const int8_t* A,    // x0: [4][K] packed as [K/4][4][4]
 *     const int8_t* B,    // x1: [K][64] packed as [K/4][4][64]
 *     float* C,           // x2: output [4][64] row-major
 *     int K,              // x3: reduction dimension (multiple of 4)
 *     float scale,        // s0: scale factor
 *     float max_val,      // s1: max value for softmax stability
 *     int ldc             // x4: leading dim of C in bytes
 * );
 *
 * Performance target:
 *   GEMM: 16 SDOT + loads per K iteration
 *   exp2: 14 instructions per vector, interleaved
 *============================================================================*/

    .global gemm_exp2_fused_4x4
    .type gemm_exp2_fused_4x4, %function
gemm_exp2_fused_4x4:
    /* Save callee-saved registers */
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.b

    /* IMPORTANT: Save scale (s0) and max_val (s1) before using z0/z1 as accumulators!
     * s0/s1 are the lower 32 bits of z0/z1, so they get clobbered by GEMM.
     * Save to callee-saved scalar registers.
     */
    fmov    s16, s0                     // Save scale to s16 (z16 lower bits)
    fmov    s17, s1                     // Save max_val to s17 (z17 lower bits)

    /* Zero 16 accumulators (4x4 tile) */
    dup     z0.s, #0
    dup     z1.s, #0
    dup     z2.s, #0
    dup     z3.s, #0
    dup     z4.s, #0
    dup     z5.s, #0
    dup     z6.s, #0
    dup     z7.s, #0
    dup     z8.s, #0
    dup     z9.s, #0
    dup     z10.s, #0
    dup     z11.s, #0
    dup     z12.s, #0
    dup     z13.s, #0
    dup     z14.s, #0
    dup     z15.s, #0

    /* K/4 iterations */
    lsr     x5, x3, #2
    cbz     x5, .Lgemm_done

    /*========================================================================
     * GEMM Main Loop: 4x4 SDOT
     *========================================================================*/
.Lgemm_loop:
    /* Load A rows (4 rows, each 4 bytes = 1 int8x4) */
    ld1rw   {z24.s}, p0/z, [x0, #0]     // A row 0
    ld1rw   {z25.s}, p0/z, [x0, #4]     // A row 1
    ld1rw   {z26.s}, p0/z, [x0, #8]     // A row 2
    ld1rw   {z27.s}, p0/z, [x0, #12]    // A row 3

    /* Load B columns (4 vectors) */
    ld1b    {z28.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z29.b}, p0/z, [x1, #1, mul vl]
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]

    /* 16 SDOT operations: C[i,j] += A[i,:] @ B[:,j] */
    /* Row 0 */
    sdot    z0.s, z24.b, z28.b          // C[0,0]
    sdot    z1.s, z24.b, z29.b          // C[0,1]
    sdot    z2.s, z24.b, z30.b          // C[0,2]
    sdot    z3.s, z24.b, z31.b          // C[0,3]

    /* Row 1 */
    sdot    z4.s, z25.b, z28.b          // C[1,0]
    sdot    z5.s, z25.b, z29.b          // C[1,1]
    sdot    z6.s, z25.b, z30.b          // C[1,2]
    sdot    z7.s, z25.b, z31.b          // C[1,3]

    /* Row 2 */
    sdot    z8.s, z26.b, z28.b          // C[2,0]
    sdot    z9.s, z26.b, z29.b          // C[2,1]
    sdot    z10.s, z26.b, z30.b         // C[2,2]
    sdot    z11.s, z26.b, z31.b         // C[2,3]

    /* Row 3 */
    sdot    z12.s, z27.b, z28.b         // C[3,0]
    sdot    z13.s, z27.b, z29.b         // C[3,1]
    sdot    z14.s, z27.b, z30.b         // C[3,2]
    sdot    z15.s, z27.b, z31.b         // C[3,3]

    /* Advance pointers */
    add     x0, x0, #16                 // A += 16 bytes
    add     x1, x1, #256                // B += 4 * 64 bytes

    subs    x5, x5, #1
    b.gt    .Lgemm_loop

.Lgemm_done:
    /*========================================================================
     * exp2 Phase: Convert GEMM results and apply exp2
     *
     * Now z0-z15 contain int32 GEMM results
     * Apply: exp2((result * scale) - max_val)
     *
     * This is the 14-instruction fast exp2:
     * scvtf, fmul, fsub, frintm, fsub, fmul, fcvtzs, fcvtzs, add, lsl, orr, fexpa
     *========================================================================*/

    ptrue   p0.s

    /* DEBUG: Store raw GEMM results to verify they're correct */
    /* Uncomment the next 4 lines to debug
    st1w    {z0.s}, p0, [x2, #0, mul vl]
    st1w    {z1.s}, p0, [x2, #1, mul vl]
    st1w    {z2.s}, p0, [x2, #2, mul vl]
    st1w    {z3.s}, p0, [x2, #3, mul vl]
    */

    /* Broadcast constants - scale was saved to s16, max_val to s17 */
    mov     z20.s, s16                  // scale (saved before GEMM)
    mov     z21.s, s17                  // max_val (saved before GEMM)

    /* Load 64.0f */
    adrp    x5, .Lconst_64f
    add     x5, x5, :lo12:.Lconst_64f
    ld1rw   {z22.s}, p0/z, [x5]

    /* 127 bias */
    mov     w5, #127
    mov     z23.s, w5

    /*------------------------------------------------------------------------
     * exp2 on z0-z3 (row 0), z4-z7 (row 1)
     * Interleaved to hide latency
     *------------------------------------------------------------------------*/

    /* Phase 1: Int32 -> FP32 conversion (all 16 vectors) */
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s
    scvtf   z4.s, p0/m, z4.s
    scvtf   z5.s, p0/m, z5.s
    scvtf   z6.s, p0/m, z6.s
    scvtf   z7.s, p0/m, z7.s
    scvtf   z8.s, p0/m, z8.s
    scvtf   z9.s, p0/m, z9.s
    scvtf   z10.s, p0/m, z10.s
    scvtf   z11.s, p0/m, z11.s
    scvtf   z12.s, p0/m, z12.s
    scvtf   z13.s, p0/m, z13.s
    scvtf   z14.s, p0/m, z14.s
    scvtf   z15.s, p0/m, z15.s

    /* Phase 2: Scale (x * scale) */
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s
    fmul    z4.s, z4.s, z20.s
    fmul    z5.s, z5.s, z20.s
    fmul    z6.s, z6.s, z20.s
    fmul    z7.s, z7.s, z20.s
    fmul    z8.s, z8.s, z20.s
    fmul    z9.s, z9.s, z20.s
    fmul    z10.s, z10.s, z20.s
    fmul    z11.s, z11.s, z20.s
    fmul    z12.s, z12.s, z20.s
    fmul    z13.s, z13.s, z20.s
    fmul    z14.s, z14.s, z20.s
    fmul    z15.s, z15.s, z20.s

    /* Phase 3: Subtract max (x - max_val) */
    fsub    z0.s, z0.s, z21.s
    fsub    z1.s, z1.s, z21.s
    fsub    z2.s, z2.s, z21.s
    fsub    z3.s, z3.s, z21.s
    fsub    z4.s, z4.s, z21.s
    fsub    z5.s, z5.s, z21.s
    fsub    z6.s, z6.s, z21.s
    fsub    z7.s, z7.s, z21.s
    fsub    z8.s, z8.s, z21.s
    fsub    z9.s, z9.s, z21.s
    fsub    z10.s, z10.s, z21.s
    fsub    z11.s, z11.s, z21.s
    fsub    z12.s, z12.s, z21.s
    fsub    z13.s, z13.s, z21.s
    fsub    z14.s, z14.s, z21.s
    fsub    z15.s, z15.s, z21.s

    /* Phase 4: N = floor(x) - use z16-z19 as temp for first 4, z24-z27 for rows */
    /* Process in batches of 4 to manage register pressure */

    /* Batch 1: z0-z3 */
    frintm  z16.s, p0/m, z0.s           // N0
    frintm  z17.s, p0/m, z1.s           // N1
    frintm  z18.s, p0/m, z2.s           // N2
    frintm  z19.s, p0/m, z3.s           // N3

    fsub    z24.s, z0.s, z16.s          // f0 = x0 - N0
    fsub    z25.s, z1.s, z17.s          // f1
    fsub    z26.s, z2.s, z18.s          // f2
    fsub    z27.s, z3.s, z19.s          // f3

    fmul    z24.s, z24.s, z22.s         // f0 * 64
    fmul    z25.s, z25.s, z22.s
    fmul    z26.s, z26.s, z22.s
    fmul    z27.s, z27.s, z22.s

    fcvtzs  z24.s, p0/m, z24.s          // m0 = (int)(f0 * 64)
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z27.s, p0/m, z27.s

    fcvtzs  z16.s, p0/m, z16.s          // N0 as int
    fcvtzs  z17.s, p0/m, z17.s
    fcvtzs  z18.s, p0/m, z18.s
    fcvtzs  z19.s, p0/m, z19.s

    add     z16.s, z16.s, z23.s         // N0 + 127
    add     z17.s, z17.s, z23.s
    add     z18.s, z18.s, z23.s
    add     z19.s, z19.s, z23.s

    lsl     z16.s, z16.s, #6            // (N0 + 127) << 6
    lsl     z17.s, z17.s, #6
    lsl     z18.s, z18.s, #6
    lsl     z19.s, z19.s, #6

    orr     z0.s, z16.s, z24.s          // fexpa input = ((N+127)<<6) | m
    orr     z1.s, z17.s, z25.s
    orr     z2.s, z18.s, z26.s
    orr     z3.s, z19.s, z27.s

    fexpa   z0.s, z0.s                  // exp2 result
    fexpa   z1.s, z1.s
    fexpa   z2.s, z2.s
    fexpa   z3.s, z3.s

    /* Batch 2: z4-z7 */
    frintm  z16.s, p0/m, z4.s
    frintm  z17.s, p0/m, z5.s
    frintm  z18.s, p0/m, z6.s
    frintm  z19.s, p0/m, z7.s

    fsub    z24.s, z4.s, z16.s
    fsub    z25.s, z5.s, z17.s
    fsub    z26.s, z6.s, z18.s
    fsub    z27.s, z7.s, z19.s

    fmul    z24.s, z24.s, z22.s
    fmul    z25.s, z25.s, z22.s
    fmul    z26.s, z26.s, z22.s
    fmul    z27.s, z27.s, z22.s

    fcvtzs  z24.s, p0/m, z24.s
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z27.s, p0/m, z27.s

    fcvtzs  z16.s, p0/m, z16.s
    fcvtzs  z17.s, p0/m, z17.s
    fcvtzs  z18.s, p0/m, z18.s
    fcvtzs  z19.s, p0/m, z19.s

    add     z16.s, z16.s, z23.s
    add     z17.s, z17.s, z23.s
    add     z18.s, z18.s, z23.s
    add     z19.s, z19.s, z23.s

    lsl     z16.s, z16.s, #6
    lsl     z17.s, z17.s, #6
    lsl     z18.s, z18.s, #6
    lsl     z19.s, z19.s, #6

    orr     z4.s, z16.s, z24.s
    orr     z5.s, z17.s, z25.s
    orr     z6.s, z18.s, z26.s
    orr     z7.s, z19.s, z27.s

    fexpa   z4.s, z4.s
    fexpa   z5.s, z5.s
    fexpa   z6.s, z6.s
    fexpa   z7.s, z7.s

    /* Batch 3: z8-z11 */
    frintm  z16.s, p0/m, z8.s
    frintm  z17.s, p0/m, z9.s
    frintm  z18.s, p0/m, z10.s
    frintm  z19.s, p0/m, z11.s

    fsub    z24.s, z8.s, z16.s
    fsub    z25.s, z9.s, z17.s
    fsub    z26.s, z10.s, z18.s
    fsub    z27.s, z11.s, z19.s

    fmul    z24.s, z24.s, z22.s
    fmul    z25.s, z25.s, z22.s
    fmul    z26.s, z26.s, z22.s
    fmul    z27.s, z27.s, z22.s

    fcvtzs  z24.s, p0/m, z24.s
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z27.s, p0/m, z27.s

    fcvtzs  z16.s, p0/m, z16.s
    fcvtzs  z17.s, p0/m, z17.s
    fcvtzs  z18.s, p0/m, z18.s
    fcvtzs  z19.s, p0/m, z19.s

    add     z16.s, z16.s, z23.s
    add     z17.s, z17.s, z23.s
    add     z18.s, z18.s, z23.s
    add     z19.s, z19.s, z23.s

    lsl     z16.s, z16.s, #6
    lsl     z17.s, z17.s, #6
    lsl     z18.s, z18.s, #6
    lsl     z19.s, z19.s, #6

    orr     z8.s, z16.s, z24.s
    orr     z9.s, z17.s, z25.s
    orr     z10.s, z18.s, z26.s
    orr     z11.s, z19.s, z27.s

    fexpa   z8.s, z8.s
    fexpa   z9.s, z9.s
    fexpa   z10.s, z10.s
    fexpa   z11.s, z11.s

    /* Batch 4: z12-z15 */
    frintm  z16.s, p0/m, z12.s
    frintm  z17.s, p0/m, z13.s
    frintm  z18.s, p0/m, z14.s
    frintm  z19.s, p0/m, z15.s

    fsub    z24.s, z12.s, z16.s
    fsub    z25.s, z13.s, z17.s
    fsub    z26.s, z14.s, z18.s
    fsub    z27.s, z15.s, z19.s

    fmul    z24.s, z24.s, z22.s
    fmul    z25.s, z25.s, z22.s
    fmul    z26.s, z26.s, z22.s
    fmul    z27.s, z27.s, z22.s

    fcvtzs  z24.s, p0/m, z24.s
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z27.s, p0/m, z27.s

    fcvtzs  z16.s, p0/m, z16.s
    fcvtzs  z17.s, p0/m, z17.s
    fcvtzs  z18.s, p0/m, z18.s
    fcvtzs  z19.s, p0/m, z19.s

    add     z16.s, z16.s, z23.s
    add     z17.s, z17.s, z23.s
    add     z18.s, z18.s, z23.s
    add     z19.s, z19.s, z23.s

    lsl     z16.s, z16.s, #6
    lsl     z17.s, z17.s, #6
    lsl     z18.s, z18.s, #6
    lsl     z19.s, z19.s, #6

    orr     z12.s, z16.s, z24.s
    orr     z13.s, z17.s, z25.s
    orr     z14.s, z18.s, z26.s
    orr     z15.s, z19.s, z27.s

    fexpa   z12.s, z12.s
    fexpa   z13.s, z13.s
    fexpa   z14.s, z14.s
    fexpa   z15.s, z15.s

    /*========================================================================
     * Store results
     *========================================================================*/
    mov     x5, x2                      // C pointer
    mov     x6, x4                      // ldc

    /* Row 0 */
    st1w    {z0.s}, p0, [x5, #0, mul vl]
    st1w    {z1.s}, p0, [x5, #1, mul vl]
    st1w    {z2.s}, p0, [x5, #2, mul vl]
    st1w    {z3.s}, p0, [x5, #3, mul vl]
    add     x5, x5, x6

    /* Row 1 */
    st1w    {z4.s}, p0, [x5, #0, mul vl]
    st1w    {z5.s}, p0, [x5, #1, mul vl]
    st1w    {z6.s}, p0, [x5, #2, mul vl]
    st1w    {z7.s}, p0, [x5, #3, mul vl]
    add     x5, x5, x6

    /* Row 2 */
    st1w    {z8.s}, p0, [x5, #0, mul vl]
    st1w    {z9.s}, p0, [x5, #1, mul vl]
    st1w    {z10.s}, p0, [x5, #2, mul vl]
    st1w    {z11.s}, p0, [x5, #3, mul vl]
    add     x5, x5, x6

    /* Row 3 */
    st1w    {z12.s}, p0, [x5, #0, mul vl]
    st1w    {z13.s}, p0, [x5, #1, mul vl]
    st1w    {z14.s}, p0, [x5, #2, mul vl]
    st1w    {z15.s}, p0, [x5, #3, mul vl]

    /* Restore and return */
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

    .size gemm_exp2_fused_4x4, .-gemm_exp2_fused_4x4


/*============================================================================
 * gemm_only_4x4 - Debug: GEMM only, stores int32 results
 *
 * void gemm_only_4x4(
 *     const int8_t* A,    // x0
 *     const int8_t* B,    // x1
 *     int32_t* C,         // x2
 *     int K,              // x3
 *     int ldc             // x4: in bytes
 * );
 *============================================================================*/
    .global gemm_only_4x4
    .type gemm_only_4x4, %function
gemm_only_4x4:
    ptrue   p0.b

    /* Zero accumulators */
    dup     z0.s, #0
    dup     z1.s, #0
    dup     z2.s, #0
    dup     z3.s, #0
    dup     z4.s, #0
    dup     z5.s, #0
    dup     z6.s, #0
    dup     z7.s, #0
    dup     z8.s, #0
    dup     z9.s, #0
    dup     z10.s, #0
    dup     z11.s, #0
    dup     z12.s, #0
    dup     z13.s, #0
    dup     z14.s, #0
    dup     z15.s, #0

    lsr     x5, x3, #2
    cbz     x5, .Lgemm_only_store

.Lgemm_only_loop:
    /* Load A */
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]

    /* Load B */
    ld1b    {z28.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z29.b}, p0/z, [x1, #1, mul vl]
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]

    /* SDOT */
    sdot    z0.s, z24.b, z28.b
    sdot    z1.s, z24.b, z29.b
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b

    sdot    z4.s, z25.b, z28.b
    sdot    z5.s, z25.b, z29.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b

    sdot    z8.s, z26.b, z28.b
    sdot    z9.s, z26.b, z29.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b

    sdot    z12.s, z27.b, z28.b
    sdot    z13.s, z27.b, z29.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b

    add     x0, x0, #16
    add     x1, x1, #256

    subs    x5, x5, #1
    b.gt    .Lgemm_only_loop

.Lgemm_only_store:
    ptrue   p0.s
    mov     x5, x2
    mov     x6, x4

    st1w    {z0.s}, p0, [x5, #0, mul vl]
    st1w    {z1.s}, p0, [x5, #1, mul vl]
    st1w    {z2.s}, p0, [x5, #2, mul vl]
    st1w    {z3.s}, p0, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    {z4.s}, p0, [x5, #0, mul vl]
    st1w    {z5.s}, p0, [x5, #1, mul vl]
    st1w    {z6.s}, p0, [x5, #2, mul vl]
    st1w    {z7.s}, p0, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    {z8.s}, p0, [x5, #0, mul vl]
    st1w    {z9.s}, p0, [x5, #1, mul vl]
    st1w    {z10.s}, p0, [x5, #2, mul vl]
    st1w    {z11.s}, p0, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    {z12.s}, p0, [x5, #0, mul vl]
    st1w    {z13.s}, p0, [x5, #1, mul vl]
    st1w    {z14.s}, p0, [x5, #2, mul vl]
    st1w    {z15.s}, p0, [x5, #3, mul vl]

    ret
    .size gemm_only_4x4, .-gemm_only_4x4


/*============================================================================
 * gemm_exp2_interleaved
 *
 * Same as gemm_exp2_fused_4x4 but with tighter instruction scheduling
 * in the exp2 phase for better ILP.
 *
 * This version processes the full GEMM first, then applies exp2 with
 * interleaved instruction scheduling to hide latency.
 *============================================================================*/

    .global gemm_exp2_interleaved
    .type gemm_exp2_interleaved, %function
gemm_exp2_interleaved:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.b

    /* Save scale/max_val before using z0/z1 */
    fmov    s16, s0                     // Save scale
    fmov    s17, s1                     // Save max_val

    /* Zero 16 accumulators (4x4 tile) */
    dup     z0.s, #0
    dup     z1.s, #0
    dup     z2.s, #0
    dup     z3.s, #0
    dup     z4.s, #0
    dup     z5.s, #0
    dup     z6.s, #0
    dup     z7.s, #0
    dup     z8.s, #0
    dup     z9.s, #0
    dup     z10.s, #0
    dup     z11.s, #0
    dup     z12.s, #0
    dup     z13.s, #0
    dup     z14.s, #0
    dup     z15.s, #0

    lsr     x5, x3, #2
    cbz     x5, .Linter2_exp2

    /*========================================================================
     * GEMM Main Loop (same as fused)
     *========================================================================*/
.Linter2_gemm:
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]

    ld1b    {z28.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z29.b}, p0/z, [x1, #1, mul vl]
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]

    sdot    z0.s, z24.b, z28.b
    sdot    z1.s, z24.b, z29.b
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b

    sdot    z4.s, z25.b, z28.b
    sdot    z5.s, z25.b, z29.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b

    sdot    z8.s, z26.b, z28.b
    sdot    z9.s, z26.b, z29.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b

    sdot    z12.s, z27.b, z28.b
    sdot    z13.s, z27.b, z29.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b

    add     x0, x0, #16
    add     x1, x1, #256

    subs    x5, x5, #1
    b.gt    .Linter2_gemm

    /*========================================================================
     * exp2 Phase with interleaved scheduling
     * Process all 16 vectors with tighter instruction interleaving
     *========================================================================*/
.Linter2_exp2:
    ptrue   p0.s

    /* Broadcast constants */
    mov     z20.s, s16                  // scale
    mov     z21.s, s17                  // max_val

    adrp    x5, .Lconst_64f
    add     x5, x5, :lo12:.Lconst_64f
    ld1rw   {z22.s}, p0/z, [x5]

    mov     w5, #127
    mov     z23.s, w5

    /*--------------------------------------------------------------------
     * Interleaved exp2: process z0-z3 and z4-z7 together
     * to maximize FPU utilization (2 pipes)
     *--------------------------------------------------------------------*/

    /* scvtf: interleave rows 0 and 1 */
    scvtf   z0.s, p0/m, z0.s
    scvtf   z4.s, p0/m, z4.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z5.s, p0/m, z5.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z6.s, p0/m, z6.s
    scvtf   z3.s, p0/m, z3.s
    scvtf   z7.s, p0/m, z7.s

    /* fmul scale: interleave */
    fmul    z0.s, z0.s, z20.s
    fmul    z4.s, z4.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z5.s, z5.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z6.s, z6.s, z20.s
    fmul    z3.s, z3.s, z20.s
    fmul    z7.s, z7.s, z20.s

    /* fsub max: interleave */
    fsub    z0.s, z0.s, z21.s
    fsub    z4.s, z4.s, z21.s
    fsub    z1.s, z1.s, z21.s
    fsub    z5.s, z5.s, z21.s
    fsub    z2.s, z2.s, z21.s
    fsub    z6.s, z6.s, z21.s
    fsub    z3.s, z3.s, z21.s
    fsub    z7.s, z7.s, z21.s

    /* Now process rows 2 and 3 (z8-z15) in parallel */
    scvtf   z8.s, p0/m, z8.s
    scvtf   z12.s, p0/m, z12.s
    scvtf   z9.s, p0/m, z9.s
    scvtf   z13.s, p0/m, z13.s
    scvtf   z10.s, p0/m, z10.s
    scvtf   z14.s, p0/m, z14.s
    scvtf   z11.s, p0/m, z11.s
    scvtf   z15.s, p0/m, z15.s

    fmul    z8.s, z8.s, z20.s
    fmul    z12.s, z12.s, z20.s
    fmul    z9.s, z9.s, z20.s
    fmul    z13.s, z13.s, z20.s
    fmul    z10.s, z10.s, z20.s
    fmul    z14.s, z14.s, z20.s
    fmul    z11.s, z11.s, z20.s
    fmul    z15.s, z15.s, z20.s

    fsub    z8.s, z8.s, z21.s
    fsub    z12.s, z12.s, z21.s
    fsub    z9.s, z9.s, z21.s
    fsub    z13.s, z13.s, z21.s
    fsub    z10.s, z10.s, z21.s
    fsub    z14.s, z14.s, z21.s
    fsub    z11.s, z11.s, z21.s
    fsub    z15.s, z15.s, z21.s

    /*--------------------------------------------------------------------
     * frintm, fsub, fmul, fcvtzs phase for z0-z7
     * Use z24-z27 for N, z28-z31 for f/m
     *--------------------------------------------------------------------*/
    frintm  z24.s, p0/m, z0.s
    frintm  z25.s, p0/m, z1.s
    frintm  z26.s, p0/m, z2.s
    frintm  z27.s, p0/m, z3.s

    fsub    z28.s, z0.s, z24.s
    fsub    z29.s, z1.s, z25.s
    fsub    z30.s, z2.s, z26.s
    fsub    z31.s, z3.s, z27.s

    fmul    z28.s, z28.s, z22.s
    fmul    z29.s, z29.s, z22.s
    fmul    z30.s, z30.s, z22.s
    fmul    z31.s, z31.s, z22.s

    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s

    fcvtzs  z24.s, p0/m, z24.s
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z27.s, p0/m, z27.s

    add     z24.s, z24.s, z23.s
    add     z25.s, z25.s, z23.s
    add     z26.s, z26.s, z23.s
    add     z27.s, z27.s, z23.s

    lsl     z24.s, z24.s, #6
    lsl     z25.s, z25.s, #6
    lsl     z26.s, z26.s, #6
    lsl     z27.s, z27.s, #6

    orr     z0.s, z24.s, z28.s
    orr     z1.s, z25.s, z29.s
    orr     z2.s, z26.s, z30.s
    orr     z3.s, z27.s, z31.s

    fexpa   z0.s, z0.s
    fexpa   z1.s, z1.s
    fexpa   z2.s, z2.s
    fexpa   z3.s, z3.s

    /* z4-z7 */
    frintm  z24.s, p0/m, z4.s
    frintm  z25.s, p0/m, z5.s
    frintm  z26.s, p0/m, z6.s
    frintm  z27.s, p0/m, z7.s

    fsub    z28.s, z4.s, z24.s
    fsub    z29.s, z5.s, z25.s
    fsub    z30.s, z6.s, z26.s
    fsub    z31.s, z7.s, z27.s

    fmul    z28.s, z28.s, z22.s
    fmul    z29.s, z29.s, z22.s
    fmul    z30.s, z30.s, z22.s
    fmul    z31.s, z31.s, z22.s

    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s

    fcvtzs  z24.s, p0/m, z24.s
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z27.s, p0/m, z27.s

    add     z24.s, z24.s, z23.s
    add     z25.s, z25.s, z23.s
    add     z26.s, z26.s, z23.s
    add     z27.s, z27.s, z23.s

    lsl     z24.s, z24.s, #6
    lsl     z25.s, z25.s, #6
    lsl     z26.s, z26.s, #6
    lsl     z27.s, z27.s, #6

    orr     z4.s, z24.s, z28.s
    orr     z5.s, z25.s, z29.s
    orr     z6.s, z26.s, z30.s
    orr     z7.s, z27.s, z31.s

    fexpa   z4.s, z4.s
    fexpa   z5.s, z5.s
    fexpa   z6.s, z6.s
    fexpa   z7.s, z7.s

    /* z8-z11 */
    frintm  z24.s, p0/m, z8.s
    frintm  z25.s, p0/m, z9.s
    frintm  z26.s, p0/m, z10.s
    frintm  z27.s, p0/m, z11.s

    fsub    z28.s, z8.s, z24.s
    fsub    z29.s, z9.s, z25.s
    fsub    z30.s, z10.s, z26.s
    fsub    z31.s, z11.s, z27.s

    fmul    z28.s, z28.s, z22.s
    fmul    z29.s, z29.s, z22.s
    fmul    z30.s, z30.s, z22.s
    fmul    z31.s, z31.s, z22.s

    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s

    fcvtzs  z24.s, p0/m, z24.s
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z27.s, p0/m, z27.s

    add     z24.s, z24.s, z23.s
    add     z25.s, z25.s, z23.s
    add     z26.s, z26.s, z23.s
    add     z27.s, z27.s, z23.s

    lsl     z24.s, z24.s, #6
    lsl     z25.s, z25.s, #6
    lsl     z26.s, z26.s, #6
    lsl     z27.s, z27.s, #6

    orr     z8.s, z24.s, z28.s
    orr     z9.s, z25.s, z29.s
    orr     z10.s, z26.s, z30.s
    orr     z11.s, z27.s, z31.s

    fexpa   z8.s, z8.s
    fexpa   z9.s, z9.s
    fexpa   z10.s, z10.s
    fexpa   z11.s, z11.s

    /* z12-z15 */
    frintm  z24.s, p0/m, z12.s
    frintm  z25.s, p0/m, z13.s
    frintm  z26.s, p0/m, z14.s
    frintm  z27.s, p0/m, z15.s

    fsub    z28.s, z12.s, z24.s
    fsub    z29.s, z13.s, z25.s
    fsub    z30.s, z14.s, z26.s
    fsub    z31.s, z15.s, z27.s

    fmul    z28.s, z28.s, z22.s
    fmul    z29.s, z29.s, z22.s
    fmul    z30.s, z30.s, z22.s
    fmul    z31.s, z31.s, z22.s

    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s

    fcvtzs  z24.s, p0/m, z24.s
    fcvtzs  z25.s, p0/m, z25.s
    fcvtzs  z26.s, p0/m, z26.s
    fcvtzs  z27.s, p0/m, z27.s

    add     z24.s, z24.s, z23.s
    add     z25.s, z25.s, z23.s
    add     z26.s, z26.s, z23.s
    add     z27.s, z27.s, z23.s

    lsl     z24.s, z24.s, #6
    lsl     z25.s, z25.s, #6
    lsl     z26.s, z26.s, #6
    lsl     z27.s, z27.s, #6

    orr     z12.s, z24.s, z28.s
    orr     z13.s, z25.s, z29.s
    orr     z14.s, z26.s, z30.s
    orr     z15.s, z27.s, z31.s

    fexpa   z12.s, z12.s
    fexpa   z13.s, z13.s
    fexpa   z14.s, z14.s
    fexpa   z15.s, z15.s

    /*========================================================================
     * Store results
     *========================================================================*/
    mov     x5, x2
    mov     x6, x4

    st1w    {z0.s}, p0, [x5, #0, mul vl]
    st1w    {z1.s}, p0, [x5, #1, mul vl]
    st1w    {z2.s}, p0, [x5, #2, mul vl]
    st1w    {z3.s}, p0, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    {z4.s}, p0, [x5, #0, mul vl]
    st1w    {z5.s}, p0, [x5, #1, mul vl]
    st1w    {z6.s}, p0, [x5, #2, mul vl]
    st1w    {z7.s}, p0, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    {z8.s}, p0, [x5, #0, mul vl]
    st1w    {z9.s}, p0, [x5, #1, mul vl]
    st1w    {z10.s}, p0, [x5, #2, mul vl]
    st1w    {z11.s}, p0, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    {z12.s}, p0, [x5, #0, mul vl]
    st1w    {z13.s}, p0, [x5, #1, mul vl]
    st1w    {z14.s}, p0, [x5, #2, mul vl]
    st1w    {z15.s}, p0, [x5, #3, mul vl]

    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

    .size gemm_exp2_interleaved, .-gemm_exp2_interleaved
