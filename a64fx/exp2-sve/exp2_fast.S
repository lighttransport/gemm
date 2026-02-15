/*
 * Ultra-fast exp2 for softmax
 *
 * Since softmax normalizes by sum(exp2), we only need relative accuracy.
 * This version minimizes instructions at the cost of some accuracy.
 *
 * Target: < 8 instructions per 16 elements
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*============================================================================
 * exp2_fast_vec - Minimal exp2 for softmax
 *
 * void exp2_fast_vec(
 *     const int32_t* S,   // x0: input [n]
 *     float* P,           // x1: output [n]
 *     int n,              // x2: count (multiple of 16)
 *     float scale,        // s0: scale factor
 *     float neg_max       // s1: -max (negated for fmla)
 * );
 *
 * Computes: P[i] = exp2(S[i] * scale + neg_max)
 *         = exp2(S[i] * scale - max)
 *============================================================================*/

    .global exp2_fast_vec
    .type exp2_fast_vec, %function
exp2_fast_vec:
    ptrue   p0.s

    /* Broadcast constants */
    mov     z20.s, s0                   // scale
    mov     z21.s, s1                   // -max

    /* For FEXPA: input = ((N + 127) << 6) | m
     * where N = floor(x), m = int((x - N) * 64)
     *
     * Simplified: input ≈ int(x * 64) + (127 << 6)
     * This works for x in range [-20, 20] approximately
     */
    mov     w3, #0x42800000             // 64.0f
    mov     z22.s, w3

    mov     w3, #8128                   // 127 << 6
    mov     z23.s, w3

    cbz     x2, .Lfast_done
    lsr     x2, x2, #4                  // n / 16

.Lfast_loop:
    /* Load S as int32 */
    ld1w    {z0.s}, p0/z, [x0]

    /* Convert and scale: x = S * scale - max */
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z21.s           // + neg_max = - max

    /* Simplified FEXPA input: int(x * 64) + bias */
    fmul    z0.s, z0.s, z22.s           // x * 64
    fcvtzs  z0.s, p0/m, z0.s            // to int
    add     z0.s, z0.s, z23.s           // + (127 << 6)

    /* FEXPA lookup */
    fexpa   z0.s, z0.s

    /* Store */
    st1w    {z0.s}, p0, [x1]

    add     x0, x0, #64
    add     x1, x1, #64

    subs    x2, x2, #1
    b.gt    .Lfast_loop

.Lfast_done:
    ret

    .size exp2_fast_vec, .-exp2_fast_vec


/*============================================================================
 * exp2_fast_rows - Fast exp2 for M rows
 *
 * void exp2_fast_rows(
 *     const int32_t* S,   // x0: [M][Nc]
 *     float* P,           // x1: [M][Nc]
 *     int M,              // x2
 *     int Nc,             // x3 (multiple of 16)
 *     float scale,        // s0
 *     float neg_max,      // s1: -max
 *     int ld_s,           // x4: S stride in elements
 *     int ld_p            // x5: P stride in elements
 * );
 *============================================================================*/

    .global exp2_fast_rows
    .type exp2_fast_rows, %function
exp2_fast_rows:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8, d9, [sp, #32]

    mov     x19, x2                     // M
    mov     x20, x3                     // Nc
    fmov    s8, s0                      // scale
    fmov    s9, s1                      // neg_max
    lsl     x4, x4, #2                  // ld_s in bytes
    lsl     x5, x5, #2                  // ld_p in bytes

    cbz     x19, .Lfr_done

.Lfr_row_loop:
    mov     x6, x0                      // S row start
    mov     x7, x1                      // P row start
    mov     x8, x20                     // Nc counter

    ptrue   p0.s
    mov     z20.s, s8                   // scale
    mov     z21.s, s9                   // neg_max
    mov     w9, #0x42800000
    mov     z22.s, w9                   // 64.0f
    mov     w9, #8128
    mov     z23.s, w9                   // bias

    lsr     x8, x8, #4                  // Nc / 16

.Lfr_vec_loop:
    ld1w    {z0.s}, p0/z, [x6]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z21.s
    fmul    z0.s, z0.s, z22.s
    fcvtzs  z0.s, p0/m, z0.s
    add     z0.s, z0.s, z23.s
    fexpa   z0.s, z0.s
    st1w    {z0.s}, p0, [x7]

    add     x6, x6, #64
    add     x7, x7, #64
    subs    x8, x8, #1
    b.gt    .Lfr_vec_loop

    add     x0, x0, x4                  // next S row
    add     x1, x1, x5                  // next P row
    subs    x19, x19, #1
    b.gt    .Lfr_row_loop

.Lfr_done:
    ldp     d8, d9, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .size exp2_fast_rows, .-exp2_fast_rows
