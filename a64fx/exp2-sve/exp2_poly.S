/*
 * Polynomial exp2 approximation for softmax
 *
 * For softmax, we only need relative accuracy since we normalize.
 * Use a low-degree polynomial to minimize instructions.
 *
 * exp2(x) ≈ 2^floor(x) * poly(frac(x))
 *
 * For frac in [0,1):
 * exp2(f) ≈ 1 + f*0.693 + f²*0.240 + f³*0.056
 *
 * But this still needs range reduction. Alternative:
 * For small |x|, use Taylor series directly:
 * exp2(x) = e^(x*ln2) ≈ 1 + x*ln2 + (x*ln2)²/2 + ...
 *
 * Even simpler for softmax: since we normalize, we can use
 * exp2(x) ≈ a*x + b for the linear region and clamp elsewhere
 *
 * Ultra-minimal: 4 instructions per vector
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*============================================================================
 * exp2_ultra_fast - Minimal exp2 for softmax (4 instructions per vector)
 *
 * Uses bit manipulation to construct approximate exp2 directly.
 * For softmax where x is typically in [-10, 0] range after subtracting max.
 *
 * void exp2_ultra_fast(
 *     const int32_t* S,   // x0
 *     float* P,           // x1
 *     int n,              // x2 (multiple of 16)
 *     float scale,        // s0
 *     float neg_max       // s1
 * );
 *============================================================================*/

    .global exp2_ultra_fast
    .type exp2_ultra_fast, %function
exp2_ultra_fast:
    ptrue   p0.s

    mov     z20.s, s0                   // scale
    mov     z21.s, s1                   // neg_max

    /* For IEEE754 float: exp2(x) = 2^x
     * The exponent field is biased by 127.
     * So 2^x as float has exponent = x + 127.
     *
     * Direct construction:
     * 1. Compute x = S * scale - max
     * 2. Add 127 to get biased exponent
     * 3. Shift left 23 bits into exponent position
     *
     * This gives exact 2^floor(x), ignoring fractional part.
     * For softmax normalization, this approximation is often sufficient.
     */
    mov     w3, #127
    mov     z22.s, w3                   // bias

    cbz     x2, .Luf_done
    lsr     x2, x2, #4

.Luf_loop:
    /* Load and convert */
    ld1w    {z0.s}, p0/z, [x0]
    scvtf   z0.s, p0/m, z0.s            // 1: int32 → fp32

    /* Scale and subtract max */
    fmul    z0.s, z0.s, z20.s           // 2: * scale
    fadd    z0.s, z0.s, z21.s           // 3: + neg_max (= - max)

    /* Convert to integer exponent */
    fcvtzs  z0.s, p0/m, z0.s            // 4: to int (floor)

    /* Construct 2^floor(x) */
    add     z0.s, z0.s, z22.s           // 5: + 127
    lsl     z0.s, z0.s, #23             // 6: shift to exponent

    /* z0 now contains 2^floor(x) as IEEE754 float */
    st1w    {z0.s}, p0, [x1]

    add     x0, x0, #64
    add     x1, x1, #64
    subs    x2, x2, #1
    b.gt    .Luf_loop

.Luf_done:
    ret

    .size exp2_ultra_fast, .-exp2_ultra_fast


/*============================================================================
 * exp2_ultra_rows - Ultra fast exp2 for M rows
 *============================================================================*/

    .global exp2_ultra_rows
    .type exp2_ultra_rows, %function
exp2_ultra_rows:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8, d9, [sp, #32]

    mov     x19, x2                     // M
    mov     x20, x3                     // Nc
    fmov    s8, s0
    fmov    s9, s1
    lsl     x4, x4, #2
    lsl     x5, x5, #2

    cbz     x19, .Lur_done

.Lur_row_loop:
    mov     x6, x0
    mov     x7, x1
    mov     x8, x20

    ptrue   p0.s
    mov     z20.s, s8
    mov     z21.s, s9
    mov     w9, #127
    mov     z22.s, w9

    lsr     x8, x8, #4

.Lur_vec_loop:
    ld1w    {z0.s}, p0/z, [x6]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z21.s
    fcvtzs  z0.s, p0/m, z0.s
    add     z0.s, z0.s, z22.s
    lsl     z0.s, z0.s, #23
    st1w    {z0.s}, p0, [x7]

    add     x6, x6, #64
    add     x7, x7, #64
    subs    x8, x8, #1
    b.gt    .Lur_vec_loop

    add     x0, x0, x4
    add     x1, x1, x5
    subs    x19, x19, #1
    b.gt    .Lur_row_loop

.Lur_done:
    ldp     d8, d9, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .size exp2_ultra_rows, .-exp2_ultra_rows


/*============================================================================
 * exp2_minimal - Even more minimal: inline in GEMM
 *
 * The absolute minimum for softmax: just 2^floor(x)
 * Only 6 instructions per vector: ld, scvtf, fmul, fadd, fcvtzs+add+lsl combined?, st
 *
 * Actually, the bottleneck is that we need:
 * - Load (1)
 * - Convert int→float (1)
 * - Scale and offset (2)
 * - Convert float→int (1)
 * - Bias and shift (2)
 * - Store (1)
 * = 8 instructions minimum
 *
 * Can we do better? If S is already float:
 * - Load (1)
 * - Scale and offset (fmad) (1)
 * - To int (1)
 * - Bias+shift (2)
 * - Store (1)
 * = 6 instructions
 *============================================================================*/

    .global exp2_minimal_float
    .type exp2_minimal_float, %function
exp2_minimal_float:
    /* Input: float S instead of int32 */
    /* void exp2_minimal_float(float* S, float* P, int n, float scale, float neg_max) */
    ptrue   p0.s

    mov     z20.s, s0                   // scale
    mov     z21.s, s1                   // neg_max
    mov     w3, #127
    mov     z22.s, w3

    cbz     x2, .Lmf_done
    lsr     x2, x2, #4

.Lmf_loop:
    ld1w    {z0.s}, p0/z, [x0]          // 1: load float
    fmul    z0.s, z0.s, z20.s           // 2: * scale
    fadd    z0.s, z0.s, z21.s           // 3: - max
    fcvtzs  z0.s, p0/m, z0.s            // 4: floor
    add     z0.s, z0.s, z22.s           // 5: + 127
    lsl     z0.s, z0.s, #23             // 6: to exponent
    st1w    {z0.s}, p0, [x1]            // 7: store

    add     x0, x0, #64
    add     x1, x1, #64
    subs    x2, x2, #1
    b.gt    .Lmf_loop

.Lmf_done:
    ret

    .size exp2_minimal_float, .-exp2_minimal_float
