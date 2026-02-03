/*
 * SVE exp2 kernel using Estrin's polynomial with software pipelining
 *
 * Overlaps loads of next batch with computation of current batch
 * to hide memory latency and improve throughput.
 */

.arch armv8.2-a+sve

.text
.align 4

/*
 * exp2_estrin_pipe: Software-pipelined Estrin exp2
 *
 * Arguments:
 *   x0: input pointer (float*)
 *   x1: output pointer (float*)
 *   x2: count (int)
 *
 * Register allocation:
 *   z0-z3:   current batch input / scratch
 *   z4-z7:   next batch input (prefetched)
 *   z8-z11:  fractional parts f
 *   z12-z15: f², then f⁴
 *   z16-z19: polynomial accumulators (result)
 *   z20-z23: q1 temporaries
 *   z24: clamp low (-126.0)
 *   z25: clamp high (127.0)
 *   z26: c1 = ln(2)
 *   z27: c2
 *   z28: c3
 *   z29: c4
 *   z30: bias = 127
 *   z31: 1.0
 *   p0: all-true predicate
 */
.global exp2_estrin_pipe
.type exp2_estrin_pipe, %function
exp2_estrin_pipe:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.s

    // ========== Load constants ==========
    mov     w19, #127
    dup     z30.s, w19

    // c1 = 0.6931472
    movz    w19, #0x7218
    movk    w19, #0x3F31, lsl #16
    dup     z26.s, w19

    // c2 = 0.2402265
    movz    w19, #0xED50
    movk    w19, #0x3E75, lsl #16
    dup     z27.s, w19

    // c3 = 0.0555041
    movz    w19, #0x6C16
    movk    w19, #0x3D63, lsl #16
    dup     z28.s, w19

    // c4 = 0.0096181
    movz    w19, #0x8DAE
    movk    w19, #0x3C1D, lsl #16
    dup     z29.s, w19

    // 1.0
    movz    w19, #0x0000
    movk    w19, #0x3F80, lsl #16
    dup     z31.s, w19

    // Clamp: -126.0
    movz    w19, #0x0000
    movk    w19, #0xC2FC, lsl #16
    dup     z24.s, w19

    // Clamp: 127.0
    movz    w19, #0x0000
    movk    w19, #0x42FE, lsl #16
    dup     z25.s, w19

    // Vector length calculations
    cntw    x4                          // vl = 16
    lsl     x5, x4, #2                  // 4 * vl = 64 (batch size)
    lsl     x6, x4, #3                  // 8 * vl = 128 (for 8x comparison)

    // Need at least 2 batches for pipelining
    lsl     x7, x5, #1                  // 2 * batch = 128
    cmp     x2, x7
    b.lt    .Lpipe_tail

    // ========== Prologue: Load first batch ==========
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]
    add     x0, x0, x5, lsl #2          // advance input pointer

    sub     x2, x2, x5                  // consumed first batch

    // ========== Main pipelined loop ==========
.Lpipe_loop:
    // --- Stage 1: Start loading NEXT batch while processing current ---
    // Prefetch next batch into z4-z7
    ld1w    z4.s, p0/z, [x0]
    ld1w    z5.s, p0/z, [x0, #1, mul vl]

    // --- Stage 2: Clamp current batch (z0-z3) ---
    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s

    ld1w    z6.s, p0/z, [x0, #2, mul vl]
    ld1w    z7.s, p0/z, [x0, #3, mul vl]

    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s
    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    // --- Stage 3: Split n = floor(x), f = x - n ---
    frintm  z8.s, p0/m, z0.s            // n0 (temp in z8)
    frintm  z9.s, p0/m, z1.s            // n1
    frintm  z10.s, p0/m, z2.s           // n2
    frintm  z11.s, p0/m, z3.s           // n3

    fsub    z12.s, z0.s, z8.s           // f0 (in z12 temporarily)
    fsub    z13.s, z1.s, z9.s           // f1
    fsub    z14.s, z2.s, z10.s          // f2
    fsub    z15.s, z3.s, z11.s          // f3

    // --- Stage 4: Compute 2^n ---
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s
    add     z8.s, z8.s, z30.s
    add     z9.s, z9.s, z30.s
    add     z10.s, z10.s, z30.s
    add     z11.s, z11.s, z30.s
    lsl     z8.s, z8.s, #23             // 2^n in z8-z11
    lsl     z9.s, z9.s, #23
    lsl     z10.s, z10.s, #23
    lsl     z11.s, z11.s, #23

    // --- Stage 5: Estrin polynomial for 2^f ---
    // f is in z12-z15, need to compute f²

    // f²
    fmul    z0.s, z12.s, z12.s          // f0² (reuse z0)
    fmul    z1.s, z13.s, z13.s          // f1²
    fmul    z2.s, z14.s, z14.s          // f2²
    fmul    z3.s, z15.s, z15.s          // f3²

    // q0 = 1 + c1*f (parallel with f²)
    mov     z16.d, z31.d
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z12.s, z26.s
    fmla    z17.s, p0/m, z13.s, z26.s
    fmla    z18.s, p0/m, z14.s, z26.s
    fmla    z19.s, p0/m, z15.s, z26.s

    // q1 = c2 + c3*f (parallel with q0)
    mov     z20.d, z27.d
    mov     z21.d, z27.d
    mov     z22.d, z27.d
    mov     z23.d, z27.d
    fmla    z20.s, p0/m, z12.s, z28.s
    fmla    z21.s, p0/m, z13.s, z28.s
    fmla    z22.s, p0/m, z14.s, z28.s
    fmla    z23.s, p0/m, z15.s, z28.s

    // f⁴ = f² * f² (z0-z3 have f², store f⁴ in z12-z15)
    fmul    z12.s, z0.s, z0.s
    fmul    z13.s, z1.s, z1.s
    fmul    z14.s, z2.s, z2.s
    fmul    z15.s, z3.s, z3.s

    // r = q0 + q1*f² (z0-z3 have f²)
    fmla    z16.s, p0/m, z20.s, z0.s
    fmla    z17.s, p0/m, z21.s, z1.s
    fmla    z18.s, p0/m, z22.s, z2.s
    fmla    z19.s, p0/m, z23.s, z3.s

    // p = r + c4*f⁴
    fmla    z16.s, p0/m, z12.s, z29.s
    fmla    z17.s, p0/m, z13.s, z29.s
    fmla    z18.s, p0/m, z14.s, z29.s
    fmla    z19.s, p0/m, z15.s, z29.s

    // --- Stage 6: Final multiply and store ---
    // result = 2^n * 2^f
    fmul    z16.s, z16.s, z8.s
    fmul    z17.s, z17.s, z9.s
    fmul    z18.s, z18.s, z10.s
    fmul    z19.s, z19.s, z11.s

    st1w    z16.s, p0, [x1]
    st1w    z17.s, p0, [x1, #1, mul vl]
    st1w    z18.s, p0, [x1, #2, mul vl]
    st1w    z19.s, p0, [x1, #3, mul vl]

    // --- Move next batch to current ---
    mov     z0.d, z4.d
    mov     z1.d, z5.d
    mov     z2.d, z6.d
    mov     z3.d, z7.d

    // Advance pointers
    add     x0, x0, x5, lsl #2
    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5

    cmp     x2, x5
    b.ge    .Lpipe_loop

    // ========== Epilogue: Process last batch in z0-z3 ==========
    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s
    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    frintm  z8.s, p0/m, z0.s
    frintm  z9.s, p0/m, z1.s
    frintm  z10.s, p0/m, z2.s
    frintm  z11.s, p0/m, z3.s

    fsub    z12.s, z0.s, z8.s
    fsub    z13.s, z1.s, z9.s
    fsub    z14.s, z2.s, z10.s
    fsub    z15.s, z3.s, z11.s

    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s
    add     z8.s, z8.s, z30.s
    add     z9.s, z9.s, z30.s
    add     z10.s, z10.s, z30.s
    add     z11.s, z11.s, z30.s
    lsl     z8.s, z8.s, #23
    lsl     z9.s, z9.s, #23
    lsl     z10.s, z10.s, #23
    lsl     z11.s, z11.s, #23

    fmul    z0.s, z12.s, z12.s
    fmul    z1.s, z13.s, z13.s
    fmul    z2.s, z14.s, z14.s
    fmul    z3.s, z15.s, z15.s

    mov     z16.d, z31.d
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z12.s, z26.s
    fmla    z17.s, p0/m, z13.s, z26.s
    fmla    z18.s, p0/m, z14.s, z26.s
    fmla    z19.s, p0/m, z15.s, z26.s

    mov     z20.d, z27.d
    mov     z21.d, z27.d
    mov     z22.d, z27.d
    mov     z23.d, z27.d
    fmla    z20.s, p0/m, z12.s, z28.s
    fmla    z21.s, p0/m, z13.s, z28.s
    fmla    z22.s, p0/m, z14.s, z28.s
    fmla    z23.s, p0/m, z15.s, z28.s

    fmul    z12.s, z0.s, z0.s
    fmul    z13.s, z1.s, z1.s
    fmul    z14.s, z2.s, z2.s
    fmul    z15.s, z3.s, z3.s

    fmla    z16.s, p0/m, z20.s, z0.s
    fmla    z17.s, p0/m, z21.s, z1.s
    fmla    z18.s, p0/m, z22.s, z2.s
    fmla    z19.s, p0/m, z23.s, z3.s

    fmla    z16.s, p0/m, z12.s, z29.s
    fmla    z17.s, p0/m, z13.s, z29.s
    fmla    z18.s, p0/m, z14.s, z29.s
    fmla    z19.s, p0/m, z15.s, z29.s

    fmul    z16.s, z16.s, z8.s
    fmul    z17.s, z17.s, z9.s
    fmul    z18.s, z18.s, z10.s
    fmul    z19.s, z19.s, z11.s

    st1w    z16.s, p0, [x1]
    st1w    z17.s, p0, [x1, #1, mul vl]
    st1w    z18.s, p0, [x1, #2, mul vl]
    st1w    z19.s, p0, [x1, #3, mul vl]

    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5

    // ========== Tail handling ==========
.Lpipe_tail:
    cbz     x2, .Lpipe_done

    // Handle remaining elements with predicate
    whilelt p1.s, xzr, x2

.Lpipe_tail_loop:
    ld1w    z0.s, p1/z, [x0]

    fmax    z0.s, p1/m, z0.s, z24.s
    fmin    z0.s, p1/m, z0.s, z25.s

    frintm  z4.s, p1/m, z0.s
    fsub    z8.s, z0.s, z4.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, z30.s
    lsl     z4.s, z4.s, #23

    fmul    z12.s, z8.s, z8.s

    mov     z16.d, z31.d
    fmla    z16.s, p1/m, z8.s, z26.s

    mov     z20.d, z27.d
    fmla    z20.s, p1/m, z8.s, z28.s

    fmul    z0.s, z12.s, z12.s
    fmla    z16.s, p1/m, z20.s, z12.s
    fmla    z16.s, p1/m, z0.s, z29.s

    fmul    z16.s, p1/m, z16.s, z4.s
    st1w    z16.s, p1, [x1]

    incw    x0
    incw    x1
    decw    x2
    whilelt p1.s, xzr, x2
    b.first .Lpipe_tail_loop

.Lpipe_done:
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret
.size exp2_estrin_pipe, .-exp2_estrin_pipe


/*
 * exp2_estrin_pipe8: 8x unroll with deeper pipelining
 *
 * Process 8 vectors with 2-stage pipeline:
 * - Load batch N+1 while computing batch N
 * - Two batches of 4 vectors each
 */
.global exp2_estrin_pipe8
.type exp2_estrin_pipe8, %function
exp2_estrin_pipe8:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.s

    // Load constants
    mov     w19, #127
    dup     z30.s, w19

    movz    w19, #0x7218
    movk    w19, #0x3F31, lsl #16
    dup     z26.s, w19

    movz    w19, #0xED50
    movk    w19, #0x3E75, lsl #16
    dup     z27.s, w19

    movz    w19, #0x6C16
    movk    w19, #0x3D63, lsl #16
    dup     z28.s, w19

    movz    w19, #0x8DAE
    movk    w19, #0x3C1D, lsl #16
    dup     z29.s, w19

    movz    w19, #0x0000
    movk    w19, #0x3F80, lsl #16
    dup     z31.s, w19

    movz    w19, #0x0000
    movk    w19, #0xC2FC, lsl #16
    dup     z24.s, w19

    movz    w19, #0x0000
    movk    w19, #0x42FE, lsl #16
    dup     z25.s, w19

    cntw    x4
    lsl     x5, x4, #3                  // 8 * vl = 128 elements
    lsl     x6, x4, #2                  // 4 * vl = 64 elements

    // Need at least 2 full iterations (256 elements)
    lsl     x7, x5, #1
    cmp     x2, x7
    b.lt    .Lpipe8_small

    // ========== Prologue: Load first 8 vectors ==========
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]
    ld1w    z4.s, p0/z, [x0, #4, mul vl]
    ld1w    z5.s, p0/z, [x0, #5, mul vl]
    ld1w    z6.s, p0/z, [x0, #6, mul vl]
    ld1w    z7.s, p0/z, [x0, #7, mul vl]
    add     x0, x0, x5, lsl #2
    sub     x2, x2, x5

    // ========== Main loop: Process 8 vectors with pipelined loads ==========
.Lpipe8_loop:
    // ===== Process first batch (z0-z3) while loading next iteration's first batch =====

    // Clamp z0-z3
    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s
    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    // Split: n = floor(x), f = x - n
    frintm  z8.s, p0/m, z0.s
    frintm  z9.s, p0/m, z1.s
    frintm  z10.s, p0/m, z2.s
    frintm  z11.s, p0/m, z3.s
    fsub    z12.s, z0.s, z8.s           // f in z12-z15
    fsub    z13.s, z1.s, z9.s
    fsub    z14.s, z2.s, z10.s
    fsub    z15.s, z3.s, z11.s

    // 2^n computation
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s
    add     z8.s, z8.s, z30.s
    add     z9.s, z9.s, z30.s
    add     z10.s, z10.s, z30.s
    add     z11.s, z11.s, z30.s
    lsl     z8.s, z8.s, #23             // 2^n in z8-z11
    lsl     z9.s, z9.s, #23
    lsl     z10.s, z10.s, #23
    lsl     z11.s, z11.s, #23

    // Estrin polynomial (f in z12-z15, 2^n in z8-z11)
    fmul    z0.s, z12.s, z12.s          // f² in z0-z3
    fmul    z1.s, z13.s, z13.s
    fmul    z2.s, z14.s, z14.s
    fmul    z3.s, z15.s, z15.s

    mov     z16.d, z31.d                // q0 = 1.0
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z12.s, z26.s   // q0 = 1 + c1*f
    fmla    z17.s, p0/m, z13.s, z26.s
    fmla    z18.s, p0/m, z14.s, z26.s
    fmla    z19.s, p0/m, z15.s, z26.s

    mov     z20.d, z27.d                // q1 = c2
    mov     z21.d, z27.d
    mov     z22.d, z27.d
    mov     z23.d, z27.d
    fmla    z20.s, p0/m, z12.s, z28.s   // q1 = c2 + c3*f
    fmla    z21.s, p0/m, z13.s, z28.s
    fmla    z22.s, p0/m, z14.s, z28.s
    fmla    z23.s, p0/m, z15.s, z28.s

    fmul    z12.s, z0.s, z0.s           // f⁴ in z12-z15
    fmul    z13.s, z1.s, z1.s
    fmul    z14.s, z2.s, z2.s
    fmul    z15.s, z3.s, z3.s

    fmla    z16.s, p0/m, z20.s, z0.s    // r = q0 + q1*f²
    fmla    z17.s, p0/m, z21.s, z1.s
    fmla    z18.s, p0/m, z22.s, z2.s
    fmla    z19.s, p0/m, z23.s, z3.s

    fmla    z16.s, p0/m, z12.s, z29.s   // p = r + c4*f⁴
    fmla    z17.s, p0/m, z13.s, z29.s
    fmla    z18.s, p0/m, z14.s, z29.s
    fmla    z19.s, p0/m, z15.s, z29.s

    fmul    z16.s, z16.s, z8.s          // result = 2^n * 2^f
    fmul    z17.s, z17.s, z9.s
    fmul    z18.s, z18.s, z10.s
    fmul    z19.s, z19.s, z11.s

    // Store first batch and start loading next iteration
    st1w    z16.s, p0, [x1]
    ld1w    z0.s, p0/z, [x0]            // Load next iter batch 1
    st1w    z17.s, p0, [x1, #1, mul vl]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    st1w    z18.s, p0, [x1, #2, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    st1w    z19.s, p0, [x1, #3, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // ===== Process second batch (z4-z7) while loading next iteration's second batch =====

    fmax    z4.s, p0/m, z4.s, z24.s
    fmax    z5.s, p0/m, z5.s, z24.s
    fmax    z6.s, p0/m, z6.s, z24.s
    fmax    z7.s, p0/m, z7.s, z24.s
    fmin    z4.s, p0/m, z4.s, z25.s
    fmin    z5.s, p0/m, z5.s, z25.s
    fmin    z6.s, p0/m, z6.s, z25.s
    fmin    z7.s, p0/m, z7.s, z25.s

    frintm  z8.s, p0/m, z4.s
    frintm  z9.s, p0/m, z5.s
    frintm  z10.s, p0/m, z6.s
    frintm  z11.s, p0/m, z7.s
    fsub    z12.s, z4.s, z8.s
    fsub    z13.s, z5.s, z9.s
    fsub    z14.s, z6.s, z10.s
    fsub    z15.s, z7.s, z11.s

    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s
    add     z8.s, z8.s, z30.s
    add     z9.s, z9.s, z30.s
    add     z10.s, z10.s, z30.s
    add     z11.s, z11.s, z30.s
    lsl     z8.s, z8.s, #23
    lsl     z9.s, z9.s, #23
    lsl     z10.s, z10.s, #23
    lsl     z11.s, z11.s, #23

    fmul    z4.s, z12.s, z12.s
    fmul    z5.s, z13.s, z13.s
    fmul    z6.s, z14.s, z14.s
    fmul    z7.s, z15.s, z15.s

    mov     z16.d, z31.d
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z12.s, z26.s
    fmla    z17.s, p0/m, z13.s, z26.s
    fmla    z18.s, p0/m, z14.s, z26.s
    fmla    z19.s, p0/m, z15.s, z26.s

    mov     z20.d, z27.d
    mov     z21.d, z27.d
    mov     z22.d, z27.d
    mov     z23.d, z27.d
    fmla    z20.s, p0/m, z12.s, z28.s
    fmla    z21.s, p0/m, z13.s, z28.s
    fmla    z22.s, p0/m, z14.s, z28.s
    fmla    z23.s, p0/m, z15.s, z28.s

    fmul    z12.s, z4.s, z4.s
    fmul    z13.s, z5.s, z5.s
    fmul    z14.s, z6.s, z6.s
    fmul    z15.s, z7.s, z7.s

    fmla    z16.s, p0/m, z20.s, z4.s
    fmla    z17.s, p0/m, z21.s, z5.s
    fmla    z18.s, p0/m, z22.s, z6.s
    fmla    z19.s, p0/m, z23.s, z7.s

    fmla    z16.s, p0/m, z12.s, z29.s
    fmla    z17.s, p0/m, z13.s, z29.s
    fmla    z18.s, p0/m, z14.s, z29.s
    fmla    z19.s, p0/m, z15.s, z29.s

    fmul    z16.s, z16.s, z8.s
    fmul    z17.s, z17.s, z9.s
    fmul    z18.s, z18.s, z10.s
    fmul    z19.s, z19.s, z11.s

    // Store second batch and load next iteration's second batch
    st1w    z16.s, p0, [x1, #4, mul vl]
    ld1w    z4.s, p0/z, [x0, #4, mul vl]
    st1w    z17.s, p0, [x1, #5, mul vl]
    ld1w    z5.s, p0/z, [x0, #5, mul vl]
    st1w    z18.s, p0, [x1, #6, mul vl]
    ld1w    z6.s, p0/z, [x0, #6, mul vl]
    st1w    z19.s, p0, [x1, #7, mul vl]
    ld1w    z7.s, p0/z, [x0, #7, mul vl]

    // Advance pointers
    add     x0, x0, x5, lsl #2
    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5

    cmp     x2, x5
    b.ge    .Lpipe8_loop

    // ========== Epilogue: Process remaining loaded data ==========
    // z0-z7 contain the last loaded batch, process it

    // First batch z0-z3
    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s
    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    frintm  z8.s, p0/m, z0.s
    frintm  z9.s, p0/m, z1.s
    frintm  z10.s, p0/m, z2.s
    frintm  z11.s, p0/m, z3.s
    fsub    z12.s, z0.s, z8.s
    fsub    z13.s, z1.s, z9.s
    fsub    z14.s, z2.s, z10.s
    fsub    z15.s, z3.s, z11.s

    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s
    add     z8.s, z8.s, z30.s
    add     z9.s, z9.s, z30.s
    add     z10.s, z10.s, z30.s
    add     z11.s, z11.s, z30.s
    lsl     z8.s, z8.s, #23
    lsl     z9.s, z9.s, #23
    lsl     z10.s, z10.s, #23
    lsl     z11.s, z11.s, #23

    fmul    z0.s, z12.s, z12.s
    fmul    z1.s, z13.s, z13.s
    fmul    z2.s, z14.s, z14.s
    fmul    z3.s, z15.s, z15.s

    mov     z16.d, z31.d
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z12.s, z26.s
    fmla    z17.s, p0/m, z13.s, z26.s
    fmla    z18.s, p0/m, z14.s, z26.s
    fmla    z19.s, p0/m, z15.s, z26.s

    mov     z20.d, z27.d
    mov     z21.d, z27.d
    mov     z22.d, z27.d
    mov     z23.d, z27.d
    fmla    z20.s, p0/m, z12.s, z28.s
    fmla    z21.s, p0/m, z13.s, z28.s
    fmla    z22.s, p0/m, z14.s, z28.s
    fmla    z23.s, p0/m, z15.s, z28.s

    fmul    z12.s, z0.s, z0.s
    fmul    z13.s, z1.s, z1.s
    fmul    z14.s, z2.s, z2.s
    fmul    z15.s, z3.s, z3.s

    fmla    z16.s, p0/m, z20.s, z0.s
    fmla    z17.s, p0/m, z21.s, z1.s
    fmla    z18.s, p0/m, z22.s, z2.s
    fmla    z19.s, p0/m, z23.s, z3.s

    fmla    z16.s, p0/m, z12.s, z29.s
    fmla    z17.s, p0/m, z13.s, z29.s
    fmla    z18.s, p0/m, z14.s, z29.s
    fmla    z19.s, p0/m, z15.s, z29.s

    fmul    z16.s, z16.s, z8.s
    fmul    z17.s, z17.s, z9.s
    fmul    z18.s, z18.s, z10.s
    fmul    z19.s, z19.s, z11.s

    st1w    z16.s, p0, [x1]
    st1w    z17.s, p0, [x1, #1, mul vl]
    st1w    z18.s, p0, [x1, #2, mul vl]
    st1w    z19.s, p0, [x1, #3, mul vl]

    // Second batch z4-z7
    fmax    z4.s, p0/m, z4.s, z24.s
    fmax    z5.s, p0/m, z5.s, z24.s
    fmax    z6.s, p0/m, z6.s, z24.s
    fmax    z7.s, p0/m, z7.s, z24.s
    fmin    z4.s, p0/m, z4.s, z25.s
    fmin    z5.s, p0/m, z5.s, z25.s
    fmin    z6.s, p0/m, z6.s, z25.s
    fmin    z7.s, p0/m, z7.s, z25.s

    frintm  z8.s, p0/m, z4.s
    frintm  z9.s, p0/m, z5.s
    frintm  z10.s, p0/m, z6.s
    frintm  z11.s, p0/m, z7.s
    fsub    z12.s, z4.s, z8.s
    fsub    z13.s, z5.s, z9.s
    fsub    z14.s, z6.s, z10.s
    fsub    z15.s, z7.s, z11.s

    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s
    add     z8.s, z8.s, z30.s
    add     z9.s, z9.s, z30.s
    add     z10.s, z10.s, z30.s
    add     z11.s, z11.s, z30.s
    lsl     z8.s, z8.s, #23
    lsl     z9.s, z9.s, #23
    lsl     z10.s, z10.s, #23
    lsl     z11.s, z11.s, #23

    fmul    z4.s, z12.s, z12.s
    fmul    z5.s, z13.s, z13.s
    fmul    z6.s, z14.s, z14.s
    fmul    z7.s, z15.s, z15.s

    mov     z16.d, z31.d
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z12.s, z26.s
    fmla    z17.s, p0/m, z13.s, z26.s
    fmla    z18.s, p0/m, z14.s, z26.s
    fmla    z19.s, p0/m, z15.s, z26.s

    mov     z20.d, z27.d
    mov     z21.d, z27.d
    mov     z22.d, z27.d
    mov     z23.d, z27.d
    fmla    z20.s, p0/m, z12.s, z28.s
    fmla    z21.s, p0/m, z13.s, z28.s
    fmla    z22.s, p0/m, z14.s, z28.s
    fmla    z23.s, p0/m, z15.s, z28.s

    fmul    z12.s, z4.s, z4.s
    fmul    z13.s, z5.s, z5.s
    fmul    z14.s, z6.s, z6.s
    fmul    z15.s, z7.s, z7.s

    fmla    z16.s, p0/m, z20.s, z4.s
    fmla    z17.s, p0/m, z21.s, z5.s
    fmla    z18.s, p0/m, z22.s, z6.s
    fmla    z19.s, p0/m, z23.s, z7.s

    fmla    z16.s, p0/m, z12.s, z29.s
    fmla    z17.s, p0/m, z13.s, z29.s
    fmla    z18.s, p0/m, z14.s, z29.s
    fmla    z19.s, p0/m, z15.s, z29.s

    fmul    z16.s, z16.s, z8.s
    fmul    z17.s, z17.s, z9.s
    fmul    z18.s, z18.s, z10.s
    fmul    z19.s, z19.s, z11.s

    st1w    z16.s, p0, [x1, #4, mul vl]
    st1w    z17.s, p0, [x1, #5, mul vl]
    st1w    z18.s, p0, [x1, #6, mul vl]
    st1w    z19.s, p0, [x1, #7, mul vl]

    add     x1, x1, x5, lsl #2
    b       .Lpipe8_done

.Lpipe8_small:
    // Handle < 256 elements - use simple 4x processing
    cmp     x2, x6
    b.lt    .Lpipe8_tail1

.Lpipe8_tail4:
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s
    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    frintm  z8.s, p0/m, z0.s
    frintm  z9.s, p0/m, z1.s
    frintm  z10.s, p0/m, z2.s
    frintm  z11.s, p0/m, z3.s
    fsub    z12.s, z0.s, z8.s
    fsub    z13.s, z1.s, z9.s
    fsub    z14.s, z2.s, z10.s
    fsub    z15.s, z3.s, z11.s

    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s
    add     z8.s, z8.s, z30.s
    add     z9.s, z9.s, z30.s
    add     z10.s, z10.s, z30.s
    add     z11.s, z11.s, z30.s
    lsl     z8.s, z8.s, #23
    lsl     z9.s, z9.s, #23
    lsl     z10.s, z10.s, #23
    lsl     z11.s, z11.s, #23

    fmul    z0.s, z12.s, z12.s
    fmul    z1.s, z13.s, z13.s
    fmul    z2.s, z14.s, z14.s
    fmul    z3.s, z15.s, z15.s

    mov     z16.d, z31.d
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z12.s, z26.s
    fmla    z17.s, p0/m, z13.s, z26.s
    fmla    z18.s, p0/m, z14.s, z26.s
    fmla    z19.s, p0/m, z15.s, z26.s

    mov     z20.d, z27.d
    mov     z21.d, z27.d
    mov     z22.d, z27.d
    mov     z23.d, z27.d
    fmla    z20.s, p0/m, z12.s, z28.s
    fmla    z21.s, p0/m, z13.s, z28.s
    fmla    z22.s, p0/m, z14.s, z28.s
    fmla    z23.s, p0/m, z15.s, z28.s

    fmul    z12.s, z0.s, z0.s
    fmul    z13.s, z1.s, z1.s
    fmul    z14.s, z2.s, z2.s
    fmul    z15.s, z3.s, z3.s

    fmla    z16.s, p0/m, z20.s, z0.s
    fmla    z17.s, p0/m, z21.s, z1.s
    fmla    z18.s, p0/m, z22.s, z2.s
    fmla    z19.s, p0/m, z23.s, z3.s

    fmla    z16.s, p0/m, z12.s, z29.s
    fmla    z17.s, p0/m, z13.s, z29.s
    fmla    z18.s, p0/m, z14.s, z29.s
    fmla    z19.s, p0/m, z15.s, z29.s

    fmul    z16.s, z16.s, z8.s
    fmul    z17.s, z17.s, z9.s
    fmul    z18.s, z18.s, z10.s
    fmul    z19.s, z19.s, z11.s

    st1w    z16.s, p0, [x1]
    st1w    z17.s, p0, [x1, #1, mul vl]
    st1w    z18.s, p0, [x1, #2, mul vl]
    st1w    z19.s, p0, [x1, #3, mul vl]

    add     x0, x0, x6, lsl #2
    add     x1, x1, x6, lsl #2
    sub     x2, x2, x6

    cmp     x2, x6
    b.ge    .Lpipe8_tail4

.Lpipe8_tail1:
    cbz     x2, .Lpipe8_done

    whilelt p1.s, xzr, x2
    ld1w    z0.s, p1/z, [x0]

    fmax    z0.s, p1/m, z0.s, z24.s
    fmin    z0.s, p1/m, z0.s, z25.s

    frintm  z4.s, p1/m, z0.s
    fsub    z8.s, z0.s, z4.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, z30.s
    lsl     z4.s, z4.s, #23

    fmul    z12.s, z8.s, z8.s

    mov     z16.d, z31.d
    fmla    z16.s, p1/m, z8.s, z26.s

    mov     z20.d, z27.d
    fmla    z20.s, p1/m, z8.s, z28.s

    fmul    z0.s, z12.s, z12.s
    fmla    z16.s, p1/m, z20.s, z12.s
    fmla    z16.s, p1/m, z0.s, z29.s

    fmul    z16.s, p1/m, z16.s, z4.s
    st1w    z16.s, p1, [x1]

.Lpipe8_done:
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret
.size exp2_estrin_pipe8, .-exp2_estrin_pipe8
