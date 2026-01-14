// kernel_6x4_opt.S
// Highly optimized INT8 SDOT GEMM Microkernel for A64FX SVE
// MR=6 rows × NR=64 columns (4 SVE vectors)
// K-loop unrolled 8× (processes 32 bytes per iteration)
// Software pipelined with prefetch
//
// Target: 95%+ of 512 GOPS peak (486+ GOPS) on M=N=4096+, K=256
//
// Register allocation:
//   z0-z23:  24 accumulators (6 rows × 4 vectors)
//   z24-z29: A row buffers (6 rows)
//   z30-z31: B column buffers (2 vectors, reloaded 2× per iteration)
//   p0:      All-true predicate for byte ops
//   p1:      All-true predicate for word ops

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_6x4_opt_256
    .type kernel_6x4_opt_256, @function

// Arguments:
//   x0 = Apack pointer [6×256] int8 row-major
//   x1 = Bpack pointer [256×64] int8 K-major (SDOT layout)
//   x2 = C pointer [6×64] int32 row-major output
//   x3 = ldc in bytes

kernel_6x4_opt_256:
    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!

    // Setup predicates
    ptrue   p0.b
    ptrue   p1.s

    // Zero 24 accumulators
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
    dup     z16.s, #0
    dup     z17.s, #0
    dup     z18.s, #0
    dup     z19.s, #0
    dup     z20.s, #0
    dup     z21.s, #0
    dup     z22.s, #0
    dup     z23.s, #0

    // Setup A row pointers
    mov     x4, x0                  // Row 0
    add     x10, x0, #256           // Row 1
    add     x11, x10, #256          // Row 2
    add     x12, x11, #256          // Row 3
    add     x13, x12, #256          // Row 4
    add     x14, x13, #256          // Row 5

    mov     x5, x1                  // Bpack base
    mov     x6, #8                  // K counter (256/32 = 8 iterations, 32 bytes/iter)

    .align 6
.Lk_loop_6x4_opt:
    // ================================================================
    // K-loop processes 32 bytes (8 SDOT groups) per iteration
    // Each SDOT group is 4 bytes
    // With 6 rows × 4 vectors = 24 SDOT ops per group
    // Total: 24 × 8 = 192 SDOT instructions per iteration
    // ================================================================

    // Prefetch next iteration's data
    prfm    pldl1keep, [x4, #128]
    prfm    pldl1keep, [x10, #128]
    prfm    pldl1keep, [x11, #128]
    prfm    pldl1keep, [x12, #128]
    prfm    pldl1keep, [x13, #128]
    prfm    pldl1keep, [x14, #128]
    prfm    pldl1keep, [x5, #1024]
    prfm    pldl1keep, [x5, #1280]

    // ----------------------------------------
    // Unrolled iteration 0: k+0
    // ----------------------------------------
    ldr     w15, [x4]
    ldr     w16, [x10]
    ldr     w17, [x11]
    ldr     w19, [x12]
    ldr     w20, [x13]
    ldr     w21, [x14]

    mov     z24.s, w15
    mov     z25.s, w16
    mov     z26.s, w17
    mov     z27.s, w19
    mov     z28.s, w20
    mov     z29.s, w21

    ld1b    z30.b, p0/z, [x5, #0, mul vl]
    ld1b    z31.b, p0/z, [x5, #1, mul vl]

    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b

    ld1b    z30.b, p0/z, [x5, #2, mul vl]
    ld1b    z31.b, p0/z, [x5, #3, mul vl]

    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b

    add     x4, x4, #4
    add     x10, x10, #4
    add     x11, x11, #4
    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x5, x5, #256

    // ----------------------------------------
    // Unrolled iteration 1: k+4
    // ----------------------------------------
    ldr     w15, [x4]
    ldr     w16, [x10]
    ldr     w17, [x11]
    ldr     w19, [x12]
    ldr     w20, [x13]
    ldr     w21, [x14]

    mov     z24.s, w15
    mov     z25.s, w16
    mov     z26.s, w17
    mov     z27.s, w19
    mov     z28.s, w20
    mov     z29.s, w21

    ld1b    z30.b, p0/z, [x5, #0, mul vl]
    ld1b    z31.b, p0/z, [x5, #1, mul vl]

    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b

    ld1b    z30.b, p0/z, [x5, #2, mul vl]
    ld1b    z31.b, p0/z, [x5, #3, mul vl]

    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b

    add     x4, x4, #4
    add     x10, x10, #4
    add     x11, x11, #4
    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x5, x5, #256

    // ----------------------------------------
    // Unrolled iterations 2-7 (identical pattern)
    // ----------------------------------------

    // Iteration 2
    ldr     w15, [x4]
    ldr     w16, [x10]
    ldr     w17, [x11]
    ldr     w19, [x12]
    ldr     w20, [x13]
    ldr     w21, [x14]
    mov     z24.s, w15
    mov     z25.s, w16
    mov     z26.s, w17
    mov     z27.s, w19
    mov     z28.s, w20
    mov     z29.s, w21
    ld1b    z30.b, p0/z, [x5, #0, mul vl]
    ld1b    z31.b, p0/z, [x5, #1, mul vl]
    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b
    ld1b    z30.b, p0/z, [x5, #2, mul vl]
    ld1b    z31.b, p0/z, [x5, #3, mul vl]
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    add     x4, x4, #4
    add     x10, x10, #4
    add     x11, x11, #4
    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x5, x5, #256

    // Iteration 3
    ldr     w15, [x4]
    ldr     w16, [x10]
    ldr     w17, [x11]
    ldr     w19, [x12]
    ldr     w20, [x13]
    ldr     w21, [x14]
    mov     z24.s, w15
    mov     z25.s, w16
    mov     z26.s, w17
    mov     z27.s, w19
    mov     z28.s, w20
    mov     z29.s, w21
    ld1b    z30.b, p0/z, [x5, #0, mul vl]
    ld1b    z31.b, p0/z, [x5, #1, mul vl]
    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b
    ld1b    z30.b, p0/z, [x5, #2, mul vl]
    ld1b    z31.b, p0/z, [x5, #3, mul vl]
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    add     x4, x4, #4
    add     x10, x10, #4
    add     x11, x11, #4
    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x5, x5, #256

    // Iteration 4
    ldr     w15, [x4]
    ldr     w16, [x10]
    ldr     w17, [x11]
    ldr     w19, [x12]
    ldr     w20, [x13]
    ldr     w21, [x14]
    mov     z24.s, w15
    mov     z25.s, w16
    mov     z26.s, w17
    mov     z27.s, w19
    mov     z28.s, w20
    mov     z29.s, w21
    ld1b    z30.b, p0/z, [x5, #0, mul vl]
    ld1b    z31.b, p0/z, [x5, #1, mul vl]
    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b
    ld1b    z30.b, p0/z, [x5, #2, mul vl]
    ld1b    z31.b, p0/z, [x5, #3, mul vl]
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    add     x4, x4, #4
    add     x10, x10, #4
    add     x11, x11, #4
    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x5, x5, #256

    // Iteration 5
    ldr     w15, [x4]
    ldr     w16, [x10]
    ldr     w17, [x11]
    ldr     w19, [x12]
    ldr     w20, [x13]
    ldr     w21, [x14]
    mov     z24.s, w15
    mov     z25.s, w16
    mov     z26.s, w17
    mov     z27.s, w19
    mov     z28.s, w20
    mov     z29.s, w21
    ld1b    z30.b, p0/z, [x5, #0, mul vl]
    ld1b    z31.b, p0/z, [x5, #1, mul vl]
    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b
    ld1b    z30.b, p0/z, [x5, #2, mul vl]
    ld1b    z31.b, p0/z, [x5, #3, mul vl]
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    add     x4, x4, #4
    add     x10, x10, #4
    add     x11, x11, #4
    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x5, x5, #256

    // Iteration 6
    ldr     w15, [x4]
    ldr     w16, [x10]
    ldr     w17, [x11]
    ldr     w19, [x12]
    ldr     w20, [x13]
    ldr     w21, [x14]
    mov     z24.s, w15
    mov     z25.s, w16
    mov     z26.s, w17
    mov     z27.s, w19
    mov     z28.s, w20
    mov     z29.s, w21
    ld1b    z30.b, p0/z, [x5, #0, mul vl]
    ld1b    z31.b, p0/z, [x5, #1, mul vl]
    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b
    ld1b    z30.b, p0/z, [x5, #2, mul vl]
    ld1b    z31.b, p0/z, [x5, #3, mul vl]
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    add     x4, x4, #4
    add     x10, x10, #4
    add     x11, x11, #4
    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x5, x5, #256

    // Iteration 7
    ldr     w15, [x4]
    ldr     w16, [x10]
    ldr     w17, [x11]
    ldr     w19, [x12]
    ldr     w20, [x13]
    ldr     w21, [x14]
    mov     z24.s, w15
    mov     z25.s, w16
    mov     z26.s, w17
    mov     z27.s, w19
    mov     z28.s, w20
    mov     z29.s, w21
    ld1b    z30.b, p0/z, [x5, #0, mul vl]
    ld1b    z31.b, p0/z, [x5, #1, mul vl]
    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b
    ld1b    z30.b, p0/z, [x5, #2, mul vl]
    ld1b    z31.b, p0/z, [x5, #3, mul vl]
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    add     x4, x4, #4
    add     x10, x10, #4
    add     x11, x11, #4
    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x5, x5, #256

    // Decrement counter (processed 32 bytes)
    subs    x6, x6, #1
    b.gt    .Lk_loop_6x4_opt

    // ========================================
    // Store results: 6 rows × 4 vectors = 24 SVE vectors
    // ========================================
    mov     x7, x2
    mov     x8, x3

    // Row 0
    st1w    z0.s, p1, [x7, #0, mul vl]
    st1w    z1.s, p1, [x7, #1, mul vl]
    st1w    z2.s, p1, [x7, #2, mul vl]
    st1w    z3.s, p1, [x7, #3, mul vl]

    // Row 1
    add     x7, x7, x8
    st1w    z4.s, p1, [x7, #0, mul vl]
    st1w    z5.s, p1, [x7, #1, mul vl]
    st1w    z6.s, p1, [x7, #2, mul vl]
    st1w    z7.s, p1, [x7, #3, mul vl]

    // Row 2
    add     x7, x7, x8
    st1w    z8.s, p1, [x7, #0, mul vl]
    st1w    z9.s, p1, [x7, #1, mul vl]
    st1w    z10.s, p1, [x7, #2, mul vl]
    st1w    z11.s, p1, [x7, #3, mul vl]

    // Row 3
    add     x7, x7, x8
    st1w    z12.s, p1, [x7, #0, mul vl]
    st1w    z13.s, p1, [x7, #1, mul vl]
    st1w    z14.s, p1, [x7, #2, mul vl]
    st1w    z15.s, p1, [x7, #3, mul vl]

    // Row 4
    add     x7, x7, x8
    st1w    z16.s, p1, [x7, #0, mul vl]
    st1w    z17.s, p1, [x7, #1, mul vl]
    st1w    z18.s, p1, [x7, #2, mul vl]
    st1w    z19.s, p1, [x7, #3, mul vl]

    // Row 5
    add     x7, x7, x8
    st1w    z20.s, p1, [x7, #0, mul vl]
    st1w    z21.s, p1, [x7, #1, mul vl]
    st1w    z22.s, p1, [x7, #2, mul vl]
    st1w    z23.s, p1, [x7, #3, mul vl]

    // Restore registers
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    ret

    .size kernel_6x4_opt_256, .-kernel_6x4_opt_256
