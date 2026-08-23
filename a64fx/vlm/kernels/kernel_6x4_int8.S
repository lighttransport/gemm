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

    .global kernel_int8_6x4_256
    .type kernel_int8_6x4_256, @function

// Arguments:
//   x0 = Apack pointer [6×256] int8 row-major
//   x1 = Bpack pointer [256×64] int8 K-major (SDOT layout)
//   x2 = C pointer [6×64] int32 row-major output
//   x3 = ldc in bytes

kernel_int8_6x4_256:
    // Save callee-saved SVE vectors (z8-z15, z28-z31). The kernel clobbers
    // z0-z31; without saving the callee-saved ones, concurrent C callers that
    // keep live SVE values across the call get corrupted -> a data race.
    // Reserve 12*VL (VL=64 on this 512-bit node); VL-relative stores keep it
    // correct for any VL.
    sub     sp, sp, #768
    str     z8,  [sp, #0,  mul vl]
    str     z9,  [sp, #1,  mul vl]
    str     z10, [sp, #2,  mul vl]
    str     z11, [sp, #3,  mul vl]
    str     z12, [sp, #4,  mul vl]
    str     z13, [sp, #5,  mul vl]
    str     z14, [sp, #6,  mul vl]
    str     z15, [sp, #7,  mul vl]
    str     z28, [sp, #8,  mul vl]
    str     z29, [sp, #9,  mul vl]
    str     z30, [sp, #10, mul vl]
    str     z31, [sp, #11, mul vl]
    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    // Setup predicates
    ptrue   p0.b
    ptrue   p1.s

    // ACCUMULATE: load C (24 vectors) into the accumulators so this call
    // computes C += A x B^T (needed to sum over K/256 chunks). x2=C, x3=ldc(bytes).
    mov     x7, x2
    mov     x8, x3
    ld1w    z0.s,  p1/z, [x7, #0, mul vl]
    ld1w    z1.s,  p1/z, [x7, #1, mul vl]
    ld1w    z2.s,  p1/z, [x7, #2, mul vl]
    ld1w    z3.s,  p1/z, [x7, #3, mul vl]
    add     x7, x7, x8
    ld1w    z4.s,  p1/z, [x7, #0, mul vl]
    ld1w    z5.s,  p1/z, [x7, #1, mul vl]
    ld1w    z6.s,  p1/z, [x7, #2, mul vl]
    ld1w    z7.s,  p1/z, [x7, #3, mul vl]
    add     x7, x7, x8
    ld1w    z8.s,  p1/z, [x7, #0, mul vl]
    ld1w    z9.s,  p1/z, [x7, #1, mul vl]
    ld1w    z10.s, p1/z, [x7, #2, mul vl]
    ld1w    z11.s, p1/z, [x7, #3, mul vl]
    add     x7, x7, x8
    ld1w    z12.s, p1/z, [x7, #0, mul vl]
    ld1w    z13.s, p1/z, [x7, #1, mul vl]
    ld1w    z14.s, p1/z, [x7, #2, mul vl]
    ld1w    z15.s, p1/z, [x7, #3, mul vl]
    add     x7, x7, x8
    ld1w    z16.s, p1/z, [x7, #0, mul vl]
    ld1w    z17.s, p1/z, [x7, #1, mul vl]
    ld1w    z18.s, p1/z, [x7, #2, mul vl]
    ld1w    z19.s, p1/z, [x7, #3, mul vl]
    add     x7, x7, x8
    ld1w    z20.s, p1/z, [x7, #0, mul vl]
    ld1w    z21.s, p1/z, [x7, #1, mul vl]
    ld1w    z22.s, p1/z, [x7, #2, mul vl]
    ld1w    z23.s, p1/z, [x7, #3, mul vl]

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
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    // Restore callee-saved SVE vectors
    ldr     z8,  [sp, #0,  mul vl]
    ldr     z9,  [sp, #1,  mul vl]
    ldr     z10, [sp, #2,  mul vl]
    ldr     z11, [sp, #3,  mul vl]
    ldr     z12, [sp, #4,  mul vl]
    ldr     z13, [sp, #5,  mul vl]
    ldr     z14, [sp, #6,  mul vl]
    ldr     z15, [sp, #7,  mul vl]
    ldr     z28, [sp, #8,  mul vl]
    ldr     z29, [sp, #9,  mul vl]
    ldr     z30, [sp, #10, mul vl]
    ldr     z31, [sp, #11, mul vl]
    add     sp, sp, #768

    ret

    .size kernel_int8_6x4_256, .-kernel_int8_6x4_256

// ============================================================================
// FUSED variant: zeroes accumulators, loops over ALL K/256 chunks (register
// accumulation, NO int32 C buffer), then dequantizes to float Y and stores.
// Eliminates the C buffer (alloc + memset + int32 write + int32 read) and the
// separate dequant pass.
//
// Arguments:
//   x0 = A_base  (mb's kc=0 tile, [KC][6][256] int8, 1536 bytes/kc)
//   x1 = B_base  (nb's kc=0 tile, [KC][256][64] int8, 16384 bytes/kc)
//   x2 = KC
//   x3 = Y_tile  (float*, Y + (mb*6)*ldc + nb*64)  -- top-left of the 6x64 tile
//   x4 = ldc_bytes (N*4)
//   x5 = a_scale_ptr (float*, 6 values = a_scale[mb*6 .. mb*6+5])
//   x6 = w_scale_ptr (float*, 64 values = w_scale[nb*64 .. nb*64+63])
//   x7 = mr  (1..6, number of valid rows to store; tail rows are zero-padded
//            in A and their accumulators are computed but NOT stored, since Y
//            is only M rows, not MB*6)
//
// Register plan: x22=A cur, x23=B cur, x24=KC counter (inner loop clobbers
// w19/w20/w21, so the outer pointers avoid x19-x21). Y_tile/ldc/ws/as/mr
// saved on the stack. Inner SDOT loop uses x4,x5,x6,x10-x14 + z0-z31 as
// before. (x6 is reused as the inner K counter, so w_scale_ptr is saved.)
    .global kernel_int8_6x4_fused
    .type kernel_int8_6x4_fused, @function

kernel_int8_6x4_fused:
    // 768 (SVE z8-15,z28-31) + x19-x24 (48) + d8-d15 (64) + 4 ptrs + mr (40)
    sub     sp, sp, #976
    str     z8,  [sp, #0,  mul vl]
    str     z9,  [sp, #1,  mul vl]
    str     z10, [sp, #2,  mul vl]
    str     z11, [sp, #3,  mul vl]
    str     z12, [sp, #4,  mul vl]
    str     z13, [sp, #5,  mul vl]
    str     z14, [sp, #6,  mul vl]
    str     z15, [sp, #7,  mul vl]
    str     z28, [sp, #8,  mul vl]
    str     z29, [sp, #9,  mul vl]
    str     z30, [sp, #10, mul vl]
    str     z31, [sp, #11, mul vl]
    // x19-x24
    str     x19, [sp, #800]
    str     x20, [sp, #808]
    str     x21, [sp, #816]
    str     x22, [sp, #824]
    str     x23, [sp, #832]
    str     x24, [sp, #840]
    // d8-d15
    str     d8,  [sp, #864]
    str     d9,  [sp, #872]
    str     d10, [sp, #880]
    str     d11, [sp, #888]
    str     d12, [sp, #896]
    str     d13, [sp, #904]
    str     d14, [sp, #912]
    str     d15, [sp, #920]
    // constant pointers (needed in the epilogue)
    str     x3, [sp, #928]   // Y_tile
    str     x4, [sp, #936]   // ldc_bytes
    str     x5, [sp, #944]   // a_scale_ptr
    str     x6, [sp, #952]   // w_scale_ptr
    str     w7,  [sp, #960]  // mr

    ptrue   p0.b
    ptrue   p1.s

    // zero the 24 accumulators (register accumulation across all kc)
    mov     z0.s, #0
    mov     z1.s, #0
    mov     z2.s, #0
    mov     z3.s, #0
    mov     z4.s, #0
    mov     z5.s, #0
    mov     z6.s, #0
    mov     z7.s, #0
    mov     z8.s, #0
    mov     z9.s, #0
    mov     z10.s, #0
    mov     z11.s, #0
    mov     z12.s, #0
    mov     z13.s, #0
    mov     z14.s, #0
    mov     z15.s, #0
    mov     z16.s, #0
    mov     z17.s, #0
    mov     z18.s, #0
    mov     z19.s, #0
    mov     z20.s, #0
    mov     z21.s, #0
    mov     z22.s, #0
    mov     z23.s, #0

    mov     x22, x0        // A cur
    mov     x23, x1        // B cur
    mov     x24, x2        // KC

    .align 6
.Lkc_outer_fused:
    // set A rows (x4, x10-x14) from A cur
    mov     x4, x22
    add     x10, x22, #256
    add     x11, x22, #512
    add     x12, x22, #768
    add     x13, x22, #1024
    add     x14, x22, #1280
    mov     x5, x23        // B cur
    mov     x6, #8         // inner K counter (256/32)

    .align 6
.Lk_loop_fused:
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
    b.gt    .Lk_loop_fused

    // advance to next kc chunk
    add     x22, x22, #1536
    add     x23, x23, #16384
    subs    x24, x24, #1
    b.gt    .Lkc_outer_fused

    // ================================================================
    // Dequant epilogue: z0-z23 (int32) -> float Y = C * (a_scale[m] * w_scale[n])
    // Multiplication order matches gemm_int8_BTP's SVE dequant EXACTLY
    // (t = a_scale*w_scale per column, then C*t) so results are bit-identical.
    ldr     x3, [sp, #928]   // Y_tile
    ldr     x4, [sp, #936]   // ldc_bytes
    ldr     x5, [sp, #944]   // a_scale_ptr
    ldr     x6, [sp, #952]   // w_scale_ptr
    ldr     w9,  [sp, #960]  // mr (valid rows); x9 is caller-saved scratch
    // w_scale (64 cols = 4 vectors), shared by all 6 rows
    ld1w    {z24.s}, p1/z, [x6, #0, mul vl]
    ld1w    {z25.s}, p1/z, [x6, #1, mul vl]
    ld1w    {z26.s}, p1/z, [x6, #2, mul vl]
    ld1w    {z27.s}, p1/z, [x6, #3, mul vl]
    // a_scale (6 rows) into w-regs (w16-w21)
    ldr     w16, [x5, #0]
    ldr     w17, [x5, #4]
    ldr     w18, [x5, #8]
    ldr     w19, [x5, #12]
    ldr     w20, [x5, #16]
    ldr     w21, [x5, #20]
    // Row 0: z0-z3, a_scale w16  (store only if mr >= 1)
    cmp     w9, #1
    b.lt    .Lfused_r1
    dup     z28.s, w16
    fmul    z29.s, z28.s, z24.s   // t = a_scale*w_scale
    scvtf   z0.s, p1/m, z0.s
    fmul    z0.s, z0.s, z29.s
    st1w    {z0.s}, p1, [x3, #0, mul vl]
    fmul    z29.s, z28.s, z25.s   // t = a_scale*w_scale
    scvtf   z1.s, p1/m, z1.s
    fmul    z1.s, z1.s, z29.s
    st1w    {z1.s}, p1, [x3, #1, mul vl]
    fmul    z29.s, z28.s, z26.s   // t = a_scale*w_scale
    scvtf   z2.s, p1/m, z2.s
    fmul    z2.s, z2.s, z29.s
    st1w    {z2.s}, p1, [x3, #2, mul vl]
    fmul    z29.s, z28.s, z27.s   // t = a_scale*w_scale
    scvtf   z3.s, p1/m, z3.s
    fmul    z3.s, z3.s, z29.s
    st1w    {z3.s}, p1, [x3, #3, mul vl]
.Lfused_r1:
    add     x3, x3, x4
    // Row 1: z4-z7, a_scale w17  (store only if mr >= 2)
    cmp     w9, #2
    b.lt    .Lfused_r2
    dup     z28.s, w17
    fmul    z29.s, z28.s, z24.s   // t = a_scale*w_scale
    scvtf   z4.s, p1/m, z4.s
    fmul    z4.s, z4.s, z29.s
    st1w    {z4.s}, p1, [x3, #0, mul vl]
    fmul    z29.s, z28.s, z25.s   // t = a_scale*w_scale
    scvtf   z5.s, p1/m, z5.s
    fmul    z5.s, z5.s, z29.s
    st1w    {z5.s}, p1, [x3, #1, mul vl]
    fmul    z29.s, z28.s, z26.s   // t = a_scale*w_scale
    scvtf   z6.s, p1/m, z6.s
    fmul    z6.s, z6.s, z29.s
    st1w    {z6.s}, p1, [x3, #2, mul vl]
    fmul    z29.s, z28.s, z27.s   // t = a_scale*w_scale
    scvtf   z7.s, p1/m, z7.s
    fmul    z7.s, z7.s, z29.s
    st1w    {z7.s}, p1, [x3, #3, mul vl]
.Lfused_r2:
    add     x3, x3, x4
    // Row 2: z8-z11, a_scale w18  (store only if mr >= 3)
    cmp     w9, #3
    b.lt    .Lfused_r3
    dup     z28.s, w18
    fmul    z29.s, z28.s, z24.s   // t = a_scale*w_scale
    scvtf   z8.s, p1/m, z8.s
    fmul    z8.s, z8.s, z29.s
    st1w    {z8.s}, p1, [x3, #0, mul vl]
    fmul    z29.s, z28.s, z25.s   // t = a_scale*w_scale
    scvtf   z9.s, p1/m, z9.s
    fmul    z9.s, z9.s, z29.s
    st1w    {z9.s}, p1, [x3, #1, mul vl]
    fmul    z29.s, z28.s, z26.s   // t = a_scale*w_scale
    scvtf   z10.s, p1/m, z10.s
    fmul    z10.s, z10.s, z29.s
    st1w    {z10.s}, p1, [x3, #2, mul vl]
    fmul    z29.s, z28.s, z27.s   // t = a_scale*w_scale
    scvtf   z11.s, p1/m, z11.s
    fmul    z11.s, z11.s, z29.s
    st1w    {z11.s}, p1, [x3, #3, mul vl]
.Lfused_r3:
    add     x3, x3, x4
    // Row 3: z12-z15, a_scale w19  (store only if mr >= 4)
    cmp     w9, #4
    b.lt    .Lfused_r4
    dup     z28.s, w19
    fmul    z29.s, z28.s, z24.s   // t = a_scale*w_scale
    scvtf   z12.s, p1/m, z12.s
    fmul    z12.s, z12.s, z29.s
    st1w    {z12.s}, p1, [x3, #0, mul vl]
    fmul    z29.s, z28.s, z25.s   // t = a_scale*w_scale
    scvtf   z13.s, p1/m, z13.s
    fmul    z13.s, z13.s, z29.s
    st1w    {z13.s}, p1, [x3, #1, mul vl]
    fmul    z29.s, z28.s, z26.s   // t = a_scale*w_scale
    scvtf   z14.s, p1/m, z14.s
    fmul    z14.s, z14.s, z29.s
    st1w    {z14.s}, p1, [x3, #2, mul vl]
    fmul    z29.s, z28.s, z27.s   // t = a_scale*w_scale
    scvtf   z15.s, p1/m, z15.s
    fmul    z15.s, z15.s, z29.s
    st1w    {z15.s}, p1, [x3, #3, mul vl]
.Lfused_r4:
    add     x3, x3, x4
    // Row 4: z16-z19, a_scale w20  (store only if mr >= 5)
    cmp     w9, #5
    b.lt    .Lfused_r5
    dup     z28.s, w20
    fmul    z29.s, z28.s, z24.s   // t = a_scale*w_scale
    scvtf   z16.s, p1/m, z16.s
    fmul    z16.s, z16.s, z29.s
    st1w    {z16.s}, p1, [x3, #0, mul vl]
    fmul    z29.s, z28.s, z25.s   // t = a_scale*w_scale
    scvtf   z17.s, p1/m, z17.s
    fmul    z17.s, z17.s, z29.s
    st1w    {z17.s}, p1, [x3, #1, mul vl]
    fmul    z29.s, z28.s, z26.s   // t = a_scale*w_scale
    scvtf   z18.s, p1/m, z18.s
    fmul    z18.s, z18.s, z29.s
    st1w    {z18.s}, p1, [x3, #2, mul vl]
    fmul    z29.s, z28.s, z27.s   // t = a_scale*w_scale
    scvtf   z19.s, p1/m, z19.s
    fmul    z19.s, z19.s, z29.s
    st1w    {z19.s}, p1, [x3, #3, mul vl]
.Lfused_r5:
    add     x3, x3, x4
    // Row 5: z20-z23, a_scale w21  (store only if mr >= 6)
    cmp     w9, #6
    b.lt    .Lfused_r6
    dup     z28.s, w21
    fmul    z29.s, z28.s, z24.s   // t = a_scale*w_scale
    scvtf   z20.s, p1/m, z20.s
    fmul    z20.s, z20.s, z29.s
    st1w    {z20.s}, p1, [x3, #0, mul vl]
    fmul    z29.s, z28.s, z25.s   // t = a_scale*w_scale
    scvtf   z21.s, p1/m, z21.s
    fmul    z21.s, z21.s, z29.s
    st1w    {z21.s}, p1, [x3, #1, mul vl]
    fmul    z29.s, z28.s, z26.s   // t = a_scale*w_scale
    scvtf   z22.s, p1/m, z22.s
    fmul    z22.s, z22.s, z29.s
    st1w    {z22.s}, p1, [x3, #2, mul vl]
    fmul    z29.s, z28.s, z27.s   // t = a_scale*w_scale
    scvtf   z23.s, p1/m, z23.s
    fmul    z23.s, z23.s, z29.s
    st1w    {z23.s}, p1, [x3, #3, mul vl]
.Lfused_r6:
    // restore
    ldr     d15, [sp, #920]
    ldr     d14, [sp, #912]
    ldr     d13, [sp, #904]
    ldr     d12, [sp, #896]
    ldr     d11, [sp, #888]
    ldr     d10, [sp, #880]
    ldr     d9,  [sp, #872]
    ldr     d8,  [sp, #864]
    ldr     x24, [sp, #840]
    ldr     x23, [sp, #832]
    ldr     x22, [sp, #824]
    ldr     x21, [sp, #816]
    ldr     x20, [sp, #808]
    ldr     x19, [sp, #800]
    ldr     z8,  [sp, #0,  mul vl]
    ldr     z9,  [sp, #1,  mul vl]
    ldr     z10, [sp, #2,  mul vl]
    ldr     z11, [sp, #3,  mul vl]
    ldr     z12, [sp, #4,  mul vl]
    ldr     z13, [sp, #5,  mul vl]
    ldr     z14, [sp, #6,  mul vl]
    ldr     z15, [sp, #7,  mul vl]
    ldr     z28, [sp, #8,  mul vl]
    ldr     z29, [sp, #9,  mul vl]
    ldr     z30, [sp, #10, mul vl]
    ldr     z31, [sp, #11, mul vl]
    add     sp, sp, #976
    ret

    .size kernel_int8_6x4_fused, .-kernel_int8_6x4_fused
