// int16_sdot_kernel.S
// INT16 SDOT GEMM kernel for A64FX SVE
// 5(rows) x 4(simd) x 2(double-buffering) blocking
//
// INT16 SDOT: SDOT Zda.D, Zn.H, Zm.H
//   - Accumulates 2 int16 products into int64
//   - 8 int64 lanes per 512-bit vector
//   - 32 int16 elements per vector
//
// Register allocation:
//   z0-z19:  5x4 = 20 accumulators (int64)
//   z20-z24: A row loads buffer 0 (5 rows)
//   z25-z29: A row loads buffer 1 (partial)
//   z28-z31: B column loads

    .arch armv8.2-a+sve
    .text

// ============================================
// L1 Throughput Kernel: int16_sdot_kernel_5x4_l1
// ============================================
// Args:
//   x0: A pointer (L1 resident)
//   x1: B pointer (L1 resident)
//   x2: C pointer
//   x3: iterations

    .align 6
    .global int16_sdot_kernel_5x4_l1
    .type int16_sdot_kernel_5x4_l1, %function

int16_sdot_kernel_5x4_l1:
    stp     x19, x20, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.h                    // All int16 lanes
    ptrue   p1.d                    // All int64 lanes

    // Initialize 20 accumulators to zero
    fmov    z0.d, #0
    fmov    z1.d, #0
    fmov    z2.d, #0
    fmov    z3.d, #0
    fmov    z4.d, #0
    fmov    z5.d, #0
    fmov    z6.d, #0
    fmov    z7.d, #0
    fmov    z8.d, #0
    fmov    z9.d, #0
    fmov    z10.d, #0
    fmov    z11.d, #0
    fmov    z12.d, #0
    fmov    z13.d, #0
    fmov    z14.d, #0
    fmov    z15.d, #0
    fmov    z16.d, #0
    fmov    z17.d, #0
    fmov    z18.d, #0
    fmov    z19.d, #0

    mov     x4, x0                  // A base
    mov     x5, x1                  // B base

    // Initial loads
    // A: use ld1rqh to load 8 int16 (16 bytes) and replicate
    ld1rqh  z20.h, p0/z, [x4, #0]   // A row 0
    ld1rqh  z21.h, p0/z, [x4, #16]  // A row 1
    ld1rqh  z22.h, p0/z, [x4, #32]  // A row 2
    ld1rqh  z23.h, p0/z, [x4, #48]  // A row 3
    ld1rqh  z24.h, p0/z, [x4, #64]  // A row 4

    // B: 4 vectors
    ld1h    z28.h, p0/z, [x5, #0, mul vl]
    ld1h    z29.h, p0/z, [x5, #1, mul vl]
    ld1h    z30.h, p0/z, [x5, #2, mul vl]
    ld1h    z31.h, p0/z, [x5, #3, mul vl]

    lsr     x3, x3, #1              // iterations / 2
    cbz     x3, .Ll1_16_epilog

    .align 6
.Ll1_16_loop:
    // === Iteration 0: Compute with buffer 0, load buffer 1 ===
    ld1rqh  z25.h, p0/z, [x4, #0]
    sdot    z0.d, z20.h, z28.h
    sdot    z1.d, z20.h, z29.h
    sdot    z2.d, z20.h, z30.h
    sdot    z3.d, z20.h, z31.h

    ld1rqh  z26.h, p0/z, [x4, #16]
    sdot    z4.d, z21.h, z28.h
    sdot    z5.d, z21.h, z29.h
    sdot    z6.d, z21.h, z30.h
    sdot    z7.d, z21.h, z31.h

    ld1rqh  z27.h, p0/z, [x4, #32]
    sdot    z8.d, z22.h, z28.h
    sdot    z9.d, z22.h, z29.h
    sdot    z10.d, z22.h, z30.h
    sdot    z11.d, z22.h, z31.h

    ld1rqh  z20.h, p0/z, [x4, #48]
    sdot    z12.d, z23.h, z28.h
    sdot    z13.d, z23.h, z29.h
    sdot    z14.d, z23.h, z30.h
    sdot    z15.d, z23.h, z31.h

    ld1rqh  z21.h, p0/z, [x4, #64]
    sdot    z16.d, z24.h, z28.h
    sdot    z17.d, z24.h, z29.h
    sdot    z18.d, z24.h, z30.h
    sdot    z19.d, z24.h, z31.h

    // === Iteration 1: Compute with buffer 1, load buffer 0 ===
    ld1rqh  z22.h, p0/z, [x4, #0]
    sdot    z0.d, z25.h, z28.h
    sdot    z1.d, z25.h, z29.h
    sdot    z2.d, z25.h, z30.h
    sdot    z3.d, z25.h, z31.h

    ld1rqh  z23.h, p0/z, [x4, #16]
    sdot    z4.d, z26.h, z28.h
    sdot    z5.d, z26.h, z29.h
    sdot    z6.d, z26.h, z30.h
    sdot    z7.d, z26.h, z31.h

    ld1rqh  z24.h, p0/z, [x4, #32]
    sdot    z8.d, z27.h, z28.h
    sdot    z9.d, z27.h, z29.h
    sdot    z10.d, z27.h, z30.h
    sdot    z11.d, z27.h, z31.h

    ld1rqh  z20.h, p0/z, [x4, #48]
    sdot    z12.d, z20.h, z28.h
    sdot    z13.d, z20.h, z29.h
    sdot    z14.d, z20.h, z30.h
    sdot    z15.d, z20.h, z31.h

    ld1rqh  z21.h, p0/z, [x4, #64]
    sdot    z16.d, z21.h, z28.h
    sdot    z17.d, z21.h, z29.h
    sdot    z18.d, z21.h, z30.h
    sdot    z19.d, z21.h, z31.h

    // Reload for next iteration
    ld1rqh  z20.h, p0/z, [x4, #0]
    ld1rqh  z21.h, p0/z, [x4, #16]
    ld1rqh  z22.h, p0/z, [x4, #32]
    ld1rqh  z23.h, p0/z, [x4, #48]
    ld1rqh  z24.h, p0/z, [x4, #64]

    subs    x3, x3, #1
    b.ne    .Ll1_16_loop

.Ll1_16_epilog:
    // Store results
    mov     x6, #256                // Row stride = 32 int64 * 8 bytes

    st1d    z0.d, p1, [x2, #0, mul vl]
    st1d    z1.d, p1, [x2, #1, mul vl]
    st1d    z2.d, p1, [x2, #2, mul vl]
    st1d    z3.d, p1, [x2, #3, mul vl]
    add     x2, x2, x6

    st1d    z4.d, p1, [x2, #0, mul vl]
    st1d    z5.d, p1, [x2, #1, mul vl]
    st1d    z6.d, p1, [x2, #2, mul vl]
    st1d    z7.d, p1, [x2, #3, mul vl]
    add     x2, x2, x6

    st1d    z8.d, p1, [x2, #0, mul vl]
    st1d    z9.d, p1, [x2, #1, mul vl]
    st1d    z10.d, p1, [x2, #2, mul vl]
    st1d    z11.d, p1, [x2, #3, mul vl]
    add     x2, x2, x6

    st1d    z12.d, p1, [x2, #0, mul vl]
    st1d    z13.d, p1, [x2, #1, mul vl]
    st1d    z14.d, p1, [x2, #2, mul vl]
    st1d    z15.d, p1, [x2, #3, mul vl]
    add     x2, x2, x6

    st1d    z16.d, p1, [x2, #0, mul vl]
    st1d    z17.d, p1, [x2, #1, mul vl]
    st1d    z18.d, p1, [x2, #2, mul vl]
    st1d    z19.d, p1, [x2, #3, mul vl]

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x19, x20, [sp], #16

    ret
    .size int16_sdot_kernel_5x4_l1, .-int16_sdot_kernel_5x4_l1


// ============================================
// Optimized L1 Kernel: Pure SDOT loop
// ============================================
// All data pre-loaded, just 20 SDOTs per iteration

    .align 6
    .global int16_sdot_kernel_5x4_l1_opt
    .type int16_sdot_kernel_5x4_l1_opt, %function

int16_sdot_kernel_5x4_l1_opt:
    stp     x19, x20, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.h
    ptrue   p1.d

    // Initialize accumulators
    fmov    z0.d, #0
    fmov    z1.d, #0
    fmov    z2.d, #0
    fmov    z3.d, #0
    fmov    z4.d, #0
    fmov    z5.d, #0
    fmov    z6.d, #0
    fmov    z7.d, #0
    fmov    z8.d, #0
    fmov    z9.d, #0
    fmov    z10.d, #0
    fmov    z11.d, #0
    fmov    z12.d, #0
    fmov    z13.d, #0
    fmov    z14.d, #0
    fmov    z15.d, #0
    fmov    z16.d, #0
    fmov    z17.d, #0
    fmov    z18.d, #0
    fmov    z19.d, #0

    // Load A and B (constant throughout)
    ld1rqh  z20.h, p0/z, [x0, #0]
    ld1rqh  z21.h, p0/z, [x0, #16]
    ld1rqh  z22.h, p0/z, [x0, #32]
    ld1rqh  z23.h, p0/z, [x0, #48]
    ld1rqh  z24.h, p0/z, [x0, #64]

    ld1h    z28.h, p0/z, [x1, #0, mul vl]
    ld1h    z29.h, p0/z, [x1, #1, mul vl]
    ld1h    z30.h, p0/z, [x1, #2, mul vl]
    ld1h    z31.h, p0/z, [x1, #3, mul vl]

    cbz     x3, .Lopt_16_epilog

    .align 6
.Lopt_16_loop:
    // All 20 SDOTs
    sdot    z0.d, z20.h, z28.h
    sdot    z1.d, z20.h, z29.h
    sdot    z2.d, z20.h, z30.h
    sdot    z3.d, z20.h, z31.h

    sdot    z4.d, z21.h, z28.h
    sdot    z5.d, z21.h, z29.h
    sdot    z6.d, z21.h, z30.h
    sdot    z7.d, z21.h, z31.h

    sdot    z8.d, z22.h, z28.h
    sdot    z9.d, z22.h, z29.h
    sdot    z10.d, z22.h, z30.h
    sdot    z11.d, z22.h, z31.h

    sdot    z12.d, z23.h, z28.h
    sdot    z13.d, z23.h, z29.h
    sdot    z14.d, z23.h, z30.h
    sdot    z15.d, z23.h, z31.h

    sdot    z16.d, z24.h, z28.h
    sdot    z17.d, z24.h, z29.h
    sdot    z18.d, z24.h, z30.h
    sdot    z19.d, z24.h, z31.h

    subs    x3, x3, #1
    b.ne    .Lopt_16_loop

.Lopt_16_epilog:
    mov     x6, #256

    st1d    z0.d, p1, [x2, #0, mul vl]
    st1d    z1.d, p1, [x2, #1, mul vl]
    st1d    z2.d, p1, [x2, #2, mul vl]
    st1d    z3.d, p1, [x2, #3, mul vl]
    add     x2, x2, x6

    st1d    z4.d, p1, [x2, #0, mul vl]
    st1d    z5.d, p1, [x2, #1, mul vl]
    st1d    z6.d, p1, [x2, #2, mul vl]
    st1d    z7.d, p1, [x2, #3, mul vl]
    add     x2, x2, x6

    st1d    z8.d, p1, [x2, #0, mul vl]
    st1d    z9.d, p1, [x2, #1, mul vl]
    st1d    z10.d, p1, [x2, #2, mul vl]
    st1d    z11.d, p1, [x2, #3, mul vl]
    add     x2, x2, x6

    st1d    z12.d, p1, [x2, #0, mul vl]
    st1d    z13.d, p1, [x2, #1, mul vl]
    st1d    z14.d, p1, [x2, #2, mul vl]
    st1d    z15.d, p1, [x2, #3, mul vl]
    add     x2, x2, x6

    st1d    z16.d, p1, [x2, #0, mul vl]
    st1d    z17.d, p1, [x2, #1, mul vl]
    st1d    z18.d, p1, [x2, #2, mul vl]
    st1d    z19.d, p1, [x2, #3, mul vl]

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x19, x20, [sp], #16

    ret
    .size int16_sdot_kernel_5x4_l1_opt, .-int16_sdot_kernel_5x4_l1_opt
