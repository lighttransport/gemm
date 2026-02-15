// FP32 FMLA Micro-kernel: 6×4 broadcast tile
// 6 rows (broadcast A) × 4 column vectors (64 FP32 outputs per row)
// Total: 6 × 64 = 384 FP32 outputs, 24 FMLAs per K
//
// Arithmetic intensity: 768 / 280 = 2.74 FLOPS/byte
//   A: 6 × 4 = 24 bytes/K, B: 4 × 64 = 256 bytes/K
//
// Input format:
//   A: packed [K][6] column-major (6 FP32 per K step, 24 bytes)
//   B: packed [K][64] row-major (4 SVE vectors per K step, 256 bytes)
//   C: [6][64] row-major with ldc stride between rows
//   K: must be even (2× unroll)
//
// Register allocation:
//   Z0-Z3:   accum row 0, col vectors 0-3
//   Z4-Z7:   accum row 1, col vectors 0-3
//   Z8-Z11:  accum row 2, col vectors 0-3
//   Z12-Z15: accum row 3, col vectors 0-3
//   Z16-Z19: accum row 4, col vectors 0-3
//   Z20-Z23: accum row 5, col vectors 0-3
//   Z24-Z27: B operands (4 column vectors)
//   Z28-Z31: A operands (broadcast scalars, 4 at a time)
//
// Interface:
//   x0: A (float*), packed [K][6]
//   x1: B (float*), packed [K][64]
//   x2: C (float*), output [6][64] with stride x5
//   x3: K (must be even)
//   x4: unused
//   x5: ldc in bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp32_6x4_bcast
    .type micro_kernel_fp32_6x4_bcast, %function

micro_kernel_fp32_6x4_bcast:
    // Save callee-saved D8-D15
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.s

    // Zero 24 accumulators
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    eor z15.d, z15.d, z15.d
    eor z16.d, z16.d, z16.d
    eor z17.d, z17.d, z17.d
    eor z18.d, z18.d, z18.d
    eor z19.d, z19.d, z19.d
    eor z20.d, z20.d, z20.d
    eor z21.d, z21.d, z21.d
    eor z22.d, z22.d, z22.d
    eor z23.d, z23.d, z23.d

    // Load initial B (4 column vectors)
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]
    ld1w {z27.s}, p0/z, [x1, #3, mul vl]

    // K loop counter (process 2 K per iteration)
    lsr x6, x3, #1

.L6x4_loop:
    // ═══════════════════════════════════════════════════
    // K iteration 0
    // ═══════════════════════════════════════════════════

    // Load A[0..3][k] broadcast
    ld1rw {z28.s}, p0/z, [x0, #0]    // A[0,k]
    ld1rw {z29.s}, p0/z, [x0, #4]    // A[1,k]
    ld1rw {z30.s}, p0/z, [x0, #8]    // A[2,k]
    ld1rw {z31.s}, p0/z, [x0, #12]   // A[3,k]

    // Row 0: C[0,:] += A[0,k] * B[k,:]
    fmla z0.s, p0/m, z28.s, z24.s
    fmla z1.s, p0/m, z28.s, z25.s
    fmla z2.s, p0/m, z28.s, z26.s
    fmla z3.s, p0/m, z28.s, z27.s

    // Row 1
    fmla z4.s, p0/m, z29.s, z24.s
    fmla z5.s, p0/m, z29.s, z25.s
    fmla z6.s, p0/m, z29.s, z26.s
    fmla z7.s, p0/m, z29.s, z27.s

    // Row 2
    fmla z8.s, p0/m, z30.s, z24.s
    fmla z9.s, p0/m, z30.s, z25.s
    fmla z10.s, p0/m, z30.s, z26.s
    fmla z11.s, p0/m, z30.s, z27.s

    // Row 3
    fmla z12.s, p0/m, z31.s, z24.s
    fmla z13.s, p0/m, z31.s, z25.s
    fmla z14.s, p0/m, z31.s, z26.s
    fmla z15.s, p0/m, z31.s, z27.s

    // Load A[4..5][k] broadcast
    ld1rw {z28.s}, p0/z, [x0, #16]   // A[4,k]
    ld1rw {z29.s}, p0/z, [x0, #20]   // A[5,k]

    // Row 4
    fmla z16.s, p0/m, z28.s, z24.s
    fmla z17.s, p0/m, z28.s, z25.s
    fmla z18.s, p0/m, z28.s, z26.s
    fmla z19.s, p0/m, z28.s, z27.s

    // Row 5
    fmla z20.s, p0/m, z29.s, z24.s
    fmla z21.s, p0/m, z29.s, z25.s
    fmla z22.s, p0/m, z29.s, z26.s
    fmla z23.s, p0/m, z29.s, z27.s

    // Load next B (k+1)
    ld1w {z24.s}, p0/z, [x1, #4, mul vl]
    ld1w {z25.s}, p0/z, [x1, #5, mul vl]
    ld1w {z26.s}, p0/z, [x1, #6, mul vl]
    ld1w {z27.s}, p0/z, [x1, #7, mul vl]

    // ═══════════════════════════════════════════════════
    // K iteration 1
    // ═══════════════════════════════════════════════════

    // Load A[0..3][k+1] broadcast
    ld1rw {z28.s}, p0/z, [x0, #24]   // A[0,k+1]
    ld1rw {z29.s}, p0/z, [x0, #28]   // A[1,k+1]
    ld1rw {z30.s}, p0/z, [x0, #32]   // A[2,k+1]
    ld1rw {z31.s}, p0/z, [x0, #36]   // A[3,k+1]

    // Row 0
    fmla z0.s, p0/m, z28.s, z24.s
    fmla z1.s, p0/m, z28.s, z25.s
    fmla z2.s, p0/m, z28.s, z26.s
    fmla z3.s, p0/m, z28.s, z27.s

    // Row 1
    fmla z4.s, p0/m, z29.s, z24.s
    fmla z5.s, p0/m, z29.s, z25.s
    fmla z6.s, p0/m, z29.s, z26.s
    fmla z7.s, p0/m, z29.s, z27.s

    // Row 2
    fmla z8.s, p0/m, z30.s, z24.s
    fmla z9.s, p0/m, z30.s, z25.s
    fmla z10.s, p0/m, z30.s, z26.s
    fmla z11.s, p0/m, z30.s, z27.s

    // Row 3
    fmla z12.s, p0/m, z31.s, z24.s
    fmla z13.s, p0/m, z31.s, z25.s
    fmla z14.s, p0/m, z31.s, z26.s
    fmla z15.s, p0/m, z31.s, z27.s

    // Load A[4..5][k+1] broadcast
    ld1rw {z28.s}, p0/z, [x0, #40]   // A[4,k+1]
    ld1rw {z29.s}, p0/z, [x0, #44]   // A[5,k+1]

    // Row 4
    fmla z16.s, p0/m, z28.s, z24.s
    fmla z17.s, p0/m, z28.s, z25.s
    fmla z18.s, p0/m, z28.s, z26.s
    fmla z19.s, p0/m, z28.s, z27.s

    // Row 5
    fmla z20.s, p0/m, z29.s, z24.s
    fmla z21.s, p0/m, z29.s, z25.s
    fmla z22.s, p0/m, z29.s, z26.s
    fmla z23.s, p0/m, z29.s, z27.s

    // Advance pointers: A += 2*6*4=48, B += 2*4*64=512
    add x0, x0, #48
    add x1, x1, #512

    subs x6, x6, #1
    beq .L6x4_store

    // Load next B for next iteration
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]
    ld1w {z27.s}, p0/z, [x1, #3, mul vl]

    b .L6x4_loop

.L6x4_store:
    // Store 6 rows × 4 column vectors
    // Row 0
    st1w {z0.s}, p0, [x2]
    st1w {z1.s}, p0, [x2, #1, mul vl]
    st1w {z2.s}, p0, [x2, #2, mul vl]
    st1w {z3.s}, p0, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 1
    st1w {z4.s}, p0, [x2]
    st1w {z5.s}, p0, [x2, #1, mul vl]
    st1w {z6.s}, p0, [x2, #2, mul vl]
    st1w {z7.s}, p0, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 2
    st1w {z8.s}, p0, [x2]
    st1w {z9.s}, p0, [x2, #1, mul vl]
    st1w {z10.s}, p0, [x2, #2, mul vl]
    st1w {z11.s}, p0, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 3
    st1w {z12.s}, p0, [x2]
    st1w {z13.s}, p0, [x2, #1, mul vl]
    st1w {z14.s}, p0, [x2, #2, mul vl]
    st1w {z15.s}, p0, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 4
    st1w {z16.s}, p0, [x2]
    st1w {z17.s}, p0, [x2, #1, mul vl]
    st1w {z18.s}, p0, [x2, #2, mul vl]
    st1w {z19.s}, p0, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 5
    st1w {z20.s}, p0, [x2]
    st1w {z21.s}, p0, [x2, #1, mul vl]
    st1w {z22.s}, p0, [x2, #2, mul vl]
    st1w {z23.s}, p0, [x2, #3, mul vl]

    // Restore callee-saved D8-D15
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp32_6x4_bcast, .-micro_kernel_fp32_6x4_bcast
