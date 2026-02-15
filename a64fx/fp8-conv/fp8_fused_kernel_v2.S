/*
 * FP8 GEMM Kernel with Fused Conversion - Version 2
 *
 * Instead of using mov z.s, z.s[i] for broadcast (slow),
 * we store gathered values to stack and use ld1rw (fast).
 *
 * Per K iteration:
 *   1. Load 8 FP8 bytes
 *   2. Gather 8 FP32 from LUT
 *   3. Store to stack (32 bytes)
 *   4. Use ld1rw to broadcast each value
 *   5. 24 FMLA operations
 *
 * Register allocation:
 *   Z0-Z23:  24 accumulators
 *   Z24-Z26: 3 B vectors
 *   Z27-Z30: 4 A broadcast values
 *   Z31:     temp for gather
 *
 * Arguments:
 *   x0: A_fp8 pointer ([K][MR] packed)
 *   x1: B_fp32 pointer ([K][N])
 *   x2: C pointer
 *   x3: ldc in bytes
 *   x4: K
 *   x5: LUT pointer
 */

    .arch armv8.2-a+sve
    .text
    .align 4
    .global fp8_fused_kernel_v2
    .type fp8_fused_kernel_v2, %function

fp8_fused_kernel_v2:
    // Save frame pointer and allocate stack for A temps
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    // Stack layout: [sp+16] = 32 bytes for 8 floats

    ptrue p0.s

    // Initialize 24 accumulators
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

    // Predicate for 8 elements
    mov x6, #8
    whilelt p1.s, xzr, x6

    // K loop counter
    mov x6, x4
    cbz x6, .Lv2_store

    // Pointer to stack temp area
    add x7, sp, #16

    // Load first B vectors
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]

.Lv2_loop:
    // Load 8 FP8 bytes
    ld1b {z31.s}, p1/z, [x0]

    // Gather FP32 from LUT
    ld1w {z31.s}, p1/z, [x5, z31.s, uxtw #2]

    // Store 8 floats to stack
    st1w {z31.s}, p1, [x7]

    // Load with broadcast and compute
    // Row 0
    ld1rw {z27.s}, p0/z, [x7, #0]
    fmla z0.s, p0/m, z27.s, z24.s
    fmla z1.s, p0/m, z27.s, z25.s
    fmla z2.s, p0/m, z27.s, z26.s

    // Row 1
    ld1rw {z28.s}, p0/z, [x7, #4]
    fmla z3.s, p0/m, z28.s, z24.s
    fmla z4.s, p0/m, z28.s, z25.s
    fmla z5.s, p0/m, z28.s, z26.s

    // Row 2
    ld1rw {z29.s}, p0/z, [x7, #8]
    fmla z6.s, p0/m, z29.s, z24.s
    fmla z7.s, p0/m, z29.s, z25.s
    fmla z8.s, p0/m, z29.s, z26.s

    // Row 3
    ld1rw {z30.s}, p0/z, [x7, #12]
    fmla z9.s, p0/m, z30.s, z24.s
    fmla z10.s, p0/m, z30.s, z25.s
    fmla z11.s, p0/m, z30.s, z26.s

    // Row 4
    ld1rw {z27.s}, p0/z, [x7, #16]
    fmla z12.s, p0/m, z27.s, z24.s
    fmla z13.s, p0/m, z27.s, z25.s
    fmla z14.s, p0/m, z27.s, z26.s

    // Row 5
    ld1rw {z28.s}, p0/z, [x7, #20]
    fmla z15.s, p0/m, z28.s, z24.s
    fmla z16.s, p0/m, z28.s, z25.s
    fmla z17.s, p0/m, z28.s, z26.s

    // Row 6
    ld1rw {z29.s}, p0/z, [x7, #24]
    fmla z18.s, p0/m, z29.s, z24.s
    fmla z19.s, p0/m, z29.s, z25.s
    fmla z20.s, p0/m, z29.s, z26.s

    // Row 7
    ld1rw {z30.s}, p0/z, [x7, #28]
    fmla z21.s, p0/m, z30.s, z24.s
    fmla z22.s, p0/m, z30.s, z25.s
    fmla z23.s, p0/m, z30.s, z26.s

    // Advance pointers
    add x0, x0, #8     // A: 8 bytes
    add x1, x1, #192   // B: 3 vectors

    subs x6, x6, #1
    beq .Lv2_store

    // Load next B
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]

    b .Lv2_loop

.Lv2_store:
    // Store results
    st1w {z0.s}, p0, [x2]
    st1w {z1.s}, p0, [x2, #1, mul vl]
    st1w {z2.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z3.s}, p0, [x2]
    st1w {z4.s}, p0, [x2, #1, mul vl]
    st1w {z5.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z6.s}, p0, [x2]
    st1w {z7.s}, p0, [x2, #1, mul vl]
    st1w {z8.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z9.s}, p0, [x2]
    st1w {z10.s}, p0, [x2, #1, mul vl]
    st1w {z11.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z12.s}, p0, [x2]
    st1w {z13.s}, p0, [x2, #1, mul vl]
    st1w {z14.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z15.s}, p0, [x2]
    st1w {z16.s}, p0, [x2, #1, mul vl]
    st1w {z17.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z18.s}, p0, [x2]
    st1w {z19.s}, p0, [x2, #1, mul vl]
    st1w {z20.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z21.s}, p0, [x2]
    st1w {z22.s}, p0, [x2, #1, mul vl]
    st1w {z23.s}, p0, [x2, #2, mul vl]

    // Restore frame
    ldp x29, x30, [sp], #48
    ret
    .size fp8_fused_kernel_v2, .-fp8_fused_kernel_v2
