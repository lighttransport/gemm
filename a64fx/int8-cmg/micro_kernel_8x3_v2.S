// 8x3 INT8 SDOT Microkernel v2 - Register-efficient approach
// 8 rows x 3 vectors = 24 accumulators
// Strategy: Process in 2 phases of 4 rows each, reusing registers
// Phase 1: rows 0-3 with B[0-2]
// Phase 2: rows 4-7 with B[0-2] (B reloaded)

    .arch armv8.2-a+sve
    .text
    .align 6

    .global micro_kernel_8x3_v2
    .type micro_kernel_8x3_v2, @function

// Arguments:
// x0 = A pointer (packed, MR=8)
// x1 = B pointer (packed, NR=48)
// x2 = C pointer
// x3 = K (dimension)
// x4 = unused
// x5 = C stride (bytes)

micro_kernel_8x3_v2:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.b

    // Sector cache hint for B
    mov     x19, #1
    lsl     x19, x19, #56
    orr     x1, x1, x19

    // Save original pointers
    mov     x20, x0                 // A base
    mov     x21, x1                 // B base

    // Zero 24 accumulators (z0-z23)
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    eor     z16.d, z16.d, z16.d
    eor     z17.d, z17.d, z17.d
    eor     z18.d, z18.d, z18.d
    eor     z19.d, z19.d, z19.d
    eor     z20.d, z20.d, z20.d
    eor     z21.d, z21.d, z21.d
    eor     z22.d, z22.d, z22.d
    eor     z23.d, z23.d, z23.d

    // K/4 iterations
    lsr     x6, x3, #2
    cbz     x6, .Ldone_8x3v2

    // Register allocation for loop:
    // z0-z11: rows 0-3, columns 0-2 (12 accumulators)
    // z12-z23: rows 4-7, columns 0-2 (12 accumulators)
    // z24-z27: A[0-3] or A[4-7] (4 rows at a time)
    // z28-z30: B[0-2] (3 columns)
    // z31: spare

    // ========== MAIN LOOP ==========
    .align 6
.Lmain_8x3v2:
    // === Phase 1: Rows 0-3 ===
    // Load A[0-3]
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]

    // Load B[0-2]
    ld1b    {z28.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z29.b}, p0/z, [x1, #1, mul vl]
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]

    // Row 0: z0, z1, z2
    sdot    z0.s, z24.b, z28.b
    sdot    z1.s, z24.b, z29.b
    sdot    z2.s, z24.b, z30.b

    // Row 1: z3, z4, z5
    sdot    z3.s, z25.b, z28.b
    sdot    z4.s, z25.b, z29.b
    sdot    z5.s, z25.b, z30.b

    // Row 2: z6, z7, z8
    sdot    z6.s, z26.b, z28.b
    sdot    z7.s, z26.b, z29.b
    sdot    z8.s, z26.b, z30.b

    // Row 3: z9, z10, z11
    sdot    z9.s, z27.b, z28.b
    sdot    z10.s, z27.b, z29.b
    sdot    z11.s, z27.b, z30.b

    // === Phase 2: Rows 4-7 ===
    // Load A[4-7]
    ld1rw   {z24.s}, p0/z, [x0, #16]
    ld1rw   {z25.s}, p0/z, [x0, #20]
    ld1rw   {z26.s}, p0/z, [x0, #24]
    ld1rw   {z27.s}, p0/z, [x0, #28]

    // B is still in z28-z30

    // Row 4: z12, z13, z14
    sdot    z12.s, z24.b, z28.b
    sdot    z13.s, z24.b, z29.b
    sdot    z14.s, z24.b, z30.b

    // Row 5: z15, z16, z17
    sdot    z15.s, z25.b, z28.b
    sdot    z16.s, z25.b, z29.b
    sdot    z17.s, z25.b, z30.b

    // Row 6: z18, z19, z20
    sdot    z18.s, z26.b, z28.b
    sdot    z19.s, z26.b, z29.b
    sdot    z20.s, z26.b, z30.b

    // Row 7: z21, z22, z23
    sdot    z21.s, z27.b, z28.b
    sdot    z22.s, z27.b, z29.b
    sdot    z23.s, z27.b, z30.b

    // Advance pointers
    add     x0, x0, #32             // 8 * 4 bytes
    add     x1, x1, #192            // 3 * 64 bytes

    subs    x6, x6, #1
    b.gt    .Lmain_8x3v2

    // ========== STORE RESULTS ==========
.Ldone_8x3v2:
    ptrue   p0.s
    mov     x7, x2
    mov     x8, x5

    // Row 0
    st1w    {z0.s}, p0, [x7, #0, mul vl]
    st1w    {z1.s}, p0, [x7, #1, mul vl]
    st1w    {z2.s}, p0, [x7, #2, mul vl]
    add     x7, x7, x8

    // Row 1
    st1w    {z3.s}, p0, [x7, #0, mul vl]
    st1w    {z4.s}, p0, [x7, #1, mul vl]
    st1w    {z5.s}, p0, [x7, #2, mul vl]
    add     x7, x7, x8

    // Row 2
    st1w    {z6.s}, p0, [x7, #0, mul vl]
    st1w    {z7.s}, p0, [x7, #1, mul vl]
    st1w    {z8.s}, p0, [x7, #2, mul vl]
    add     x7, x7, x8

    // Row 3
    st1w    {z9.s}, p0, [x7, #0, mul vl]
    st1w    {z10.s}, p0, [x7, #1, mul vl]
    st1w    {z11.s}, p0, [x7, #2, mul vl]
    add     x7, x7, x8

    // Row 4
    st1w    {z12.s}, p0, [x7, #0, mul vl]
    st1w    {z13.s}, p0, [x7, #1, mul vl]
    st1w    {z14.s}, p0, [x7, #2, mul vl]
    add     x7, x7, x8

    // Row 5
    st1w    {z15.s}, p0, [x7, #0, mul vl]
    st1w    {z16.s}, p0, [x7, #1, mul vl]
    st1w    {z17.s}, p0, [x7, #2, mul vl]
    add     x7, x7, x8

    // Row 6
    st1w    {z18.s}, p0, [x7, #0, mul vl]
    st1w    {z19.s}, p0, [x7, #1, mul vl]
    st1w    {z20.s}, p0, [x7, #2, mul vl]
    add     x7, x7, x8

    // Row 7
    st1w    {z21.s}, p0, [x7, #0, mul vl]
    st1w    {z22.s}, p0, [x7, #1, mul vl]
    st1w    {z23.s}, p0, [x7, #2, mul vl]

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size micro_kernel_8x3_v2, .-micro_kernel_8x3_v2
