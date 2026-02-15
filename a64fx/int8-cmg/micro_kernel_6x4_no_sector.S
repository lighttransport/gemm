// 6x4 INT8 SDOT Microkernel WITHOUT sector cache hints
// Same as v5 but without L2 streaming mode
// Test if normal caching performs better for L2-sized working sets

    .arch armv8.2-a+sve
    .text
    .align 6

    .global micro_kernel_6x4_no_sector
    .type micro_kernel_6x4_no_sector, @function

// Arguments:
// x0 = A pointer (packed, MR=6, K-interleaved)
// x1 = B pointer (packed, NR=64, 4 vectors per K-group)
// x2 = C pointer
// x3 = K dimension
// x4 = unused
// x5 = C stride (bytes)

micro_kernel_6x4_no_sector:
    stp     x19, x20, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.b

    // NO sector cache hint - use normal caching
    // (removed: orr x1, x1, x19)

    // Zero accumulators z0-z23 using EOR (faster than MOV)
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

    // K/8 iterations (2 K-groups per iteration)
    lsr     x6, x3, #3
    cbz     x6, .Ldone_ns

    // Prologue: Load first A values
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]
    ld1rw   {z28.s}, p0/z, [x0, #16]
    ld1rw   {z29.s}, p0/z, [x0, #20]

    // ========== MAIN LOOP (2 K-groups per iteration) ==========
    .align 6
.Lmain_ns:
    // ===== K-group 0 =====
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]
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

    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    ld1rw   {z24.s}, p0/z, [x0, #24]
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    ld1rw   {z25.s}, p0/z, [x0, #28]
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    ld1rw   {z26.s}, p0/z, [x0, #32]
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    ld1rw   {z27.s}, p0/z, [x0, #36]
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    ld1rw   {z28.s}, p0/z, [x0, #40]
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    ld1rw   {z29.s}, p0/z, [x0, #44]

    // ===== K-group 1 =====
    ld1b    {z30.b}, p0/z, [x1, #4, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #5, mul vl]
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

    ld1b    {z30.b}, p0/z, [x1, #6, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #7, mul vl]
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    ld1rw   {z24.s}, p0/z, [x0, #48]
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    ld1rw   {z25.s}, p0/z, [x0, #52]
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    ld1rw   {z26.s}, p0/z, [x0, #56]
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    ld1rw   {z27.s}, p0/z, [x0, #60]
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    ld1rw   {z28.s}, p0/z, [x0, #64]
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    ld1rw   {z29.s}, p0/z, [x0, #68]

    // Advance pointers
    add     x0, x0, #48
    add     x1, x1, #512

    subs    x6, x6, #1
    b.gt    .Lmain_ns

    // ========== STORE RESULTS ==========
.Ldone_ns:
    ptrue   p0.s
    mov     x7, x2
    mov     x8, x5

    st1w    {z0.s}, p0, [x7, #0, mul vl]
    st1w    {z1.s}, p0, [x7, #1, mul vl]
    st1w    {z2.s}, p0, [x7, #2, mul vl]
    st1w    {z3.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x8

    st1w    {z4.s}, p0, [x7, #0, mul vl]
    st1w    {z5.s}, p0, [x7, #1, mul vl]
    st1w    {z6.s}, p0, [x7, #2, mul vl]
    st1w    {z7.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x8

    st1w    {z8.s}, p0, [x7, #0, mul vl]
    st1w    {z9.s}, p0, [x7, #1, mul vl]
    st1w    {z10.s}, p0, [x7, #2, mul vl]
    st1w    {z11.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x8

    st1w    {z12.s}, p0, [x7, #0, mul vl]
    st1w    {z13.s}, p0, [x7, #1, mul vl]
    st1w    {z14.s}, p0, [x7, #2, mul vl]
    st1w    {z15.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x8

    st1w    {z16.s}, p0, [x7, #0, mul vl]
    st1w    {z17.s}, p0, [x7, #1, mul vl]
    st1w    {z18.s}, p0, [x7, #2, mul vl]
    st1w    {z19.s}, p0, [x7, #3, mul vl]
    add     x7, x7, x8

    st1w    {z20.s}, p0, [x7, #0, mul vl]
    st1w    {z21.s}, p0, [x7, #1, mul vl]
    st1w    {z22.s}, p0, [x7, #2, mul vl]
    st1w    {z23.s}, p0, [x7, #3, mul vl]

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size micro_kernel_6x4_no_sector, .-micro_kernel_6x4_no_sector
