// kernel_6x4_prfm.S
// Baseline kernel with aggressive prefetching
// Simple approach: add prefetch instructions to existing efficient kernel

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_6x4_prfm
    .type kernel_6x4_prfm, @function

kernel_6x4_prfm:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!

    ptrue   p0.b

    // Zero accumulators
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

    // K/4 iterations
    lsr     x5, x3, #2

    // Initial prefetch burst for L2 (A64FX L2 latency ~30 cycles)
    prfm    pldl1strm, [x0, #0]
    prfm    pldl1strm, [x0, #64]
    prfm    pldl1strm, [x1, #0]
    prfm    pldl1strm, [x1, #256]
    prfm    pldl1strm, [x1, #512]
    prfm    pldl1strm, [x1, #768]

    .align 6
.Lk_loop_prfm:
    // Prefetch ahead: A needs 24 bytes/iter, B needs 256 bytes/iter
    // Prefetch ~8 iterations ahead for B (2KB)
    prfm    pldl1strm, [x1, #2048]
    // Prefetch ~10 iterations ahead for A (240 bytes)
    prfm    pldl1strm, [x0, #256]

    // Load A (6 rows × 4 bytes each)
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]
    ld1rw   {z28.s}, p0/z, [x0, #16]
    ld1rw   {z29.s}, p0/z, [x0, #20]

    // Load B cols 0-1 and compute
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]

    sdot    z0.s, z24.b, z30.b
    sdot    z4.s, z25.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z12.s, z27.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z20.s, z29.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z21.s, z29.b, z31.b

    // Load B cols 2-3 and compute
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]

    sdot    z2.s, z24.b, z30.b
    sdot    z6.s, z25.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z14.s, z27.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z22.s, z29.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z23.s, z29.b, z31.b

    // Advance pointers
    add     x0, x0, #24
    add     x1, x1, #256

    subs    x5, x5, #1
    b.gt    .Lk_loop_prfm

    // Store results
    ptrue   p0.s

    st1w    {z0.s}, p0, [x2, #0, mul vl]
    st1w    {z1.s}, p0, [x2, #1, mul vl]
    st1w    {z2.s}, p0, [x2, #2, mul vl]
    st1w    {z3.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4

    st1w    {z4.s}, p0, [x2, #0, mul vl]
    st1w    {z5.s}, p0, [x2, #1, mul vl]
    st1w    {z6.s}, p0, [x2, #2, mul vl]
    st1w    {z7.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4

    st1w    {z8.s}, p0, [x2, #0, mul vl]
    st1w    {z9.s}, p0, [x2, #1, mul vl]
    st1w    {z10.s}, p0, [x2, #2, mul vl]
    st1w    {z11.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4

    st1w    {z12.s}, p0, [x2, #0, mul vl]
    st1w    {z13.s}, p0, [x2, #1, mul vl]
    st1w    {z14.s}, p0, [x2, #2, mul vl]
    st1w    {z15.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4

    st1w    {z16.s}, p0, [x2, #0, mul vl]
    st1w    {z17.s}, p0, [x2, #1, mul vl]
    st1w    {z18.s}, p0, [x2, #2, mul vl]
    st1w    {z19.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4

    st1w    {z20.s}, p0, [x2, #0, mul vl]
    st1w    {z21.s}, p0, [x2, #1, mul vl]
    st1w    {z22.s}, p0, [x2, #2, mul vl]
    st1w    {z23.s}, p0, [x2, #3, mul vl]

    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size kernel_6x4_prfm, .-kernel_6x4_prfm
