// kernel_6x4_deep.S
// INT8 SDOT Microkernel with 4× unroll and deep software pipelining
// Target: >85% efficiency by hiding L2 latency (~30 cycles)
//
// Strategy:
// - 4× K-group unroll (process 16 K elements per iteration)
// - Load A[k+16:k+31] while computing A[k:k+15]
// - Aggressive prefetch 2KB ahead for L2
// - Interleave B loads between SDOT pairs

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_6x4_deep
    .type kernel_6x4_deep, @function

kernel_6x4_deep:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!

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

    // K/16 iterations (4 K-groups per iter)
    lsr     x5, x3, #4

    // Prefetch initial data
    prfm    pldl1strm, [x0, #0]
    prfm    pldl1strm, [x0, #96]
    prfm    pldl1strm, [x1, #0]
    prfm    pldl1strm, [x1, #256]
    prfm    pldl1strm, [x1, #512]
    prfm    pldl1strm, [x1, #768]

    .align 6
.Lk_loop_deep:
    // Prefetch next iteration (2KB ahead for B, 192B for A)
    prfm    pldl1strm, [x1, #2048]
    prfm    pldl1strm, [x0, #192]

    // ============ K-group 0 ============
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]

    sdot    z0.s, z24.b, z30.b
    sdot    z4.s, z25.b, z30.b
    ld1rw   {z28.s}, p0/z, [x0, #16]
    sdot    z1.s, z24.b, z31.b
    sdot    z5.s, z25.b, z31.b
    ld1rw   {z29.s}, p0/z, [x0, #20]

    sdot    z8.s, z26.b, z30.b
    sdot    z12.s, z27.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z20.s, z29.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z21.s, z29.b, z31.b

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

    // ============ K-group 1 ============
    ld1rw   {z24.s}, p0/z, [x0, #24]
    ld1rw   {z25.s}, p0/z, [x0, #28]
    ld1b    {z30.b}, p0/z, [x1, #4, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #5, mul vl]
    ld1rw   {z26.s}, p0/z, [x0, #32]
    ld1rw   {z27.s}, p0/z, [x0, #36]

    sdot    z0.s, z24.b, z30.b
    sdot    z4.s, z25.b, z30.b
    ld1rw   {z28.s}, p0/z, [x0, #40]
    sdot    z1.s, z24.b, z31.b
    sdot    z5.s, z25.b, z31.b
    ld1rw   {z29.s}, p0/z, [x0, #44]

    sdot    z8.s, z26.b, z30.b
    sdot    z12.s, z27.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z20.s, z29.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z21.s, z29.b, z31.b

    ld1b    {z30.b}, p0/z, [x1, #6, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #7, mul vl]

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

    // ============ K-group 2 ============
    // Advance B base for offset range
    add     x19, x1, #512

    ld1rw   {z24.s}, p0/z, [x0, #48]
    ld1rw   {z25.s}, p0/z, [x0, #52]
    ld1b    {z30.b}, p0/z, [x19, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x19, #1, mul vl]
    ld1rw   {z26.s}, p0/z, [x0, #56]
    ld1rw   {z27.s}, p0/z, [x0, #60]

    sdot    z0.s, z24.b, z30.b
    sdot    z4.s, z25.b, z30.b
    ld1rw   {z28.s}, p0/z, [x0, #64]
    sdot    z1.s, z24.b, z31.b
    sdot    z5.s, z25.b, z31.b
    ld1rw   {z29.s}, p0/z, [x0, #68]

    sdot    z8.s, z26.b, z30.b
    sdot    z12.s, z27.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z20.s, z29.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z21.s, z29.b, z31.b

    ld1b    {z30.b}, p0/z, [x19, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x19, #3, mul vl]

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

    // ============ K-group 3 ============
    ld1rw   {z24.s}, p0/z, [x0, #72]
    ld1rw   {z25.s}, p0/z, [x0, #76]
    ld1b    {z30.b}, p0/z, [x19, #4, mul vl]
    ld1b    {z31.b}, p0/z, [x19, #5, mul vl]
    ld1rw   {z26.s}, p0/z, [x0, #80]
    ld1rw   {z27.s}, p0/z, [x0, #84]

    sdot    z0.s, z24.b, z30.b
    sdot    z4.s, z25.b, z30.b
    ld1rw   {z28.s}, p0/z, [x0, #88]
    sdot    z1.s, z24.b, z31.b
    sdot    z5.s, z25.b, z31.b
    ld1rw   {z29.s}, p0/z, [x0, #92]

    sdot    z8.s, z26.b, z30.b
    sdot    z12.s, z27.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z20.s, z29.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z21.s, z29.b, z31.b

    ld1b    {z30.b}, p0/z, [x19, #6, mul vl]
    ld1b    {z31.b}, p0/z, [x19, #7, mul vl]

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
    add     x0, x0, #96             // 4 K-groups × 24 bytes
    add     x1, x1, #1024           // 4 K-groups × 256 bytes

    subs    x5, x5, #1
    b.gt    .Lk_loop_deep

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

    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size kernel_6x4_deep, .-kernel_6x4_deep
