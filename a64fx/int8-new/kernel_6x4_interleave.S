// kernel_6x4_interleave.S
// Better interleaved kernel - spread A loads across SDOT sequence
// Goal: Start SDOT earlier while other A loads are still in flight
//
// Observation: Current kernel loads all 6 A before any SDOT
// This wastes cycles waiting for all A to be ready
//
// New approach: Load A[0:1], B[0:1], start SDOT, load A[2:3], more SDOT, etc.

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_6x4_interleave
    .type kernel_6x4_interleave, @function

kernel_6x4_interleave:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.b

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

    lsr     x5, x3, #2              // K/4

    .align 6
.Lk_loop_interleave:
    // Phase 1: Load A[0:1], B[0:1], start computing
    ld1rw   {z24.s}, p0/z, [x0, #0]     // A row 0
    ld1rw   {z25.s}, p0/z, [x0, #4]     // A row 1
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]  // B col 0
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]  // B col 1

    // Compute row 0-1, col 0-1 (4 SDOTs) while loading A[2:3]
    sdot    z0.s, z24.b, z30.b      // C[0,0]
    ld1rw   {z26.s}, p0/z, [x0, #8]     // A row 2
    sdot    z1.s, z24.b, z31.b      // C[0,1]
    ld1rw   {z27.s}, p0/z, [x0, #12]    // A row 3
    sdot    z4.s, z25.b, z30.b      // C[1,0]
    sdot    z5.s, z25.b, z31.b      // C[1,1]

    // Phase 2: Compute row 2-3 while loading A[4:5]
    sdot    z8.s, z26.b, z30.b      // C[2,0]
    ld1rw   {z28.s}, p0/z, [x0, #16]    // A row 4
    sdot    z9.s, z26.b, z31.b      // C[2,1]
    ld1rw   {z29.s}, p0/z, [x0, #20]    // A row 5
    sdot    z12.s, z27.b, z30.b     // C[3,0]
    sdot    z13.s, z27.b, z31.b     // C[3,1]

    // Phase 3: Compute row 4-5, col 0-1
    sdot    z16.s, z28.b, z30.b     // C[4,0]
    sdot    z17.s, z28.b, z31.b     // C[4,1]
    sdot    z20.s, z29.b, z30.b     // C[5,0]
    sdot    z21.s, z29.b, z31.b     // C[5,1]

    // Phase 4: Load B[2:3], compute all rows col 2-3
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]  // B col 2
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]  // B col 3

    sdot    z2.s, z24.b, z30.b      // C[0,2]
    sdot    z3.s, z24.b, z31.b      // C[0,3]
    sdot    z6.s, z25.b, z30.b      // C[1,2]
    sdot    z7.s, z25.b, z31.b      // C[1,3]
    sdot    z10.s, z26.b, z30.b     // C[2,2]
    sdot    z11.s, z26.b, z31.b     // C[2,3]
    sdot    z14.s, z27.b, z30.b     // C[3,2]
    sdot    z15.s, z27.b, z31.b     // C[3,3]
    sdot    z18.s, z28.b, z30.b     // C[4,2]
    sdot    z19.s, z28.b, z31.b     // C[4,3]
    sdot    z22.s, z29.b, z30.b     // C[5,2]
    sdot    z23.s, z29.b, z31.b     // C[5,3]

    // Advance pointers
    add     x0, x0, #24             // A += 6 rows × 4 bytes
    add     x1, x1, #256            // B += 4 vectors × 64 bytes

    subs    x5, x5, #1
    b.gt    .Lk_loop_interleave

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

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size kernel_6x4_interleave, .-kernel_6x4_interleave


// Same-address test version for measuring pure throughput
    .global kernel_6x4_interleave_same
    .type kernel_6x4_interleave_same, @function

kernel_6x4_interleave_same:
    stp     x19, x20, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.b

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

    mov     x5, x3                  // K_iters

    .align 6
.Lloop_interleave_same:
    // Same interleaving but A/B pointers don't advance
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]

    sdot    z0.s, z24.b, z30.b
    ld1rw   {z26.s}, p0/z, [x0, #8]
    sdot    z1.s, z24.b, z31.b
    ld1rw   {z27.s}, p0/z, [x0, #12]
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b

    sdot    z8.s, z26.b, z30.b
    ld1rw   {z28.s}, p0/z, [x0, #16]
    sdot    z9.s, z26.b, z31.b
    ld1rw   {z29.s}, p0/z, [x0, #20]
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

    subs    x5, x5, #1
    b.gt    .Lloop_interleave_same

    // Store
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

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size kernel_6x4_interleave_same, .-kernel_6x4_interleave_same
