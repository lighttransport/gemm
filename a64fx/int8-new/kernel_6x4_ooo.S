// kernel_6x4_ooo.S
// 6x4 kernel exploiting Out-of-Order execution and register renaming
//
// Key insight: After issuing SDOT using z24, we can immediately load
// a new value into z24. Register renaming preserves the old value
// for the in-flight SDOT.
//
// Strategy: Issue all 24 SDOTs using A[k], THEN load A[k+1]
// The 11-cycle latency for A[k+1] is hidden by:
// 1. The remaining SDOTs of A[k] (some cycles)
// 2. B loads for next iteration (some cycles)
// 3. Loop overhead (some cycles)
//
// Per K-iteration:
//   - 6 A loads + 4 B loads = 10 loads
//   - 24 SDOT = 12 cycles @ 2/cycle
//   - Need to hide 11 cycle latency for A

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_6x4_ooo
    .type kernel_6x4_ooo, @function

kernel_6x4_ooo:
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

    lsr     x5, x3, #2              // K/4 iterations

    // Prologue: Load A[0]
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]
    ld1rw   {z28.s}, p0/z, [x0, #16]
    ld1rw   {z29.s}, p0/z, [x0, #20]
    add     x0, x0, #24

    sub     x5, x5, #1
    cbz     x5, .Lepilogue_ooo

    .align 6
.Lk_loop_ooo:
    // A[k] is ready in z24-z29 (from prologue or previous iteration)

    // Load B[k,0:1]
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]

    // 12 SDOTs for cols 0-1 using A[k]
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

    // Load B[k,2:3]
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]

    // 12 SDOTs for cols 2-3 using A[k]
    // Interleave with A[k+1] loads - OoO will handle the renaming
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    ld1rw   {z24.s}, p0/z, [x0, #0]     // A[k+1] row 0 - z24 renamed
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    ld1rw   {z25.s}, p0/z, [x0, #4]     // A[k+1] row 1
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    ld1rw   {z26.s}, p0/z, [x0, #8]     // A[k+1] row 2
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    ld1rw   {z27.s}, p0/z, [x0, #12]    // A[k+1] row 3
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    ld1rw   {z28.s}, p0/z, [x0, #16]    // A[k+1] row 4
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    ld1rw   {z29.s}, p0/z, [x0, #20]    // A[k+1] row 5

    // Advance pointers
    add     x0, x0, #24
    add     x1, x1, #256

    subs    x5, x5, #1
    b.gt    .Lk_loop_ooo

.Lepilogue_ooo:
    // Final iteration - A already loaded
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
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size kernel_6x4_ooo, .-kernel_6x4_ooo


// Same-address version for pure compute testing
    .global kernel_6x4_ooo_same
    .type kernel_6x4_ooo_same, @function

kernel_6x4_ooo_same:
    stp     x19, x20, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

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

    mov     x5, x3                  // K_iters

    // Pre-load A[0]
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]
    ld1rw   {z28.s}, p0/z, [x0, #16]
    ld1rw   {z29.s}, p0/z, [x0, #20]

    .align 6
.Lloop_ooo_same:
    // Load B[0:1] - same address
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]

    // 12 SDOTs for cols 0-1
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

    // Load B[2:3] - same address
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]

    // 12 SDOTs for cols 2-3, interleaved with A loads (same address)
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    ld1rw   {z24.s}, p0/z, [x0, #0]     // Reload same A - OoO handles it
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    ld1rw   {z25.s}, p0/z, [x0, #4]
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    ld1rw   {z26.s}, p0/z, [x0, #8]
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    ld1rw   {z27.s}, p0/z, [x0, #12]
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    ld1rw   {z28.s}, p0/z, [x0, #16]
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    ld1rw   {z29.s}, p0/z, [x0, #20]

    subs    x5, x5, #1
    b.gt    .Lloop_ooo_same

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

    .size kernel_6x4_ooo_same, .-kernel_6x4_ooo_same
