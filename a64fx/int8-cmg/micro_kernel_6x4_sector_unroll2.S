// 6x4 INT8 SDOT Microkernel with Sector Cache + 2x K-unrolling
// Combines OoO unroll2 structure with sector cache hints
// Target: >70% efficiency across all K values
//
// A64FX sector cache: addr[57:56]=sec_id
// Sector 0: A/Q data (default, keep in cache)
// Sector 1: B data (streaming, evict-first)

    .arch armv8.2-a+sve
    .text
    .align 6

    .global micro_kernel_6x4_sector_unroll2
    .type micro_kernel_6x4_sector_unroll2, @function

micro_kernel_6x4_sector_unroll2:
    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.b

    // Tag B pointer with sector 1 (streaming)
    mov     x19, #1
    lsl     x19, x19, #56       // Shift to bits 57:56
    orr     x1, x1, x19         // B_tagged = B | (1 << 56)

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

    // K/8 iterations (process 2 K-groups per iteration)
    lsr     x6, x3, #3              // K / 8
    cbz     x6, .Lremainder_su2

    // Prologue: Load first A (sector 0, default - keep in cache)
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]
    ld1rw   {z28.s}, p0/z, [x0, #16]
    ld1rw   {z29.s}, p0/z, [x0, #20]
    add     x0, x0, #24

    sub     x6, x6, #1
    cbz     x6, .Lepilogue_su2_single

    // ========== MAIN LOOP (2 K-groups per iteration) ==========
    .align 6
.Lmain_su2:
    // ===== K-group 0 =====
    // Load B with sector 1 tag (streaming - evict first when cache is full)
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
    ld1rw   {z24.s}, p0/z, [x0, #0]
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

    // Advance pointers by 2 K-groups
    add     x0, x0, #48             // 2 * 24 bytes
    add     x1, x1, #512            // 2 * 256 bytes

    subs    x6, x6, #1
    b.gt    .Lmain_su2

    // ===== Epilogue: last 2 K-groups =====
.Lepilogue_su2_single:
    // K-group 0
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
    ld1rw   {z24.s}, p0/z, [x0, #0]
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

    // K-group 1 (final)
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

    b       .Ldone_su2

    // Handle K % 8 remainder (single K-group)
.Lremainder_su2:
    and     x7, x3, #4              // Check if K/4 is odd
    cbz     x7, .Ldone_su2

    // Load A for single K-group
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]
    ld1rw   {z28.s}, p0/z, [x0, #16]
    ld1rw   {z29.s}, p0/z, [x0, #20]

    // Process single K-group with sector-tagged B
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

    // ===== STORE RESULTS =====
.Ldone_su2:
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

    // Restore callee-saved registers
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size micro_kernel_6x4_sector_unroll2, .-micro_kernel_6x4_sector_unroll2
