// 6x4 INT8 SDOT Microkernel with Sector Cache Hints (Fixed)
// A64FX sector cache: addr[63:60]=pf_func, addr[57:56]=sec_id
// Sector 0: A data (default, keep in cache)
// Sector 1: B data (streaming, evict-first)

    .arch armv8.2-a+sve
    .text
    .global micro_kernel_6x4_sector
    .type micro_kernel_6x4_sector, %function

// x0 = A (Q), x1 = B (K), x2 = C, x3 = K_dim, x4 = unused, x5 = C_stride
// Sector 1 for B (streaming): bits 57:56 = 01

    .align 6
micro_kernel_6x4_sector:
    // Save callee-saved registers (d8-d15 are callee-saved)
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.b

    // Tag B pointer with sector 1 (streaming)
    // Sector ID in bits 57:56
    mov     x11, #1
    lsl     x11, x11, #56       // Shift to bits 57:56
    orr     x1, x1, x11         // B_tagged = B | (1 << 56)

    // Zero accumulators (z0-z23)
    mov     z0.s, #0
    mov     z1.s, #0
    mov     z2.s, #0
    mov     z3.s, #0
    mov     z4.s, #0
    mov     z5.s, #0
    mov     z6.s, #0
    mov     z7.s, #0
    mov     z8.s, #0
    mov     z9.s, #0
    mov     z10.s, #0
    mov     z11.s, #0
    mov     z12.s, #0
    mov     z13.s, #0
    mov     z14.s, #0
    mov     z15.s, #0
    mov     z16.s, #0
    mov     z17.s, #0
    mov     z18.s, #0
    mov     z19.s, #0
    mov     z20.s, #0
    mov     z21.s, #0
    mov     z22.s, #0
    mov     z23.s, #0

    // K iterations
    lsr     x8, x3, #2      // K/4

    .align 5
.Lloop:
    // Load B with sector 1 tag (streaming)
    ld1b    {z24.b}, p0/z, [x1]
    ld1b    {z25.b}, p0/z, [x1, #1, mul vl]
    ld1b    {z26.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z27.b}, p0/z, [x1, #3, mul vl]

    // Load A with sector 0 (default, keep in cache)
    // Row 0
    ld1rw   {z28.s}, p0/z, [x0]
    sdot    z0.s, z24.b, z28.b
    sdot    z1.s, z25.b, z28.b
    sdot    z2.s, z26.b, z28.b
    sdot    z3.s, z27.b, z28.b

    // Row 1
    ld1rw   {z29.s}, p0/z, [x0, #4]
    sdot    z4.s, z24.b, z29.b
    sdot    z5.s, z25.b, z29.b
    sdot    z6.s, z26.b, z29.b
    sdot    z7.s, z27.b, z29.b

    // Row 2
    ld1rw   {z30.s}, p0/z, [x0, #8]
    sdot    z8.s, z24.b, z30.b
    sdot    z9.s, z25.b, z30.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z27.b, z30.b

    // Row 3
    ld1rw   {z31.s}, p0/z, [x0, #12]
    sdot    z12.s, z24.b, z31.b
    sdot    z13.s, z25.b, z31.b
    sdot    z14.s, z26.b, z31.b
    sdot    z15.s, z27.b, z31.b

    // Row 4
    ld1rw   {z28.s}, p0/z, [x0, #16]
    sdot    z16.s, z24.b, z28.b
    sdot    z17.s, z25.b, z28.b
    sdot    z18.s, z26.b, z28.b
    sdot    z19.s, z27.b, z28.b

    // Row 5
    ld1rw   {z29.s}, p0/z, [x0, #20]
    sdot    z20.s, z24.b, z29.b
    sdot    z21.s, z25.b, z29.b
    sdot    z22.s, z26.b, z29.b
    sdot    z23.s, z27.b, z29.b

    // Advance pointers
    add     x0, x0, #24
    add     x1, x1, #256

    subs    x8, x8, #1
    bne     .Lloop

    // Store results - row-major within each row
    // x5 = C_stride (offset between rows)
    // z0-z3 = row 0, z4-z7 = row 1, etc.
    ptrue   p0.s
    mov     x6, x2

    // Row 0: z0-z3 at consecutive 64-byte offsets
    st1w    {z0.s}, p0, [x6, #0, mul vl]
    st1w    {z1.s}, p0, [x6, #1, mul vl]
    st1w    {z2.s}, p0, [x6, #2, mul vl]
    st1w    {z3.s}, p0, [x6, #3, mul vl]
    add     x6, x6, x5

    // Row 1: z4-z7
    st1w    {z4.s}, p0, [x6, #0, mul vl]
    st1w    {z5.s}, p0, [x6, #1, mul vl]
    st1w    {z6.s}, p0, [x6, #2, mul vl]
    st1w    {z7.s}, p0, [x6, #3, mul vl]
    add     x6, x6, x5

    // Row 2: z8-z11
    st1w    {z8.s}, p0, [x6, #0, mul vl]
    st1w    {z9.s}, p0, [x6, #1, mul vl]
    st1w    {z10.s}, p0, [x6, #2, mul vl]
    st1w    {z11.s}, p0, [x6, #3, mul vl]
    add     x6, x6, x5

    // Row 3: z12-z15
    st1w    {z12.s}, p0, [x6, #0, mul vl]
    st1w    {z13.s}, p0, [x6, #1, mul vl]
    st1w    {z14.s}, p0, [x6, #2, mul vl]
    st1w    {z15.s}, p0, [x6, #3, mul vl]
    add     x6, x6, x5

    // Row 4: z16-z19
    st1w    {z16.s}, p0, [x6, #0, mul vl]
    st1w    {z17.s}, p0, [x6, #1, mul vl]
    st1w    {z18.s}, p0, [x6, #2, mul vl]
    st1w    {z19.s}, p0, [x6, #3, mul vl]
    add     x6, x6, x5

    // Row 5: z20-z23
    st1w    {z20.s}, p0, [x6, #0, mul vl]
    st1w    {z21.s}, p0, [x6, #1, mul vl]
    st1w    {z22.s}, p0, [x6, #2, mul vl]
    st1w    {z23.s}, p0, [x6, #3, mul vl]

    // Restore callee-saved registers
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16

    ret

    .size micro_kernel_6x4_sector, .-micro_kernel_6x4_sector
