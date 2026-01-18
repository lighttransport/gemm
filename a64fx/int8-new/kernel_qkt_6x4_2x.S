// kernel_qkt_6x4_2x.S
// 2x K-unrolled 6x4 kernel to hide 11-cycle load latency
//
// Key insight: A64FX can issue 2 loads + 2 SDOT per cycle (different ports)
// SDOT reads source operands at ISSUE time, so we can interleave loads
// as long as we complete all SDOT using a register before reloading it.
//
// Strategy: For K group N cols 2-3, interleave SDOT with loads for N+1
// Each Q register is fully consumed (2 SDOT) before being reloaded.
//
// Register allocation:
//   z0-z23:  24 accumulators (6 rows × 4 vectors)
//   z24-z29: Q broadcast (6 rows)
//   z30-z31: K vectors (2 at a time)

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_qkt_6x4_2x
    .type kernel_qkt_6x4_2x, @function

kernel_qkt_6x4_2x:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.b
    ptrue   p1.s

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

    lsr     x5, x3, #2              // K_groups = K / 4
    cbz     x5, .L2x_done

    // ========== PROLOGUE: Load first K group ==========
    ld1rw   z24.s, p1/z, [x0, #0]
    ld1rw   z25.s, p1/z, [x0, #4]
    ld1rw   z26.s, p1/z, [x0, #8]
    ld1rw   z27.s, p1/z, [x0, #12]
    ld1rw   z28.s, p1/z, [x0, #16]
    ld1rw   z29.s, p1/z, [x0, #20]

    ld1b    z30.b, p0/z, [x1, #0, mul vl]
    ld1b    z31.b, p0/z, [x1, #1, mul vl]

    sub     x5, x5, #1              // Account for prologue
    add     x0, x0, #24             // Advance Q for next load
    cbz     x5, .L2x_epilogue

    .align 6
.L2x_loop:
    // ===== PHASE 1: K cols 0-1 SDOT, then load K cols 2-3 =====
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

    // Load K cols 2-3 (z30/z31 free after all SDOT issued)
    ld1b    z30.b, p0/z, [x1, #2, mul vl]
    ld1b    z31.b, p0/z, [x1, #3, mul vl]

    // ===== PHASE 2: K cols 2-3 SDOT, interleaved with next Q loads =====
    // Complete all SDOT using each Q register before reloading it

    // Row 0: z24
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    ld1rw   z24.s, p1/z, [x0, #0]   // Safe: z24 fully consumed

    // Row 1: z25
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    ld1rw   z25.s, p1/z, [x0, #4]   // Safe: z25 fully consumed

    // Row 2: z26
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    ld1rw   z26.s, p1/z, [x0, #8]   // Safe: z26 fully consumed

    // Row 3: z27
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    ld1rw   z27.s, p1/z, [x0, #12]  // Safe: z27 fully consumed

    // Row 4: z28
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    ld1rw   z28.s, p1/z, [x0, #16]  // Safe: z28 fully consumed

    // Row 5: z29
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    ld1rw   z29.s, p1/z, [x0, #20]  // Safe: z29 fully consumed

    // Advance pointers
    add     x1, x1, #256            // K += 4 cols × 64 bytes
    add     x0, x0, #24             // Q += 6 rows × 4 bytes

    // Load K cols 0-1 for next iteration
    ld1b    z30.b, p0/z, [x1, #0, mul vl]
    ld1b    z31.b, p0/z, [x1, #1, mul vl]

    subs    x5, x5, #1
    b.gt    .L2x_loop

.L2x_epilogue:
    // Process final K group (Q already loaded, K cols 0-1 loaded)
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

    ld1b    z30.b, p0/z, [x1, #2, mul vl]
    ld1b    z31.b, p0/z, [x1, #3, mul vl]

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

.L2x_done:
    mov     x5, x2
    mov     x6, x4

    st1w    z0.s, p1, [x5, #0, mul vl]
    st1w    z1.s, p1, [x5, #1, mul vl]
    st1w    z2.s, p1, [x5, #2, mul vl]
    st1w    z3.s, p1, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    z4.s, p1, [x5, #0, mul vl]
    st1w    z5.s, p1, [x5, #1, mul vl]
    st1w    z6.s, p1, [x5, #2, mul vl]
    st1w    z7.s, p1, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    z8.s, p1, [x5, #0, mul vl]
    st1w    z9.s, p1, [x5, #1, mul vl]
    st1w    z10.s, p1, [x5, #2, mul vl]
    st1w    z11.s, p1, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    z12.s, p1, [x5, #0, mul vl]
    st1w    z13.s, p1, [x5, #1, mul vl]
    st1w    z14.s, p1, [x5, #2, mul vl]
    st1w    z15.s, p1, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    z16.s, p1, [x5, #0, mul vl]
    st1w    z17.s, p1, [x5, #1, mul vl]
    st1w    z18.s, p1, [x5, #2, mul vl]
    st1w    z19.s, p1, [x5, #3, mul vl]
    add     x5, x5, x6

    st1w    z20.s, p1, [x5, #0, mul vl]
    st1w    z21.s, p1, [x5, #1, mul vl]
    st1w    z22.s, p1, [x5, #2, mul vl]
    st1w    z23.s, p1, [x5, #3, mul vl]

    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

    .size kernel_qkt_6x4_2x, .-kernel_qkt_6x4_2x
