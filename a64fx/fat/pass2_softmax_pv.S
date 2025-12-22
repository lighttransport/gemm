// pass2_softmax_pv.S
// Pass 2: O = softmax(S) @ V
//
// Strategy: exp amortization
//   - Load S[4, 16] chunk (4 vectors)
//   - Compute P = exp(S - m)
//   - Reuse P 4 times (D/16 = 4 slices)
//   - Each exp cost amortized by 4x
//
// Register allocation:
//   z0-z3:   O[0, 0:64]  (row 0, 4 vectors)
//   z4-z7:   O[1, 0:64]
//   z8-z11:  O[2, 0:64]
//   z12-z15: O[3, 0:64]
//   z16-z19: P[0:4, jj:jj+16] (exp results)
//   z20-z23: V[j, d:d+16] (D slice)
//   z24-z27: broadcast temps (time-multiplexed)
//   z28:     1/ln(2) constant
//   z29:     scratch
//   z30-z31: exp scratch

    .arch armv8.2-a+sve
    .text
    .align 6
    .global pass2_softmax_pv
    .type pass2_softmax_pv, %function

// Macro for exp(x) computation using range reduction and polynomial
// Input: z_in (contains x values)
// Output: z_in (contains exp(x) values)
// Clobbers: z29, z30, z31
// Uses: z28 = 1/ln(2)
.macro compute_exp z_in
    // exp(x) = 2^(x * log2(e)) = 2^(x / ln(2))
    // Range reduction: n = round(x / ln(2)), r = x - n * ln(2)
    // exp(x) = 2^n * exp(r) where |r| <= ln(2)/2

    // n = round(x / ln(2))
    fmul    z30.s, \z_in\().s, z28.s    // x * (1/ln(2))
    frintn  z31.s, p0/m, z30.s          // n = round(...)

    // r = x - n * ln(2)
    // ln(2) = 0.6931471805599453 = 0x3f317218
    movz    w15, #0x7218
    movk    w15, #0x3f31, lsl #16
    fmov    s29, w15
    dup     z29.s, z29.s[0]

    // r = x - n * ln(2) using fnmls: z_in = z_in - z31 * z29
    fnmls   \z_in\().s, p0/m, z31.s, z29.s

    // Now compute exp(r) using polynomial approximation
    // exp(r) ≈ 1 + r * (1 + r * (0.5 + r * (0.16667 + r * 0.04167)))
    // Using Horner's method starting from innermost coefficient

    // c4 = 1/24 ≈ 0.041666667 = 0x3d2aaaab
    movz    w15, #0xaaab
    movk    w15, #0x3d2a, lsl #16
    fmov    s30, w15
    dup     z30.s, z30.s[0]

    // c3 = 1/6 ≈ 0.166666667 = 0x3e2aaaab
    movz    w15, #0xaaab
    movk    w15, #0x3e2a, lsl #16
    fmov    s29, w15
    dup     z29.s, z29.s[0]

    // p = c4 * r + c3 = r * (1/24) + (1/6)
    fmla    z29.s, p0/m, z30.s, \z_in\().s

    // c2 = 0.5 = 0x3f000000
    fmov    z30.s, #0.5

    // p = p * r + c2 = r * p + 0.5
    fmla    z30.s, p0/m, z29.s, \z_in\().s

    // c1 = 1.0
    fmov    z29.s, #1.0

    // p = p * r + c1 = r * p + 1
    fmla    z29.s, p0/m, z30.s, \z_in\().s

    // p = p * r + 1 = r * p + 1
    fmov    z30.s, #1.0
    fmla    z30.s, p0/m, z29.s, \z_in\().s

    // Now z30 = exp(r)
    // Scale by 2^n: result = exp(r) * 2^n
    fscale  z30.s, p0/m, z30.s, z31.s

    mov     \z_in\().s, p0/m, z30.s
.endm

// Args: x0=S, x1=V, x2=m, x3=O, x4=l
pass2_softmax_pv:
    ptrue   p0.s

    // Constants
    mov     x5, #256            // row stride = 64 * 4
    mov     x6, #64             // VL bytes = 16 * 4

    // 1/ln(2) = 1.4426950408889634 = 0x3fb8aa3b
    movz    w10, #0xaa3b
    movk    w10, #0x3fb8, lsl #16
    fmov    s28, w10
    dup     z28.s, z28.s[0]     // z28 = 1/ln(2)

    // ========================================
    // Initialize O[4, 64] = 0
    // ========================================
    fmov    z0.s, #0.0
    fmov    z1.s, #0.0
    fmov    z2.s, #0.0
    fmov    z3.s, #0.0
    fmov    z4.s, #0.0
    fmov    z5.s, #0.0
    fmov    z6.s, #0.0
    fmov    z7.s, #0.0
    fmov    z8.s, #0.0
    fmov    z9.s, #0.0
    fmov    z10.s, #0.0
    fmov    z11.s, #0.0
    fmov    z12.s, #0.0
    fmov    z13.s, #0.0
    fmov    z14.s, #0.0
    fmov    z15.s, #0.0

    // Initialize l[4] = 0 (in memory)
    fmov    s30, wzr
    str     s30, [x4]
    str     s30, [x4, #4]
    str     s30, [x4, #8]
    str     s30, [x4, #12]

    // Load m[4] into stack for easy access
    // Stack layout after pushes:
    //   sp+0:  m[2]
    //   sp+4:  m[3]
    //   sp+16: m[0]
    //   sp+20: m[1]
    ldp     s30, s31, [x2]      // m[0], m[1]
    stp     s30, s31, [sp, #-16]!
    ldp     s30, s31, [x2, #8]  // m[2], m[3]
    stp     s30, s31, [sp, #-16]!

    // S pointers (4 rows)
    mov     x7, x0              // S[0]
    add     x8, x0, x5          // S[1]
    add     x9, x8, x5          // S[2]
    add     x10, x9, x5         // S[3]

    // V base
    mov     x11, x1

    // ========================================
    // Main loop: process S in chunks of 16
    // BC=64, so 4 chunks
    // ========================================
    mov     x12, #4

.Lchunk_loop:
    // ========================================
    // Load S[0:4, jj:jj+16] and compute exp(S - m)
    // ========================================

    // Load S chunks
    ld1w    z16.s, p0/z, [x7]
    ld1w    z17.s, p0/z, [x8]
    ld1w    z18.s, p0/z, [x9]
    ld1w    z19.s, p0/z, [x10]

    // Advance S pointers
    add     x7, x7, x6
    add     x8, x8, x6
    add     x9, x9, x6
    add     x10, x10, x6

    // ----------------------------------------
    // exp(S[0] - m[0]) -> z16
    // ----------------------------------------
    ldr     s29, [sp, #16]      // m[0]
    dup     z29.s, z29.s[0]
    fsub    z16.s, z16.s, z29.s
    compute_exp z16

    // Accumulate l[0]
    faddv   s30, p0, z16.s
    ldr     s31, [x4]
    fadd    s31, s31, s30
    str     s31, [x4]

    // ----------------------------------------
    // exp(S[1] - m[1]) -> z17
    // ----------------------------------------
    ldr     s29, [sp, #20]      // m[1]
    dup     z29.s, z29.s[0]
    fsub    z17.s, z17.s, z29.s
    compute_exp z17

    faddv   s30, p0, z17.s
    ldr     s31, [x4, #4]
    fadd    s31, s31, s30
    str     s31, [x4, #4]

    // ----------------------------------------
    // exp(S[2] - m[2]) -> z18
    // ----------------------------------------
    ldr     s29, [sp]           // m[2]
    dup     z29.s, z29.s[0]
    fsub    z18.s, z18.s, z29.s
    compute_exp z18

    faddv   s30, p0, z18.s
    ldr     s31, [x4, #8]
    fadd    s31, s31, s30
    str     s31, [x4, #8]

    // ----------------------------------------
    // exp(S[3] - m[3]) -> z19
    // ----------------------------------------
    ldr     s29, [sp, #4]       // m[3]
    dup     z29.s, z29.s[0]
    fsub    z19.s, z19.s, z29.s
    compute_exp z19

    faddv   s30, p0, z19.s
    ldr     s31, [x4, #12]
    fadd    s31, s31, s30
    str     s31, [x4, #12]

    // ========================================
    // O += P @ V for this chunk
    // P[0:4, jj:jj+16] in z16-z19
    // Process 16 j values, each P element used 4 times (D slices)
    // ========================================

    // V pointer for this chunk (16 rows starting at jj)
    // Current chunk number: 4 - x12
    // V offset: (4 - x12) * 16 * 256 bytes
    mov     x13, #4
    sub     x13, x13, x12       // chunk index
    lsl     x13, x13, #12       // * 4096 = 16 * 256
    add     x13, x11, x13       // V[jj, 0]

    // Unroll 16 j values
.macro process_j lane
    // Broadcast P[0:4, jj+lane]
    dup     z24.s, z16.s[\lane]     // P[0, j]
    dup     z25.s, z17.s[\lane]     // P[1, j]
    dup     z26.s, z18.s[\lane]     // P[2, j]
    dup     z27.s, z19.s[\lane]     // P[3, j]

    // Load V[jj+lane, 0:64]
    ld1w    z20.s, p0/z, [x13]
    ld1w    z21.s, p0/z, [x13, #1, mul vl]
    ld1w    z22.s, p0/z, [x13, #2, mul vl]
    ld1w    z23.s, p0/z, [x13, #3, mul vl]
    add     x13, x13, x5

    // O[0, :] += P[0, j] * V[j, :]
    fmla    z0.s, p0/m, z24.s, z20.s
    fmla    z1.s, p0/m, z24.s, z21.s
    fmla    z2.s, p0/m, z24.s, z22.s
    fmla    z3.s, p0/m, z24.s, z23.s

    // O[1, :] += P[1, j] * V[j, :]
    fmla    z4.s, p0/m, z25.s, z20.s
    fmla    z5.s, p0/m, z25.s, z21.s
    fmla    z6.s, p0/m, z25.s, z22.s
    fmla    z7.s, p0/m, z25.s, z23.s

    // O[2, :] += P[2, j] * V[j, :]
    fmla    z8.s, p0/m, z26.s, z20.s
    fmla    z9.s, p0/m, z26.s, z21.s
    fmla    z10.s, p0/m, z26.s, z22.s
    fmla    z11.s, p0/m, z26.s, z23.s

    // O[3, :] += P[3, j] * V[j, :]
    fmla    z12.s, p0/m, z27.s, z20.s
    fmla    z13.s, p0/m, z27.s, z21.s
    fmla    z14.s, p0/m, z27.s, z22.s
    fmla    z15.s, p0/m, z27.s, z23.s
.endm

    process_j 0
    process_j 1
    process_j 2
    process_j 3
    process_j 4
    process_j 5
    process_j 6
    process_j 7
    process_j 8
    process_j 9
    process_j 10
    process_j 11
    process_j 12
    process_j 13
    process_j 14
    process_j 15

    subs    x12, x12, #1
    b.ne    .Lchunk_loop

    // ========================================
    // Store O[4, 64]
    // ========================================
    mov     x14, x3
    st1w    z0.s, p0, [x14]
    st1w    z1.s, p0, [x14, #1, mul vl]
    st1w    z2.s, p0, [x14, #2, mul vl]
    st1w    z3.s, p0, [x14, #3, mul vl]
    add     x14, x14, x5

    st1w    z4.s, p0, [x14]
    st1w    z5.s, p0, [x14, #1, mul vl]
    st1w    z6.s, p0, [x14, #2, mul vl]
    st1w    z7.s, p0, [x14, #3, mul vl]
    add     x14, x14, x5

    st1w    z8.s, p0, [x14]
    st1w    z9.s, p0, [x14, #1, mul vl]
    st1w    z10.s, p0, [x14, #2, mul vl]
    st1w    z11.s, p0, [x14, #3, mul vl]
    add     x14, x14, x5

    st1w    z12.s, p0, [x14]
    st1w    z13.s, p0, [x14, #1, mul vl]
    st1w    z14.s, p0, [x14, #2, mul vl]
    st1w    z15.s, p0, [x14, #3, mul vl]

    // Clean up stack
    add     sp, sp, #32

    ret
    .size pass2_softmax_pv, .-pass2_softmax_pv
