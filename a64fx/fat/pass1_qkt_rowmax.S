// pass1_qkt_rowmax.S
// Pass 1: S = Q @ K^T with row max computation
//
// Q[4, 64] register-resident
// K[64, 64] streamed
// S[4, 64] output to L1
// m[4] row max output
//
// Register allocation:
//   z0-z3:   Q[0, 0:64]
//   z4-z7:   Q[1, 0:64]
//   z8-z11:  Q[2, 0:64]
//   z12-z15: Q[3, 0:64]
//   z16-z19: K[j, 0:64] current K row
//   z20-z23: dot product temps
//   z24-z27: row max accumulators
//   z28-z31: scratch

    .arch armv8.2-a+sve
    .text
    .align 6
    .global pass1_qkt_rowmax
    .type pass1_qkt_rowmax, %function

// Args: x0=Q, x1=K, x2=S, x3=m
pass1_qkt_rowmax:
    ptrue   p0.s

    mov     x4, #256            // row stride = 64 * 4

    // ========================================
    // Load Q[4, 64] into z0-z15
    // ========================================
    mov     x10, x0
    ld1w    z0.s, p0/z, [x10]
    ld1w    z1.s, p0/z, [x10, #1, mul vl]
    ld1w    z2.s, p0/z, [x10, #2, mul vl]
    ld1w    z3.s, p0/z, [x10, #3, mul vl]
    add     x10, x10, x4

    ld1w    z4.s, p0/z, [x10]
    ld1w    z5.s, p0/z, [x10, #1, mul vl]
    ld1w    z6.s, p0/z, [x10, #2, mul vl]
    ld1w    z7.s, p0/z, [x10, #3, mul vl]
    add     x10, x10, x4

    ld1w    z8.s, p0/z, [x10]
    ld1w    z9.s, p0/z, [x10, #1, mul vl]
    ld1w    z10.s, p0/z, [x10, #2, mul vl]
    ld1w    z11.s, p0/z, [x10, #3, mul vl]
    add     x10, x10, x4

    ld1w    z12.s, p0/z, [x10]
    ld1w    z13.s, p0/z, [x10, #1, mul vl]
    ld1w    z14.s, p0/z, [x10, #2, mul vl]
    ld1w    z15.s, p0/z, [x10, #3, mul vl]

    // ========================================
    // Initialize row max = -inf
    // ========================================
    mov     w10, #0xff800000    // -inf IEEE754
    fmov    s24, w10
    fmov    s25, w10
    fmov    s26, w10
    fmov    s27, w10

    // ========================================
    // Compute S = Q @ K^T
    // Process K rows in groups of 4 for better pipelining
    // ========================================
    mov     x5, #16             // 64 / 4 = 16 groups
    mov     x6, x1              // K ptr
    mov     x7, x2              // S ptr

.Lgroup_loop:
    // Process 4 consecutive K rows
    
.macro compute_sj offset
    // Load K[j, 0:64]
    ld1w    z16.s, p0/z, [x6]
    ld1w    z17.s, p0/z, [x6, #1, mul vl]
    ld1w    z18.s, p0/z, [x6, #2, mul vl]
    ld1w    z19.s, p0/z, [x6, #3, mul vl]
    add     x6, x6, x4

    // S[0, j] = Q[0, :] · K[j, :]
    fmul    z20.s, z0.s, z16.s
    fmla    z20.s, p0/m, z1.s, z17.s
    fmla    z20.s, p0/m, z2.s, z18.s
    fmla    z20.s, p0/m, z3.s, z19.s

    // S[1, j]
    fmul    z21.s, z4.s, z16.s
    fmla    z21.s, p0/m, z5.s, z17.s
    fmla    z21.s, p0/m, z6.s, z18.s
    fmla    z21.s, p0/m, z7.s, z19.s

    // S[2, j]
    fmul    z22.s, z8.s, z16.s
    fmla    z22.s, p0/m, z9.s, z17.s
    fmla    z22.s, p0/m, z10.s, z18.s
    fmla    z22.s, p0/m, z11.s, z19.s

    // S[3, j]
    fmul    z23.s, z12.s, z16.s
    fmla    z23.s, p0/m, z13.s, z17.s
    fmla    z23.s, p0/m, z14.s, z18.s
    fmla    z23.s, p0/m, z15.s, z19.s

    // Horizontal reduction + store + max update
    faddv   s28, p0, z20.s
    faddv   s29, p0, z21.s
    faddv   s30, p0, z22.s
    faddv   s31, p0, z23.s

    // Store S[0:4, j]
    str     s28, [x7, #(\offset * 4)]
    str     s29, [x7, #(\offset * 4 + 256)]      // S[1] row offset
    str     s30, [x7, #(\offset * 4 + 512)]      // S[2] row offset
    str     s31, [x7, #(\offset * 4 + 768)]      // S[3] row offset

    // Update row max
    fmax    s24, s24, s28
    fmax    s25, s25, s29
    fmax    s26, s26, s30
    fmax    s27, s27, s31
.endm

    compute_sj 0
    compute_sj 1
    compute_sj 2
    compute_sj 3

    add     x7, x7, #16         // advance S ptr by 4 floats

    subs    x5, x5, #1
    b.ne    .Lgroup_loop

    // ========================================
    // Store row max m[4]
    // ========================================
    stp     s24, s25, [x3]
    stp     s26, s27, [x3, #8]

    ret
    .size pass1_qkt_rowmax, .-pass1_qkt_rowmax
