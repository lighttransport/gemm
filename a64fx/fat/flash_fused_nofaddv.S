// flash_fused_nofaddv.S
// Fused Flash Attention kernel using transposed K (no faddv)
//
// Computes: O = softmax(Q @ K^T) @ V
//
// Key optimizations:
// 1. Uses Kt (transposed K) to avoid faddv in Q@K^T
// 2. Online softmax: compute max and exp incrementally
// 3. Exp reuse: each exp(S) value used for both sum and P@V
//
// Algorithm (online softmax):
//   m = -inf, l = 0, O = 0
//   for j in 0..BC:
//       S[:,j] = Q @ Kt[:,j]   # outer product style, no faddv
//       m_new = max(m, S[:,j])
//       l = l * exp(m - m_new) + exp(S[:,j] - m_new)
//       O = O * exp(m - m_new) + exp(S[:,j] - m_new) * V[j,:]
//       m = m_new
//   O = O / l
//
// Dimensions:
//   Q[BR, D]   = [4, 64]
//   Kt[D, BC]  = [64, 64]  (K transposed)
//   V[BC, D]   = [64, 64]
//   O[BR, D]   = [4, 64]
//
// Register allocation:
//   z0-z3:   O[0, 0:64]
//   z4-z7:   O[1, 0:64]
//   z8-z11:  O[2, 0:64]
//   z12-z15: O[3, 0:64]
//   z16-z19: V[j, 0:64] after S computation done
//   s20-s23: m[0:3] (row maxes) - scalar only
//   z24-z27: temporaries for exp, correction, p
//   z28-z31: temporaries for exp computation
//
// Stack layout (16 bytes):
//   [sp, #0]:  l[0]
//   [sp, #4]:  l[1]
//   [sp, #8]:  l[2]
//   [sp, #12]: l[3]
// Stack layout (additional 16 bytes for S values):
//   [sp, #16]: S[0,j]
//   [sp, #20]: S[1,j]
//   [sp, #24]: S[2,j]
//   [sp, #28]: S[3,j]

    .arch armv8.2-a+sve
    .text

// Compute exp of a single scalar in s_reg, result in s_reg
// Uses s28, s29, s30, s31 as scratch, w14, w15 as integer scratch
// For very negative inputs (< -87), returns ~0
// For very positive inputs (> 88), returns inf
// Uses higher precision polynomial for better accuracy
.macro compute_exp_scalar s_reg
    // Clamp input to reasonable range first to avoid FP issues
    // exp(-87.3) ≈ 1e-38 (min positive float)
    // exp(88.7) ≈ 1e38 (max finite float)
    movz    w15, #0xc000
    movk    w15, #0xc2ae, lsl #16   // -87.0 in float
    fmov    s30, w15
    fmax    \s_reg, \s_reg, s30    // Clamp to >= -87

    movz    w15, #0x0000
    movk    w15, #0x42b0, lsl #16   // 88.0 in float
    fmov    s30, w15
    fmin    \s_reg, \s_reg, s30    // Clamp to <= 88

    // n = round(x / ln(2))
    // Using more precise constant: 1/ln(2) = 0x3fb8aa3b
    movz    w15, #0xaa3b
    movk    w15, #0x3fb8, lsl #16
    fmov    s28, w15
    fmul    s28, \s_reg, s28
    frintn  s31, s28

    // r = x - n * ln(2) using precise constants
    // ln(2)_hi = 0x3f317218 (high precision part)
    // ln(2)_lo = 0x353bce50 (low precision part for Cody-Waite)
    movz    w15, #0x7218
    movk    w15, #0x3f31, lsl #16   // ln(2)_hi
    fmov    s29, w15
    fnmsub  \s_reg, s31, s29, \s_reg   // r = x - n * ln2_hi

    // Higher precision polynomial: exp(r) for r in [-ln(2)/2, ln(2)/2]
    // p(r) = 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
    // Coefficients:
    // c5 = 1/120 = 0x3c088889
    // c4 = 1/24  = 0x3d2aaaab
    // c3 = 1/6   = 0x3e2aaaab
    // c2 = 1/2   = 0x3f000000
    // c1 = 1     = 0x3f800000
    // c0 = 1     = 0x3f800000

    movz    w15, #0x8889
    movk    w15, #0x3c08, lsl #16   // c5 = 1/120
    fmov    s28, w15
    movz    w15, #0xaaab
    movk    w15, #0x3d2a, lsl #16   // c4 = 1/24
    fmov    s29, w15
    fmadd   s29, s28, \s_reg, s29   // p = c5*r + c4

    movz    w15, #0xaaab
    movk    w15, #0x3e2a, lsl #16   // c3 = 1/6
    fmov    s28, w15
    fmadd   s28, s29, \s_reg, s28   // p = p*r + c3

    fmov    s29, #0.5               // c2 = 1/2
    fmadd   s29, s28, \s_reg, s29   // p = p*r + 0.5
    fmov    s28, #1.0               // c1 = 1
    fmadd   s28, s29, \s_reg, s28   // p = p*r + 1
    fmov    s29, #1.0               // c0 = 1
    fmadd   s28, s28, \s_reg, s29   // p = p*r + 1

    // Scale by 2^n using integer manipulation of IEEE754 exponent
    // 2^n = reinterpret((n + 127) << 23) for float
    fcvtzs  w15, s31               // n as integer
    add     w15, w15, #127         // n + 127 (float exponent bias)
    lsl     w15, w15, #23          // shift to exponent position
    fmov    s30, w15               // reinterpret as float (this is 2^n)

    fmul    s28, s28, s30          // p * 2^n

    fmov    \s_reg, s28
.endm

    .align 6
    .global flash_fused_nofaddv
    .type flash_fused_nofaddv, %function

// Args: x0=Q, x1=Kt, x2=V, x3=O
flash_fused_nofaddv:
    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!
    // Allocate stack space for l[0:4] and S[0:4,j]
    sub     sp, sp, #32

    ptrue   p0.s

    mov     x4, #256            // row stride = 64 * 4 bytes

    // ========================================
    // Initialize O = 0, m = -inf, l = 0
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

    // m[0:4] = -1e30 (approximation of -inf)
    movz    w10, #0x0000
    movk    w10, #0xf149, lsl #16  // -1e30 approximately
    fmov    s20, w10
    fmov    s21, s20
    fmov    s22, s20
    fmov    s23, s20

    // l[0:4] = 0 (on stack)
    str     wzr, [sp, #0]       // l[0]
    str     wzr, [sp, #4]       // l[1]
    str     wzr, [sp, #8]       // l[2]
    str     wzr, [sp, #12]      // l[3]

    // Q row pointers
    mov     x6, x0              // Q[0]
    add     x7, x0, x4          // Q[1]
    add     x8, x7, x4          // Q[2]
    add     x9, x8, x4          // Q[3]

    // ========================================
    // Main loop: for j in 0..BC (64 columns of Kt)
    // ========================================
    mov     x10, x1             // Kt pointer (columns)
    mov     x11, x2             // V pointer
    mov     x12, #64            // BC loop counter

.Lmain_loop:
    // ----------------------------------------
    // Step 1: Compute S[0:4, j] = Q @ Kt[:, j]
    // S is stored as scalars on stack
    // ----------------------------------------

    // Initialize S accumulators (scalars)
    fmov    s24, wzr            // S[0, j]
    fmov    s25, wzr            // S[1, j]
    fmov    s26, wzr            // S[2, j]
    fmov    s27, wzr            // S[3, j]

    // Save Q row pointers for reuse
    mov     x13, x6
    mov     x14, x7
    mov     x15, x8
    mov     x19, x9

    // Inner loop: for k in 0..D (64 iterations)
    mov     x21, x10            // Kt column pointer
    mov     x22, #64            // D loop counter

.Linner_k_loop:
    // Load Kt[k, j] - scalar
    // Kt is [D, BC] = [64, 64], row-major
    // Kt[k, j] = Kt[k * BC + j], stride between k values is BC * 4 = 256
    ldr     s28, [x21]
    add     x21, x21, x4        // next k (stride = BC * 4 = 256)

    // Load Q[i, k] as scalars and accumulate S[i, j]
    ldr     s29, [x13], #4      // Q[0, k]
    ldr     s30, [x14], #4      // Q[1, k]
    ldr     s31, [x15], #4      // Q[2, k]

    fmadd   s24, s29, s28, s24  // S[0,j] += Q[0,k] * Kt[k,j]
    fmadd   s25, s30, s28, s25  // S[1,j] += Q[1,k] * Kt[k,j]
    fmadd   s26, s31, s28, s26  // S[2,j] += Q[2,k] * Kt[k,j]

    ldr     s29, [x19], #4      // Q[3, k]
    fmadd   s27, s29, s28, s27  // S[3,j] += Q[3,k] * Kt[k,j]

    subs    x22, x22, #1
    b.ne    .Linner_k_loop

    // Save S[0:4, j] to stack
    str     s24, [sp, #16]
    str     s25, [sp, #20]
    str     s26, [sp, #24]
    str     s27, [sp, #28]

    // ----------------------------------------
    // Step 2: Load V[j, :] into z16-z19
    // ----------------------------------------
    ld1w    z16.s, p0/z, [x11]
    ld1w    z17.s, p0/z, [x11, #1, mul vl]
    ld1w    z18.s, p0/z, [x11, #2, mul vl]
    ld1w    z19.s, p0/z, [x11, #3, mul vl]

    // ----------------------------------------
    // Step 3: Online softmax update for row 0
    // ----------------------------------------
    ldr     s24, [sp, #16]       // S[0, j]

    fmax    s25, s20, s24        // m_new = max(m, S)
    fsub    s26, s20, s25        // m - m_new (for correction)
    compute_exp_scalar s26       // correction = exp(m - m_new)

    fsub    s27, s24, s25        // S - m_new
    compute_exp_scalar s27       // p = exp(S - m_new)

    // l[0] = l[0] * correction + p
    ldr     s24, [sp, #0]
    fmul    s24, s24, s26
    fadd    s24, s24, s27
    str     s24, [sp, #0]

    // O[0,:] = O[0,:] * correction + p * V[j,:]
    dup     z26.s, z26.s[0]      // broadcast correction
    dup     z27.s, z27.s[0]      // broadcast p

    fmul    z0.s, z0.s, z26.s
    fmul    z1.s, z1.s, z26.s
    fmul    z2.s, z2.s, z26.s
    fmul    z3.s, z3.s, z26.s

    fmla    z0.s, p0/m, z27.s, z16.s
    fmla    z1.s, p0/m, z27.s, z17.s
    fmla    z2.s, p0/m, z27.s, z18.s
    fmla    z3.s, p0/m, z27.s, z19.s

    fmov    s20, s25             // m[0] = m_new

    // ----------------------------------------
    // Step 4: Online softmax update for row 1
    // ----------------------------------------
    ldr     s24, [sp, #20]       // S[1, j]

    fmax    s25, s21, s24
    fsub    s26, s21, s25
    compute_exp_scalar s26

    fsub    s27, s24, s25
    compute_exp_scalar s27

    ldr     s24, [sp, #4]
    fmul    s24, s24, s26
    fadd    s24, s24, s27
    str     s24, [sp, #4]

    dup     z26.s, z26.s[0]
    dup     z27.s, z27.s[0]

    fmul    z4.s, z4.s, z26.s
    fmul    z5.s, z5.s, z26.s
    fmul    z6.s, z6.s, z26.s
    fmul    z7.s, z7.s, z26.s

    fmla    z4.s, p0/m, z27.s, z16.s
    fmla    z5.s, p0/m, z27.s, z17.s
    fmla    z6.s, p0/m, z27.s, z18.s
    fmla    z7.s, p0/m, z27.s, z19.s

    fmov    s21, s25

    // ----------------------------------------
    // Step 5: Online softmax update for row 2
    // ----------------------------------------
    ldr     s24, [sp, #24]       // S[2, j]

    fmax    s25, s22, s24
    fsub    s26, s22, s25
    compute_exp_scalar s26

    fsub    s27, s24, s25
    compute_exp_scalar s27

    ldr     s24, [sp, #8]
    fmul    s24, s24, s26
    fadd    s24, s24, s27
    str     s24, [sp, #8]

    dup     z26.s, z26.s[0]
    dup     z27.s, z27.s[0]

    fmul    z8.s, z8.s, z26.s
    fmul    z9.s, z9.s, z26.s
    fmul    z10.s, z10.s, z26.s
    fmul    z11.s, z11.s, z26.s

    fmla    z8.s, p0/m, z27.s, z16.s
    fmla    z9.s, p0/m, z27.s, z17.s
    fmla    z10.s, p0/m, z27.s, z18.s
    fmla    z11.s, p0/m, z27.s, z19.s

    fmov    s22, s25

    // ----------------------------------------
    // Step 6: Online softmax update for row 3
    // ----------------------------------------
    ldr     s24, [sp, #28]       // S[3, j]

    fmax    s25, s23, s24
    fsub    s26, s23, s25
    compute_exp_scalar s26

    fsub    s27, s24, s25
    compute_exp_scalar s27

    ldr     s24, [sp, #12]
    fmul    s24, s24, s26
    fadd    s24, s24, s27
    str     s24, [sp, #12]

    dup     z26.s, z26.s[0]
    dup     z27.s, z27.s[0]

    fmul    z12.s, z12.s, z26.s
    fmul    z13.s, z13.s, z26.s
    fmul    z14.s, z14.s, z26.s
    fmul    z15.s, z15.s, z26.s

    fmla    z12.s, p0/m, z27.s, z16.s
    fmla    z13.s, p0/m, z27.s, z17.s
    fmla    z14.s, p0/m, z27.s, z18.s
    fmla    z15.s, p0/m, z27.s, z19.s

    fmov    s23, s25

    // Advance Kt column pointer (next j)
    add     x10, x10, #4        // Kt[:, j+1]

    // Advance V row pointer
    add     x11, x11, x4

    subs    x12, x12, #1
    b.ne    .Lmain_loop

    // ========================================
    // Final normalization: O = O / l
    // ========================================

    // 1/l[0]
    ldr     s24, [sp, #0]
    fmov    s25, #1.0
    fdiv    s25, s25, s24
    dup     z25.s, z25.s[0]
    fmul    z0.s, z0.s, z25.s
    fmul    z1.s, z1.s, z25.s
    fmul    z2.s, z2.s, z25.s
    fmul    z3.s, z3.s, z25.s

    // 1/l[1]
    ldr     s24, [sp, #4]
    fmov    s25, #1.0
    fdiv    s25, s25, s24
    dup     z25.s, z25.s[0]
    fmul    z4.s, z4.s, z25.s
    fmul    z5.s, z5.s, z25.s
    fmul    z6.s, z6.s, z25.s
    fmul    z7.s, z7.s, z25.s

    // 1/l[2]
    ldr     s24, [sp, #8]
    fmov    s25, #1.0
    fdiv    s25, s25, s24
    dup     z25.s, z25.s[0]
    fmul    z8.s, z8.s, z25.s
    fmul    z9.s, z9.s, z25.s
    fmul    z10.s, z10.s, z25.s
    fmul    z11.s, z11.s, z25.s

    // 1/l[3]
    ldr     s24, [sp, #12]
    fmov    s25, #1.0
    fdiv    s25, s25, s24
    dup     z25.s, z25.s[0]
    fmul    z12.s, z12.s, z25.s
    fmul    z13.s, z13.s, z25.s
    fmul    z14.s, z14.s, z25.s
    fmul    z15.s, z15.s, z25.s

    // ========================================
    // Store O[4, 64]
    // ========================================
    st1w    z0.s, p0, [x3]
    st1w    z1.s, p0, [x3, #1, mul vl]
    st1w    z2.s, p0, [x3, #2, mul vl]
    st1w    z3.s, p0, [x3, #3, mul vl]
    add     x3, x3, x4

    st1w    z4.s, p0, [x3]
    st1w    z5.s, p0, [x3, #1, mul vl]
    st1w    z6.s, p0, [x3, #2, mul vl]
    st1w    z7.s, p0, [x3, #3, mul vl]
    add     x3, x3, x4

    st1w    z8.s, p0, [x3]
    st1w    z9.s, p0, [x3, #1, mul vl]
    st1w    z10.s, p0, [x3, #2, mul vl]
    st1w    z11.s, p0, [x3, #3, mul vl]
    add     x3, x3, x4

    st1w    z12.s, p0, [x3]
    st1w    z13.s, p0, [x3, #1, mul vl]
    st1w    z14.s, p0, [x3, #2, mul vl]
    st1w    z15.s, p0, [x3, #3, mul vl]

    // Deallocate stack space
    add     sp, sp, #32

    // Restore callee-saved registers
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    ret
    .size flash_fused_nofaddv, .-flash_fused_nofaddv
