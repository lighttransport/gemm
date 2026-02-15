/*
 * Simplified exp2f using FEXPA - minimal register usage
 */
.arch armv8.2-a+sve

.text
.align 4

.global exp2_fexpa_simple
.type exp2_fexpa_simple, %function
exp2_fexpa_simple:
    // x0 = input ptr, x1 = output ptr, x2 = n elements

    ptrue   p0.s

    // Load shift constant inline
    mov     w3, #0x1fc0
    movk    w3, #0x4848, lsl #16    // 0x48481fc0
    dup     z30.s, w3               // shift

    // Load c0 = ln(2)
    mov     w3, #0x7211
    movk    w3, #0x3f31, lsl #16    // 0x3f317211
    dup     z31.s, w3               // c0

    cntw    x4                      // elements per vector
    cmp     x2, x4
    b.lt    .Lsimple_tail

.Lsimple_loop:
    // Load input
    ld1w    z0.s, p0/z, [x0]

    // z = x + shift
    fadd    z1.s, z0.s, z30.s

    // n = z - shift
    fsub    z2.s, z1.s, z30.s

    // r = x - n
    fsub    z3.s, z0.s, z2.s

    // scale = fexpa(z)
    fexpa   z4.s, z1.s

    // Simple approximation: poly = r * c0 (just first term)
    fmul    z5.s, z3.s, z31.s

    // result = scale + scale * poly
    fmla    z4.s, p0/m, z4.s, z5.s

    // Store
    st1w    z4.s, p0, [x1]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x2, x2, x4

    cmp     x2, x4
    b.ge    .Lsimple_loop

.Lsimple_tail:
    cbz     x2, .Lsimple_done

    whilelt p1.s, xzr, x2
    ld1w    z0.s, p1/z, [x0]

    fadd    z1.s, z0.s, z30.s
    fsub    z2.s, z1.s, z30.s
    fsub    z3.s, z0.s, z2.s
    fexpa   z4.s, z1.s
    fmul    z5.s, z3.s, z31.s
    fmla    z4.s, p1/m, z4.s, z5.s
    st1w    z4.s, p1, [x1]

.Lsimple_done:
    ret
.size exp2_fexpa_simple, .-exp2_fexpa_simple
