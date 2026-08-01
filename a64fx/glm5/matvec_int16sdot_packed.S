/* Bit-exact group-64 int16-SDOT matvec for the experimental packed 8-row
 * layout in test_matvec_int16sdot.c. Each 32-column panel occupies one
 * 256-byte cache line: row0[32], row1[32], ... row7[32]. */

    .arch armv8.2-a+sve
    .text
    .align 6
    .global glm5_matvec_int16sdot_packed_asm
    .type glm5_matvec_int16sdot_packed_asm, @function

/* x0: float dst[8]
 * x1: packed_asm_job
 *   +0  packed weights
 *   +8  scale pointers s0..s7
 *   +72 xq
 *   +80 xgsum
 *   +88 cols (int)
 *
 * z0..z7   per-group integer dots
 * d8..d15  scalar bias accumulators (callee-saved low halves)
 * z16..z23 persistent f64 scaled-dot accumulators
 * z24      activation, z25 weight/scale broadcast, z26/z27 scalar scratch
 */
glm5_matvec_int16sdot_packed_asm:
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.d
    ptrue   p1.h

    ldr     x2, [x1, #0]
    ldp     x5, x6, [x1, #8]
    ldp     x7, x8, [x1, #24]
    ldp     x9, x10, [x1, #40]
    ldp     x11, x12, [x1, #56]
    ldr     x3, [x1, #72]
    ldr     x4, [x1, #80]
    ldr     w13, [x1, #88]
    lsr     w13, w13, #6

    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
    dup     z20.d, #0
    dup     z21.d, #0
    dup     z22.d, #0
    dup     z23.d, #0
    fmov    d8, xzr
    fmov    d9, xzr
    fmov    d10, xzr
    fmov    d11, xzr
    fmov    d12, xzr
    fmov    d13, xzr
    fmov    d14, xzr
    fmov    d15, xzr

.Lpacked_group:
    prfm    pldl1keep, [x2, #1024]
    dup     z0.d, #0
    dup     z1.d, #0
    dup     z2.d, #0
    dup     z3.d, #0
    dup     z4.d, #0
    dup     z5.d, #0
    dup     z6.d, #0
    dup     z7.d, #0

    mov     x14, x2
    add     x15, x2, #256
    ld1h    { z24.h }, p1/z, [x3]
    ld1b    { z25.h }, p1/z, [x14, #0, mul vl]
    sdot    z0.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x14, #1, mul vl]
    sdot    z1.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x14, #2, mul vl]
    sdot    z2.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x14, #3, mul vl]
    sdot    z3.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x14, #4, mul vl]
    sdot    z4.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x14, #5, mul vl]
    sdot    z5.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x14, #6, mul vl]
    sdot    z6.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x14, #7, mul vl]
    sdot    z7.d, z25.h, z24.h

    ld1h    { z24.h }, p1/z, [x3, #1, mul vl]
    ld1b    { z25.h }, p1/z, [x15, #0, mul vl]
    sdot    z0.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x15, #1, mul vl]
    sdot    z1.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x15, #2, mul vl]
    sdot    z2.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x15, #3, mul vl]
    sdot    z3.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x15, #4, mul vl]
    sdot    z4.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x15, #5, mul vl]
    sdot    z5.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x15, #6, mul vl]
    sdot    z6.d, z25.h, z24.h
    ld1b    { z25.h }, p1/z, [x15, #7, mul vl]
    sdot    z7.d, z25.h, z24.h

    ldr     x15, [x4], #8
    scvtf   d27, x15

    ldr     s26, [x5], #4
    fcvt    d26, s26
    scvtf   z0.d, p0/m, z0.d
    mov     z25.d, d26
    fmla    z16.d, p0/m, z0.d, z25.d
    fmadd   d8, d26, d27, d8

    ldr     s26, [x6], #4
    fcvt    d26, s26
    scvtf   z1.d, p0/m, z1.d
    mov     z25.d, d26
    fmla    z17.d, p0/m, z1.d, z25.d
    fmadd   d9, d26, d27, d9

    ldr     s26, [x7], #4
    fcvt    d26, s26
    scvtf   z2.d, p0/m, z2.d
    mov     z25.d, d26
    fmla    z18.d, p0/m, z2.d, z25.d
    fmadd   d10, d26, d27, d10

    ldr     s26, [x8], #4
    fcvt    d26, s26
    scvtf   z3.d, p0/m, z3.d
    mov     z25.d, d26
    fmla    z19.d, p0/m, z3.d, z25.d
    fmadd   d11, d26, d27, d11

    ldr     s26, [x9], #4
    fcvt    d26, s26
    scvtf   z4.d, p0/m, z4.d
    mov     z25.d, d26
    fmla    z20.d, p0/m, z4.d, z25.d
    fmadd   d12, d26, d27, d12

    ldr     s26, [x10], #4
    fcvt    d26, s26
    scvtf   z5.d, p0/m, z5.d
    mov     z25.d, d26
    fmla    z21.d, p0/m, z5.d, z25.d
    fmadd   d13, d26, d27, d13

    ldr     s26, [x11], #4
    fcvt    d26, s26
    scvtf   z6.d, p0/m, z6.d
    mov     z25.d, d26
    fmla    z22.d, p0/m, z6.d, z25.d
    fmadd   d14, d26, d27, d14

    ldr     s26, [x12], #4
    fcvt    d26, s26
    scvtf   z7.d, p0/m, z7.d
    mov     z25.d, d26
    fmla    z23.d, p0/m, z7.d, z25.d
    fmadd   d15, d26, d27, d15

    add     x2, x2, #512
    add     x3, x3, #128
    subs    w13, w13, #1
    b.ne    .Lpacked_group

    mov     x14, #128
    scvtf   d26, x14

    faddv   d24, p0, z16.d
    fmul    d25, d8, d26
    fsub    d24, d24, d25
    fcvt    s24, d24
    str     s24, [x0, #0]

    faddv   d24, p0, z17.d
    fmul    d25, d9, d26
    fsub    d24, d24, d25
    fcvt    s24, d24
    str     s24, [x0, #4]

    faddv   d24, p0, z18.d
    fmul    d25, d10, d26
    fsub    d24, d24, d25
    fcvt    s24, d24
    str     s24, [x0, #8]

    faddv   d24, p0, z19.d
    fmul    d25, d11, d26
    fsub    d24, d24, d25
    fcvt    s24, d24
    str     s24, [x0, #12]

    faddv   d24, p0, z20.d
    fmul    d25, d12, d26
    fsub    d24, d24, d25
    fcvt    s24, d24
    str     s24, [x0, #16]

    faddv   d24, p0, z21.d
    fmul    d25, d13, d26
    fsub    d24, d24, d25
    fcvt    s24, d24
    str     s24, [x0, #20]

    faddv   d24, p0, z22.d
    fmul    d25, d14, d26
    fsub    d24, d24, d25
    fcvt    s24, d24
    str     s24, [x0, #24]

    faddv   d24, p0, z23.d
    fmul    d25, d15, d26
    fsub    d24, d24, d25
    fcvt    s24, d24
    str     s24, [x0, #28]

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ret

    .size glm5_matvec_int16sdot_packed_asm, .-glm5_matvec_int16sdot_packed_asm
