/* A64FX Q6A8 matvec micro-kernel used by test_matvec_q6.c.
 *
 * q6_block layout: low nibbles, high 2-bit planes, then one f32 scale.
 * One call contracts eight independent rows.  The assembly keeps all eight
 * f32 accumulators in registers and decodes/dots one row at a time, avoiding
 * the scalable-vector spills emitted for the equivalent ACLE expression. */

    .arch armv8.2-a+sve
    .text
    .align 6
#ifndef Q6_GROUP_SIZE
#define Q6_GROUP_SIZE 128
#endif
#if Q6_GROUP_SIZE == 64
#define Q6_QH_OFFSET 32
#define Q6_SCALE_OFFSET 48
#define Q6_ACT_BYTES 64
#ifndef Q6_BLOCK_BYTES
#define Q6_BLOCK_BYTES 52
#endif
#elif Q6_GROUP_SIZE == 128
#define Q6_QH_OFFSET 64
#define Q6_SCALE_OFFSET 96
#define Q6_ACT_BYTES 128
#ifndef Q6_BLOCK_BYTES
#define Q6_BLOCK_BYTES 100
#endif
#elif Q6_GROUP_SIZE == 256
#define Q6_QH_OFFSET 128
#define Q6_SCALE_OFFSET 192
#define Q6_ACT_BYTES 256
#ifndef Q6_BLOCK_BYTES
#define Q6_BLOCK_BYTES 196
#endif
#else
#error unsupported Q6_GROUP_SIZE
#endif
#ifndef Q6_PF_DIST
#define Q6_PF_DIST 376
#endif
    .global glm5_matvec_q6_a8_asm
    .type glm5_matvec_q6_a8_asm, @function

/* x0: float dst[8]
 * x1: q6_a8_asm_job
 *   +0   q6_block *w[8]
 *   +64  int8_t *xq
 *   +72  float *xd
 *   +80  int nb
 *
 * z16..z23: persistent f32 accumulators
 * z24: high-bit byte indices; z25: per-lane shifts
 * z26/z27/z28: 15/3/32 constants; z29: activation
 * z0..z7: row decode, dot, and scale scratch
 */
glm5_matvec_q6_a8_asm:
    ptrue   p0.s
    ptrue   p1.b
    mov     x14, #16
    whilelo p2.b, xzr, x14
    mov     x14, #32
    whilelo p3.b, xzr, x14

    ldp     x2, x3, [x1, #0]
    ldp     x4, x5, [x1, #16]
    ldp     x6, x7, [x1, #32]
    ldp     x8, x9, [x1, #48]
    ldr     x10, [x1, #64]
    ldr     x11, [x1, #72]
    ldr     w12, [x1, #80]

    index   z24.b, #0, #1
    mov     z25.d, z24.d
    lsr     z24.b, p1/m, z24.b, #2
    mov     z27.b, #3
    and     z25.b, p1/m, z25.b, z27.b
    lsl     z25.b, p1/m, z25.b, #1
    mov     z26.b, #15
    mov     z28.b, #32

    dup     z16.s, #0
    dup     z17.s, #0
    dup     z18.s, #0
    dup     z19.s, #0
    dup     z20.s, #0
    dup     z21.s, #0
    dup     z22.s, #0
    dup     z23.s, #0

    .macro  Q6A8_ROW wp, acc
    dup     z3.s, #0
    ld1b    { z0.b }, p3/z, [\wp]
    mov     z1.d, z0.d
    and     z1.b, p1/m, z1.b, z26.b
    lsr     z0.b, p1/m, z0.b, #4
    zip1    z1.b, z1.b, z0.b
    add     x14, \wp, #Q6_QH_OFFSET
    ld1b    { z2.b }, p2/z, [x14]
    tbl     z2.b, { z2.b }, z24.b
    lsr     z2.b, p1/m, z2.b, z25.b
    and     z2.b, p1/m, z2.b, z27.b
    lsl     z2.b, p1/m, z2.b, #4
    orr     z1.b, p1/m, z1.b, z2.b
    sub     z1.b, p1/m, z1.b, z28.b
    sdot    z3.s, z1.b, z29.b
#if Q6_GROUP_SIZE >= 128
    add     x14, \wp, #32
    ld1b    { z0.b }, p3/z, [x14]
    mov     z1.d, z0.d
    and     z1.b, p1/m, z1.b, z26.b
    lsr     z0.b, p1/m, z0.b, #4
    zip1    z1.b, z1.b, z0.b
    add     x14, \wp, #(Q6_QH_OFFSET + 16)
    ld1b    { z2.b }, p2/z, [x14]
    tbl     z2.b, { z2.b }, z24.b
    lsr     z2.b, p1/m, z2.b, z25.b
    and     z2.b, p1/m, z2.b, z27.b
    lsl     z2.b, p1/m, z2.b, #4
    orr     z1.b, p1/m, z1.b, z2.b
    sub     z1.b, p1/m, z1.b, z28.b
    sdot    z3.s, z1.b, z30.b
#endif
#if Q6_GROUP_SIZE == 256
    add     x14, \wp, #64
    ld1b    { z0.b }, p3/z, [x14]
    mov     z1.d, z0.d
    and     z1.b, p1/m, z1.b, z26.b
    lsr     z0.b, p1/m, z0.b, #4
    zip1    z1.b, z1.b, z0.b
    add     x14, \wp, #(Q6_QH_OFFSET + 32)
    ld1b    { z2.b }, p2/z, [x14]
    tbl     z2.b, { z2.b }, z24.b
    lsr     z2.b, p1/m, z2.b, z25.b
    and     z2.b, p1/m, z2.b, z27.b
    lsl     z2.b, p1/m, z2.b, #4
    orr     z1.b, p1/m, z1.b, z2.b
    sub     z1.b, p1/m, z1.b, z28.b
    sdot    z3.s, z1.b, z31.b

    add     x14, \wp, #96
    ld1b    { z0.b }, p3/z, [x14]
    mov     z1.d, z0.d
    and     z1.b, p1/m, z1.b, z26.b
    lsr     z0.b, p1/m, z0.b, #4
    zip1    z1.b, z1.b, z0.b
    add     x14, \wp, #(Q6_QH_OFFSET + 48)
    ld1b    { z2.b }, p2/z, [x14]
    tbl     z2.b, { z2.b }, z24.b
    lsr     z2.b, p1/m, z2.b, z25.b
    and     z2.b, p1/m, z2.b, z27.b
    lsl     z2.b, p1/m, z2.b, #4
    orr     z1.b, p1/m, z1.b, z2.b
    sub     z1.b, p1/m, z1.b, z28.b
    sdot    z3.s, z1.b, z7.b
#endif
    scvtf   z3.s, p0/m, z3.s
    ldr     s4, [\wp, #Q6_SCALE_OFFSET]
#if Q6_GROUP_SIZE == 256
    fmov    s5, w13
    fmul    s4, s4, s5
#else
    fmul    s4, s4, s7
#endif
    mov     z4.s, s4
    fmla    z\acc\().s, p0/m, z3.s, z4.s
    .endm

.Lq6a8_group:
    prfm    pldl1keep, [x2, #Q6_PF_DIST]
    prfm    pldl1keep, [x3, #Q6_PF_DIST]
    prfm    pldl1keep, [x4, #Q6_PF_DIST]
    prfm    pldl1keep, [x5, #Q6_PF_DIST]
    prfm    pldl1keep, [x6, #Q6_PF_DIST]
    prfm    pldl1keep, [x7, #Q6_PF_DIST]
    prfm    pldl1keep, [x8, #Q6_PF_DIST]
    prfm    pldl1keep, [x9, #Q6_PF_DIST]
    ld1b    { z29.b }, p1/z, [x10]
#if Q6_GROUP_SIZE >= 128
    add     x14, x10, #64
    ld1b    { z30.b }, p1/z, [x14]
#endif
#if Q6_GROUP_SIZE == 256
    add     x14, x10, #128
    ld1b    { z31.b }, p1/z, [x14]
    add     x14, x10, #192
    ld1b    { z7.b }, p1/z, [x14]
#endif
#if Q6_GROUP_SIZE == 256
    ldr     w13, [x11]
#else
    ldr     s7, [x11]
#endif

    Q6A8_ROW x2, 16
    Q6A8_ROW x3, 17
    Q6A8_ROW x4, 18
    Q6A8_ROW x5, 19
    Q6A8_ROW x6, 20
    Q6A8_ROW x7, 21
    Q6A8_ROW x8, 22
    Q6A8_ROW x9, 23

    add     x2, x2, #Q6_BLOCK_BYTES
    add     x3, x3, #Q6_BLOCK_BYTES
    add     x4, x4, #Q6_BLOCK_BYTES
    add     x5, x5, #Q6_BLOCK_BYTES
    add     x6, x6, #Q6_BLOCK_BYTES
    add     x7, x7, #Q6_BLOCK_BYTES
    add     x8, x8, #Q6_BLOCK_BYTES
    add     x9, x9, #Q6_BLOCK_BYTES
    add     x10, x10, #Q6_ACT_BYTES
    add     x11, x11, #4
    subs    w12, w12, #1
    b.ne    .Lq6a8_group

    faddv   s0, p0, z16.s
    faddv   s1, p0, z17.s
    stp     s0, s1, [x0, #0]
    faddv   s0, p0, z18.s
    faddv   s1, p0, z19.s
    stp     s0, s1, [x0, #8]
    faddv   s0, p0, z20.s
    faddv   s1, p0, z21.s
    stp     s0, s1, [x0, #16]
    faddv   s0, p0, z22.s
    faddv   s1, p0, z23.s
    stp     s0, s1, [x0, #24]
    ret

    .size glm5_matvec_q6_a8_asm, .-glm5_matvec_q6_a8_asm
