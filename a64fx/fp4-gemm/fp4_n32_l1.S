/* A64FX FP4 -> FP16 producer/consumer microkernel.
 * NEON integer dequantization writes alternating L1 cache lines; SVE consumes
 * the preceding line.  This deliberately avoids transfers between V and Z
 * register views, which serialize on A64FX.
 */
    .arch armv8.2-a+sve
    .text
    .align 4

/* struct argument offsets (see fp4_gemm.c). */
#define ARG_CP       0
#define ARG_SP       8
#define ARG_A       16
#define ARG_OUT     24
#define ARG_K       32
#define ARG_KBEGIN  36
#define ARG_KEND    40
#define ARG_BS      44

/* Expand 16 packed bytes at x9 to 32 FP16 bit patterns at DST.  This producer
 * intentionally uses only scalar AArch64 GPR/load/store instructions. x16 is
 * a 256-entry LUT mapping one packed byte to two packed FP16 bit patterns. */
    .macro BYTE SRC, OFF, OUT
    and     x17, \SRC, #255
    ldr     w18, [x16, x17, lsl #2]
    str     w18, [\OUT, #\OFF]
    lsr     \SRC, \SRC, #8
    .endm
    .macro DEQUANT DST
    ldp     x14, x15, [x9], #16
    BYTE x14, 0, \DST
    BYTE x14, 4, \DST
    BYTE x14, 8, \DST
    BYTE x14, 12, \DST
    BYTE x14, 16, \DST
    BYTE x14, 20, \DST
    BYTE x14, 24, \DST
    BYTE x14, 28, \DST
    BYTE x15, 32, \DST
    BYTE x15, 36, \DST
    BYTE x15, 40, \DST
    BYTE x15, 44, \DST
    BYTE x15, 48, \DST
    BYTE x15, 52, \DST
    BYTE x15, 56, \DST
    BYTE x15, 60, \DST
    .endm

/* void fp4_n32_m6_l1_asm(const struct fp4_l1_args *); */
    .global fp4_n32_m6_l1_asm
    .type fp4_n32_m6_l1_asm, %function
fp4_n32_m6_l1_asm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #128
    mov     x10, sp
    add     x11, sp, #64

    ldr     x9, [x0, #ARG_CP]
    ldr     x8, [x0, #ARG_SP]
    ldr     x7, [x0, #ARG_A]
    ldr     x6, [x0, #ARG_OUT]
    ldr     w5, [x0, #ARG_K]
    ldr     w4, [x0, #ARG_KBEGIN]
    ldr     w3, [x0, #ARG_KEND]
    ldr     w2, [x0, #ARG_BS]
    lsl     x12, x4, #4
    add     x9, x9, x12             // cp + kbegin*16
    add     x7, x7, x4, lsl #1      // A + kbegin
    udiv    w12, w4, w2
    lsl     x12, x12, #6
    add     x8, x8, x12             // scale block

    ptrue   p0.h
    adrp    x16, fp4_pair_lut
    add     x16, x16, :lo12:fp4_pair_lut
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d

.Lblock:
    ld1h    z16.h, p0/z, [x8]
    add     x8, x8, #64
    DEQUANT x10
    mov     w12, #0
.Linner:
    add     w12, w12, #1
    cmp     w12, w2
    b.eq    .Lconsume_last
    DEQUANT x11
    ld1h    z17.h, p0/z, [x10]
    fmul    z17.h, p0/m, z17.h, z16.h
    ld1rh   z18.h, p0/z, [x7]
    fmla    z0.h, p0/m, z18.h, z17.h
    add     x13, x7, x5, lsl #1
    ld1rh   z18.h, p0/z, [x13]
    fmla    z1.h, p0/m, z18.h, z17.h
    add     x13, x13, x5, lsl #1
    ld1rh   z18.h, p0/z, [x13]
    fmla    z2.h, p0/m, z18.h, z17.h
    add     x13, x13, x5, lsl #1
    ld1rh   z18.h, p0/z, [x13]
    fmla    z3.h, p0/m, z18.h, z17.h
    add     x13, x13, x5, lsl #1
    ld1rh   z18.h, p0/z, [x13]
    fmla    z4.h, p0/m, z18.h, z17.h
    add     x13, x13, x5, lsl #1
    ld1rh   z18.h, p0/z, [x13]
    fmla    z5.h, p0/m, z18.h, z17.h
    add     x7, x7, #2
    mov     x13, x10
    mov     x10, x11
    mov     x11, x13
    b       .Linner
.Lconsume_last:
    ld1h    z17.h, p0/z, [x10]
    fmul    z17.h, p0/m, z17.h, z16.h
    ld1rh   z18.h, p0/z, [x7]
    fmla    z0.h, p0/m, z18.h, z17.h
    add     x13, x7, x5, lsl #1
    ld1rh   z18.h, p0/z, [x13]
    fmla    z1.h, p0/m, z18.h, z17.h
    add     x13, x13, x5, lsl #1
    ld1rh   z18.h, p0/z, [x13]
    fmla    z2.h, p0/m, z18.h, z17.h
    add     x13, x13, x5, lsl #1
    ld1rh   z18.h, p0/z, [x13]
    fmla    z3.h, p0/m, z18.h, z17.h
    add     x13, x13, x5, lsl #1
    ld1rh   z18.h, p0/z, [x13]
    fmla    z4.h, p0/m, z18.h, z17.h
    add     x13, x13, x5, lsl #1
    ld1rh   z18.h, p0/z, [x13]
    fmla    z5.h, p0/m, z18.h, z17.h
    add     x7, x7, #2
    sub     w3, w3, w2
    cmp     w4, w3
    b.lt    .Lblock

    st1h    z0.h, p0, [x6]
    add     x6, x6, #64
    st1h    z1.h, p0, [x6]
    add     x6, x6, #64
    st1h    z2.h, p0, [x6]
    add     x6, x6, #64
    st1h    z3.h, p0, [x6]
    add     x6, x6, #64
    st1h    z4.h, p0, [x6]
    add     x6, x6, #64
    st1h    z5.h, p0, [x6]
    add     sp, sp, #128
    ldp     x29, x30, [sp], #16
    ret
    .size fp4_n32_m6_l1_asm, .-fp4_n32_m6_l1_asm

    .section .note.GNU-stack,"",%progbits
