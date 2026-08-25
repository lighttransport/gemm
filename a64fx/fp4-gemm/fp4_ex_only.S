/* Scalar AArch64-only E2M1 kernels.  No SIMD, SVE, FP, or table lookup. */
    .arch armv8.2-a+sve
    .text
    .align 4

/* Decode nibble Q into signed integer OUT representing 2*E2M1.
 * Clobbers M,E,T,S. */
    .macro DECODE Q, OUT, M, E, T, S
    and     \M, \Q, #7
    lsr     \E, \M, #2
    and     \T, \M, #3
    and     \S, \T, \T, lsr #1
    add     \T, \T, \S, lsl #1
    madd    \OUT, \E, \T, \M
    ubfx    \S, \Q, #3, #1
    neg     \E, \S
    eor     \OUT, \OUT, \E
    add     \OUT, \OUT, \S
    .endm

/* uint64_t fp4_ex_dequant_i16(src, dst, packed_bytes, repetitions); */
    .global fp4_ex_dequant_i16
    .type fp4_ex_dequant_i16, %function
fp4_ex_dequant_i16:
    mov     x8, #0
    mov     x9, x3
1:  mov     x10, x0
    mov     x11, x1
    mov     x12, x2
2:  ldrb    w13, [x10], #1
    DECODE w13, w14, w15, w16, w17, w18
    strh    w14, [x11], #2
    add     x8, x8, x14
    lsr     w13, w13, #4
    DECODE w13, w14, w15, w16, w17, w18
    strh    w14, [x11], #2
    add     x8, x8, x14
    subs    x12, x12, #1
    b.ne    2b
    subs    x9, x9, #1
    b.ne    1b
    mov     x0, x8
    ret
    .size fp4_ex_dequant_i16, .-fp4_ex_dequant_i16

/* uint64_t fp4_ex_mul_i16(a, b, dst, packed_bytes, repetitions); */
    .global fp4_ex_mul_i16
    .type fp4_ex_mul_i16, %function
fp4_ex_mul_i16:
    stp     x19, x20, [sp, #-64]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    str     x25, [sp, #48]
    mov     x8, x4                  // fifth argument
    mov     x9, #0
3:  mov     x10, x0
    mov     x11, x1
    mov     x12, x2
    mov     x13, x3
4:  ldrb    w14, [x10], #1
    ldrb    w15, [x11], #1
    DECODE w14, w16, w17, w18, w19, w20
    DECODE w15, w21, w22, w23, w24, w25
    mul     w16, w16, w21
    strh    w16, [x12], #2
    add     x9, x9, x16
    lsr     w14, w14, #4
    lsr     w15, w15, #4
    DECODE w14, w16, w17, w18, w19, w20
    DECODE w15, w21, w22, w23, w24, w25
    mul     w16, w16, w21
    strh    w16, [x12], #2
    add     x9, x9, x16
    subs    x13, x13, #1
    b.ne    4b
    subs    x8, x8, #1
    b.ne    3b
    mov     x0, x9
    ldr     x25, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #64
    ret
    .size fp4_ex_mul_i16, .-fp4_ex_mul_i16
    .section .note.GNU-stack,"",%progbits
