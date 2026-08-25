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

/* Layout-friendly 8 packed bytes -> 16 nibble bytes. Low-nibble group first. */
    .global fp4_ex_unpack_u8
    .type fp4_ex_unpack_u8, %function
fp4_ex_unpack_u8:
    mov     x8, #0x0f0f0f0f0f0f0f0f
    mov     x9, x3
5:  mov     x10, x0
    mov     x11, x1
    mov     x12, x2
6:  ldr     x13, [x10], #8
    and     x14, x13, x8
    and     x15, x8, x13, lsr #4
    stp     x14, x15, [x11], #16
    subs    x12, x12, #8
    b.ne    6b
    subs    x9, x9, #1
    b.ne    5b
    eor     x0, x14, x15
    ret
    .size fp4_ex_unpack_u8, .-fp4_ex_unpack_u8

/* Exact SWAR nibble -> signed byte coefficient representing 2*E2M1. */
    .macro SWAR_E2M1 Q, OUT
    and     x16, \Q, x8              // magnitude bytes
    and     x17, x16, x9             // b0
    and     x18, x16, x10            // b1
    and     x19, x16, x11            // b2
    lsr     x20, x19, #2
    bic     x21, x17, x20            // output bit 0
    lsr     x20, x19, #1
    bic     x22, x18, x20
    lsl     x23, x17, #1
    and     x23, x23, x20
    bic     x23, x23, x18
    orr     x21, x21, x22
    orr     x21, x21, x23            // output bit 1
    lsl     x20, x18, #1
    bic     x22, x19, x20
    lsl     x23, x17, #2
    and     x23, x23, x19
    orr     x21, x21, x22
    orr     x21, x21, x23            // output bit 2
    and     x20, x19, x18, lsl #1
    orr     x21, x21, x20, lsl #1    // output bit 3
    lsr     x20, \Q, #3
    and     x20, x20, x9
    mul     x22, x20, x12            // per-byte sign mask
    eor     \OUT, x21, x22
    add     \OUT, \OUT, x20
    .endm

    .global fp4_ex_unpack_e2m1_i8
    .type fp4_ex_unpack_e2m1_i8, %function
fp4_ex_unpack_e2m1_i8:
    stp     x19, x20, [sp, #-64]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    str     x25, [sp, #48]
    mov     x8, #0x0707070707070707
    mov     x9, #0x0101010101010101
    mov     x10, #0x0202020202020202
    mov     x11, #0x0404040404040404
    mov     x12, #255
    mov     x13, x3
7:  mov     x14, x0
    mov     x15, x1
    mov     x24, x2
8:  ldr     x25, [x14], #8
    and     x6, x25, #0x0f0f0f0f0f0f0f0f
    SWAR_E2M1 x6, x7
    lsr     x6, x25, #4
    SWAR_E2M1 x6, x5
    stp     x7, x5, [x15], #16
    subs    x24, x24, #8
    b.ne    8b
    subs    x13, x13, #1
    b.ne    7b
    eor     x0, x7, x5
    ldr     x25, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #64
    ret
    .size fp4_ex_unpack_e2m1_i8, .-fp4_ex_unpack_e2m1_i8
    .section .note.GNU-stack,"",%progbits
