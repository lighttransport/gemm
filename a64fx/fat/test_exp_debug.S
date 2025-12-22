// Debug exp_f64 step by step
    .arch armv8.2-a+sve
    .text

// Input: x0 = input ptr (single double)
// Output: x1 = ptr to 5-element output array:
//   [0] = n (after frintn)
//   [1] = r (after range reduction)
//   [2] = 2^n
//   [3] = exp(r) polynomial
//   [4] = final result
    .global exp_f64_debug
    .type exp_f64_debug, %function
exp_f64_debug:
    ptrue   p0.d

    // Load constants
    adrp    x10, .Lconst
    add     x10, x10, :lo12:.Lconst
    ld1rd   z28.d, p0/z, [x10]              // 1/ln(2)
    ld1rd   z29.d, p0/z, [x10, #8]          // ln(2)
    ld1rd   z30.d, p0/z, [x10, #16]         // 1.0

    // Load input x
    ld1rd   z0.d, p0/z, [x0]

    // n = round(x / ln2)
    fmul    z4.d, z0.d, z28.d
    frintn  z4.d, p0/m, z4.d

    // Store n
    st1d    z4.d, p0, [x1]

    // r = x - n * ln2
    fmls    z0.d, p0/m, z4.d, z29.d

    // Store r
    st1d    z0.d, p0, [x1, #1, mul vl]

    // 2^n
    fcvtzs  z4.d, p0/m, z4.d
    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    lsl     z4.d, z4.d, #52

    // Store 2^n
    st1d    z4.d, p0, [x1, #2, mul vl]

    // 1-term polynomial: p = 1 + r
    fadd    z8.d, z30.d, z0.d

    // Store poly
    st1d    z8.d, p0, [x1, #3, mul vl]

    // final = 2^n * p
    fmul    z8.d, z8.d, z4.d

    // Store final
    st1d    z8.d, p0, [x1, #4, mul vl]

    ret
    .size exp_f64_debug, .-exp_f64_debug

    .section .rodata
    .align 3
.Lconst:
    .quad   0x3ff71547652b82fe      // 1/ln(2)
    .quad   0x3fe62e42fefa39ef      // ln(2)
    .quad   0x3ff0000000000000      // 1.0
