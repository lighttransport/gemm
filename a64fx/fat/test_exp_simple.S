// test_exp_simple.S
// Simple exp test that outputs intermediate values for debugging
// Args: x0 = input (float), x1 = output array (7 floats for debug values)

    .arch armv8.2-a+sve
    .text

    .align 6
    .global test_exp_debug
    .type test_exp_debug, %function
test_exp_debug:
    ptrue   p0.s

    // Load constants
    movz    w10, #0xaa3b
    movk    w10, #0x3fb8, lsl #16   // 1/ln(2) = 1.4426950
    dup     z28.s, w10

    movz    w10, #0x7218
    movk    w10, #0x3f31, lsl #16   // ln(2) = 0.6931471
    dup     z29.s, w10

    // Load single input value and broadcast
    ldr     s0, [x0]
    dup     z0.s, z0.s[0]

    // Store x
    str     s0, [x1]               // out[0] = x

    // n = round(x / ln(2))
    fmul    z4.s, z0.s, z28.s
    frintn  z4.s, p0/m, z4.s

    // Store n
    str     s4, [x1, #4]           // out[1] = n

    // r = x - n * ln(2)
    fmls    z0.s, p0/m, z4.s, z29.s

    // Store r
    str     s0, [x1, #8]           // out[2] = r

    // Convert n to integer
    fcvtzs  z4.s, p0/m, z4.s

    // Store n_int (as float for viewing)
    scvtf   z5.s, p0/m, z4.s
    str     s5, [x1, #12]          // out[3] = n as float

    // 2^n via IEEE754
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    lsl     z4.s, z4.s, #23

    // Store 2^n
    str     s4, [x1, #16]          // out[4] = 2^n

    // Polynomial: start with 1/24
    movz    w10, #0xaaab
    movk    w10, #0x3d2a, lsl #16
    dup     z8.s, w10

    // p = p*r + 1/6
    movz    w10, #0xaaab
    movk    w10, #0x3e2a, lsl #16
    dup     z31.s, w10
    fmad    z8.s, p0/m, z0.s, z31.s

    // p = p*r + 0.5
    fmov    z31.s, #0.5
    fmad    z8.s, p0/m, z0.s, z31.s

    // p = p*r + 1.0
    fmov    z31.s, #1.0
    fmad    z8.s, p0/m, z0.s, z31.s

    // p = p*r + 1.0
    fmad    z8.s, p0/m, z0.s, z31.s

    // Store polynomial result
    str     s8, [x1, #20]          // out[5] = poly(r)

    // Final result
    fmul    z8.s, z8.s, z4.s
    str     s8, [x1, #24]          // out[6] = exp(x)

    ret
    .size test_exp_debug, .-test_exp_debug
