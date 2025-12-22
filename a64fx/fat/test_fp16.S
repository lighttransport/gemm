// test_fp16.S - Simple FP16 test
    .arch armv8.2-a+sve
    .text

    .align 6
    .global test_fp16_simple
    .type test_fp16_simple, %function
// Input: x0 = input array (16 halfs), x1 = output array
// Just multiplies each element by 2
test_fp16_simple:
    ptrue   p0.h
    ld1h    z0.h, p0/z, [x0]
    fmov    z1.h, #2.0
    fmul    z0.h, z0.h, z1.h
    st1h    z0.h, p0, [x1]
    ret
    .size test_fp16_simple, .-test_fp16_simple

    .align 6
    .global test_fp16_const
    .type test_fp16_const, %function
// x0 = output (4 halfs): store constant loaded via different methods
test_fp16_const:
    ptrue   p0.h

    // Method 1: fmov immediate
    fmov    z0.h, #1.0
    str     h0, [x0]

    // Method 2: dup from register with known constant 0x3c00 (=1.0 in FP16)
    mov     w10, #0x3c00
    dup     z1.h, w10
    str     h1, [x0, #2]

    // Method 3: dup with 0x3dc5 (=1/ln(2) in FP16)
    mov     w10, #0x3dc5
    dup     z2.h, w10
    str     h2, [x0, #4]

    // Method 4: multiply 1.0 by constant from method 3
    fmul    z3.h, z0.h, z2.h
    str     h3, [x0, #6]

    ret
    .size test_fp16_const, .-test_fp16_const

    .align 6
    .global test_fp16_exp_step
    .type test_fp16_exp_step, %function
// Debug: just test the exp steps one at a time
// x0 = input (1 half), x1 = output (8 halfs for debug values)
test_fp16_exp_step:
    ptrue   p0.h

    // Load single value and broadcast
    ldr     h0, [x0]
    dup     z0.h, z0.h[0]

    // Store input
    str     h0, [x1]                // out[0] = x

    // Load 1/ln(2) = 1.4426950 ≈ 0x3DC5 in FP16
    mov     w10, #0x3dc5
    dup     z28.h, w10

    // n = x * (1/ln2)
    fmul    z4.h, z0.h, z28.h
    str     h4, [x1, #2]            // out[1] = x/ln2

    // n = round(n)
    frintn  z4.h, p0/m, z4.h
    str     h4, [x1, #4]            // out[2] = n

    // Load ln(2) = 0.6931471 ≈ 0x398c in FP16
    mov     w10, #0x398c
    dup     z29.h, w10

    // r = x - n * ln2
    fmls    z0.h, p0/m, z4.h, z29.h
    str     h0, [x1, #6]            // out[3] = r

    // 2^n: convert n to int, add bias, shift
    fcvtzs  z4.h, p0/m, z4.h
    str     h4, [x1, #8]            // out[4] = n_int

    mov     z31.h, #15
    add     z4.h, z4.h, z31.h
    str     h4, [x1, #10]           // out[5] = n+15

    lsl     z4.h, z4.h, #10
    str     h4, [x1, #12]           // out[6] = 2^n

    // Simple test: just return 2^n * 1.0
    fmov    z8.h, #1.0
    fmul    z8.h, z8.h, z4.h
    str     h8, [x1, #14]           // out[7] = result

    ret
    .size test_fp16_exp_step, .-test_fp16_exp_step
