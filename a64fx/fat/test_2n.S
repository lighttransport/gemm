// Test 2^n computation for FP64
    .arch armv8.2-a+sve
    .text
    .global test_2n_f64
    .type test_2n_f64, %function
test_2n_f64:
    ptrue   p0.d
    cntd    x3
    mov     x5, x2

.Lloop:
    cmp     x5, x3
    b.lt    .Ltail

    // Load n values (already as integers in double format)
    ld1d    z0.d, p0/z, [x0]

    // Convert to integer
    fcvtzs  z0.d, p0/m, z0.d

    // Add bias 1023
    mov     z31.d, #1023
    add     z0.d, z0.d, z31.d

    // Shift to exponent position
    lsl     z0.d, z0.d, #52

    // Store result (reinterpreted as double = 2^n)
    st1d    z0.d, p0, [x1]

    add     x0, x0, x3, lsl #3
    add     x1, x1, x3, lsl #3
    sub     x5, x5, x3
    b       .Lloop

.Ltail:
    cbz     x5, .Ldone
    whilelt p1.d, xzr, x5
    ld1d    z0.d, p1/z, [x0]
    fcvtzs  z0.d, p0/m, z0.d
    mov     z31.d, #1023
    add     z0.d, z0.d, z31.d
    lsl     z0.d, z0.d, #52
    st1d    z0.d, p1, [x1]

.Ldone:
    ret
    .size test_2n_f64, .-test_2n_f64
