// test_flash_s.S
// Test kernel that computes S[0:4, 0:4] like flash_fused_nofaddv
// but just outputs S values for debugging

    .arch armv8.2-a+sve
    .text

    .align 6
    .global test_flash_s
    .type test_flash_s, %function

// Args: x0=Q, x1=Kt, x2=S_out (output 4x4 S values)
test_flash_s:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!

    mov     x4, #256            // row stride = 64 * 4 bytes

    // Q row pointers
    mov     x6, x0              // Q[0]
    add     x7, x0, x4          // Q[1]
    add     x8, x7, x4          // Q[2]
    add     x9, x8, x4          // Q[3]

    mov     x10, x1             // Kt pointer
    mov     x20, #4             // 4 columns to compute

.Lj_loop:
    // Initialize S accumulators
    fmov    s24, wzr            // S[0, j]
    fmov    s25, wzr            // S[1, j]
    fmov    s26, wzr            // S[2, j]
    fmov    s27, wzr            // S[3, j]

    // Save Q row pointers
    mov     x13, x6
    mov     x14, x7
    mov     x15, x8
    mov     x19, x9

    // Inner loop: k = 0..D (64)
    mov     x21, x10            // Kt column pointer
    mov     x22, #64            // D loop counter

.Lk_loop:
    // Load Kt[k, j]
    ldr     s28, [x21]
    add     x21, x21, x4        // stride = BC * 4 = 256

    // Load Q[i, k] and accumulate
    ldr     s29, [x13], #4
    fmadd   s24, s29, s28, s24

    ldr     s29, [x14], #4
    fmadd   s25, s29, s28, s25

    ldr     s29, [x15], #4
    fmadd   s26, s29, s28, s26

    ldr     s29, [x19], #4
    fmadd   s27, s29, s28, s27

    subs    x22, x22, #1
    b.ne    .Lk_loop

    // Store S[0:4, j]
    str     s24, [x2], #4
    str     s25, [x2], #4
    str     s26, [x2], #4
    str     s27, [x2], #4

    // Next column
    add     x10, x10, #4
    subs    x20, x20, #1
    b.ne    .Lj_loop

    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret
    .size test_flash_s, .-test_flash_s
