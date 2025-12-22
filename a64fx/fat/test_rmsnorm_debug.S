// test_rmsnorm_debug.S - Debug version of rmsnorm

.arch armv8.2-a+sve

// void rmsnorm_debug(const float* input, float* debug_out, size_t dim, float eps)
// x0 = input, x1 = debug_out, x2 = dim, s0 = eps
// debug_out[0] = sum_sq
// debug_out[1] = mean_sq
// debug_out[2] = variance_eps
// debug_out[3] = inv_std
// debug_out[4] = first output element
.global rmsnorm_debug
.type rmsnorm_debug, %function
rmsnorm_debug:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    cntw    x3                      // VL in words (16)
    ptrue   p0.s
    lsl     x10, x3, #2             // VL bytes stride

    // Sum of squares
    fmov    z0.s, #0.0
    fmov    z1.s, #0.0
    fmov    z2.s, #0.0
    fmov    z3.s, #0.0

    mov     x5, x0
    mov     x6, x2                  // dim
    lsl     x7, x3, #2              // VL*4

.Ldebug_sum_loop4:
    cmp     x6, x7
    b.lt    .Ldebug_sum_rem

    ld1w    {z4.s}, p0/z, [x5]
    add     x5, x5, x10
    ld1w    {z5.s}, p0/z, [x5]
    add     x5, x5, x10
    ld1w    {z6.s}, p0/z, [x5]
    add     x5, x5, x10
    ld1w    {z7.s}, p0/z, [x5]
    add     x5, x5, x10

    fmla    z0.s, p0/m, z4.s, z4.s
    fmla    z1.s, p0/m, z5.s, z5.s
    fmla    z2.s, p0/m, z6.s, z6.s
    fmla    z3.s, p0/m, z7.s, z7.s

    sub     x6, x6, x7
    b       .Ldebug_sum_loop4

.Ldebug_sum_rem:
    cbz     x6, .Ldebug_sum_done

.Ldebug_sum_loop1:
    whilelt p1.s, xzr, x6
    ld1w    {z4.s}, p1/z, [x5]
    fmla    z0.s, p1/m, z4.s, z4.s
    add     x5, x5, x10
    subs    x6, x6, x3
    b.gt    .Ldebug_sum_loop1

.Ldebug_sum_done:
    fadd    z0.s, p0/m, z0.s, z1.s
    fadd    z2.s, p0/m, z2.s, z3.s
    fadd    z0.s, p0/m, z0.s, z2.s
    faddv   s4, p0, z0.s            // s4 = sum_sq

    // Store sum_sq
    str     s4, [x1]                // debug_out[0] = sum_sq

    // mean_sq = sum_sq / dim
    ucvtf   s5, w2                  // s5 = (float)dim
    fdiv    s4, s4, s5              // s4 = mean_sq

    // Store mean_sq
    str     s4, [x1, #4]            // debug_out[1] = mean_sq

    // variance_eps = mean_sq + eps
    fadd    s4, s4, s0

    // Store variance_eps
    str     s4, [x1, #8]            // debug_out[2] = variance_eps

    // rsqrte + 2 NR
    frsqrte s5, s4
    fmul    s6, s5, s5
    frsqrts s6, s4, s6
    fmul    s5, s5, s6

    fmul    s6, s5, s5
    frsqrts s6, s4, s6
    fmul    s5, s5, s6              // s5 = inv_std

    // Store inv_std
    str     s5, [x1, #12]           // debug_out[3] = inv_std

    // Broadcast and compute first output
    dup     z8.s, z5.s[0]

    // Load first input, gamma=1.0 implicit
    ld1w    {z0.s}, p0/z, [x0]
    fmul    z0.s, z0.s, z8.s

    // Store first element of output
    add     x4, x1, #16
    st1w    {z0.s}, p0, [x4]        // debug_out[4..] = first vector

    ldp     x29, x30, [sp], #16
    ret
.size rmsnorm_debug, .-rmsnorm_debug
