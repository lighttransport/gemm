/*
 * exp2 kernel that outputs column-major for 8x3 GEMM
 * Uses SVE FEXPA for proper exp2 approximation
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*============================================================================
 * exp2_colmajor_8row - exp2 with column-major output for 8 rows
 *
 * void exp2_colmajor_8row(
 *     const int32_t* S,    // x0: [8][Nc] row-major input
 *     float* P,            // x1: [Nc][8] col-major output
 *     int Nc,              // x2: number of columns (K dimension)
 *     float scale,         // s0
 *     float neg_max        // s1: -max
 * );
 *============================================================================*/

    .global exp2_colmajor_8row
    .type exp2_colmajor_8row, %function
exp2_colmajor_8row:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8, d9, [sp, #48]

    fmov    s8, s0              // scale
    fmov    s9, s1              // neg_max

    /* Row pointers (8 rows of S) */
    lsl     x3, x2, #2          // Nc * 4 = row stride in bytes
    mov     x10, x0             // S row 0
    add     x11, x10, x3        // S row 1
    add     x12, x11, x3        // S row 2
    add     x13, x12, x3        // S row 3
    add     x14, x13, x3        // S row 4
    add     x15, x14, x3        // S row 5
    add     x16, x15, x3        // S row 6
    add     x17, x16, x3        // S row 7

    /* Constants for FEXPA */
    ptrue   p0.s

    /* Broadcast scale and neg_max to SVE registers */
    mov     z24.s, s8           // scale
    mov     z25.s, s9           // neg_max

    /* 64.0f for FEXPA encoding */
    mov     w19, #0x42800000
    mov     z26.s, w19

    /* bias = 127 << 6 = 8128 */
    mov     w19, #8128
    mov     z27.s, w19

    mov     x19, x2             // Nc counter

.Lcm_loop:
    /* Load S[row][k] for all 8 rows into NEON vector */
    /* Use paired loads and inserts */
    ldr     s0, [x10], #4       // row 0
    ldr     s1, [x11], #4       // row 1
    ldr     s2, [x12], #4       // row 2
    ldr     s3, [x13], #4       // row 3

    /* Build first 4 elements in v0 */
    ins     v0.s[1], v1.s[0]
    ins     v0.s[2], v2.s[0]
    ins     v0.s[3], v3.s[0]

    ldr     s4, [x14], #4       // row 4
    ldr     s5, [x15], #4       // row 5
    ldr     s6, [x16], #4       // row 6
    ldr     s7, [x17], #4       // row 7

    /* Build second 4 elements in v4 */
    ins     v4.s[1], v5.s[0]
    ins     v4.s[2], v6.s[0]
    ins     v4.s[3], v7.s[0]

    /* Convert int32 to float (NEON) */
    scvtf   v0.4s, v0.4s
    scvtf   v4.4s, v4.4s

    /* x = S * scale + neg_max (NEON) */
    dup     v16.4s, v24.s[0]    // scale
    dup     v17.4s, v25.s[0]    // neg_max

    fmul    v0.4s, v0.4s, v16.4s
    fmul    v4.4s, v4.4s, v16.4s
    fadd    v0.4s, v0.4s, v17.4s
    fadd    v4.4s, v4.4s, v17.4s

    /* FEXPA encoding: int(x * 64) + bias (NEON) */
    dup     v18.4s, v26.s[0]    // 64.0f
    dup     v19.4s, v27.s[0]    // bias (as float bits, will reinterpret)

    fmul    v0.4s, v0.4s, v18.4s
    fmul    v4.4s, v4.4s, v18.4s
    fcvtzs  v0.4s, v0.4s
    fcvtzs  v4.4s, v4.4s

    /* Add bias - need to do integer add */
    mov     w20, #8128
    dup     v19.4s, w20
    add     v0.4s, v0.4s, v19.4s
    add     v4.4s, v4.4s, v19.4s

    /* Now we need FEXPA which is SVE only */
    /* Copy NEON to SVE, apply FEXPA, copy back */
    /* v0 and v4 map to z0 and z4 lower 128 bits */

    /* Use SVE FEXPA on the combined vector */
    /* First, combine v0 and v4 into z0 (lower 8 lanes) */
    /* z0[0:3] = v0, z0[4:7] = v4 */

    /* SVE splice or similar? Actually simpler: just use two FEXPA calls */
    /* and store separately */

    /* FEXPA on z0 (uses lower 128 bits = v0) */
    fexpa   z0.s, z0.s
    fexpa   z4.s, z4.s

    /* Store 8 floats: first 4 from v0, next 4 from v4 */
    str     q0, [x1, #0]        // rows 0-3
    str     q4, [x1, #16]       // rows 4-7
    add     x1, x1, #32

    subs    x19, x19, #1
    b.gt    .Lcm_loop

    ldp     d8, d9, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

    .size exp2_colmajor_8row, .-exp2_colmajor_8row
