// embedding_asm.S
// SVE-optimized Embedding Forward and Backward Kernels for A64FX
//
// Forward: output[i, :] = embedding_table[indices[i], :]
// Backward: grad_embedding[indices[i], :] += grad_output[i, :]
//
// Targeting A64FX with 512-bit SVE (VL=64 bytes, 16 FP32 per vector)

    .arch armv8.2-a+sve
    .text

//=============================================================================
// Embedding Forward FP32
// void embedding_fwd_f32_asm(const int32_t* indices, const float* emb_table,
//                            float* output, size_t batch_size, size_t hidden_dim)
// x0 = indices, x1 = emb_table, x2 = output, x3 = batch_size, x4 = hidden_dim
//
// For each i in [0, batch_size):
//     output[i * hidden_dim : (i+1) * hidden_dim] =
//         emb_table[indices[i] * hidden_dim : (indices[i]+1) * hidden_dim]
//=============================================================================
    .align 6
    .global embedding_fwd_f32_asm
    .type embedding_fwd_f32_asm, %function
embedding_fwd_f32_asm:
    stp     x29, x30, [sp, #-64]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    mov     x29, sp

    // Early exit if batch_size == 0
    cbz     x3, .Lemb_fwd_f32_done

    // Save parameters
    mov     x19, x0                 // indices
    mov     x20, x1                 // emb_table
    mov     x21, x2                 // output
    mov     x22, x3                 // batch_size
    mov     x23, x4                 // hidden_dim

    // Compute row stride in bytes: hidden_dim * 4
    lsl     x24, x4, #2             // x24 = hidden_dim * sizeof(float)

    // Get SVE vector length info
    cntw    x5                      // x5 = VL in words (16 for A64FX)
    ptrue   p0.s                    // All-true predicate

    // VL bytes = VL_words * 4
    lsl     x6, x5, #2              // x6 = VL * 4 bytes

    // 4x unroll count: VL * 4
    lsl     x7, x5, #2              // x7 = 4 * VL elements

    // Loop counter
    mov     x8, x22                 // remaining batch items

.Lemb_fwd_f32_batch_loop:
    // Load index for current item
    ldr     w9, [x19], #4           // w9 = indices[i], advance indices ptr

    // Compute source address: emb_table + index * hidden_dim * 4
    mul     x10, x9, x24            // offset = index * row_stride
    add     x10, x20, x10           // x10 = src row ptr

    // Prefetch next embedding row (if not last iteration)
    cmp     x8, #1
    b.le    .Lemb_fwd_f32_no_prefetch
    ldr     w11, [x19]              // peek next index
    mul     x12, x11, x24
    add     x12, x20, x12
    prfm    pldl1strm, [x12]        // prefetch next row
.Lemb_fwd_f32_no_prefetch:

    // Copy row: output[i, :] = emb_table[idx, :]
    mov     x11, x10                // src ptr
    mov     x12, x21                // dst ptr
    mov     x13, x23                // remaining elements

    // Main loop: 4x unrolled
.Lemb_fwd_f32_copy_loop4:
    cmp     x13, x7
    b.lt    .Lemb_fwd_f32_copy_remainder

    ld1w    {z0.s}, p0/z, [x11]
    ld1w    {z1.s}, p0/z, [x11, #1, mul vl]
    ld1w    {z2.s}, p0/z, [x11, #2, mul vl]
    ld1w    {z3.s}, p0/z, [x11, #3, mul vl]

    st1w    {z0.s}, p0, [x12]
    st1w    {z1.s}, p0, [x12, #1, mul vl]
    st1w    {z2.s}, p0, [x12, #2, mul vl]
    st1w    {z3.s}, p0, [x12, #3, mul vl]

    add     x11, x11, x7, lsl #2    // src += 4 * VL * 4
    add     x12, x12, x7, lsl #2    // dst += 4 * VL * 4
    sub     x13, x13, x7
    b       .Lemb_fwd_f32_copy_loop4

.Lemb_fwd_f32_copy_remainder:
    cbz     x13, .Lemb_fwd_f32_copy_done

.Lemb_fwd_f32_copy_loop1:
    whilelt p1.s, xzr, x13
    ld1w    {z0.s}, p1/z, [x11]
    st1w    {z0.s}, p1, [x12]
    add     x11, x11, x6
    add     x12, x12, x6
    subs    x13, x13, x5
    b.gt    .Lemb_fwd_f32_copy_loop1

.Lemb_fwd_f32_copy_done:
    // Advance output pointer
    add     x21, x21, x24

    // Next batch item
    subs    x8, x8, #1
    b.ne    .Lemb_fwd_f32_batch_loop

.Lemb_fwd_f32_done:
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret
    .size embedding_fwd_f32_asm, .-embedding_fwd_f32_asm


//=============================================================================
// Embedding Forward FP32 - Batched Version (processes 4 tokens at once)
// Better for memory bandwidth utilization
// void embedding_fwd_f32_batched_asm(const int32_t* indices, const float* emb_table,
//                                    float* output, size_t batch_size, size_t hidden_dim)
//=============================================================================
    .align 6
    .global embedding_fwd_f32_batched_asm
    .type embedding_fwd_f32_batched_asm, %function
embedding_fwd_f32_batched_asm:
    stp     x29, x30, [sp, #-80]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    mov     x29, sp

    cbz     x3, .Lemb_fwd_f32_batched_done

    mov     x19, x0                 // indices
    mov     x20, x1                 // emb_table
    mov     x21, x2                 // output
    mov     x22, x3                 // batch_size
    mov     x23, x4                 // hidden_dim

    lsl     x24, x4, #2             // row_stride = hidden_dim * 4

    cntw    x5
    ptrue   p0.s
    lsl     x6, x5, #2              // VL bytes

    // Process 4 batch items at a time
    mov     x25, x22                // remaining
    lsr     x26, x22, #2            // batch / 4

    cbz     x26, .Lemb_fwd_f32_batched_remainder

.Lemb_fwd_f32_batched_loop4:
    // Load 4 indices
    ldp     w8, w9, [x19]
    ldp     w10, w11, [x19, #8]
    add     x19, x19, #16

    // Compute 4 source addresses
    mul     x12, x8, x24
    mul     x13, x9, x24
    mul     x14, x10, x24
    mul     x15, x11, x24
    add     x12, x20, x12           // src0
    add     x13, x20, x13           // src1
    add     x14, x20, x14           // src2
    add     x15, x20, x15           // src3

    // Output addresses
    mov     x16, x21                // dst0
    add     x17, x21, x24           // dst1
    add     x8, x17, x24            // dst2 (reuse x8)
    add     x9, x8, x24             // dst3 (reuse x9)

    // Copy all 4 rows - interleave loads and stores for better pipelining
    mov     x10, x23                // remaining elements

.Lemb_fwd_f32_batched_inner:
    cmp     x10, x5
    b.lt    .Lemb_fwd_f32_batched_inner_rem

    // Load one vector from each source row
    ld1w    {z0.s}, p0/z, [x12]
    ld1w    {z1.s}, p0/z, [x13]
    ld1w    {z2.s}, p0/z, [x14]
    ld1w    {z3.s}, p0/z, [x15]

    // Store to corresponding output rows
    st1w    {z0.s}, p0, [x16]
    st1w    {z1.s}, p0, [x17]
    st1w    {z2.s}, p0, [x8]
    st1w    {z3.s}, p0, [x9]

    // Advance all pointers
    add     x12, x12, x6
    add     x13, x13, x6
    add     x14, x14, x6
    add     x15, x15, x6
    add     x16, x16, x6
    add     x17, x17, x6
    add     x8, x8, x6
    add     x9, x9, x6

    sub     x10, x10, x5
    b       .Lemb_fwd_f32_batched_inner

.Lemb_fwd_f32_batched_inner_rem:
    cbz     x10, .Lemb_fwd_f32_batched_inner_done

    whilelt p1.s, xzr, x10
    ld1w    {z0.s}, p1/z, [x12]
    ld1w    {z1.s}, p1/z, [x13]
    ld1w    {z2.s}, p1/z, [x14]
    ld1w    {z3.s}, p1/z, [x15]
    st1w    {z0.s}, p1, [x16]
    st1w    {z1.s}, p1, [x17]
    st1w    {z2.s}, p1, [x8]
    st1w    {z3.s}, p1, [x9]

.Lemb_fwd_f32_batched_inner_done:
    // Advance output base by 4 rows
    add     x21, x21, x24, lsl #2

    sub     x25, x25, #4
    subs    x26, x26, #1
    b.ne    .Lemb_fwd_f32_batched_loop4

.Lemb_fwd_f32_batched_remainder:
    // Handle remaining 0-3 items
    and     x25, x22, #3
    cbz     x25, .Lemb_fwd_f32_batched_done

.Lemb_fwd_f32_batched_rem_loop:
    ldr     w8, [x19], #4
    mul     x10, x8, x24
    add     x10, x20, x10           // src

    mov     x11, x21                // dst
    mov     x12, x23                // count

.Lemb_fwd_f32_batched_rem_copy:
    whilelt p1.s, xzr, x12
    ld1w    {z0.s}, p1/z, [x10]
    st1w    {z0.s}, p1, [x11]
    add     x10, x10, x6
    add     x11, x11, x6
    subs    x12, x12, x5
    b.gt    .Lemb_fwd_f32_batched_rem_copy

    add     x21, x21, x24
    subs    x25, x25, #1
    b.ne    .Lemb_fwd_f32_batched_rem_loop

.Lemb_fwd_f32_batched_done:
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
    .size embedding_fwd_f32_batched_asm, .-embedding_fwd_f32_batched_asm


//=============================================================================
// Embedding Backward FP32 (Gradient Accumulation)
// void embedding_bwd_f32_asm(const int32_t* indices, const float* grad_output,
//                            float* grad_embedding, size_t batch_size,
//                            size_t hidden_dim, size_t vocab_size)
// x0 = indices, x1 = grad_output, x2 = grad_embedding
// x3 = batch_size, x4 = hidden_dim, x5 = vocab_size (unused but for API consistency)
//
// For each i in [0, batch_size):
//     grad_embedding[indices[i], :] += grad_output[i, :]
//
// Note: This version assumes no duplicate indices in a batch for simplicity.
// For handling duplicates, see embedding_bwd_f32_atomic_asm below.
//=============================================================================
    .align 6
    .global embedding_bwd_f32_asm
    .type embedding_bwd_f32_asm, %function
embedding_bwd_f32_asm:
    stp     x29, x30, [sp, #-64]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    mov     x29, sp

    cbz     x3, .Lemb_bwd_f32_done

    mov     x19, x0                 // indices
    mov     x20, x1                 // grad_output
    mov     x21, x2                 // grad_embedding
    mov     x22, x3                 // batch_size
    mov     x23, x4                 // hidden_dim

    lsl     x24, x4, #2             // row_stride

    cntw    x5
    ptrue   p0.s
    lsl     x6, x5, #2              // VL bytes
    lsl     x7, x5, #2              // 4x unroll count

    mov     x8, x22                 // remaining batch items

.Lemb_bwd_f32_batch_loop:
    // Load index
    ldr     w9, [x19], #4

    // Compute grad_embedding row address
    mul     x10, x9, x24
    add     x10, x21, x10           // dst = grad_embedding + idx * stride

    // grad_output row address
    mov     x11, x20                // src = current grad_output row

    mov     x12, x23                // remaining elements

    // Main loop: 4x unrolled accumulation
.Lemb_bwd_f32_accum_loop4:
    cmp     x12, x7
    b.lt    .Lemb_bwd_f32_accum_remainder

    // Load grad_output
    ld1w    {z0.s}, p0/z, [x11]
    ld1w    {z1.s}, p0/z, [x11, #1, mul vl]
    ld1w    {z2.s}, p0/z, [x11, #2, mul vl]
    ld1w    {z3.s}, p0/z, [x11, #3, mul vl]

    // Load current grad_embedding
    ld1w    {z4.s}, p0/z, [x10]
    ld1w    {z5.s}, p0/z, [x10, #1, mul vl]
    ld1w    {z6.s}, p0/z, [x10, #2, mul vl]
    ld1w    {z7.s}, p0/z, [x10, #3, mul vl]

    // Accumulate
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s

    // Store back
    st1w    {z4.s}, p0, [x10]
    st1w    {z5.s}, p0, [x10, #1, mul vl]
    st1w    {z6.s}, p0, [x10, #2, mul vl]
    st1w    {z7.s}, p0, [x10, #3, mul vl]

    add     x10, x10, x7, lsl #2
    add     x11, x11, x7, lsl #2
    sub     x12, x12, x7
    b       .Lemb_bwd_f32_accum_loop4

.Lemb_bwd_f32_accum_remainder:
    cbz     x12, .Lemb_bwd_f32_accum_done

.Lemb_bwd_f32_accum_loop1:
    whilelt p1.s, xzr, x12
    ld1w    {z0.s}, p1/z, [x11]
    ld1w    {z4.s}, p1/z, [x10]
    fadd    z4.s, p1/m, z4.s, z0.s
    st1w    {z4.s}, p1, [x10]
    add     x10, x10, x6
    add     x11, x11, x6
    subs    x12, x12, x5
    b.gt    .Lemb_bwd_f32_accum_loop1

.Lemb_bwd_f32_accum_done:
    // Advance grad_output pointer
    add     x20, x20, x24

    subs    x8, x8, #1
    b.ne    .Lemb_bwd_f32_batch_loop

.Lemb_bwd_f32_done:
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret
    .size embedding_bwd_f32_asm, .-embedding_bwd_f32_asm


//=============================================================================
// Embedding Forward FP64
// void embedding_fwd_f64_asm(const int32_t* indices, const double* emb_table,
//                            double* output, size_t batch_size, size_t hidden_dim)
//=============================================================================
    .align 6
    .global embedding_fwd_f64_asm
    .type embedding_fwd_f64_asm, %function
embedding_fwd_f64_asm:
    stp     x29, x30, [sp, #-64]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    mov     x29, sp

    cbz     x3, .Lemb_fwd_f64_done

    mov     x19, x0                 // indices
    mov     x20, x1                 // emb_table
    mov     x21, x2                 // output
    mov     x22, x3                 // batch_size
    mov     x23, x4                 // hidden_dim

    lsl     x24, x4, #3             // row_stride = hidden_dim * 8

    cntd    x5                      // VL in doublewords (8 for A64FX)
    ptrue   p0.d
    lsl     x6, x5, #3              // VL bytes
    lsl     x7, x5, #2              // 4x unroll count

    mov     x8, x22

.Lemb_fwd_f64_batch_loop:
    ldr     w9, [x19], #4
    mul     x10, x9, x24
    add     x10, x20, x10           // src

    // Prefetch
    cmp     x8, #1
    b.le    .Lemb_fwd_f64_no_pf
    ldr     w11, [x19]
    mul     x12, x11, x24
    add     x12, x20, x12
    prfm    pldl1strm, [x12]
.Lemb_fwd_f64_no_pf:

    mov     x11, x10
    mov     x12, x21
    mov     x13, x23

.Lemb_fwd_f64_copy_loop4:
    cmp     x13, x7
    b.lt    .Lemb_fwd_f64_copy_rem

    ld1d    {z0.d}, p0/z, [x11]
    ld1d    {z1.d}, p0/z, [x11, #1, mul vl]
    ld1d    {z2.d}, p0/z, [x11, #2, mul vl]
    ld1d    {z3.d}, p0/z, [x11, #3, mul vl]

    st1d    {z0.d}, p0, [x12]
    st1d    {z1.d}, p0, [x12, #1, mul vl]
    st1d    {z2.d}, p0, [x12, #2, mul vl]
    st1d    {z3.d}, p0, [x12, #3, mul vl]

    add     x11, x11, x7, lsl #3
    add     x12, x12, x7, lsl #3
    sub     x13, x13, x7
    b       .Lemb_fwd_f64_copy_loop4

.Lemb_fwd_f64_copy_rem:
    cbz     x13, .Lemb_fwd_f64_copy_done

.Lemb_fwd_f64_copy_loop1:
    whilelt p1.d, xzr, x13
    ld1d    {z0.d}, p1/z, [x11]
    st1d    {z0.d}, p1, [x12]
    add     x11, x11, x6
    add     x12, x12, x6
    subs    x13, x13, x5
    b.gt    .Lemb_fwd_f64_copy_loop1

.Lemb_fwd_f64_copy_done:
    add     x21, x21, x24
    subs    x8, x8, #1
    b.ne    .Lemb_fwd_f64_batch_loop

.Lemb_fwd_f64_done:
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret
    .size embedding_fwd_f64_asm, .-embedding_fwd_f64_asm


//=============================================================================
// Embedding Backward FP64
// void embedding_bwd_f64_asm(const int32_t* indices, const double* grad_output,
//                            double* grad_embedding, size_t batch_size,
//                            size_t hidden_dim, size_t vocab_size)
//=============================================================================
    .align 6
    .global embedding_bwd_f64_asm
    .type embedding_bwd_f64_asm, %function
embedding_bwd_f64_asm:
    stp     x29, x30, [sp, #-64]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    mov     x29, sp

    cbz     x3, .Lemb_bwd_f64_done

    mov     x19, x0                 // indices
    mov     x20, x1                 // grad_output
    mov     x21, x2                 // grad_embedding
    mov     x22, x3                 // batch_size
    mov     x23, x4                 // hidden_dim

    lsl     x24, x4, #3             // row_stride

    cntd    x5
    ptrue   p0.d
    lsl     x6, x5, #3
    lsl     x7, x5, #2

    mov     x8, x22

.Lemb_bwd_f64_batch_loop:
    ldr     w9, [x19], #4
    mul     x10, x9, x24
    add     x10, x21, x10           // dst

    mov     x11, x20                // src
    mov     x12, x23

.Lemb_bwd_f64_accum_loop4:
    cmp     x12, x7
    b.lt    .Lemb_bwd_f64_accum_rem

    ld1d    {z0.d}, p0/z, [x11]
    ld1d    {z1.d}, p0/z, [x11, #1, mul vl]
    ld1d    {z2.d}, p0/z, [x11, #2, mul vl]
    ld1d    {z3.d}, p0/z, [x11, #3, mul vl]

    ld1d    {z4.d}, p0/z, [x10]
    ld1d    {z5.d}, p0/z, [x10, #1, mul vl]
    ld1d    {z6.d}, p0/z, [x10, #2, mul vl]
    ld1d    {z7.d}, p0/z, [x10, #3, mul vl]

    fadd    z4.d, p0/m, z4.d, z0.d
    fadd    z5.d, p0/m, z5.d, z1.d
    fadd    z6.d, p0/m, z6.d, z2.d
    fadd    z7.d, p0/m, z7.d, z3.d

    st1d    {z4.d}, p0, [x10]
    st1d    {z5.d}, p0, [x10, #1, mul vl]
    st1d    {z6.d}, p0, [x10, #2, mul vl]
    st1d    {z7.d}, p0, [x10, #3, mul vl]

    add     x10, x10, x7, lsl #3
    add     x11, x11, x7, lsl #3
    sub     x12, x12, x7
    b       .Lemb_bwd_f64_accum_loop4

.Lemb_bwd_f64_accum_rem:
    cbz     x12, .Lemb_bwd_f64_accum_done

.Lemb_bwd_f64_accum_loop1:
    whilelt p1.d, xzr, x12
    ld1d    {z0.d}, p1/z, [x11]
    ld1d    {z4.d}, p1/z, [x10]
    fadd    z4.d, p1/m, z4.d, z0.d
    st1d    {z4.d}, p1, [x10]
    add     x10, x10, x6
    add     x11, x11, x6
    subs    x12, x12, x5
    b.gt    .Lemb_bwd_f64_accum_loop1

.Lemb_bwd_f64_accum_done:
    add     x20, x20, x24
    subs    x8, x8, #1
    b.ne    .Lemb_bwd_f64_batch_loop

.Lemb_bwd_f64_done:
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret
    .size embedding_bwd_f64_asm, .-embedding_bwd_f64_asm


//=============================================================================
// Embedding Forward with Position Encoding (token + position)
// void embedding_fwd_with_pos_f32_asm(const int32_t* token_ids,
//                                     const int32_t* position_ids,
//                                     const float* token_emb,
//                                     const float* pos_emb,
//                                     float* output,
//                                     size_t batch_size, size_t hidden_dim)
// x0 = token_ids, x1 = position_ids, x2 = token_emb, x3 = pos_emb
// x4 = output, x5 = batch_size, x6 = hidden_dim
//
// output[i, :] = token_emb[token_ids[i], :] + pos_emb[position_ids[i], :]
//=============================================================================
    .align 6
    .global embedding_fwd_with_pos_f32_asm
    .type embedding_fwd_with_pos_f32_asm, %function
embedding_fwd_with_pos_f32_asm:
    stp     x29, x30, [sp, #-96]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    mov     x29, sp

    cbz     x5, .Lemb_pos_fwd_done

    mov     x19, x0                 // token_ids
    mov     x20, x1                 // position_ids
    mov     x21, x2                 // token_emb
    mov     x22, x3                 // pos_emb
    mov     x23, x4                 // output
    mov     x24, x5                 // batch_size
    mov     x25, x6                 // hidden_dim

    lsl     x26, x6, #2             // row_stride

    cntw    x7
    ptrue   p0.s
    lsl     x8, x7, #2              // VL bytes
    lsl     x9, x7, #2              // 4x unroll

    mov     x27, x24                // remaining

.Lemb_pos_fwd_loop:
    // Load token and position indices
    ldr     w10, [x19], #4          // token_id
    ldr     w11, [x20], #4          // position_id

    // Compute source addresses
    mul     x12, x10, x26
    mul     x13, x11, x26
    add     x12, x21, x12           // token_emb row
    add     x13, x22, x13           // pos_emb row

    mov     x14, x23                // output row
    mov     x15, x25                // remaining elements

.Lemb_pos_fwd_inner4:
    cmp     x15, x9
    b.lt    .Lemb_pos_fwd_inner_rem

    // Load token embeddings
    ld1w    {z0.s}, p0/z, [x12]
    ld1w    {z1.s}, p0/z, [x12, #1, mul vl]
    ld1w    {z2.s}, p0/z, [x12, #2, mul vl]
    ld1w    {z3.s}, p0/z, [x12, #3, mul vl]

    // Load position embeddings
    ld1w    {z4.s}, p0/z, [x13]
    ld1w    {z5.s}, p0/z, [x13, #1, mul vl]
    ld1w    {z6.s}, p0/z, [x13, #2, mul vl]
    ld1w    {z7.s}, p0/z, [x13, #3, mul vl]

    // Add
    fadd    z0.s, p0/m, z0.s, z4.s
    fadd    z1.s, p0/m, z1.s, z5.s
    fadd    z2.s, p0/m, z2.s, z6.s
    fadd    z3.s, p0/m, z3.s, z7.s

    // Store
    st1w    {z0.s}, p0, [x14]
    st1w    {z1.s}, p0, [x14, #1, mul vl]
    st1w    {z2.s}, p0, [x14, #2, mul vl]
    st1w    {z3.s}, p0, [x14, #3, mul vl]

    add     x12, x12, x9, lsl #2
    add     x13, x13, x9, lsl #2
    add     x14, x14, x9, lsl #2
    sub     x15, x15, x9
    b       .Lemb_pos_fwd_inner4

.Lemb_pos_fwd_inner_rem:
    cbz     x15, .Lemb_pos_fwd_inner_done

.Lemb_pos_fwd_inner1:
    whilelt p1.s, xzr, x15
    ld1w    {z0.s}, p1/z, [x12]
    ld1w    {z4.s}, p1/z, [x13]
    fadd    z0.s, p1/m, z0.s, z4.s
    st1w    {z0.s}, p1, [x14]
    add     x12, x12, x8
    add     x13, x13, x8
    add     x14, x14, x8
    subs    x15, x15, x7
    b.gt    .Lemb_pos_fwd_inner1

.Lemb_pos_fwd_inner_done:
    add     x23, x23, x26
    subs    x27, x27, #1
    b.ne    .Lemb_pos_fwd_loop

.Lemb_pos_fwd_done:
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret
    .size embedding_fwd_with_pos_f32_asm, .-embedding_fwd_with_pos_f32_asm


//=============================================================================
// Zero-initialize embedding gradient table
// void embedding_grad_zero_f32_asm(float* grad_embedding,
//                                  size_t vocab_size, size_t hidden_dim)
// x0 = grad_embedding, x1 = vocab_size, x2 = hidden_dim
//=============================================================================
    .align 6
    .global embedding_grad_zero_f32_asm
    .type embedding_grad_zero_f32_asm, %function
embedding_grad_zero_f32_asm:
    mul     x3, x1, x2              // total elements
    cbz     x3, .Lemb_zero_done

    cntw    x4
    ptrue   p0.s
    lsl     x5, x4, #2              // VL bytes
    lsl     x6, x4, #2              // 4x unroll

    fmov    z0.s, #0.0
    fmov    z1.s, #0.0
    fmov    z2.s, #0.0
    fmov    z3.s, #0.0

.Lemb_zero_loop4:
    cmp     x3, x6
    b.lt    .Lemb_zero_rem

    st1w    {z0.s}, p0, [x0]
    st1w    {z1.s}, p0, [x0, #1, mul vl]
    st1w    {z2.s}, p0, [x0, #2, mul vl]
    st1w    {z3.s}, p0, [x0, #3, mul vl]

    add     x0, x0, x6, lsl #2
    sub     x3, x3, x6
    b       .Lemb_zero_loop4

.Lemb_zero_rem:
    cbz     x3, .Lemb_zero_done

.Lemb_zero_loop1:
    whilelt p1.s, xzr, x3
    st1w    {z0.s}, p1, [x0]
    add     x0, x0, x5
    subs    x3, x3, x4
    b.gt    .Lemb_zero_loop1

.Lemb_zero_done:
    ret
    .size embedding_grad_zero_f32_asm, .-embedding_grad_zero_f32_asm


//=============================================================================
// Embedding Backward with Duplicate Index Handling
// This version uses sorting + segmented reduction approach
// void embedding_bwd_sorted_f32_asm(const int32_t* sorted_indices,
//                                   const int32_t* original_positions,
//                                   const float* grad_output,
//                                   float* grad_embedding,
//                                   size_t batch_size, size_t hidden_dim)
//
// Assumes indices are pre-sorted and original_positions tells us where
// each sorted entry came from in the original grad_output
//=============================================================================
    .align 6
    .global embedding_bwd_sorted_f32_asm
    .type embedding_bwd_sorted_f32_asm, %function
embedding_bwd_sorted_f32_asm:
    stp     x29, x30, [sp, #-80]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    mov     x29, sp

    cbz     x4, .Lemb_bwd_sorted_done

    mov     x19, x0                 // sorted_indices
    mov     x20, x1                 // original_positions
    mov     x21, x2                 // grad_output
    mov     x22, x3                 // grad_embedding
    mov     x23, x4                 // batch_size
    mov     x24, x5                 // hidden_dim

    lsl     x25, x5, #2             // row_stride

    cntw    x6
    ptrue   p0.s
    lsl     x7, x6, #2              // VL bytes

    mov     x8, #0                  // current position in sorted array

.Lemb_bwd_sorted_outer:
    cmp     x8, x23
    b.ge    .Lemb_bwd_sorted_done

    // Load current index
    ldr     w9, [x19, x8, lsl #2]   // current sorted index

    // Compute destination address
    mul     x10, x9, x25
    add     x10, x22, x10           // grad_embedding row

    // Initialize accumulator for this index
    mov     x11, x24                // hidden_dim count

.Lemb_bwd_sorted_accum_init:
    // Zero initialize the accumulator in grad_embedding row
    // (This might be redundant if already zeroed, but safe)

    // Find all consecutive entries with same index and accumulate
    mov     x12, x8                 // start of group

.Lemb_bwd_sorted_group:
    // Get original position to find grad_output row
    ldr     w13, [x20, x12, lsl #2] // original position
    mul     x14, x13, x25
    add     x14, x21, x14           // grad_output row

    // Accumulate this grad_output row into grad_embedding row
    mov     x15, x10                // dst ptr
    mov     x16, x14                // src ptr
    mov     x17, x24                // count

.Lemb_bwd_sorted_inner:
    whilelt p1.s, xzr, x17
    ld1w    {z0.s}, p1/z, [x16]
    ld1w    {z1.s}, p1/z, [x15]
    fadd    z1.s, p1/m, z1.s, z0.s
    st1w    {z1.s}, p1, [x15]
    add     x15, x15, x7
    add     x16, x16, x7
    subs    x17, x17, x6
    b.gt    .Lemb_bwd_sorted_inner

    // Move to next in group
    add     x12, x12, #1
    cmp     x12, x23
    b.ge    .Lemb_bwd_sorted_group_done

    // Check if next has same index
    ldr     w17, [x19, x12, lsl #2]
    cmp     w17, w9
    b.eq    .Lemb_bwd_sorted_group

.Lemb_bwd_sorted_group_done:
    mov     x8, x12                 // move past this group
    b       .Lemb_bwd_sorted_outer

.Lemb_bwd_sorted_done:
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
    .size embedding_bwd_sorted_f32_asm, .-embedding_bwd_sorted_f32_asm


//=============================================================================
// Embedding Forward FP32 - SVE Gather/Scatter (16 tokens column-wise)
// void embedding_fwd_f32_gather_asm(const int32_t* indices,
//                                    const float* emb_table,
//                                    float* output,
//                                    size_t seq_len, size_t hidden_dim)
// x0 = indices, x1 = emb_table, x2 = output, x3 = seq_len, x4 = hidden_dim
//
// Processes 16 tokens simultaneously using SVE gather loads and scatter stores.
// Iterates column-wise over the hidden dimension.
// Constraint: vocab_size * hidden_dim < 2^32 (32-bit word offsets)
//
// Register plan:
//   z0-z3:   gathered data (4x column unroll)
//   z16-z19: gather offsets for columns j+0..j+3
//   z20-z23: scatter offsets for columns j+0..j+3
//   z28:     loaded indices
//   z31:     dup(hidden_dim) for multiply
//   p0:      all-true (.s)
//   p1:      tail mask for remainder tokens
//=============================================================================
    .align 6
    .global embedding_fwd_f32_gather_asm
    .type embedding_fwd_f32_gather_asm, %function
embedding_fwd_f32_gather_asm:
    stp     x29, x30, [sp, #-80]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    mov     x29, sp

    cbz     x3, .Lgather_done

    mov     x19, x0             // indices
    mov     x20, x1             // emb_table
    mov     x21, x2             // output
    mov     x22, x3             // seq_len
    mov     x23, x4             // hidden_dim
    lsl     x24, x4, #2         // row_stride_bytes = dim * 4

    ptrue   p0.s

    // z31 = dup(hidden_dim) for index multiplication
    mov     w5, w4
    mov     z31.s, w5

    mov     x25, #0             // batch_start = 0

.Lgather_batch16:
    sub     x6, x22, x25        // remaining = seq_len - batch_start
    cmp     x6, #16
    b.lt    .Lgather_tail

    // Load 16 indices
    add     x7, x19, x25, lsl #2
    ld1w    {z28.s}, p0/z, [x7]

    // Gather word offsets: z16[i] = indices[i] * hidden_dim
    movprfx z16, z28
    mul     z16.s, p0/m, z16.s, z31.s

    // Scatter word offsets: z20[i] = (batch_start + i) * hidden_dim
    mul     w7, w25, w4          // w7 = batch_start * dim
    mov     w8, w4               // w8 = dim (step)
    index   z20.s, w7, w8

    // 4x column unroll: offset+1, +2, +3 (ADD imm is destructive, need movprfx)
    movprfx z17, z16
    add     z17.s, z17.s, #1
    movprfx z18, z16
    add     z18.s, z18.s, #2
    movprfx z19, z16
    add     z19.s, z19.s, #3
    movprfx z21, z20
    add     z21.s, z21.s, #1
    movprfx z22, z20
    add     z22.s, z22.s, #2
    movprfx z23, z20
    add     z23.s, z23.s, #3

    // Inner loop: iterate over columns, 4 at a time
    mov     x6, x23              // remaining columns

.Lgather_col4:
    cmp     x6, #4
    b.lt    .Lgather_col_tail

    // Gather 4 columns from 16 rows
    ld1w    {z0.s}, p0/z, [x20, z16.s, UXTW #2]
    ld1w    {z1.s}, p0/z, [x20, z17.s, UXTW #2]
    ld1w    {z2.s}, p0/z, [x20, z18.s, UXTW #2]
    ld1w    {z3.s}, p0/z, [x20, z19.s, UXTW #2]

    // Scatter 4 columns to 16 output positions
    st1w    {z0.s}, p0, [x21, z20.s, UXTW #2]
    st1w    {z1.s}, p0, [x21, z21.s, UXTW #2]
    st1w    {z2.s}, p0, [x21, z22.s, UXTW #2]
    st1w    {z3.s}, p0, [x21, z23.s, UXTW #2]

    // Advance all offset vectors by 4 columns
    add     z16.s, z16.s, #4
    add     z17.s, z17.s, #4
    add     z18.s, z18.s, #4
    add     z19.s, z19.s, #4
    add     z20.s, z20.s, #4
    add     z21.s, z21.s, #4
    add     z22.s, z22.s, #4
    add     z23.s, z23.s, #4

    sub     x6, x6, #4
    b       .Lgather_col4

.Lgather_col_tail:
    cbz     x6, .Lgather_next_batch

.Lgather_col1:
    ld1w    {z0.s}, p0/z, [x20, z16.s, UXTW #2]
    st1w    {z0.s}, p0, [x21, z20.s, UXTW #2]
    add     z16.s, z16.s, #1
    add     z20.s, z20.s, #1
    subs    x6, x6, #1
    b.gt    .Lgather_col1

.Lgather_next_batch:
    add     x25, x25, #16
    b       .Lgather_batch16

.Lgather_tail:
    // Handle remaining < 16 tokens
    sub     x6, x22, x25
    cbz     x6, .Lgather_done

    // Predicate for remaining tokens
    whilelt p1.s, xzr, x6

    // Load remaining indices
    add     x7, x19, x25, lsl #2
    ld1w    {z28.s}, p1/z, [x7]

    // Compute offsets (all lanes via p0; inactive lanes don't matter for gather/scatter with p1)
    movprfx z16, z28
    mul     z16.s, p0/m, z16.s, z31.s

    mul     w7, w25, w4
    mov     w8, w4
    index   z20.s, w7, w8

    movprfx z17, z16
    add     z17.s, z17.s, #1
    movprfx z18, z16
    add     z18.s, z18.s, #2
    movprfx z19, z16
    add     z19.s, z19.s, #3
    movprfx z21, z20
    add     z21.s, z21.s, #1
    movprfx z22, z20
    add     z22.s, z22.s, #2
    movprfx z23, z20
    add     z23.s, z23.s, #3

    mov     x6, x23

.Lgather_tail_col4:
    cmp     x6, #4
    b.lt    .Lgather_tail_col1

    ld1w    {z0.s}, p1/z, [x20, z16.s, UXTW #2]
    ld1w    {z1.s}, p1/z, [x20, z17.s, UXTW #2]
    ld1w    {z2.s}, p1/z, [x20, z18.s, UXTW #2]
    ld1w    {z3.s}, p1/z, [x20, z19.s, UXTW #2]

    st1w    {z0.s}, p1, [x21, z20.s, UXTW #2]
    st1w    {z1.s}, p1, [x21, z21.s, UXTW #2]
    st1w    {z2.s}, p1, [x21, z22.s, UXTW #2]
    st1w    {z3.s}, p1, [x21, z23.s, UXTW #2]

    add     z16.s, z16.s, #4
    add     z17.s, z17.s, #4
    add     z18.s, z18.s, #4
    add     z19.s, z19.s, #4
    add     z20.s, z20.s, #4
    add     z21.s, z21.s, #4
    add     z22.s, z22.s, #4
    add     z23.s, z23.s, #4

    sub     x6, x6, #4
    b       .Lgather_tail_col4

.Lgather_tail_col1:
    cbz     x6, .Lgather_done

.Lgather_tail_col1_loop:
    ld1w    {z0.s}, p1/z, [x20, z16.s, UXTW #2]
    st1w    {z0.s}, p1, [x21, z20.s, UXTW #2]
    add     z16.s, z16.s, #1
    add     z20.s, z20.s, #1
    subs    x6, x6, #1
    b.gt    .Lgather_tail_col1_loop

.Lgather_done:
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
    .size embedding_fwd_f32_gather_asm, .-embedding_fwd_f32_gather_asm


//=============================================================================
// Embedding Forward FP32 - Deep-Prefetch Stream
// void embedding_fwd_f32_stream_asm(const int32_t* indices,
//                                    const float* emb_table,
//                                    float* output,
//                                    size_t seq_len, size_t hidden_dim)
// x0 = indices, x1 = emb_table, x2 = output, x3 = seq_len, x4 = hidden_dim
//
// Processes tokens sequentially with 8x vector unroll (128 FP32 = 512B per iter)
// and deep prefetching of rows 8 tokens ahead to hide HBM2 latency (~260 cycles).
//
// Register plan:
//   z0-z7:   data (8x unroll)
//   x19-x26: saved parameters
//   p0:      all-true (.s)
//=============================================================================
    .align 6
    .global embedding_fwd_f32_stream_asm
    .type embedding_fwd_f32_stream_asm, %function
embedding_fwd_f32_stream_asm:
    stp     x29, x30, [sp, #-80]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    mov     x29, sp

    cbz     x3, .Lstream_done

    mov     x19, x0             // indices
    mov     x20, x1             // emb_table
    mov     x21, x2             // output
    mov     x22, x3             // seq_len
    mov     x23, x4             // hidden_dim
    lsl     x24, x4, #2         // row_stride_bytes = dim * 4

    cntw    x5                   // VL words = 16
    ptrue   p0.s
    lsl     x25, x5, #5         // 8 * VL * 4 bytes = 512 (8x unroll stride)
    lsl     x26, x5, #3         // 8 * VL = 128 elements per 8x iter

    // Issue initial prefetches for first 8 rows
    mov     x6, #0
    mov     x7, #8
    cmp     x7, x22
    csel    x7, x7, x22, lo     // min(8, seq_len)
.Lstream_init_pf:
    cmp     x6, x7
    b.ge    .Lstream_init_pf_done
    ldr     w8, [x19, x6, lsl #2]
    mul     x9, x8, x24
    add     x9, x20, x9
    prfm    pldl1strm, [x9]
    prfm    pldl1strm, [x9, #256]
    prfm    pldl1strm, [x9, #512]
    prfm    pldl1strm, [x9, #768]
    add     x6, x6, #1
    b       .Lstream_init_pf
.Lstream_init_pf_done:

    mov     x8, #0              // current token index

.Lstream_outer:
    cmp     x8, x22
    b.ge    .Lstream_done

    // Load current index
    ldr     w9, [x19, x8, lsl #2]

    // Source address: emb_table + idx * stride
    mul     x10, x9, x24
    add     x10, x20, x10

    // Destination address: output + i * stride
    mul     x11, x8, x24
    add     x11, x21, x11

    // Prefetch row for token i+8
    add     x12, x8, #8
    cmp     x12, x22
    b.ge    .Lstream_no_pf

    ldr     w13, [x19, x12, lsl #2]
    mul     x14, x13, x24
    add     x14, x20, x14
    prfm    pldl1strm, [x14]
    prfm    pldl1strm, [x14, #256]
    prfm    pldl1strm, [x14, #512]
    prfm    pldl1strm, [x14, #768]

.Lstream_no_pf:

    // Copy row: 8x unrolled inner loop
    mov     x15, x23             // remaining elements

.Lstream_inner8:
    cmp     x15, x26
    b.lt    .Lstream_inner_rem

    ld1w    {z0.s}, p0/z, [x10]
    ld1w    {z1.s}, p0/z, [x10, #1, mul vl]
    ld1w    {z2.s}, p0/z, [x10, #2, mul vl]
    ld1w    {z3.s}, p0/z, [x10, #3, mul vl]
    ld1w    {z4.s}, p0/z, [x10, #4, mul vl]
    ld1w    {z5.s}, p0/z, [x10, #5, mul vl]
    ld1w    {z6.s}, p0/z, [x10, #6, mul vl]
    ld1w    {z7.s}, p0/z, [x10, #7, mul vl]

    st1w    {z0.s}, p0, [x11]
    st1w    {z1.s}, p0, [x11, #1, mul vl]
    st1w    {z2.s}, p0, [x11, #2, mul vl]
    st1w    {z3.s}, p0, [x11, #3, mul vl]
    st1w    {z4.s}, p0, [x11, #4, mul vl]
    st1w    {z5.s}, p0, [x11, #5, mul vl]
    st1w    {z6.s}, p0, [x11, #6, mul vl]
    st1w    {z7.s}, p0, [x11, #7, mul vl]

    add     x10, x10, x25       // src += 8*VL*4
    add     x11, x11, x25       // dst += 8*VL*4
    sub     x15, x15, x26
    b       .Lstream_inner8

.Lstream_inner_rem:
    cbz     x15, .Lstream_next

.Lstream_inner1:
    whilelt p1.s, xzr, x15
    ld1w    {z0.s}, p1/z, [x10]
    st1w    {z0.s}, p1, [x11]
    add     x10, x10, x5, lsl #2
    add     x11, x11, x5, lsl #2
    subs    x15, x15, x5
    b.gt    .Lstream_inner1

.Lstream_next:
    add     x8, x8, #1
    b       .Lstream_outer

.Lstream_done:
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
    .size embedding_fwd_f32_stream_asm, .-embedding_fwd_f32_stream_asm


//=============================================================================
// Embedding Forward FP32 - Sorted Core (contiguous read, indirect write)
// void embedding_fwd_f32_sorted_core_asm(const int32_t* sorted_indices,
//                                         const int32_t* sorted_order,
//                                         const float* emb_table,
//                                         float* output,
//                                         size_t count, size_t hidden_dim)
// x0 = sorted_indices, x1 = sorted_order, x2 = emb_table
// x3 = output, x4 = count, x5 = hidden_dim
//
// For each i in [0, count):
//   src = emb_table + sorted_indices[i] * hidden_dim
//   dst = output + sorted_order[i] * hidden_dim
//   copy hidden_dim floats from src to dst
//
// Since indices are sorted, consecutive reads access nearby/identical rows
// in the embedding table, exploiting HBM2 page locality and HW prefetch.
// Writes are scattered to original token positions.
//
// Register plan:
//   z0-z7: data (8x unroll)
//   x19-x26: saved parameters
//   p0: all-true (.s)
//=============================================================================
    .align 6
    .global embedding_fwd_f32_sorted_core_asm
    .type embedding_fwd_f32_sorted_core_asm, %function
embedding_fwd_f32_sorted_core_asm:
    stp     x29, x30, [sp, #-80]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    mov     x29, sp

    cbz     x4, .Lsorted_done

    mov     x19, x0             // sorted_indices
    mov     x20, x1             // sorted_order
    mov     x21, x2             // emb_table
    mov     x22, x3             // output
    mov     x23, x4             // count
    mov     x24, x5             // hidden_dim
    lsl     x25, x5, #2         // row_stride_bytes = dim * 4

    cntw    x5                   // VL words = 16 (overwrites x5, saved in x24)
    ptrue   p0.s
    lsl     x26, x5, #5         // 8 * VL * 4 bytes = 512 (8x unroll stride)
    lsl     x6, x5, #3          // 8 * VL = 128 elements per 8x iter

    mov     x8, #0              // current index

.Lsorted_outer:
    cmp     x8, x23
    b.ge    .Lsorted_done

    // Load sorted_indices[i] and sorted_order[i]
    ldr     w9, [x19, x8, lsl #2]   // embedding index (sorted)
    ldr     w10, [x20, x8, lsl #2]  // original output position

    // Source: emb_table + sorted_indices[i] * stride
    mul     x11, x9, x25
    add     x11, x21, x11

    // Destination: output + sorted_order[i] * stride
    mul     x12, x10, x25
    add     x12, x22, x12

    // Prefetch: row for sorted token i+8 (sequential read = HW prefetch helps,
    // but SW prefetch for the initial cache line of future rows still beneficial)
    add     x13, x8, #8
    cmp     x13, x23
    b.ge    .Lsorted_no_pf

    ldr     w14, [x19, x13, lsl #2]
    mul     x15, x14, x25
    add     x15, x21, x15
    prfm    pldl1keep, [x15]
    prfm    pldl1keep, [x15, #256]

.Lsorted_no_pf:

    // Copy row: 8x unrolled
    mov     x15, x24             // remaining elements

.Lsorted_inner8:
    cmp     x15, x6
    b.lt    .Lsorted_inner_rem

    ld1w    {z0.s}, p0/z, [x11]
    ld1w    {z1.s}, p0/z, [x11, #1, mul vl]
    ld1w    {z2.s}, p0/z, [x11, #2, mul vl]
    ld1w    {z3.s}, p0/z, [x11, #3, mul vl]
    ld1w    {z4.s}, p0/z, [x11, #4, mul vl]
    ld1w    {z5.s}, p0/z, [x11, #5, mul vl]
    ld1w    {z6.s}, p0/z, [x11, #6, mul vl]
    ld1w    {z7.s}, p0/z, [x11, #7, mul vl]

    st1w    {z0.s}, p0, [x12]
    st1w    {z1.s}, p0, [x12, #1, mul vl]
    st1w    {z2.s}, p0, [x12, #2, mul vl]
    st1w    {z3.s}, p0, [x12, #3, mul vl]
    st1w    {z4.s}, p0, [x12, #4, mul vl]
    st1w    {z5.s}, p0, [x12, #5, mul vl]
    st1w    {z6.s}, p0, [x12, #6, mul vl]
    st1w    {z7.s}, p0, [x12, #7, mul vl]

    add     x11, x11, x26       // src += 8*VL*4
    add     x12, x12, x26       // dst += 8*VL*4
    sub     x15, x15, x6
    b       .Lsorted_inner8

.Lsorted_inner_rem:
    cbz     x15, .Lsorted_next

.Lsorted_inner1:
    whilelt p1.s, xzr, x15
    ld1w    {z0.s}, p1/z, [x11]
    st1w    {z0.s}, p1, [x12]
    add     x11, x11, x5, lsl #2
    add     x12, x12, x5, lsl #2
    subs    x15, x15, x5
    b.gt    .Lsorted_inner1

.Lsorted_next:
    add     x8, x8, #1
    b       .Lsorted_outer

.Lsorted_done:
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
    .size embedding_fwd_f32_sorted_core_asm, .-embedding_fwd_f32_sorted_core_asm


//=============================================================================
// Embedding Forward FP32 - Stream with Index Prefetch + Deeper Row Prefetch
// void embedding_fwd_f32_stream_ipf_asm(const int32_t* indices,
//                                        const float* emb_table,
//                                        float* output,
//                                        size_t seq_len, size_t hidden_dim)
//
// Same as stream_asm but:
// 1. Prefetches 16 embedding rows ahead (vs 8)
// 2. Prefetches index array 64 entries ahead (1 cache line of indices)
// 3. Initial prefetch primes first 16 rows
//=============================================================================
    .align 6
    .global embedding_fwd_f32_stream_ipf_asm
    .type embedding_fwd_f32_stream_ipf_asm, %function
embedding_fwd_f32_stream_ipf_asm:
    stp     x29, x30, [sp, #-80]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    mov     x29, sp

    cbz     x3, .Lsipf_done

    mov     x19, x0             // indices
    mov     x20, x1             // emb_table
    mov     x21, x2             // output
    mov     x22, x3             // seq_len
    mov     x23, x4             // hidden_dim
    lsl     x24, x4, #2         // row_stride_bytes = dim * 4

    cntw    x5                   // VL words = 16
    ptrue   p0.s
    lsl     x25, x5, #5         // 8 * VL * 4 bytes = 512 (8x unroll stride)
    lsl     x26, x5, #3         // 8 * VL = 128 elements per 8x iter

    // Issue initial prefetches for first 16 rows
    mov     x6, #0
    mov     x7, #16
    cmp     x7, x22
    csel    x7, x7, x22, lo     // min(16, seq_len)
.Lsipf_init_pf:
    cmp     x6, x7
    b.ge    .Lsipf_init_pf_done
    ldr     w8, [x19, x6, lsl #2]
    mul     x9, x8, x24
    add     x9, x20, x9
    prfm    pldl1strm, [x9]
    prfm    pldl1strm, [x9, #256]
    prfm    pldl1strm, [x9, #512]
    prfm    pldl1strm, [x9, #768]
    add     x6, x6, #1
    b       .Lsipf_init_pf
.Lsipf_init_pf_done:

    // Prefetch first index cache line
    prfm    pldl1keep, [x19]

    mov     x8, #0              // current token index

.Lsipf_outer:
    cmp     x8, x22
    b.ge    .Lsipf_done

    // Load current index
    ldr     w9, [x19, x8, lsl #2]

    // Source address: emb_table + idx * stride
    mul     x10, x9, x24
    add     x10, x20, x10

    // Destination address: output + i * stride
    mul     x11, x8, x24
    add     x11, x21, x11

    // Prefetch embedding row for token i+16
    add     x12, x8, #16
    cmp     x12, x22
    b.ge    .Lsipf_no_row_pf

    ldr     w13, [x19, x12, lsl #2]
    mul     x14, x13, x24
    add     x14, x20, x14
    prfm    pldl1strm, [x14]
    prfm    pldl1strm, [x14, #256]
    prfm    pldl1strm, [x14, #512]
    prfm    pldl1strm, [x14, #768]

.Lsipf_no_row_pf:

    // Prefetch index array 64 entries ahead (1 cache line = 256B = 64 int32)
    add     x12, x8, #64
    cmp     x12, x22
    b.ge    .Lsipf_no_idx_pf
    add     x14, x19, x12, lsl #2
    prfm    pldl1keep, [x14]
.Lsipf_no_idx_pf:

    // Copy row: 8x unrolled inner loop
    mov     x15, x23             // remaining elements

.Lsipf_inner8:
    cmp     x15, x26
    b.lt    .Lsipf_inner_rem

    ld1w    {z0.s}, p0/z, [x10]
    ld1w    {z1.s}, p0/z, [x10, #1, mul vl]
    ld1w    {z2.s}, p0/z, [x10, #2, mul vl]
    ld1w    {z3.s}, p0/z, [x10, #3, mul vl]
    ld1w    {z4.s}, p0/z, [x10, #4, mul vl]
    ld1w    {z5.s}, p0/z, [x10, #5, mul vl]
    ld1w    {z6.s}, p0/z, [x10, #6, mul vl]
    ld1w    {z7.s}, p0/z, [x10, #7, mul vl]

    st1w    {z0.s}, p0, [x11]
    st1w    {z1.s}, p0, [x11, #1, mul vl]
    st1w    {z2.s}, p0, [x11, #2, mul vl]
    st1w    {z3.s}, p0, [x11, #3, mul vl]
    st1w    {z4.s}, p0, [x11, #4, mul vl]
    st1w    {z5.s}, p0, [x11, #5, mul vl]
    st1w    {z6.s}, p0, [x11, #6, mul vl]
    st1w    {z7.s}, p0, [x11, #7, mul vl]

    add     x10, x10, x25       // src += 8*VL*4
    add     x11, x11, x25       // dst += 8*VL*4
    sub     x15, x15, x26
    b       .Lsipf_inner8

.Lsipf_inner_rem:
    cbz     x15, .Lsipf_next

.Lsipf_inner1:
    whilelt p1.s, xzr, x15
    ld1w    {z0.s}, p1/z, [x10]
    st1w    {z0.s}, p1, [x11]
    add     x10, x10, x5, lsl #2
    add     x11, x11, x5, lsl #2
    subs    x15, x15, x5
    b.gt    .Lsipf_inner1

.Lsipf_next:
    add     x8, x8, #1
    b       .Lsipf_outer

.Lsipf_done:
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
    .size embedding_fwd_f32_stream_ipf_asm, .-embedding_fwd_f32_stream_ipf_asm


//=============================================================================
// Scatter one embedding row to multiple output positions (for dedup)
// void embedding_fwd_f32_scatter_row_asm(const float* src_row,
//                                         float* output_base,
//                                         const int32_t* positions,
//                                         size_t n_positions,
//                                         size_t hidden_dim)
// x0 = src_row, x1 = output_base, x2 = positions array
// x3 = n_positions, x4 = hidden_dim
//
// For each position k in [0, n_positions):
//   dst = output_base + positions[k] * hidden_dim
//   copy hidden_dim floats from src_row to dst
//
// Optimization: loads src_row in 8x-unrolled chunks (z0-z7),
// then stores to all positions per chunk. Source data stays in
// z registers; only stores are repeated. For duplicate indices,
// this avoids re-loading the row from cache per duplicate.
//=============================================================================
    .align 6
    .global embedding_fwd_f32_scatter_row_asm
    .type embedding_fwd_f32_scatter_row_asm, %function
embedding_fwd_f32_scatter_row_asm:
    stp     x29, x30, [sp, #-80]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    mov     x29, sp

    cbz     x3, .Lscrow_done

    mov     x19, x0             // src_row
    mov     x20, x1             // output_base
    mov     x21, x2             // positions
    mov     x22, x3             // n_positions
    mov     x23, x4             // hidden_dim
    lsl     x24, x4, #2         // row_stride_bytes = dim * 4

    cntw    x5                   // VL words = 16
    ptrue   p0.s
    lsl     x25, x5, #5         // 8 * VL * 4 bytes = 512
    lsl     x26, x5, #3         // 8 * VL = 128 elements per 8x

    // Prefetch source row into L1
    prfm    pldl1keep, [x19]
    prfm    pldl1keep, [x19, #256]
    prfm    pldl1keep, [x19, #512]
    prfm    pldl1keep, [x19, #768]

    // Outer: iterate over dim in chunks of 8*VL elements
    mov     x6, x19             // src chunk ptr
    mov     x7, #0              // element offset
    mov     x8, x23             // remaining elements

.Lscrow_chunk8:
    cmp     x8, x26             // remaining >= 8*VL?
    b.lt    .Lscrow_chunk_rem

    // Load 8 vectors from source (hot in L1 or z regs)
    ld1w    {z0.s}, p0/z, [x6]
    ld1w    {z1.s}, p0/z, [x6, #1, mul vl]
    ld1w    {z2.s}, p0/z, [x6, #2, mul vl]
    ld1w    {z3.s}, p0/z, [x6, #3, mul vl]
    ld1w    {z4.s}, p0/z, [x6, #4, mul vl]
    ld1w    {z5.s}, p0/z, [x6, #5, mul vl]
    ld1w    {z6.s}, p0/z, [x6, #6, mul vl]
    ld1w    {z7.s}, p0/z, [x6, #7, mul vl]

    // Store this chunk to all output positions
    mov     x9, #0              // position index
.Lscrow_store8:
    ldr     w10, [x21, x9, lsl #2]   // positions[k]
    mul     x11, x10, x24            // byte offset = pos * stride_bytes
    add     x11, x20, x11            // dst row base
    add     x11, x11, x7, lsl #2    // dst + element_offset * 4

    st1w    {z0.s}, p0, [x11]
    st1w    {z1.s}, p0, [x11, #1, mul vl]
    st1w    {z2.s}, p0, [x11, #2, mul vl]
    st1w    {z3.s}, p0, [x11, #3, mul vl]
    st1w    {z4.s}, p0, [x11, #4, mul vl]
    st1w    {z5.s}, p0, [x11, #5, mul vl]
    st1w    {z6.s}, p0, [x11, #6, mul vl]
    st1w    {z7.s}, p0, [x11, #7, mul vl]

    add     x9, x9, #1
    cmp     x9, x22
    b.lt    .Lscrow_store8

    // Next chunk
    add     x6, x6, x25         // src += 8*VL*4
    add     x7, x7, x26         // element offset += 8*VL
    sub     x8, x8, x26         // remaining -= 8*VL
    b       .Lscrow_chunk8

.Lscrow_chunk_rem:
    // Handle remaining < 8*VL elements (1 vector at a time)
    cbz     x8, .Lscrow_done

.Lscrow_rem1:
    whilelt p1.s, xzr, x8
    ld1w    {z0.s}, p1/z, [x6]

    mov     x9, #0
.Lscrow_store_rem1:
    ldr     w10, [x21, x9, lsl #2]
    mul     x11, x10, x24
    add     x11, x20, x11
    add     x11, x11, x7, lsl #2

    st1w    {z0.s}, p1, [x11]

    add     x9, x9, #1
    cmp     x9, x22
    b.lt    .Lscrow_store_rem1

    add     x6, x6, x5, lsl #2  // src += VL*4
    add     x7, x7, x5          // element offset += VL
    subs    x8, x8, x5          // remaining -= VL
    b.gt    .Lscrow_rem1

.Lscrow_done:
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
    .size embedding_fwd_f32_scatter_row_asm, .-embedding_fwd_f32_scatter_row_asm


//=============================================================================
// Embedding Forward FP16 - Deep-Prefetch Stream
// void embedding_fwd_f16_stream_asm(const int32_t* indices,
//                                    const __fp16* emb_table,
//                                    __fp16* output,
//                                    size_t seq_len, size_t hidden_dim)
// x0 = indices, x1 = emb_table, x2 = output, x3 = seq_len, x4 = hidden_dim
//
// Same strategy as FP32 stream but for FP16 (2 bytes per element):
// - SVE ld1h/st1h: 32 FP16 elements per vector (64 bytes)
// - 8x unroll: 256 FP16 elements = 512 bytes per inner iteration
// - Deep prefetch: 8 rows ahead, 4 cache lines per row
// - Row stride: hidden_dim * 2 bytes
//=============================================================================
    .align 6
    .global embedding_fwd_f16_stream_asm
    .type embedding_fwd_f16_stream_asm, %function
embedding_fwd_f16_stream_asm:
    stp     x29, x30, [sp, #-80]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    mov     x29, sp

    cbz     x3, .Lf16s_done

    mov     x19, x0             // indices
    mov     x20, x1             // emb_table (FP16)
    mov     x21, x2             // output (FP16)
    mov     x22, x3             // seq_len
    mov     x23, x4             // hidden_dim (in elements)
    lsl     x24, x4, #1         // row_stride_bytes = dim * 2

    cnth    x5                   // VL in halfwords = 32 for A64FX
    ptrue   p0.h                 // all-true predicate for halfwords
    lsl     x25, x5, #4         // 8 * VL_h * 2 bytes = 8 * 32 * 2 = 512
    lsl     x26, x5, #3         // 8 * VL_h = 256 elements per 8x iter

    // Issue initial prefetches for first 8 rows
    mov     x6, #0
    mov     x7, #8
    cmp     x7, x22
    csel    x7, x7, x22, lo     // min(8, seq_len)
.Lf16s_init_pf:
    cmp     x6, x7
    b.ge    .Lf16s_init_pf_done
    ldr     w8, [x19, x6, lsl #2]
    mul     x9, x8, x24
    add     x9, x20, x9
    prfm    pldl1strm, [x9]
    prfm    pldl1strm, [x9, #256]
    prfm    pldl1strm, [x9, #512]
    prfm    pldl1strm, [x9, #768]
    add     x6, x6, #1
    b       .Lf16s_init_pf
.Lf16s_init_pf_done:

    mov     x8, #0              // current token index

.Lf16s_outer:
    cmp     x8, x22
    b.ge    .Lf16s_done

    // Load current index
    ldr     w9, [x19, x8, lsl #2]

    // Source: emb_table + idx * row_stride
    mul     x10, x9, x24
    add     x10, x20, x10

    // Destination: output + i * row_stride
    mul     x11, x8, x24
    add     x11, x21, x11

    // Prefetch row for token i+8
    add     x12, x8, #8
    cmp     x12, x22
    b.ge    .Lf16s_no_pf

    ldr     w13, [x19, x12, lsl #2]
    mul     x14, x13, x24
    add     x14, x20, x14
    prfm    pldl1strm, [x14]
    prfm    pldl1strm, [x14, #256]
    prfm    pldl1strm, [x14, #512]
    prfm    pldl1strm, [x14, #768]

.Lf16s_no_pf:

    // Copy row: 8x unrolled inner loop (FP16)
    mov     x15, x23             // remaining elements (halfwords)

.Lf16s_inner8:
    cmp     x15, x26
    b.lt    .Lf16s_inner_rem

    ld1h    {z0.h}, p0/z, [x10]
    ld1h    {z1.h}, p0/z, [x10, #1, mul vl]
    ld1h    {z2.h}, p0/z, [x10, #2, mul vl]
    ld1h    {z3.h}, p0/z, [x10, #3, mul vl]
    ld1h    {z4.h}, p0/z, [x10, #4, mul vl]
    ld1h    {z5.h}, p0/z, [x10, #5, mul vl]
    ld1h    {z6.h}, p0/z, [x10, #6, mul vl]
    ld1h    {z7.h}, p0/z, [x10, #7, mul vl]

    st1h    {z0.h}, p0, [x11]
    st1h    {z1.h}, p0, [x11, #1, mul vl]
    st1h    {z2.h}, p0, [x11, #2, mul vl]
    st1h    {z3.h}, p0, [x11, #3, mul vl]
    st1h    {z4.h}, p0, [x11, #4, mul vl]
    st1h    {z5.h}, p0, [x11, #5, mul vl]
    st1h    {z6.h}, p0, [x11, #6, mul vl]
    st1h    {z7.h}, p0, [x11, #7, mul vl]

    add     x10, x10, x25       // src += 8*VL_h*2
    add     x11, x11, x25       // dst += 8*VL_h*2
    sub     x15, x15, x26
    b       .Lf16s_inner8

.Lf16s_inner_rem:
    cbz     x15, .Lf16s_next

.Lf16s_inner1:
    whilelt p1.h, xzr, x15
    ld1h    {z0.h}, p1/z, [x10]
    st1h    {z0.h}, p1, [x11]
    add     x10, x10, x5, lsl #1   // src += VL_h * 2
    add     x11, x11, x5, lsl #1   // dst += VL_h * 2
    subs    x15, x15, x5
    b.gt    .Lf16s_inner1

.Lf16s_next:
    add     x8, x8, #1
    b       .Lf16s_outer

.Lf16s_done:
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
    .size embedding_fwd_f16_stream_asm, .-embedding_fwd_f16_stream_asm
