/*
 * lz_decompress_block_asm — ARM64 branchless LZ decompressor inner loop
 *
 * Prototype:
 *   size_t lz_decompress_block_asm(const uint8_t *ip, uint8_t *op,
 *                                   const uint8_t *ip_end);
 *
 * Returns decompressed size (final_op - initial_op).
 *
 * Fast path: ~20 integer instructions per token.
 * Uses only integer execution units (no SVE/NEON), so it can overlap
 * with SVE compute on A64FX's independent pipelines.
 *
 * Requires LZ_SAFETY_MARGIN (32) extra bytes in both input and output
 * buffers beyond actual content for overrun-safe ldp/stp.
 *
 * Register allocation (all caller-saved, no callee-saved regs needed):
 *   x0  = ip   (compressed input pointer, advanced)
 *   x1  = op   (decompressed output pointer, advanced)
 *   x2  = ip_end
 *   x3  = op_start (saved at entry for return value computation)
 *   x5  = match_src (temporary)
 *   x6  = target pointer (temporary)
 *   x7  = temp
 *   x8  = temp
 *   w9  = tag byte
 *   w10 = lit_code / lit_len
 *   w11 = mat_code
 *   w13 = offset
 *   w14 = match_len
 *   x16,x17 = ldp/stp temps
 *   x18 = ldp/stp temp (second pair)
 *   x15 = ldp/stp temp (second pair)
 */

    .arch armv8.2-a
    .text
    .global lz_decompress_block_asm
    .type   lz_decompress_block_asm, %function
    .p2align 6              /* align to 64-byte cache line */

lz_decompress_block_asm:
    /* Save op_start for return value */
    mov     x3, x1

    /* ============================================================ */
    /* Main loop                                                     */
    /* ============================================================ */
.Lloop:
    cmp     x0, x2
    b.hs    .Ldone

    /* 1. Load tag byte, advance ip */
    ldrb    w9, [x0], #1
    /* 2. lit_code = tag >> 4 */
    lsr     w10, w9, #4
    /* 3. mat_code = tag & 0xF */
    and     w11, w9, #0xF

    /* Check if we need slow path (either nibble == 15) */
    cmp     w10, #15
    b.eq    .Lslow_lit
    cmp     w11, #15
    b.eq    .Lslow_match_ext

    /* ============================================================ */
    /* FAST PATH: lit_code 0..14, mat_code 0..14                    */
    /* ~20 instructions for the common case                          */
    /* ============================================================ */

    /* 4-5. Copy up to 14 literal bytes (overrun-safe 16-byte load/store) */
    cbz     w10, .Lfast_no_lit
    ldp     x16, x17, [x0]
    stp     x16, x17, [x1]
.Lfast_no_lit:
    /* 6-7. Advance ip and op by lit_code */
    add     x0, x0, x10
    add     x1, x1, x10

    /* 8. No match? → next token */
    cbz     w11, .Lloop

    /* 9. Load 16-bit offset (LE), advance ip by 2 */
    ldrh    w13, [x0], #2
    /* 10. match_len = mat_code + 3 (range 4..17) */
    add     w14, w11, #3

    /* --- FAST MATCH: offset >= 16 → non-overlapping 32-byte copy --- */
    /* 11. match_src = op - offset */
    sub     x5, x1, w13, uxtw
    /* 12. Check overlap */
    cmp     w13, #16
    b.lo    .Loverlap

    /* 13-16. Two 16-byte loads cover match_len 4..17.
     * Second ldp may read from bytes just written by first stp
     * (store-to-load forwarding, correct for periodic patterns). */
    ldp     x16, x17, [x5]
    stp     x16, x17, [x1]
    ldp     x16, x17, [x5, #16]
    stp     x16, x17, [x1, #16]
    /* 17. Advance op by exact match_len */
    add     x1, x1, w14, uxtw
    /* 18. Back to top */
    b       .Lloop

    /* ============================================================ */
    /* OVERLAP: offset < 16                                          */
    /* ============================================================ */
.Loverlap:
    add     x6, x1, w14, uxtw      /* target = op + match_len */

    /* Offset == 1 fast path: RLE byte fill using 8-byte stores */
    cmp     w13, #1
    b.ne    .Loverlap_byte_init

    /* Broadcast single byte to all 8 positions of x7 */
    ldrb    w7, [x5]
    orr     w7, w7, w7, lsl #8
    orr     w7, w7, w7, lsl #16
    orr     x7, x7, x7, lsl #32
.Loverlap_rle8:
    sub     x8, x6, x1
    cmp     x8, #8
    b.lt    .Loverlap_rle_tail
    str     x7, [x1], #8
    b       .Loverlap_rle8
.Loverlap_rle_tail:
    cmp     x1, x6
    b.hs    .Lloop
    strb    w7, [x1], #1
    b       .Loverlap_rle_tail

    /* General overlap: byte-by-byte copy (offset 2..15) */
.Loverlap_byte_init:
.Loverlap_byte:
    ldrb    w7, [x5], #1
    strb    w7, [x1], #1
    cmp     x1, x6
    b.lo    .Loverlap_byte
    b       .Lloop

    /* ============================================================ */
    /* SLOW PATH: lit_code == 15 (long literal run)                  */
    /* ============================================================ */
.Lslow_lit:
    mov     w10, #15
.Lslow_lit_ext:
    ldrb    w7, [x0], #1
    add     w10, w10, w7
    cmp     w7, #255
    b.eq    .Lslow_lit_ext

    /* Copy w10 literal bytes from [x0] to [x1] */
    add     x6, x0, w10, uxtw      /* sentinel = ip + lit_len */

    /* 16-byte bulk copy */
.Lslow_lit_bulk:
    sub     x7, x6, x0
    cmp     x7, #16
    b.lt    .Lslow_lit_tail
    ldp     x16, x17, [x0]
    stp     x16, x17, [x1]
    add     x0, x0, #16
    add     x1, x1, #16
    b       .Lslow_lit_bulk

    /* Byte-by-byte tail */
.Lslow_lit_tail:
    cmp     x0, x6
    b.hs    .Lslow_lit_handle_match
    ldrb    w7, [x0], #1
    strb    w7, [x1], #1
    b       .Lslow_lit_tail

    /* Now handle the match part (mat_code still in w11) */
.Lslow_lit_handle_match:
    cbz     w11, .Lloop             /* mat_code == 0: no match */
    cmp     w11, #15
    b.eq    .Lslow_both

    /* mat_code 1..14: fast match after slow literals */
    ldrh    w13, [x0], #2
    add     w14, w11, #3
    b       .Ldo_match

    /* Both lit and match are extended */
.Lslow_both:
    ldrh    w13, [x0], #2
    mov     w14, #18                /* 15 + 3 */
.Lslow_both_ext:
    ldrb    w7, [x0], #1
    add     w14, w14, w7
    cmp     w7, #255
    b.eq    .Lslow_both_ext
    b       .Ldo_match

    /* ============================================================ */
    /* SLOW PATH: mat_code == 15 (long match), lit_code < 15         */
    /* ============================================================ */
.Lslow_match_ext:
    /* First copy the (short) literals: lit_code 0..14 */
    cbz     w10, .Lsme_no_lit
    ldp     x16, x17, [x0]
    stp     x16, x17, [x1]
.Lsme_no_lit:
    add     x0, x0, x10
    add     x1, x1, x10

    /* Read offset */
    ldrh    w13, [x0], #2
    /* match_len = 15 + 3 = 18, then add extension bytes */
    mov     w14, #18
.Lsme_ext:
    ldrb    w7, [x0], #1
    add     w14, w14, w7
    cmp     w7, #255
    b.eq    .Lsme_ext

    /* Fall through to general match copy */

    /* ============================================================ */
    /* General match copy (used by slow paths, handles any length)   */
    /* ============================================================ */
.Ldo_match:
    sub     x5, x1, w13, uxtw      /* match_src = op - offset */
    cmp     w13, #16
    b.lo    .Loverlap

    /* Non-overlapping: 16-byte loop */
    add     x6, x1, w14, uxtw      /* target = op + match_len */
.Lmatch_copy:
    ldp     x16, x17, [x5]
    stp     x16, x17, [x1]
    add     x5, x5, #16
    add     x1, x1, #16
    cmp     x1, x6
    b.lo    .Lmatch_copy
    mov     x1, x6                  /* fix over-advance */
    b       .Lloop

    /* ============================================================ */
    /* Done                                                          */
    /* ============================================================ */
.Ldone:
    sub     x0, x1, x3             /* return op - op_start */
    ret

    .size   lz_decompress_block_asm, . - lz_decompress_block_asm
