#include "lz.h"
#include "lz_fmt.h"
#include "lz_hash.h"
#include <string.h>

size_t lz_compress_bound(size_t src_size) {
    size_t nblocks = (src_size + LZ_BLOCK_SIZE - 1) / LZ_BLOCK_SIZE;
    if (nblocks == 0) nblocks = 1;
    return sizeof(lz_frame_header_t) + nblocks * (LZ_BLOCK_SIZE + 4 + 256) + 64;
}

/* Emit extension bytes for length values >= 15 */
static inline uint8_t *emit_length_ext(uint8_t *op, int length) {
    while (length >= 255) {
        *op++ = 255;
        length -= 255;
    }
    *op++ = (uint8_t)length;
    return op;
}

/* Emit a complete token: [tag] [lit_ext...] [literals] [offset] [mat_ext...] */
static uint8_t *emit_sequence(uint8_t *op,
                               const uint8_t *literals, int lit_len,
                               int offset, int match_len) {
    int lit_code = (lit_len < 15) ? lit_len : 15;
    int mat_code;
    if (match_len == 0) {
        mat_code = 0;
    } else {
        int ml = match_len - 3;
        mat_code = (ml < 15) ? ml : 15;
    }

    *op++ = (uint8_t)((lit_code << 4) | mat_code);

    if (lit_code == 15)
        op = emit_length_ext(op, lit_len - 15);

    memcpy(op, literals, lit_len);
    op += lit_len;

    if (match_len > 0) {
        lz_write16(op, (uint16_t)offset);
        op += 2;
        if (mat_code == 15)
            op = emit_length_ext(op, match_len - 3 - 15);
    }

    return op;
}

/* ------------------------------------------------------------------ */
/* Level 3: fast greedy, single hash probe, 16K table (64KB = L1)     */
/* ------------------------------------------------------------------ */
static size_t compress_block_l3(const uint8_t *src, int src_len,
                                 uint8_t *dst, int dst_capacity) {
    uint32_t htab[LZ_HASH_SIZE_L3];
    memset(htab, 0, sizeof(htab));

    const uint8_t *ip = src;
    const uint8_t *ip_end = src + src_len;
    const uint8_t *ip_limit = ip_end - LZ_MIN_MATCH;
    const uint8_t *anchor = ip;
    uint8_t *op = dst;
    uint8_t *op_end = dst + dst_capacity;

    if (src_len < LZ_MIN_MATCH + 1)
        goto emit_last;

    ip++; /* first byte is always literal */

    while (ip < ip_limit) {
        uint32_t h = lz_hash4_l3(ip);
        uint32_t ref_idx = htab[h];
        const uint8_t *ref = src + ref_idx;
        htab[h] = (uint32_t)(ip - src);

        /* Validate match: same 4 bytes, within window */
        uint32_t v1, v2;
        __builtin_memcpy(&v1, ip, 4);
        __builtin_memcpy(&v2, ref, 4);

        if (v1 != v2 || (ip - ref) > LZ_MAX_OFFSET || ref >= ip) {
            ip++;
            continue;
        }

        /* Extend match forward */
        int match_len = 4 + lz_match_length(ip + 4, ref + 4, ip_end);
        int offset = (int)(ip - ref);
        int lit_len = (int)(ip - anchor);

        /* Conservative output space check */
        int needed = 1 + (lit_len >= 15 ? lit_len / 255 + 2 : 0)
                     + lit_len + 2
                     + (match_len - 3 >= 15 ? (match_len - 3) / 255 + 2 : 0);
        if (op + needed > op_end)
            return 0;

        op = emit_sequence(op, anchor, lit_len, offset, match_len);

        ip += match_len;
        anchor = ip;

        /* Update hash for positions near end of match */
        if (ip - 2 >= src && ip < ip_limit)
            htab[lz_hash4_l3(ip - 2)] = (uint32_t)(ip - 2 - src);
    }

emit_last:
    /* Emit remaining literals as final literal-only token */
    {
        int lit_len = (int)(ip_end - anchor);
        if (lit_len > 0) {
            int needed = 1 + (lit_len >= 15 ? lit_len / 255 + 2 : 0) + lit_len;
            if (op + needed > op_end)
                return 0;
            op = emit_sequence(op, anchor, lit_len, 0, 0);
        }
    }

    return (size_t)(op - dst);
}

/* ------------------------------------------------------------------ */
/* Level 9: hash chains (depth 8) + lazy matching                     */
/* ------------------------------------------------------------------ */
#define L9_MAX_CHAIN  8

static size_t compress_block_l9(const uint8_t *src, int src_len,
                                 uint8_t *dst, int dst_capacity) {
    uint32_t htab[LZ_HASH_SIZE_L9];
    uint16_t chain[LZ_BLOCK_SIZE];
    memset(htab, 0xFF, sizeof(htab));  /* 0xFFFFFFFF = invalid */
    memset(chain, 0, sizeof(chain));

    const uint8_t *ip = src;
    const uint8_t *ip_end = src + src_len;
    const uint8_t *ip_limit = ip_end - LZ_MIN_MATCH;
    const uint8_t *anchor = ip;
    uint8_t *op = dst;
    uint8_t *op_end = dst + dst_capacity;

    if (src_len < LZ_MIN_MATCH + 1)
        goto emit_last_l9;

    /* Insert first position */
    {
        uint32_t h0 = lz_hash4_l9(src);
        chain[0] = 0;
        htab[h0] = 0;
    }
    ip++;

    while (ip < ip_limit) {
        uint32_t cur_pos = (uint32_t)(ip - src);
        uint32_t h = lz_hash4_l9(ip);
        uint32_t head = htab[h];

        /* Insert current position into chain */
        chain[cur_pos & (LZ_BLOCK_SIZE - 1)] =
            (head != 0xFFFFFFFFu) ? (uint16_t)(head & 0xFFFF) : 0;
        htab[h] = cur_pos;

        /* Find best match by walking the chain */
        int best_len = 0, best_off = 0;
        uint32_t cand = head;
        for (int d = 0; d < L9_MAX_CHAIN && cand != 0xFFFFFFFFu; d++) {
            const uint8_t *ref = src + cand;
            int dist = (int)(cur_pos - cand);
            if (dist <= 0 || dist > LZ_MAX_OFFSET)
                break;

            uint32_t v1, v2;
            __builtin_memcpy(&v1, ip, 4);
            __builtin_memcpy(&v2, ref, 4);
            if (v1 == v2) {
                int ml = 4 + lz_match_length(ip + 4, ref + 4, ip_end);
                if (ml > best_len) {
                    best_len = ml;
                    best_off = dist;
                }
            }

            uint16_t next16 = chain[cand & (LZ_BLOCK_SIZE - 1)];
            uint32_t next = (uint32_t)next16;
            if (next >= cand) break;
            cand = next;
        }

        if (best_len < LZ_MIN_MATCH) {
            ip++;
            continue;
        }

        /* Lazy matching: check if ip+1 gives a longer match */
        if (ip + 1 < ip_limit) {
            uint32_t pos1 = cur_pos + 1;
            uint32_t h2 = lz_hash4_l9(ip + 1);
            uint32_t head2 = htab[h2];

            chain[pos1 & (LZ_BLOCK_SIZE - 1)] =
                (head2 != 0xFFFFFFFFu) ? (uint16_t)(head2 & 0xFFFF) : 0;
            htab[h2] = pos1;

            int lazy_len = 0, lazy_off = 0;
            uint32_t cand2 = head2;
            for (int d = 0; d < L9_MAX_CHAIN && cand2 != 0xFFFFFFFFu; d++) {
                const uint8_t *ref = src + cand2;
                int dist = (int)(pos1 - cand2);
                if (dist <= 0 || dist > LZ_MAX_OFFSET)
                    break;

                uint32_t v1, v2;
                __builtin_memcpy(&v1, ip + 1, 4);
                __builtin_memcpy(&v2, ref, 4);
                if (v1 == v2) {
                    int ml = 4 + lz_match_length(ip + 5, ref + 4, ip_end);
                    if (ml > lazy_len) {
                        lazy_len = ml;
                        lazy_off = dist;
                    }
                }

                uint16_t next16 = chain[cand2 & (LZ_BLOCK_SIZE - 1)];
                uint32_t next = (uint32_t)next16;
                if (next >= cand2) break;
                cand2 = next;
            }

            if (lazy_len > best_len) {
                ip++;
                best_len = lazy_len;
                best_off = lazy_off;
            }
        }

        /* Emit sequence */
        int lit_len = (int)(ip - anchor);
        int needed = 1 + (lit_len >= 15 ? lit_len / 255 + 2 : 0)
                     + lit_len + 2
                     + (best_len - 3 >= 15 ? (best_len - 3) / 255 + 2 : 0);
        if (op + needed > op_end)
            return 0;

        op = emit_sequence(op, anchor, lit_len, best_off, best_len);

        /* Advance past match, inserting skipped positions into hash */
        int skip_end = best_len;
        for (int s = 1; s < skip_end && ip + s < ip_limit; s++) {
            uint32_t sp = (uint32_t)(ip + s - src);
            uint32_t sh = lz_hash4_l9(ip + s);
            uint32_t shead = htab[sh];
            chain[sp & (LZ_BLOCK_SIZE - 1)] =
                (shead != 0xFFFFFFFFu) ? (uint16_t)(shead & 0xFFFF) : 0;
            htab[sh] = sp;
        }

        ip += best_len;
        anchor = ip;
    }

emit_last_l9:
    {
        int lit_len = (int)(ip_end - anchor);
        if (lit_len > 0) {
            int needed = 1 + (lit_len >= 15 ? lit_len / 255 + 2 : 0) + lit_len;
            if (op + needed > op_end)
                return 0;
            op = emit_sequence(op, anchor, lit_len, 0, 0);
        }
    }

    return (size_t)(op - dst);
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */
size_t lz_compress(const void *src, size_t src_size,
                   void *dst, size_t dst_capacity, int level) {
    const uint8_t *sp = (const uint8_t *)src;
    uint8_t *dp = (uint8_t *)dst;

    if (dst_capacity < sizeof(lz_frame_header_t))
        return 0;

    /* Write frame header */
    lz_frame_header_t hdr;
    hdr.magic = LZ_MAGIC;
    hdr.original_size = (uint32_t)src_size;
    memcpy(dp, &hdr, sizeof(hdr));

    uint8_t *op = dp + sizeof(lz_frame_header_t);
    uint8_t *op_end = dp + dst_capacity;
    size_t remaining = src_size;

    while (remaining > 0) {
        int block_len = (remaining > LZ_BLOCK_SIZE)
                        ? LZ_BLOCK_SIZE : (int)remaining;

        /* Reserve 4 bytes for block header */
        uint8_t *block_dst = op + 4;
        int block_cap = (int)(op_end - block_dst);
        if (block_cap <= 0) return 0;

        size_t csize = 0;
        if (level >= LZ_LEVEL_BEST)
            csize = compress_block_l9(sp, block_len, block_dst, block_cap);
        else
            csize = compress_block_l3(sp, block_len, block_dst, block_cap);

        if (csize > 0 && (int)csize < block_len) {
            /* Compressed block */
            lz_write32(op, (uint32_t)csize);
            op = block_dst + csize;
        } else {
            /* Stored (uncompressed) block */
            if (op + 4 + block_len > op_end) return 0;
            lz_write32(op, (uint32_t)block_len | LZ_BLOCK_STORED);
            memcpy(op + 4, sp, block_len);
            op = op + 4 + block_len;
        }

        sp += block_len;
        remaining -= block_len;
    }

    return (size_t)(op - dp);
}
