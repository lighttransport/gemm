#include "lz.h"
#include "lz_fmt.h"
#include <string.h>

/* ASM block decompressor (defined in lz_decompress_a64fx.S) */
extern size_t lz_decompress_block_asm(const uint8_t *ip, uint8_t *op,
                                       const uint8_t *ip_end);

/* ------------------------------------------------------------------ */
/* C reference decompressor                                            */
/* ------------------------------------------------------------------ */
static inline int read_ext(const uint8_t **ip, const uint8_t *ip_end) {
    int val = 0;
    while (*ip < ip_end) {
        uint8_t b = *(*ip)++;
        val += b;
        if (b != 255) break;
    }
    return val;
}

static size_t decompress_block_c(const uint8_t *src, size_t src_size,
                                  uint8_t *dst, size_t dst_capacity) {
    const uint8_t *ip = src;
    const uint8_t *ip_end = src + src_size;
    uint8_t *op = dst;
    uint8_t *op_end = dst + dst_capacity;

    while (ip < ip_end) {
        uint8_t tag = *ip++;
        int lit_code = tag >> 4;
        int mat_code = tag & 0xF;

        /* Literal length */
        int lit_len = lit_code;
        if (lit_code == 15)
            lit_len += read_ext(&ip, ip_end);

        /* Copy literals */
        if (op + lit_len > op_end || ip + lit_len > ip_end)
            return 0;
        memcpy(op, ip, lit_len);
        ip += lit_len;
        op += lit_len;

        /* No match → literal-only token (end of block) */
        if (mat_code == 0)
            continue;

        /* Read offset */
        if (ip + 2 > ip_end)
            return 0;
        int offset = lz_read16(ip);
        ip += 2;

        /* Match length */
        int match_len = mat_code + 3;
        if (mat_code == 15)
            match_len += read_ext(&ip, ip_end);

        if (offset == 0 || (size_t)(op - dst) < (size_t)offset)
            return 0;
        if (op + match_len > op_end)
            return 0;

        /* Copy match (handles overlap for RLE patterns) */
        const uint8_t *match_src = op - offset;
        if (offset == 1) {
            /* RLE: replicate single byte */
            memset(op, *match_src, match_len);
            op += match_len;
        } else if (offset >= 8) {
            /* Fast path: non-overlapping or mild overlap */
            int rem = match_len;
            while (rem >= 16) {
                memcpy(op, match_src, 16);
                op += 16;
                match_src += 16;
                rem -= 16;
            }
            while (rem-- > 0)
                *op++ = *match_src++;
        } else {
            /* Overlapping (small offset 2..7) → byte-by-byte */
            for (int i = 0; i < match_len; i++)
                op[i] = match_src[i];
            op += match_len;
        }
    }

    return (size_t)(op - dst);
}

/* ------------------------------------------------------------------ */
/* Frame / block parsing (shared between C and ASM paths)              */
/* ------------------------------------------------------------------ */
typedef size_t (*block_decompress_fn)(const uint8_t *src, size_t src_size,
                                       uint8_t *dst, size_t dst_capacity);

static size_t decompress_frame(const void *src, size_t src_size,
                                void *dst, size_t dst_capacity,
                                block_decompress_fn block_fn) {
    const uint8_t *sp = (const uint8_t *)src;

    if (src_size < sizeof(lz_frame_header_t))
        return 0;

    lz_frame_header_t hdr;
    memcpy(&hdr, sp, sizeof(hdr));

    if (hdr.magic != LZ_MAGIC)
        return 0;
    if (hdr.original_size > dst_capacity)
        return 0;

    const uint8_t *ip = sp + sizeof(lz_frame_header_t);
    const uint8_t *ip_end = sp + src_size;
    uint8_t *op = (uint8_t *)dst;
    size_t total = 0;

    while (total < hdr.original_size && ip + 4 <= ip_end) {
        uint32_t block_hdr = lz_read32(ip);
        ip += 4;

        int stored = (block_hdr & LZ_BLOCK_STORED) != 0;
        uint32_t block_size = block_hdr & ~LZ_BLOCK_STORED;

        if (ip + block_size > ip_end)
            return 0;

        if (stored) {
            if (total + block_size > dst_capacity) return 0;
            memcpy(op, ip, block_size);
            op += block_size;
            total += block_size;
        } else {
            size_t remaining = hdr.original_size - total;
            size_t cap = (remaining > LZ_BLOCK_SIZE) ? LZ_BLOCK_SIZE : remaining;
            size_t dsize = block_fn(ip, block_size, op, cap);
            if (dsize == 0 && cap > 0) return 0;
            op += dsize;
            total += dsize;
        }

        ip += block_size;
    }

    return total;
}

/* C decompressor */
size_t lz_decompress(const void *src, size_t src_size,
                     void *dst, size_t dst_capacity) {
    return decompress_frame(src, src_size, dst, dst_capacity,
                            decompress_block_c);
}

/* ASM block wrapper: adapts asm calling convention to block_decompress_fn */
static size_t decompress_block_asm_wrapper(const uint8_t *src, size_t src_size,
                                            uint8_t *dst, size_t dst_capacity) {
    (void)dst_capacity;
    return lz_decompress_block_asm(src, dst, src + src_size);
}

/* ASM decompressor (frame parsing in C, block decompression in ASM) */
size_t lz_decompress_asm(const void *src, size_t src_size,
                         void *dst, size_t dst_capacity) {
    return decompress_frame(src, src_size, dst, dst_capacity,
                            decompress_block_asm_wrapper);
}
