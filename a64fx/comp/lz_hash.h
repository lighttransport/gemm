#ifndef LZ_HASH_H
#define LZ_HASH_H

#include <stdint.h>
#include <string.h>

/* Hash table sizes */
#define LZ_HASH_BITS_L3  14
#define LZ_HASH_SIZE_L3  (1 << LZ_HASH_BITS_L3)  /* 16K entries */

#define LZ_HASH_BITS_L9  17
#define LZ_HASH_SIZE_L9  (1 << LZ_HASH_BITS_L9)  /* 128K entries */

static inline uint32_t lz_hash4(uint32_t v, int bits) {
    return (v * 2654435761u) >> (32 - bits);
}

static inline uint32_t lz_hash4_l3(const uint8_t *p) {
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return lz_hash4(v, LZ_HASH_BITS_L3);
}

static inline uint32_t lz_hash4_l9(const uint8_t *p) {
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return lz_hash4(v, LZ_HASH_BITS_L9);
}

/*
 * Branchless match extension using EOR + RBIT + CLZ.
 * Counts how many bytes match starting from p1 and p2,
 * up to (limit - p1) bytes.
 * EOR finds differing bits, RBIT flips for LE byte order,
 * CLZ counts leading zeros = matching bits from LSB.
 */
static inline int lz_match_length(const uint8_t *p1, const uint8_t *p2,
                                   const uint8_t *limit) {
    int len = 0;

    /* 8-byte chunks */
    while (p1 + len + 8 <= limit) {
        uint64_t v1, v2;
        __builtin_memcpy(&v1, p1 + len, 8);
        __builtin_memcpy(&v2, p2 + len, 8);
        uint64_t diff = v1 ^ v2;
        if (diff == 0) {
            len += 8;
            continue;
        }
        uint64_t rev, bits;
        __asm__("rbit %0, %1" : "=r"(rev) : "r"(diff));
        __asm__("clz  %0, %1" : "=r"(bits) : "r"(rev));
        return len + (int)(bits >> 3);
    }

    /* Byte-by-byte tail */
    while (p1 + len < limit && p1[len] == p2[len])
        len++;

    return len;
}

#endif /* LZ_HASH_H */
