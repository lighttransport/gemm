#ifndef Q38D_KV_I6_H
#define Q38D_KV_I6_H

/* Symmetric signed six-bit K/V row with one FP32 scale per 256 values.
 * Codes occupy 192 consecutive bytes; [-31,31] maps to [-31,31] and the
 * unused -32 code is never produced. This cache trades precision for enough
 * HBM headroom to retain independent long contexts in TP4 groups. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline size_t q38d_kv_i6_row_bytes(size_t dim) { return (dim * 6 + 7) / 8; }

static inline int q38d_kv_i6_get(const uint8_t *row, size_t dim, size_t d) {
    size_t bit = d * 6, byte = bit >> 3, nbytes = q38d_kv_i6_row_bytes(dim);
    unsigned word = row[byte];
    if (byte + 1 < nbytes) word |= (unsigned)row[byte + 1] << 8;
    unsigned code = (word >> (bit & 7)) & 63u;
    return code < 32 ? (int)code : (int)code - 64;
}

/* Decode four signed values from each three-byte group. Q38D attention uses
 * 256-wide rows, so this avoids per-element bit offsets in its hot loop. */
static inline void q38d_kv_i6_unpack_f32_256(const uint8_t *row, float *dst) {
    for (size_t i = 0; i < 64; i++) {
        unsigned b0 = row[3 * i], b1 = row[3 * i + 1], b2 = row[3 * i + 2];
        unsigned c0 = b0 & 63u;
        unsigned c1 = (b0 >> 6) | ((b1 & 15u) << 2);
        unsigned c2 = (b1 >> 4) | ((b2 & 3u) << 4);
        unsigned c3 = b2 >> 2;
        dst[4 * i] = (float)((int)(c0 ^ 32u) - 32);
        dst[4 * i + 1] = (float)((int)(c1 ^ 32u) - 32);
        dst[4 * i + 2] = (float)((int)(c2 ^ 32u) - 32);
        dst[4 * i + 3] = (float)((int)(c3 ^ 32u) - 32);
    }
}

static inline int q38d_kv_i6_pack_row(const float *src, uint8_t *dst,
                                       size_t dim, float *scale_out) {
    if (!src || !dst || !scale_out || !dim) return -1;
    float amax = 0.0f;
    for (size_t d = 0; d < dim; d++) {
        float a = fabsf(src[d]);
        if (!isfinite(a)) return -1;
        if (a > amax) amax = a;
    }
    float scale = amax > 0.0f ? amax / 31.0f : 1.0f;
    memset(dst, 0, q38d_kv_i6_row_bytes(dim));
    for (size_t d = 0; d < dim; d++) {
        int q = (int)lrintf(src[d] / scale);
        if (q > 31) q = 31;
        if (q < -31) q = -31;
        unsigned code = (unsigned)q & 63u;
        size_t bit = d * 6, byte = bit >> 3;
        unsigned shift = (unsigned)(bit & 7);
        dst[byte] |= (uint8_t)(code << shift);
        if (shift > 2) dst[byte + 1] |= (uint8_t)(code >> (8 - shift));
    }
    *scale_out = scale;
    return 0;
}

#endif
