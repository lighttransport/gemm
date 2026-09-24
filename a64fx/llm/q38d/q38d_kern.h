/* Qwen3.8 dedicated decode: pair-interleaved low-bit GEMV kernels.
 *
 * Every matrix is stored as independent 8-row groups. Within a group the
 * columns are processed as 32-column "pairs" p. A pair holds two 16-column
 * quantization units u = 0, 1 for all eight rows. One SDOT result vector
 * then has lane 2r+u = row r, unit u, so one SCVTF+FMLA retires two
 * units and a scale vector needs no gather:
 *
 *   code byte (pair p, vector h, row r, unit u, b = 0..3) at 64h+8r+4u+b
 *     low nibble  : column 32p + 16u + 4h + b
 *     high nibble : column 32p + 16u + 8 + 4h + b
 *
 * Group streams (np = cols / 32):
 *   F4  : codes[np][128] | sc[np][16] UE4M3 (lane 2r+u)              144 B/pair
 *   F6  : {low[128] high[64]}[np] | sc[np][8] E8M0 (row r)           200 B/pair
 *   Q6K : {low[128] high[64]}[np] | sc[np][16] int8 | d[np/8][8]     212 B/pair
 * (6-bit formats interleave each pair's code and high-plane bytes into one
 * 192-byte record so that the weights form a single stream.)
 *   Q4K : codes[np][128] | sc[np][8] f32 | off[np][8] f32            192 B/pair
 * High planes (6-bit codes): 64 bytes per pair; byte i describes code byte i
 * of both code vectors: bits [1:0]/[3:2] give the high two bits of the
 * low/high nibble code of vector 0, bits [5:4]/[7:6] those of vector 1.
 * Every field is placed at bits [5:4] by one shift and a 0x30 mask.
 *
 * Activations are quantized per 16 columns (A8: |q| <= 127; A16: centered
 * radix-256 digits, |q| <= 32639) and stored per pair as 32 permuted bytes
 * (A16: 32 low then 32 high digits), plus a float pair of scales multiplied
 * by 2^64 and a float pair of scaled digit sums (for Q4K offsets).
 */
#ifndef Q38D_KERN_H
#define Q38D_KERN_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

/* Q8K: Q6_K weights expanded to exact signed bytes (q - 32), no decode;
 * sub-scales and d as Q6K. 272 B/pair + d. */
enum { Q38D_F4 = 1, Q38D_F6 = 2, Q38D_Q6K = 3, Q38D_Q4K = 4, Q38D_Q8K = 5 };
enum { Q38D_F32 = 0, Q38D_A8 = 8, Q38D_A16 = 16 };

#define Q38D_ACT_EXP 64 /* activation scales are stored multiplied by 2^64 */

typedef struct {
    int cols, arith;     /* cols is a multiple of 32 */
    int8_t *q;           /* [np][32] (A8) or [np][64] (A16) */
    float *sc;           /* [np][2] scale * 2^64 */
    float *sum;          /* [np][2] scale * 2^64 * sum(q) */
    const float *x;      /* F32 arithmetic: original activation */
} q38d_act;

static inline size_t q38d_pair_bytes(int fmt) {
    return fmt == Q38D_F4 ? 144 : fmt == Q38D_F6 ? 200 : fmt == Q38D_Q6K ? 208 :
           fmt == Q38D_Q4K ? 192 : fmt == Q38D_Q8K ? 272 : 0;
}
/* Bytes of one 8-row group. Q6K requires cols % 256 == 0. */
static inline size_t q38d_group_bytes(int fmt, int cols) {
    size_t np = (size_t)cols / 32;
    return np * q38d_pair_bytes(fmt) + (fmt == Q38D_Q6K || fmt == Q38D_Q8K ? np / 8 * 32 : 0);
}

static const int8_t q38d_lut_f4[64] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static const int8_t q38d_lut_f6[64] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,
    0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,
    -16,-18,-20,-22,-24,-26,-28,-30,-32,-36,-40,-44,-48,-52,-56,-60};
static const int8_t q38d_lut_q6[64] = {
    -32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,
    -16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
static const int8_t q38d_lut_q4[64] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

/* ------------------------------------------------------------------ */
/* Stream accessors and portable packing                                */

static inline uint8_t *q38d_codes(uint8_t *g) { return g; }
static inline size_t q38d_code_index(int r, int kk, int *nibble) {
    /* kk = column within a 32-column pair */
    int u = kk / 16, c = kk % 16, half = c / 8, j = c % 8, h = j / 4, b = j % 4;
    *nibble = half;
    return (size_t)(64 * h + 8 * r + 4 * u + b);
}

/* Write code (4 or 6 bits) of group row r, column k. Streams zeroed first. */
/* Byte offset of pair p's low code bytes and high plane within a group. */
static inline size_t q38d_code_off(int fmt, size_t p) {
    return p * (fmt == Q38D_F6 || fmt == Q38D_Q6K ? 192 : fmt == Q38D_Q8K ? 256 : 128);
}
static inline size_t q38d_high_off(size_t p) { return p * 192 + 128; }
static inline size_t q38d_q8k_index(int r, int kk) {
    int u = kk / 16, c = kk % 16, half = c / 8, j = c % 8, h = j / 4, b = j % 4;
    return (size_t)(64 * (2 * h + half) + 8 * r + 4 * u + b);
}
static inline void q38d_put_code(uint8_t *g, int fmt, int cols, int r, int k, unsigned code) {
    size_t np = (size_t)cols / 32, p = (size_t)k / 32;
    int nib;
    if (fmt == Q38D_Q8K) { g[p * 256 + q38d_q8k_index(r, k % 32)] = (uint8_t)(int8_t)((int)code - 32); return; }
    size_t i = q38d_code_index(r, k % 32, &nib);
    (void)np;
    uint8_t *low = g + q38d_code_off(fmt, p);
    low[i] |= (uint8_t)((code & 15) << (nib * 4));
    if (fmt == Q38D_F6 || fmt == Q38D_Q6K) {
        uint8_t *high = g + q38d_high_off(p);
        high[i % 64] |= (uint8_t)(((code >> 4) & 3) << ((i / 64) * 4 + nib * 2));
    }
}
static inline unsigned q38d_get_code(const uint8_t *g, int fmt, int cols, int r, int k) {
    size_t np = (size_t)cols / 32, p = (size_t)k / 32;
    int nib;
    if (fmt == Q38D_Q8K) return (unsigned)((int)(int8_t)g[p * 256 + q38d_q8k_index(r, k % 32)] + 32);
    size_t i = q38d_code_index(r, k % 32, &nib);
    (void)np;
    unsigned c = (g[q38d_code_off(fmt, p) + i] >> (nib * 4)) & 15;
    if (fmt == Q38D_F6 || fmt == Q38D_Q6K) {
        c |= ((g[q38d_high_off(p) + i % 64] >> ((i / 64) * 4 + nib * 2)) & 3) << 4;
    }
    return c;
}
static inline uint8_t *q38d_scale_stream(uint8_t *g, int fmt, int cols) {
    size_t np = (size_t)cols / 32;
    return g + np * (fmt == Q38D_F4 || fmt == Q38D_Q4K ? 128 : fmt == Q38D_Q8K ? 256 : 192);
}

/* Exact FP32 weight of (r, k). */
static inline float q38d_weight(const uint8_t *g, int fmt, int cols, int r, int k) {
    unsigned c = q38d_get_code(g, fmt, cols, r, k);
    const uint8_t *s = q38d_scale_stream((uint8_t *)g, fmt, cols);
    size_t np = (size_t)cols / 32, p = (size_t)k / 32;
    int u = (k % 32) / 16;
    if (fmt == Q38D_F4) {
        unsigned code = s[p * 16 + 2 * r + u];
        float sc = !code ? 0.f : ldexpf((float)(8 + (code & 7)), (int)(code >> 3) - 21);
        return q38d_lut_f4[c] * sc;
    }
    if (fmt == Q38D_F6) return ldexpf((float)q38d_lut_f6[c], (int)s[p * 8 + r] - 130);
    if (fmt == Q38D_Q6K || fmt == Q38D_Q8K) {
        const float *d = (const float *)(s + np * 16);
        return d[(p / 8) * 8 + r] * (float)(int8_t)s[p * 16 + 2 * r + u] * (float)q38d_lut_q6[c];
    }
    const float *sc = (const float *)s, *off = sc + np * 8;
    return sc[p * 8 + r] * (float)c - off[p * 8 + r];
}

/* UE4M3 block scales are re-encoded as E5M3 bytes (exponent bias 20,
 * dec(u) = (8 + (u & 7)) * 2^((u >> 3) - 20)) so that the kernels' decode
 * as_float(u << 20) = dec(u) * 2^-110 is a *normal* float for every code,
 * including UE4M3 subnormals. A64FX pays ~70 cycles per instruction for
 * subnormal FP operands; the source 0x7f sentinel and zero both map to 0. */
static inline uint8_t q38d_e5m3_from_ue4m3(uint8_t u) {
    if (!u || u == 127) return 0;
    unsigned e = u >> 3, m = u & 7;
    if (e) return (uint8_t)(((e + 10) << 3) | m);
    unsigned E = 11, mm = m;          /* m * 2^-9 = (m << k) * 2^(-9-k), E = 11 - k */
    while (mm < 8) { mm <<= 1; E--; }
    return (uint8_t)((E << 3) | (mm - 8));
}

/* Repack eight rows of the original compact NVFP4 tiles (288 B per 64
 * columns, codes[4][64], scale[4][8]) into the F4 group layout. The UE4M3
 * 0x7f sentinel decodes to zero in the source, so it is stored as zero. */
static inline void q38d_repack_f4(uint8_t *dst, const uint8_t *tiles, int cols) {
    int nb = cols / 64;
    size_t np = (size_t)cols / 32;
    uint8_t *sc = dst + np * 128;
    for (int b = 0; b < nb; b++) {
        const uint8_t *t = tiles + (size_t)b * 288;
        for (int s = 0; s < 4; s++) {
            int p = 2 * b + s / 2, u = s % 2;
            for (int r = 0; r < 8; r++) {
                for (int j = 0; j < 8; j++) {
                    int h = j / 4, bb = j % 4;
                    dst[(size_t)p * 128 + 64 * h + 8 * r + 4 * u + bb] = t[s * 64 + r * 8 + j];
                }
                sc[(size_t)p * 16 + 2 * r + u] = q38d_e5m3_from_ue4m3(t[256 + s * 8 + r]);
            }
        }
    }
}

/* Repack eight rows of original FP6 tiles (400 B per 64 columns, low[4][64],
 * high[4][32], scale[2][8]) into the F6 group layout. */
static inline void q38d_repack_f6(uint8_t *dst, const uint8_t *tiles, int cols) {
    int nb = cols / 64;
    size_t np = (size_t)cols / 32;
    memset(dst, 0, q38d_group_bytes(Q38D_F6, cols));
    uint8_t *sc = dst + np * 192;
    for (int b = 0; b < nb; b++) {
        const uint8_t *t = tiles + (size_t)b * 400;
        for (int r = 0; r < 8; r++)
            for (int k = 0; k < 64; k++) {
                int s = k / 16, lane = r * 8 + k % 8, half = (k % 16) / 8;
                unsigned lo = (t[s * 64 + lane] >> (half * 4)) & 15;
                unsigned hi = (t[256 + s * 32 + lane / 2] >> ((lane & 1) * 4 + half * 2)) & 3;
                q38d_put_code(dst, Q38D_F6, cols, r, b * 64 + k, lo | hi << 4);
            }
        for (int h = 0; h < 2; h++)
            for (int r = 0; r < 8; r++) sc[(size_t)(2 * b + h) * 8 + r] = t[384 + h * 8 + r];
    }
}

static inline float q38d_half_to_float(uint16_t h) {
    unsigned e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? (m ? NAN : INFINITY)
                                                       : ldexpf((float)(m | 1024), (int)e - 25);
    return h & 0x8000 ? -v : v;
}

/* Repack eight GGML Q6_K rows (210 B per 256 columns). */
static inline void q38d_repack_q6k_fmt(uint8_t *dst, const uint8_t *src, size_t row_bytes,
                                       int rows, int cols, int fmt) {
    size_t np = (size_t)cols / 32;
    memset(dst, 0, q38d_group_bytes(fmt, cols));
    uint8_t *sc = q38d_scale_stream(dst, fmt, cols);
    float *d = (float *)(sc + np * 16);
    for (int r = 0; r < rows && r < 8; r++) {
        const uint8_t *row = src + (size_t)r * row_bytes;
        for (int sb = 0; sb < cols / 256; sb++) {
            const uint8_t *blk = row + (size_t)sb * 210;
            const uint8_t *ql = blk, *qh = blk + 128;
            const int8_t *scales = (const int8_t *)(blk + 192);
            uint16_t hd; memcpy(&hd, blk + 208, 2);
            d[(size_t)sb * 8 + r] = q38d_half_to_float(hd);
            for (int n = 0; n < 2; n++)
                for (int l = 0; l < 32; l++) {
                    const uint8_t *L = ql + 64 * n, *H = qh + 32 * n;
                    unsigned q1 = (L[l] & 15) | ((H[l] >> 0) & 3) << 4;
                    unsigned q2 = (L[l + 32] & 15) | ((H[l] >> 2) & 3) << 4;
                    unsigned q3 = (L[l] >> 4) | ((H[l] >> 4) & 3) << 4;
                    unsigned q4 = (L[l + 32] >> 4) | ((H[l] >> 6) & 3) << 4;
                    int base = sb * 256 + n * 128;
                    q38d_put_code(dst, fmt, cols, r, base + l, q1);
                    q38d_put_code(dst, fmt, cols, r, base + l + 32, q2);
                    q38d_put_code(dst, fmt, cols, r, base + l + 64, q3);
                    q38d_put_code(dst, fmt, cols, r, base + l + 96, q4);
                }
            for (int j = 0; j < 16; j++) {
                int k = sb * 256 + j * 16, p = k / 32, u = (k % 32) / 16;
                sc[(size_t)p * 16 + 2 * r + u] = (uint8_t)scales[j];
            }
        }
    }
}

static inline void q38d_repack_q6k(uint8_t *dst, const uint8_t *src, size_t row_bytes, int rows, int cols) {
    q38d_repack_q6k_fmt(dst, src, row_bytes, rows, cols, Q38D_Q6K);
}
static inline void q38d_repack_q8k(uint8_t *dst, const uint8_t *src, size_t row_bytes, int rows, int cols) {
    q38d_repack_q6k_fmt(dst, src, row_bytes, rows, cols, Q38D_Q8K);
}

/* Repack eight GGML Q4_K rows (144 B per 256 columns). */
static inline void q38d_repack_q4k(uint8_t *dst, const uint8_t *src, size_t row_bytes,
                                   int rows, int cols) {
    size_t np = (size_t)cols / 32;
    memset(dst, 0, q38d_group_bytes(Q38D_Q4K, cols));
    float *sc = (float *)(dst + np * 128), *off = sc + np * 8;
    for (int r = 0; r < rows && r < 8; r++) {
        const uint8_t *row = src + (size_t)r * row_bytes;
        for (int sb = 0; sb < cols / 256; sb++) {
            const uint8_t *blk = row + (size_t)sb * 144;
            uint16_t hd, hm; memcpy(&hd, blk, 2); memcpy(&hm, blk + 2, 2);
            float d = q38d_half_to_float(hd), dmin = q38d_half_to_float(hm);
            const uint8_t *q6 = blk + 4, *qs = blk + 16;
            for (int j = 0; j < 8; j++) {
                unsigned s, m;
                if (j < 4) { s = q6[j] & 63; m = q6[j + 4] & 63; }
                else {
                    s = (q6[j + 4] & 15) | ((q6[j - 4] >> 6) << 4);
                    m = (q6[j + 4] >> 4) | ((q6[j] >> 6) << 4);
                }
                int p = sb * 8 + j;
                sc[(size_t)p * 8 + r] = d * (float)s;
                off[(size_t)p * 8 + r] = dmin * (float)m;
                const uint8_t *q = qs + (j / 2) * 32;
                for (int l = 0; l < 32; l++)
                    q38d_put_code(dst, Q38D_Q4K, cols, r, p * 32 + l,
                                  j & 1 ? q[l] >> 4 : q[l] & 15);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Activation preparation                                               */

static inline size_t q38d_act_qbytes(int cols, int arith) {
    return (size_t)cols / 32 * (arith == Q38D_A16 ? 64 : 32);
}

/* Portable reference quantizer: per-16 amax, q = rint(x * (limit/amax)). */
static inline void q38d_prepare_ref(q38d_act *a, const float *x) {
    int np = a->cols / 32, limit = a->arith == Q38D_A16 ? 32639 : 127;
    a->x = x;
    if (a->arith == Q38D_F32) return;
    for (int p = 0; p < np; p++) {
        for (int u = 0; u < 2; u++) {
            const float *v = x + 32 * p + 16 * u;
            float amax = 0;
            for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(v[j]));
            float inv = amax > 0 ? (float)limit / amax : 0.f;
            float scale = amax > 0 ? amax / (float)limit : 0.f;
            int sum = 0;
            for (int j = 0; j < 16; j++) {
                int q = (int)rintf(v[j] * inv);
                if (q > limit) q = limit;
                if (q < -limit) q = -limit;
                sum += q;
                int half = j / 8, h = (j % 8) / 4, b = j % 4;
                int idx = (2 * h + half) * 8 + 4 * u + b;
                if (a->arith == Q38D_A8) a->q[(size_t)p * 32 + idx] = (int8_t)q;
                else {
                    int hi = (q + 128) >> 8, lo = q - 256 * hi;
                    a->q[(size_t)p * 64 + idx] = (int8_t)lo;
                    a->q[(size_t)p * 64 + 32 + idx] = (int8_t)hi;
                }
            }
            a->sc[2 * p + u] = ldexpf(scale, Q38D_ACT_EXP);
            a->sum[2 * p + u] = ldexpf(scale, Q38D_ACT_EXP) * (float)sum;
        }
    }
}

/* Reference dot with the quantized activations, double accumulation. */
static inline void q38d_group_ref(float *out, const uint8_t *g, int fmt, const q38d_act *a) {
    int cols = a->cols;
    for (int r = 0; r < 8; r++) {
        double s = 0;
        for (int k = 0; k < cols; k++) {
            double w = q38d_weight(g, fmt, cols, r, k);
            double xv;
            if (a->arith == Q38D_F32) xv = a->x[k];
            else {
                int p = k / 32, u = (k % 32) / 16, j = k % 16;
                int half = j / 8, h = (j % 8) / 4, b = j % 4;
                int idx = (2 * h + half) * 8 + 4 * u + b, q;
                if (a->arith == Q38D_A8) q = a->q[(size_t)p * 32 + idx];
                else q = a->q[(size_t)p * 64 + idx] + 256 * a->q[(size_t)p * 64 + 32 + idx];
                xv = (double)q * ldexp((double)a->sc[2 * p + u], -Q38D_ACT_EXP);
            }
            s += w * xv;
        }
        out[r] = (float)s;
    }
}

#ifdef __ARM_FEATURE_SVE
/* ------------------------------------------------------------------ */
/* SVE (512-bit) implementation                                         */

/* Permutation from natural 32-column order (two int8 halves of 16) to the
 * pair order: out[(2h+half)*8 + 4u + b] = in[16u + 8half + 4h + b]. */
static const uint8_t q38d_act_perm[64] = {
    0,1,2,3,16,17,18,19, 8,9,10,11,24,25,26,27, 4,5,6,7,20,21,22,23, 12,13,14,15,28,29,30,31,
    32,33,34,35,48,49,50,51, 40,41,42,43,56,57,58,59, 36,37,38,39,52,53,54,55, 44,45,46,47,60,61,62,63};

/* Quantize pairs [p0, p1) of x (optionally x * mul * w, the fused RMSNorm). */
static inline void q38d_prepare_pairs(q38d_act *a, const float *x, float mul, const float *w, int p0, int p1);
static inline void q38d_prepare_sve(q38d_act *a, const float *x, float mul, const float *w) {
    a->x = x;
    if (a->arith == Q38D_F32) return;
    q38d_prepare_pairs(a, x, mul, w, 0, a->cols / 32);
}
static inline void q38d_prepare_pairs(q38d_act *a, const float *x, float mul, const float *w, int p0, int p1) {
    const svbool_t pf = svptrue_b32(), pb = svptrue_b8();
    const svbool_t p32 = svwhilelt_b8((uint32_t)0, (uint32_t)32);
    const svuint8_t perm = svld1_u8(pb, q38d_act_perm);
    const int a16 = a->arith == Q38D_A16;
    const float limit = a16 ? 32639.f : 127.f;
    const int lim = (int)limit;
    /* Batches of 16 pairs (32 blocks): independent reductions first, then
     * vector divisions, then quantization. Arithmetic matches the scalar
     * reference exactly (IEEE division, rint to even, same products). */
    float v[32 * 16] __attribute__((aligned(64)));
    float mx[32] __attribute__((aligned(64))), inv[32] __attribute__((aligned(64)));
    float scl[32] __attribute__((aligned(64))), sums[32] __attribute__((aligned(64)));
    for (int pb0 = p0; pb0 < p1; pb0 += 16) {
        int n = p1 - pb0 < 16 ? p1 - pb0 : 16, nb = 2 * n;
        for (int b = 0; b < nb; b++) {
            const float *xs = x + 32 * pb0 + 16 * b;
            svfloat32_t t = svld1_f32(pf, xs);
            if (w) t = svmul_f32_x(pf, svmul_n_f32_x(pf, t, mul), svld1_f32(pf, w + 32 * pb0 + 16 * b));
            svst1_f32(pf, v + 16 * b, t);
            mx[b] = svmaxv_f32(pf, svabs_f32_x(pf, t));
        }
        for (int b = nb; b < 32; b++) mx[b] = 0.f;
        for (int b = 0; b < 32; b += 16) {
            svfloat32_t m = svld1_f32(pf, mx + b);
            svbool_t pos = svcmpgt_n_f32(pf, m, 0.f);
            svfloat32_t iv = svdiv_f32_z(pos, svdup_n_f32(limit), m);
            svfloat32_t sc = svmul_n_f32_z(pos, svdiv_n_f32_x(pf, m, limit), 0x1p64f);
            svst1_f32(pf, inv + b, iv);
            svst1_f32(pf, scl + b, sc);
        }
        for (int q = 0; q < n; q++) {
            int p = pb0 + q;
            svint32_t q0 = svcvt_s32_f32_x(pf, svrintn_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, v + 32 * q), inv[2 * q])));
            svint32_t q1 = svcvt_s32_f32_x(pf, svrintn_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, v + 32 * q + 16), inv[2 * q + 1])));
            q0 = svmax_n_s32_x(pf, svmin_n_s32_x(pf, q0, lim), -lim);
            q1 = svmax_n_s32_x(pf, svmin_n_s32_x(pf, q1, lim), -lim);
            sums[2 * q] = (float)svaddv_s32(pf, q0);
            sums[2 * q + 1] = (float)svaddv_s32(pf, q1);
            if (!a16) {
                svint16_t h = svuzp1_s16(svreinterpret_s16_s32(q0), svreinterpret_s16_s32(q1));
                svint8_t b8 = svuzp1_s8(svreinterpret_s8_s16(h), svreinterpret_s8_s16(h));
                svst1_s8(p32, a->q + (size_t)p * 32, svtbl_s8(b8, perm));
            } else {
                svint32_t h0 = svasr_n_s32_x(pf, svadd_n_s32_x(pf, q0, 128), 8);
                svint32_t h1 = svasr_n_s32_x(pf, svadd_n_s32_x(pf, q1, 128), 8);
                svint32_t l0 = svsub_s32_x(pf, q0, svlsl_n_s32_x(pf, h0, 8));
                svint32_t l1 = svsub_s32_x(pf, q1, svlsl_n_s32_x(pf, h1, 8));
                svint16_t hl = svuzp1_s16(svreinterpret_s16_s32(l0), svreinterpret_s16_s32(l1));
                svint16_t hh = svuzp1_s16(svreinterpret_s16_s32(h0), svreinterpret_s16_s32(h1));
                svint8_t b8 = svuzp1_s8(svreinterpret_s8_s16(hl), svreinterpret_s8_s16(hh));
                svst1_s8(pb, a->q + (size_t)p * 64, svtbl_s8(b8, perm));
            }
        }
        for (int b = 0; b < nb; b += 16) {
            svbool_t pg = svwhilelt_b32(b, nb);
            svst1_f32(pg, a->sc + 2 * pb0 + b, svld1_f32(pg, scl + b));
            svst1_f32(pg, a->sum + 2 * pb0 + b, svmul_f32_x(pg, svld1_f32(pg, scl + b), svld1_f32(pg, sums + b)));
        }
    }
}

static inline svint8_t q38d_rep8(const int8_t *p) {
    int64_t v;
    memcpy(&v, p, 8);
    return svreinterpret_s8_s64(svdup_n_s64(v));
}
static inline svfloat32_t q38d_rep2f(const float *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return svreinterpret_f32_u64(svdup_n_u64(v));
}

/* Decode one 64-byte code vector (and 32-byte high plane) to two int8
 * vectors of weights for the low and high nibble positions. */
#define Q38D_DECODE(FMT, LUT, Z, HP, V, L, H) do {                                  \
    if ((FMT) == Q38D_F4 || (FMT) == Q38D_Q4K) {                                  \
        L = svtbl_s8(LUT, svand_n_u8_x(pb, Z, 15));                               \
        H = svtbl_s8(LUT, svlsr_n_u8_x(pb, Z, 4));                                \
    } else {                                                                      \
        svuint8_t hp_ = svld1_u8(pb, HP);                                         \
        svuint8_t tl_ = (V) ? hp_ : svlsl_n_u8_x(pb, hp_, 4);                      \
        svuint8_t th_ = (V) ? svlsr_n_u8_x(pb, hp_, 2) : svlsl_n_u8_x(pb, hp_, 2); \
        svuint8_t il_ = svorr_u8_x(pb, svand_n_u8_x(pb, Z, 15), svand_n_u8_x(pb, tl_, 0x30)); \
        svuint8_t ih_ = svorr_u8_x(pb, svlsr_n_u8_x(pb, Z, 4), svand_n_u8_x(pb, th_, 0x30)); \
        L = svtbl_s8(LUT, il_);                                                   \
        H = svtbl_s8(LUT, ih_);                                                   \
    }                                                                             \
} while (0)

/* One pair: integer dot in lanes 2r+u. */
static inline __attribute__((always_inline)) svint32_t
q38d_pair_dot(int fmt, int arith, svint8_t lut, const uint8_t *code, const uint8_t *high,
              const int8_t *aq) {
    const svbool_t pb = svptrue_b8(), pf = svptrue_b32();
    const svbool_t p32 = svwhilelt_b8((uint32_t)0, (uint32_t)32);
    svint8_t l0, h0, l1, h1;
    if (fmt == Q38D_Q8K) {
        l0 = svld1_s8(pb, (const int8_t *)code); h0 = svld1_s8(pb, (const int8_t *)code + 64);
        l1 = svld1_s8(pb, (const int8_t *)code + 128); h1 = svld1_s8(pb, (const int8_t *)code + 192);
    } else {
        svuint8_t z0 = svld1_u8(pb, code), z1 = svld1_u8(pb, code + 64);
        Q38D_DECODE(fmt, lut, z0, high, 0, l0, h0);
        Q38D_DECODE(fmt, lut, z1, high, 1, l1, h1);
    }
    svint32_t d = svdup_n_s32(0);
    if (arith == Q38D_A16) {
        const int8_t *ah = aq + 32;
        d = svdot_s32(d, l0, q38d_rep8(ah));
        d = svdot_s32(d, h0, q38d_rep8(ah + 8));
        d = svdot_s32(d, l1, q38d_rep8(ah + 16));
        d = svdot_s32(d, h1, q38d_rep8(ah + 24));
        d = svlsl_n_s32_x(pf, d, 8);
    }
    d = svdot_s32(d, l0, q38d_rep8(aq));
    d = svdot_s32(d, h0, q38d_rep8(aq + 8));
    d = svdot_s32(d, l1, q38d_rep8(aq + 16));
    d = svdot_s32(d, h1, q38d_rep8(aq + 24));
    return d;
}

/* Scale vector (lanes 2r+u) for pair p, already multiplied by the
 * activation scale pair. */
static inline __attribute__((always_inline)) svfloat32_t
q38d_pair_scale(int fmt, const uint8_t *sc, size_t p, const float *asc, svfloat32_t dvec) {
    const svbool_t pf = svptrue_b32();
    svfloat32_t a = q38d_rep2f(asc + 2 * p);
    if (fmt == Q38D_F4) {
        svuint32_t u = svld1ub_u32(pf, sc + p * 16);
        return svmul_f32_x(pf, svreinterpret_f32_u32(svlsl_n_u32_x(pf, u, 20)), a);
    }
    if (fmt == Q38D_F6) {
        svuint32_t u = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc + p * 8);
        u = svzip1_u32(u, u);
        return svmul_f32_x(pf, svreinterpret_f32_u32(svlsl_n_u32_x(pf, u, 23)), a);
    }
    if (fmt == Q38D_Q6K || fmt == Q38D_Q8K) {
        svint32_t s = svld1sb_s32(pf, (const int8_t *)sc + p * 16);
        return svmul_f32_x(pf, svmul_f32_x(pf, svcvt_f32_s32_x(pf, s), dvec), a);
    }
    svfloat32_t s = svld1_f32(svptrue_pat_b32(SV_VL8), (const float *)sc + p * 8);
    return svmul_f32_x(pf, svzip1_f32(s, s), a);
}

/* Final per-format exponent correction. */
static inline float q38d_out_scale(int fmt) {
    return fmt == Q38D_F4 ? 0x1p45f : fmt == Q38D_F6 ? 0x1p-67f : 0x1p-64f;
}

/* Compute 8 outputs of one group. mode 0: store, 1: add into out. */
static inline __attribute__((always_inline)) void
q38d_group_sve(float *out, const uint8_t *g, int fmt, int arith, const q38d_act *a, int mode, int nrows) {
    const svbool_t pf = svptrue_b32(), pb = svptrue_b8();
    const int cols = a->cols;
    const size_t np = (size_t)cols / 32;
    const int8_t *lutp = fmt == Q38D_F4 ? q38d_lut_f4 : fmt == Q38D_F6 ? q38d_lut_f6 :
                         fmt == Q38D_Q6K ? q38d_lut_q6 : q38d_lut_q4;
    const svint8_t lut = svld1_s8(pb, lutp);
    const uint8_t *high = g + np * 128;
    const uint8_t *sc = q38d_scale_stream((uint8_t *)g, fmt, cols);
    const float *dq = fmt == Q38D_Q6K || fmt == Q38D_Q8K ? (const float *)(sc + np * 16) : NULL;
    const float *off = fmt == Q38D_Q4K ? (const float *)sc + np * 8 : NULL;
    const size_t aqs = arith == Q38D_A16 ? 64 : 32;
    const size_t cps = q38d_code_off(fmt, 1); /* code(+high) bytes per pair */
    svfloat32_t acc0 = svdup_n_f32(0), acc1 = acc0, offacc = acc0, dvec = acc0;
    size_t p = 0;
    for (; p + 2 <= np; p += 2) {
        if (dq && !(p % 8)) {
            svfloat32_t d8 = svld1_f32(svptrue_pat_b32(SV_VL8), dq + (p / 8) * 8);
            dvec = svzip1_f32(d8, d8);
        }
        svint32_t d0 = q38d_pair_dot(fmt, arith, lut, g + p * cps, g + p * cps + 128, a->q + p * aqs);
        svint32_t d1 = q38d_pair_dot(fmt, arith, lut, g + (p + 1) * cps, g + (p + 1) * cps + 128,
                                     a->q + (p + 1) * aqs);
        acc0 = svmla_f32_x(pf, acc0, svcvt_f32_s32_x(pf, d0), q38d_pair_scale(fmt, sc, p, a->sc, dvec));
        acc1 = svmla_f32_x(pf, acc1, svcvt_f32_s32_x(pf, d1), q38d_pair_scale(fmt, sc, p + 1, a->sc, dvec));
        if (fmt == Q38D_Q4K) {
            svfloat32_t o0 = svld1_f32(svptrue_pat_b32(SV_VL8), off + p * 8);
            svfloat32_t o1 = svld1_f32(svptrue_pat_b32(SV_VL8), off + (p + 1) * 8);
            offacc = svmla_f32_x(pf, offacc, svzip1_f32(o0, o0), q38d_rep2f(a->sum + 2 * p));
            offacc = svmla_f32_x(pf, offacc, svzip1_f32(o1, o1), q38d_rep2f(a->sum + 2 * p + 2));
        }
    }
    for (; p < np; p++) {
        if (dq && !(p % 8)) {
            svfloat32_t d8 = svld1_f32(svptrue_pat_b32(SV_VL8), dq + (p / 8) * 8);
            dvec = svzip1_f32(d8, d8);
        }
        svint32_t d0 = q38d_pair_dot(fmt, arith, lut, g + p * cps, g + p * cps + 128, a->q + p * aqs);
        acc0 = svmla_f32_x(pf, acc0, svcvt_f32_s32_x(pf, d0), q38d_pair_scale(fmt, sc, p, a->sc, dvec));
        if (fmt == Q38D_Q4K) {
            svfloat32_t o0 = svld1_f32(svptrue_pat_b32(SV_VL8), off + p * 8);
            offacc = svmla_f32_x(pf, offacc, svzip1_f32(o0, o0), q38d_rep2f(a->sum + 2 * p));
        }
    }
    svfloat32_t acc = svadd_f32_x(pf, acc0, acc1);
    if (fmt == Q38D_Q4K) acc = svsub_f32_x(pf, acc, offacc);
    svfloat32_t r = svadd_f32_x(pf, svuzp1_f32(acc, acc), svuzp2_f32(acc, acc));
    r = svmul_n_f32_x(pf, r, q38d_out_scale(fmt));
    svbool_t p8 = svwhilelt_b32((uint32_t)0, (uint32_t)nrows);
    if (mode) r = svadd_f32_x(p8, r, svld1_f32(p8, out));
    svst1_f32(p8, out, r);
}

/* Exact-weight FP32 reference path (unquantized activations). */
static inline void q38d_group_f32(float *out, const uint8_t *g, int fmt, const float *x, int cols,
                                  int mode, int nrows) {
    for (int r = 0; r < nrows; r++) {
        const svbool_t pf = svptrue_b32();
        svfloat32_t acc = svdup_n_f32(0);
        float wrow[64];
        for (int k0 = 0; k0 < cols; k0 += 64) {
            for (int k = 0; k < 64; k++) wrow[k] = q38d_weight(g, fmt, cols, r, k0 + k);
            for (int k = 0; k < 64; k += 16)
                acc = svmla_f32_x(pf, acc, svld1_f32(pf, wrow + k), svld1_f32(pf, x + k0 + k));
        }
        float v = svaddv_f32(pf, acc);
        out[r] = mode ? out[r] + v : v;
    }
}

/* Hand-scheduled assembly pair kernels (q38d_kern_sve.S). */
void q38d_asm_f4_a8(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                    const float *, long, float *, const int8_t *);
void q38d_asm_f4_a16(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                     const float *, long, float *, const int8_t *);
void q38d_asm_f6_a8(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                    const float *, long, float *, const int8_t *);
void q38d_asm_f6_a16(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                     const float *, long, float *, const int8_t *);

void q38d_asm_q6k_a8(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                     const float *, long, float *, const int8_t *);
void q38d_asm_q6k_a16(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                      const float *, long, float *, const int8_t *);

#define Q38D_ASM_DECL(n) void n(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *, \
                                const float *, long, float *, const int8_t *);
Q38D_ASM_DECL(q38d_asm2_f4_a8) Q38D_ASM_DECL(q38d_asm2_f4_a16) Q38D_ASM_DECL(q38d_asm2_f6_a8)
Q38D_ASM_DECL(q38d_asm2_f6_a16) Q38D_ASM_DECL(q38d_asm2_q6k_a8) Q38D_ASM_DECL(q38d_asm2_q6k_a16)
Q38D_ASM_DECL(q38d_asm4_f4_a8) Q38D_ASM_DECL(q38d_asm4_f4_a16)
Q38D_ASM_DECL(q38d_asm5_f4_a8) Q38D_ASM_DECL(q38d_asm5_f4_a16) Q38D_ASM_DECL(q38d_asm5_f6_a8)
Q38D_ASM_DECL(q38d_asm5_f6_a16) Q38D_ASM_DECL(q38d_asm5_q6k_a8) Q38D_ASM_DECL(q38d_asm5_q6k_a16)
/* 1: loads one pair ahead; 2: two pairs ahead; 4: four pairs ahead (F4);
 * 5: two ahead with the combine/convert chain split over two stages
 * 6: F4 two pairs interleaved per slot (other formats use 5) */
Q38D_ASM_DECL(q38d_asm6_f4_a8) Q38D_ASM_DECL(q38d_asm6_f4_a16)
Q38D_ASM_DECL(q38d_asm5_q8k_a8) Q38D_ASM_DECL(q38d_asm5_q8k_a16)
Q38D_ASM_DECL(q38d_asm7_f4_a8) Q38D_ASM_DECL(q38d_asm7_f4_a16)
Q38D_ASM_DECL(q38d_asm7c_f4_a8) Q38D_ASM_DECL(q38d_asm7c_f4_a16)
Q38D_ASM_DECL(q38d_asm9_f6_a8) Q38D_ASM_DECL(q38d_asm9_f6_a16)
static int q38d_f6_variant = 5; /* 9: two pairs per slot; 5: single pair */
void q38d_asm5_q4k_a8(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                      const float *, long, float *, const int8_t *, const float *);
void q38d_asm5_q4k_a16(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                       const float *, long, float *, const int8_t *, const float *);
static int q38d_asm_variant = 7;

/* F4/F6/Q6K group through the assembly kernels. */
static inline void q38d_group_asm(float *out, const uint8_t *g, int fmt, int arith,
                                  const q38d_act *a, int mode, int nrows) {
    const size_t np = (size_t)a->cols / 32;
    float acc[16] __attribute__((aligned(64)));
    const uint8_t *high = g + np * 128, *sc = g + np * (fmt == Q38D_F4 ? 128 : 192);
    if (fmt == Q38D_Q8K)
        (arith == Q38D_A16 ? q38d_asm5_q8k_a16 : q38d_asm5_q8k_a8)(
            g, high, g + np * 256, a->q, a->sc, (long)np, acc, q38d_lut_q6);
    else if (fmt == Q38D_Q4K)
        (arith == Q38D_A16 ? q38d_asm5_q4k_a16 : q38d_asm5_q4k_a8)(
            g, high, g + np * 128, a->q, a->sc, (long)np, acc, q38d_lut_q4, a->sum);
    else if (q38d_asm_variant == 8 && fmt == Q38D_F4 && np % 2 == 0 && np >= 8)
        (arith == Q38D_A16 ? q38d_asm7c_f4_a16 : q38d_asm7c_f4_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f4);
    else if (q38d_asm_variant == 7 && fmt == Q38D_F4 && np % 2 == 0 && np >= 8)
        (arith == Q38D_A16 ? q38d_asm7_f4_a16 : q38d_asm7_f4_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f4);
    else if (q38d_asm_variant >= 6 && fmt == Q38D_F4 && np % 2 == 0 && np >= 6)
        (arith == Q38D_A16 ? q38d_asm6_f4_a16 : q38d_asm6_f4_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f4);
    else if (fmt == Q38D_F6 && q38d_f6_variant == 9 && np % 2 == 0 && np >= 6)
        (arith == Q38D_A16 ? q38d_asm9_f6_a16 : q38d_asm9_f6_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f6);
    else if (q38d_asm_variant >= 5 && np % 2 == 0 && np >= 6) {
        if (fmt == Q38D_F4) (arith == Q38D_A16 ? q38d_asm5_f4_a16 : q38d_asm5_f4_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f4);
        else if (fmt == Q38D_F6) (arith == Q38D_A16 ? q38d_asm5_f6_a16 : q38d_asm5_f6_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f6);
        else (arith == Q38D_A16 ? q38d_asm5_q6k_a16 : q38d_asm5_q6k_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_q6);
    } else
    if (q38d_asm_variant == 4 && fmt == Q38D_F4 && np % 4 == 0 && np >= 8)
        (arith == Q38D_A16 ? q38d_asm4_f4_a16 : q38d_asm4_f4_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f4);
    else if (q38d_asm_variant >= 2) {
        if (fmt == Q38D_F4) (arith == Q38D_A16 ? q38d_asm2_f4_a16 : q38d_asm2_f4_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f4);
        else if (fmt == Q38D_F6) (arith == Q38D_A16 ? q38d_asm2_f6_a16 : q38d_asm2_f6_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f6);
        else (arith == Q38D_A16 ? q38d_asm2_q6k_a16 : q38d_asm2_q6k_a8)(
            g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_q6);
    } else
    if (fmt == Q38D_F4) (arith == Q38D_A16 ? q38d_asm_f4_a16 : q38d_asm_f4_a8)(
        g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f4);
    else if (fmt == Q38D_F6) (arith == Q38D_A16 ? q38d_asm_f6_a16 : q38d_asm_f6_a8)(
        g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_f6);
    else (arith == Q38D_A16 ? q38d_asm_q6k_a16 : q38d_asm_q6k_a8)(
        g, high, sc, a->q, a->sc, (long)np, acc, q38d_lut_q6);
    const float e = q38d_out_scale(fmt == Q38D_Q8K ? Q38D_Q6K : fmt);
    for (int r = 0; r < nrows; r++) {
        float v = (acc[2 * r] + acc[2 * r + 1]) * e;
        out[r] = mode ? out[r] + v : v;
    }
}

/* Row groups [g0, g1) of a matrix whose groups are contiguous at base. */
static inline void q38d_gemv(float *out, const uint8_t *base, int fmt, const q38d_act *a,
                             int g0, int g1, int rows, int mode) {
    size_t gb = q38d_group_bytes(fmt, a->cols);
    for (int g = g0; g < g1; g++) {
        int nr = rows - g * 8 < 8 ? rows - g * 8 : 8;
        const uint8_t *w = base + (size_t)g * gb;
        float *o = out + (size_t)g * 8;
        if (a->arith == Q38D_F32) { q38d_group_f32(o, w, fmt, a->x, a->cols, mode, nr); continue; }
        switch (fmt * 32 + a->arith) {
        case Q38D_F4 * 32 + Q38D_A8:   q38d_group_sve(o, w, Q38D_F4, Q38D_A8, a, mode, nr); break;
        case Q38D_F4 * 32 + Q38D_A16:  q38d_group_sve(o, w, Q38D_F4, Q38D_A16, a, mode, nr); break;
        case Q38D_F6 * 32 + Q38D_A8:   q38d_group_sve(o, w, Q38D_F6, Q38D_A8, a, mode, nr); break;
        case Q38D_F6 * 32 + Q38D_A16:  q38d_group_sve(o, w, Q38D_F6, Q38D_A16, a, mode, nr); break;
        case Q38D_Q6K * 32 + Q38D_A8:  q38d_group_sve(o, w, Q38D_Q6K, Q38D_A8, a, mode, nr); break;
        case Q38D_Q6K * 32 + Q38D_A16: q38d_group_sve(o, w, Q38D_Q6K, Q38D_A16, a, mode, nr); break;
        case Q38D_Q4K * 32 + Q38D_A8:  q38d_group_sve(o, w, Q38D_Q4K, Q38D_A8, a, mode, nr); break;
        case Q38D_Q8K * 32 + Q38D_A8:  q38d_group_sve(o, w, Q38D_Q8K, Q38D_A8, a, mode, nr); break;
        case Q38D_Q8K * 32 + Q38D_A16: q38d_group_sve(o, w, Q38D_Q8K, Q38D_A16, a, mode, nr); break;
        default:                       q38d_group_sve(o, w, Q38D_Q4K, Q38D_A16, a, mode, nr); break;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Engine helpers                                                        */

/* expf with ~1 ulp error: n = rint(x*log2e), r = x - n*ln2 (two parts),
 * degree-7 Taylor polynomial, then FSCALE. Inputs are clamped. */
static inline svfloat32_t q38d_exp_sve(svbool_t pg, svfloat32_t x) {
    x = svmin_n_f32_x(pg, svmax_n_f32_x(pg, x, -87.f), 88.7f); /* no subnormal results */
    svfloat32_t n = svrintn_f32_x(pg, svmul_n_f32_x(pg, x, 1.44269504088896341f));
    svfloat32_t r = svmls_n_f32_x(pg, x, n, 0.693145751953125f);
    r = svmls_n_f32_x(pg, r, n, 1.428606765330187045e-06f);
    svfloat32_t p = svdup_n_f32(1.f / 5040.f);
    p = svmad_n_f32_x(pg, p, r, 1.f / 720.f);
    p = svmad_n_f32_x(pg, p, r, 1.f / 120.f);
    p = svmad_n_f32_x(pg, p, r, 1.f / 24.f);
    p = svmad_n_f32_x(pg, p, r, 1.f / 6.f);
    p = svmad_n_f32_x(pg, p, r, 0.5f);
    p = svmad_n_f32_x(pg, p, r, 1.f);
    p = svmad_n_f32_x(pg, p, r, 1.f);
    return svscale_f32_x(pg, p, svcvt_s32_f32_x(pg, n));
}
static inline svfloat32_t q38d_sigmoid_sve(svbool_t pg, svfloat32_t x) {
    svfloat32_t d = svadd_n_f32_x(pg, q38d_exp_sve(pg, svneg_f32_x(pg, x)), 1.f);
    svfloat32_t r = svrecpe_f32(d);
    r = svmul_f32_x(pg, r, svrecps_f32(d, r));
    r = svmul_f32_x(pg, r, svrecps_f32(d, r));
    return svmul_f32_x(pg, r, svrecps_f32(d, r));
}
static inline float q38d_expf(float x) {
    return svlasta_f32(svpfalse_b(), q38d_exp_sve(svptrue_b32(), svdup_n_f32(x)));
}

/* Byte placement of one 16-column unit inside its 32-byte pair block:
 * slot i = (2h+half)*8 + 4u + b takes source lane 8*half + 4h + b. */
static const uint8_t q38d_unit_idx[2][64] = {
    {0,1,2,3,255,255,255,255, 8,9,10,11,255,255,255,255, 4,5,6,7,255,255,255,255, 12,13,14,15,255,255,255,255,
     16,17,18,19,255,255,255,255, 24,25,26,27,255,255,255,255, 20,21,22,23,255,255,255,255, 28,29,30,31,255,255,255,255},
    {255,255,255,255,0,1,2,3, 255,255,255,255,8,9,10,11, 255,255,255,255,4,5,6,7, 255,255,255,255,12,13,14,15,
     255,255,255,255,16,17,18,19, 255,255,255,255,24,25,26,27, 255,255,255,255,20,21,22,23, 255,255,255,255,28,29,30,31}};

/* Quantize one 16-column unit u (0/1) of pair p from 16 floats v. */
static inline void q38d_prepare_unit(q38d_act *a, int p, int u, const float *v) {
    if (a->arith == Q38D_F32) return;
    const svbool_t pf = svptrue_b32(), pb = svptrue_b8();
    const int a16 = a->arith == Q38D_A16;
    const float limit = a16 ? 32639.f : 127.f;
    svfloat32_t x = svld1_f32(pf, v);
    float m = svmaxv_f32(pf, svabs_f32_x(pf, x));
    float inv = m > 0 ? limit / m : 0.f;
    svint32_t q = svcvt_s32_f32_x(pf, svrintn_f32_x(pf, svmul_n_f32_x(pf, x, inv)));
    q = svmax_n_s32_x(pf, svmin_n_s32_x(pf, q, (int)limit), -(int)limit);
    int sum = (int)svaddv_s32(pf, q);
    svuint8_t idx = svld1_u8(pb, q38d_unit_idx[u]);
    svbool_t keep = svcmplt_n_u8(pb, idx, 64); /* 255 marks slots of the other unit */
    int8_t *dst = a->q + (size_t)p * (a16 ? 64 : 32);
    svint32_t lo = q, hi = svdup_n_s32(0);
    if (a16) {
        hi = svasr_n_s32_x(pf, svadd_n_s32_x(pf, q, 128), 8);
        lo = svsub_s32_x(pf, q, svlsl_n_s32_x(pf, hi, 8));
    }
    /* bytes 0..15 = lo digits, 16..31 = hi digits (natural column order) */
    svint16_t h16 = svuzp1_s16(svreinterpret_s16_s32(lo), svreinterpret_s16_s32(hi));
    svint8_t b8 = svuzp1_s8(svreinterpret_s8_s16(h16), svreinterpret_s8_s16(h16));
    svint8_t placed = svtbl_s8(b8, idx);
    svbool_t lim = svwhilelt_b8((uint32_t)0, (uint32_t)(a16 ? 64 : 32));
    svst1_s8(svand_b_z(pb, keep, lim), dst, placed);
    float sc = m > 0 ? m / limit * 0x1p64f : 0.f;
    a->sc[2 * p + u] = sc;
    a->sum[2 * p + u] = sc * (float)sum;
}
/* Quantize columns [32*p0, 32*p1) from x (x indexed from column 0). */
static inline void q38d_prepare_range(q38d_act *a, const float *x, int p0, int p1) {
    for (int p = p0; p < p1; p++) {
        q38d_prepare_unit(a, p, 0, x + 32 * p);
        q38d_prepare_unit(a, p, 1, x + 32 * p + 16);
    }
}

/* Exact-weight FP32 path: decode each pair to int8, convert exactly and
 * scale, then FMA with the unquantized activation. */
#define Q38D_F32_ROWPAIR(ACC, W, SIDX) do {                                           \
    svfloat32_t wf_ = svmul_f32_x(pf, svcvt_f32_s32_x(pf, W), svtbl_f32(s16, SIDX));  \
    if (fmt == Q38D_Q4K) wf_ = svsub_f32_x(pf, wf_, svtbl_f32(o16, SIDX));            \
    ACC = svmla_f32_x(pf, ACC, wf_, xp);                                               \
} while (0)
#define Q38D_F32_VEC(V, H, HALF) do {                                                  \
    const float *xa_ = x + 32 * p + 8 * (HALF) + 4 * (H);                              \
    svfloat32_t xp = svsel_f32(qsel, svld1rq_f32(pf, xa_ + 16), svld1rq_f32(pf, xa_)); \
    svint16_t w0_ = svunpklo_s16(V), w1_ = svunpkhi_s16(V);                            \
    Q38D_F32_ROWPAIR(acc0, svunpklo_s32(w0_), sidx0);                                  \
    Q38D_F32_ROWPAIR(acc1, svunpkhi_s32(w0_), sidx1);                                  \
    Q38D_F32_ROWPAIR(acc2, svunpklo_s32(w1_), sidx2);                                  \
    Q38D_F32_ROWPAIR(acc3, svunpkhi_s32(w1_), sidx3);                                  \
} while (0)

static inline svuint32_t q38d_f32_sidx(int rp) {
    const svbool_t pf = svptrue_b32();
    svuint32_t i = svindex_u32(0, 1);
    svuint32_t r = svadd_n_u32_x(pf, svlsr_n_u32_x(pf, i, 3), 2 * rp);
    svuint32_t u = svand_n_u32_x(pf, svlsr_n_u32_x(pf, i, 2), 1);
    return svadd_u32_x(pf, svlsl_n_u32_x(pf, r, 1), u);
}

static inline void q38d_group_f32_sve(float *out, const uint8_t *g, int fmt, const float *x,
                                      int cols, int mode, int nrows) {
    const svbool_t pb = svptrue_b8(), pf = svptrue_b32();
    const svbool_t p32 = svwhilelt_b8((uint32_t)0, (uint32_t)32);
    const size_t np = (size_t)cols / 32;
    const int8_t *lutp = fmt == Q38D_F4 ? q38d_lut_f4 : fmt == Q38D_F6 ? q38d_lut_f6 :
                         fmt == Q38D_Q6K ? q38d_lut_q6 : q38d_lut_q4;
    const svint8_t lut = svld1_s8(pb, lutp);
    const uint8_t *high = g + np * 128;
    const uint8_t *sc = q38d_scale_stream((uint8_t *)g, fmt, cols);
    const float *dq = fmt == Q38D_Q6K || fmt == Q38D_Q8K ? (const float *)(sc + np * 16) : NULL;
    const float *off = fmt == Q38D_Q4K ? (const float *)sc + np * 8 : NULL;
    /* lanes 4..7 and 12..15 take the second activation quad */
    const svbool_t qsel = svcmpne_n_u32(pf, svand_n_u32_x(pf, svindex_u32(0, 1), 4), 0);
    const svuint32_t sidx0 = q38d_f32_sidx(0), sidx1 = q38d_f32_sidx(1);
    const svuint32_t sidx2 = q38d_f32_sidx(2), sidx3 = q38d_f32_sidx(3);
    svfloat32_t acc0 = svdup_n_f32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (size_t p = 0; p < np; p++) {
        svint8_t v0, v1, v2, v3;
        const size_t co = q38d_code_off(fmt, p);
        svuint8_t z0 = svld1_u8(pb, g + co), z1 = svld1_u8(pb, g + co + 64);
        const uint8_t *hp = g + co + 128;
        if (fmt == Q38D_Q8K) {
            const int8_t *cq = (const int8_t *)g + p * 256;
            v0 = svld1_s8(pb, cq); v1 = svld1_s8(pb, cq + 64); v2 = svld1_s8(pb, cq + 128); v3 = svld1_s8(pb, cq + 192);
        } else if (fmt == Q38D_F4 || fmt == Q38D_Q4K) {
            v0 = svtbl_s8(lut, svand_n_u8_x(pb, z0, 15)); v1 = svtbl_s8(lut, svlsr_n_u8_x(pb, z0, 4));
            v2 = svtbl_s8(lut, svand_n_u8_x(pb, z1, 15)); v3 = svtbl_s8(lut, svlsr_n_u8_x(pb, z1, 4));
        } else {
            Q38D_DECODE(fmt, lut, z0, hp, 0, v0, v1);
            Q38D_DECODE(fmt, lut, z1, hp, 1, v2, v3);
        }
        svfloat32_t s16, o16 = svdup_n_f32(0);
        if (fmt == Q38D_F4)
            s16 = svmul_n_f32_x(pf, svreinterpret_f32_u32(svlsl_n_u32_x(pf, svld1ub_u32(pf, sc + p * 16), 20)), 0x1p109f);
        else if (fmt == Q38D_F6) {
            svuint32_t u = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc + p * 8);
            s16 = svmul_n_f32_x(pf, svreinterpret_f32_u32(svlsl_n_u32_x(pf, svzip1_u32(u, u), 23)), 0.125f);
        } else if (fmt == Q38D_Q6K || fmt == Q38D_Q8K) {
            svfloat32_t d8 = svld1_f32(svptrue_pat_b32(SV_VL8), dq + (p / 8) * 8);
            s16 = svmul_f32_x(pf, svcvt_f32_s32_x(pf, svld1sb_s32(pf, (const int8_t *)sc + p * 16)), svzip1_f32(d8, d8));
        } else {
            svfloat32_t s8 = svld1_f32(svptrue_pat_b32(SV_VL8), (const float *)sc + p * 8);
            svfloat32_t f8 = svld1_f32(svptrue_pat_b32(SV_VL8), off + p * 8);
            s16 = svzip1_f32(s8, s8); o16 = svzip1_f32(f8, f8);
        }
        Q38D_F32_VEC(v0, 0, 0);
        Q38D_F32_VEC(v1, 0, 1);
        Q38D_F32_VEC(v2, 1, 0);
        Q38D_F32_VEC(v3, 1, 1);
    }
    const svbool_t lo8 = svptrue_pat_b32(SV_VL8), hi8 = svnot_b_z(pf, lo8);
    float r8[8] = {svaddv_f32(lo8, acc0), svaddv_f32(hi8, acc0), svaddv_f32(lo8, acc1), svaddv_f32(hi8, acc1),
                   svaddv_f32(lo8, acc2), svaddv_f32(hi8, acc2), svaddv_f32(lo8, acc3), svaddv_f32(hi8, acc3)};
    for (int r = 0; r < nrows; r++) out[r] = mode ? out[r] + r8[r] : r8[r];
}

void q38d_asm8_f4_a8(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                     const float *, long, float *, const int8_t *, const uint8_t *);
void q38d_asm8_f4_a16(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                      const float *, long, float *, const int8_t *, const uint8_t *);
void q38d_asm8_f6_a8(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                     const float *, long, float *, const int8_t *, const uint8_t *);
void q38d_asm8_f6_a16(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                      const float *, long, float *, const int8_t *, const uint8_t *);
/* Two F4 (or F6) matrices with identical shape and activation: groups
 * [g0, g1) of each (outA/outB start at the groups' first rows). */
static inline void q38d_gemv_dual_fmt(float *outA, float *outB, const uint8_t *baseA, const uint8_t *baseB,
                                      const q38d_act *a, int g0, int g1, int fmt) {
    const size_t np = (size_t)a->cols / 32, gb = q38d_group_bytes(fmt, a->cols);
    const size_t so = np * (fmt == Q38D_F6 ? 192 : 128);
    float acc[32] __attribute__((aligned(64)));
    const float e = q38d_out_scale(fmt);
    for (int g = g0; g < g1; g++) {
        const uint8_t *wa = baseA + (size_t)g * gb, *wb = baseB + (size_t)g * gb;
        if (fmt == Q38D_F6)
            (a->arith == Q38D_A16 ? q38d_asm8_f6_a16 : q38d_asm8_f6_a8)(
                wa, wb, wa + so, a->q, a->sc, (long)np, acc, q38d_lut_f6, wb + so);
        else
            (a->arith == Q38D_A16 ? q38d_asm8_f4_a16 : q38d_asm8_f4_a8)(
                wa, wb, wa + so, a->q, a->sc, (long)np, acc, q38d_lut_f4, wb + so);
        for (int r = 0; r < 8; r++) {
            outA[g * 8 + r] = (acc[2 * r] + acc[2 * r + 1]) * e;
            outB[g * 8 + r] = (acc[16 + 2 * r] + acc[16 + 2 * r + 1]) * e;
        }
    }
}

static inline void q38d_gemv_dual_f4(float *outA, float *outB, const uint8_t *baseA, const uint8_t *baseB,
                                     const q38d_act *a, int g0, int g1) {
    q38d_gemv_dual_fmt(outA, outB, baseA, baseB, a, g0, g1, Q38D_F4);
}

/* Dispatcher used by the engine. F4 groups are processed two at a time
 * through the dual kernel (consecutive groups share activation loads). */
static int q38d_pair_groups = 0;
static inline void q38d_gemv_any(float *out, const uint8_t *base, int fmt, const q38d_act *a,
                                 int g0, int g1, int rows, int mode) {
    size_t gb = q38d_group_bytes(fmt, a->cols);
    if (q38d_pair_groups && fmt == Q38D_F4 && a->arith != Q38D_F32 && a->cols % 64 == 0 &&
        a->cols >= 256 && g1 * 8 <= rows && g1 - g0 >= 2) {
        /* pair group g0+i with g0+half+i: two long sequential streams */
        int half = (g1 - g0) / 2;
        float tmp[16];
        for (int i = 0; i < half; i++) {
            int ga = g0 + i, gb2 = g0 + half + i;
            const uint8_t *wa = base + (size_t)ga * gb, *wb = base + (size_t)gb2 * gb;
            float *oa = mode ? tmp : out + (size_t)ga * 8, *ob = mode ? tmp + 8 : out + (size_t)gb2 * 8;
            q38d_gemv_dual_f4(oa, ob, wa, wb, a, 0, 1);
            if (mode) for (int r = 0; r < 8; r++) { out[(size_t)ga * 8 + r] += tmp[r]; out[(size_t)gb2 * 8 + r] += tmp[8 + r]; }
        }
        g0 += 2 * half;
    }
    for (int g = g0; g < g1; g++) {
        int nr = rows - g * 8 < 8 ? rows - g * 8 : 8;
        const uint8_t *w = base + (size_t)g * gb;
        float *o = out + (size_t)g * 8;
        if (a->arith == Q38D_F32) q38d_group_f32_sve(o, w, fmt, a->x, a->cols, mode, nr);
        else if (fmt != Q38D_Q4K || q38d_asm_variant >= 5) q38d_group_asm(o, w, fmt, a->arith, a, mode, nr);
        else q38d_gemv(o, w, fmt, a, 0, 1, nr, mode);
    }
}
#endif /* __ARM_FEATURE_SVE */

#endif
