/* E0M3 is sign-magnitude -7..7, not two's-complement s4. */
#ifndef INT4_NUMERIC_H
#define INT4_NUMERIC_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define INT4_HD __host__ __device__ __forceinline__
#else
#define INT4_HD static inline
#endif

INT4_HD float int4_scale_decode(unsigned code) {
    unsigned e = (code >> 3) & 15, m = code & 7;
    return e ? ldexpf(1.0f + m * 0.125f, (int)e - 7) : m * (1.0f / 512.0f);
}

INT4_HD int int4_round_even(float x) {
    float f = floorf(x), d = x - f;
    return (int)f + (d > 0.5f || (d == 0.5f && ((int)f & 1)));
}

/* Positive finite E4M3, clamped to [2^-9, 448]. Zero blocks use scale 1. */
INT4_HD unsigned int4_scale_encode(float x) {
    if (x <= 1.0f / 512.0f) return 1;
    if (x >= 448.0f) return 126;
    if (x < 1.0f / 64.0f) return (unsigned)int4_round_even(x * 512.0f);
    uint32_t bits;
#ifdef __CUDA_ARCH__
    bits = __float_as_uint(x);
#else
    memcpy(&bits, &x, sizeof(bits));
#endif
    bits += 0x7ffffu + ((bits >> 20) & 1u);
    return (bits >> 20) - 960u;
}

INT4_HD float int4_div(float x, float y) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(x, y);
#else
    return x / y;
#endif
}

INT4_HD float int4_code_decode(unsigned code, int e0m3) {
    unsigned m = code & 7;
    float v;
    if (e0m3) v = (float)m;
    else {
        /* 0, .5, 1, 1.5, 2, 3, 4, 6. */
        v = m < 4 ? m * 0.5f : ldexpf(1.0f + (m & 1) * 0.5f, (int)(m >> 1) - 1);
    }
    return code & 8 ? -v : v;
}

INT4_HD unsigned int4_code_encode(float x, float scale, int e0m3) {
    float v = int4_div(fabsf(x), scale);
    unsigned m = 0;
    if (e0m3) {
        m = v >= 7.0f ? 7u : (unsigned)int4_round_even(v);
    } else if (v >= 6.0f) {
        m = 7;
    } else {
        float best = v;
        for (unsigned i = 1; i < 8; i++) {
            float d = fabsf(v - int4_code_decode(i, 0));
            if (d < best || (d == best && !(i & 1))) { best = d; m = i; }
        }
    }
    return m && x < 0 ? m | 8u : m;
}

#undef INT4_HD
#endif
