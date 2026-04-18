/*
 * philox_rng.h - Shared Philox RNG utilities (CPU-side)
 *
 * Header-only implementation of Philox4x32-10 with deterministic
 * float32 uniform/normal generators for cross-backend reproducibility.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef PHILOX_RNG_H
#define PHILOX_RNG_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t seed;
    uint64_t offset_u32; /* offset in 32-bit words */
} philox_rng_state;

static inline uint32_t philox_mulhi_u32(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}

static inline void philox4x32_10(uint32_t c[4], uint32_t k0, uint32_t k1) {
    const uint32_t M0 = 0xD2511F53u;
    const uint32_t M1 = 0xCD9E8D57u;
    const uint32_t W0 = 0x9E3779B9u;
    const uint32_t W1 = 0xBB67AE85u;

    for (int r = 0; r < 10; r++) {
        const uint32_t c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
        const uint32_t hi0 = philox_mulhi_u32(M0, c0);
        const uint32_t hi1 = philox_mulhi_u32(M1, c2);
        const uint32_t lo0 = M0 * c0;
        const uint32_t lo1 = M1 * c2;

        c[0] = hi1 ^ c1 ^ k0;
        c[1] = lo1;
        c[2] = hi0 ^ c3 ^ k1;
        c[3] = lo0;

        k0 += W0;
        k1 += W1;
    }
}

static inline void philox_rng_init(philox_rng_state *st, uint64_t seed, uint64_t offset_u32) {
    if (!st) return;
    st->seed = seed;
    st->offset_u32 = offset_u32;
}

static inline void philox_next4_u32(philox_rng_state *st, uint32_t out4[4]) {
    uint32_t c[4] = {0, 0, 0, 0};
    uint32_t k0 = (uint32_t)(st->seed & 0xffffffffu);
    uint32_t k1 = (uint32_t)((st->seed >> 32) & 0xffffffffu);

    const uint64_t block = st->offset_u32 / 4u;
    c[0] = (uint32_t)(block & 0xffffffffu);
    c[1] = (uint32_t)((block >> 32) & 0xffffffffu);
    philox4x32_10(c, k0, k1);

    out4[0] = c[0];
    out4[1] = c[1];
    out4[2] = c[2];
    out4[3] = c[3];
    st->offset_u32 += 4u;
}

/* Uniform in (0, 1], using 32-bit precision. */
static inline float philox_u32_to_uniform_f32(uint32_t x) {
    return (float)((x + 1.0) * (1.0 / 4294967296.0));
}

static inline void philox_fill_uniform_f32(float *out, size_t n, uint64_t seed, uint64_t offset_u32) {
    philox_rng_state st;
    philox_rng_init(&st, seed, offset_u32);
    size_t i = 0;
    while (i < n) {
        uint32_t r[4];
        philox_next4_u32(&st, r);
        for (int j = 0; j < 4 && i < n; j++, i++) {
            out[i] = philox_u32_to_uniform_f32(r[j]);
        }
    }
}

static inline void philox_fill_normal_f32(float *out, size_t n, uint64_t seed, uint64_t offset_u32) {
    philox_rng_state st;
    philox_rng_init(&st, seed, offset_u32);

    size_t i = 0;
    while (i < n) {
        uint32_t r[4];
        philox_next4_u32(&st, r);
        for (int j = 0; j < 4 && i < n; j += 2) {
            const float u1 = philox_u32_to_uniform_f32(r[j]);
            const float u2 = philox_u32_to_uniform_f32(r[j + 1]);
            const float rr = sqrtf(-2.0f * logf(u1));
            const float th = 6.2831853071795864769f * u2; /* 2*pi */
            out[i++] = rr * cosf(th);
            if (i < n) out[i++] = rr * sinf(th);
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* PHILOX_RNG_H */
