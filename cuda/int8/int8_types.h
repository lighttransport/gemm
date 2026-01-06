/*
 * INT8 Type Definitions and Utilities
 * Supports both signed (s8) and unsigned (u8) INT8 formats
 *
 * s8: signed 8-bit integer, range [-128, 127]
 * u8: unsigned 8-bit integer, range [0, 255]
 *
 * Includes:
 * - Xoroshiro128+ PRNG for fast random number generation
 * - Stochastic rounding for int32 -> fp32 -> int8 conversion
 */

#ifndef INT8_TYPES_H
#define INT8_TYPES_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

typedef int8_t s8_t;
typedef uint8_t u8_t;

/* INT8 Constants */
#define S8_MIN  (-128)
#define S8_MAX  (127)
#define U8_MIN  (0)
#define U8_MAX  (255)

/*
 * Xoroshiro128+ PRNG
 * Fast, high-quality random number generator
 * Period: 2^128 - 1
 * Reference: https://prng.di.unimi.it/
 */
typedef struct {
    uint64_t s[2];
} xoroshiro128plus_t;

static inline uint64_t xoro_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/* Initialize xoroshiro128+ with a seed */
static inline void xoro_seed(xoroshiro128plus_t* rng, uint64_t seed) {
    /* Use splitmix64 to generate initial state from seed */
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    rng->s[0] = z ^ (z >> 31);
    z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    rng->s[1] = z ^ (z >> 31);
    /* Ensure state is not all zeros */
    if (rng->s[0] == 0 && rng->s[1] == 0) {
        rng->s[0] = 1;
    }
}

/* Generate next random uint64 */
static inline uint64_t xoro_next(xoroshiro128plus_t* rng) {
    uint64_t s0 = rng->s[0];
    uint64_t s1 = rng->s[1];
    uint64_t result = s0 + s1;

    s1 ^= s0;
    rng->s[0] = xoro_rotl(s0, 24) ^ s1 ^ (s1 << 16);
    rng->s[1] = xoro_rotl(s1, 37);

    return result;
}

/* Generate uniform float in [0, 1) */
static inline float xoro_uniform(xoroshiro128plus_t* rng) {
    uint64_t x = xoro_next(rng);
    /* Use upper 24 bits for float mantissa */
    return (x >> 40) * (1.0f / 16777216.0f);
}

/* Generate uniform double in [0, 1) */
static inline double xoro_uniform_double(xoroshiro128plus_t* rng) {
    uint64_t x = xoro_next(rng);
    /* Use upper 53 bits for double mantissa */
    return (x >> 11) * (1.0 / 9007199254740992.0);
}

/* Jump function: advance state by 2^64 calls */
static inline void xoro_jump(xoroshiro128plus_t* rng) {
    static const uint64_t JUMP[] = { 0xDF900294D8F554A5ULL, 0x170865DF4B3201FCULL };
    uint64_t s0 = 0, s1 = 0;
    for (int i = 0; i < 2; i++) {
        for (int b = 0; b < 64; b++) {
            if (JUMP[i] & (1ULL << b)) {
                s0 ^= rng->s[0];
                s1 ^= rng->s[1];
            }
            xoro_next(rng);
        }
    }
    rng->s[0] = s0;
    rng->s[1] = s1;
}

/*
 * Stochastic Rounding
 *
 * For a value x, stochastic rounding rounds to floor(x) with probability
 * ceil(x) - x, and to ceil(x) with probability x - floor(x).
 *
 * This provides an unbiased estimate: E[SR(x)] = x
 */

/* Stochastic round float to nearest integer */
static inline int32_t stochastic_round_f32(float x, xoroshiro128plus_t* rng) {
    float floor_x = floorf(x);
    float frac = x - floor_x;
    float r = xoro_uniform(rng);
    return (int32_t)floor_x + (r < frac ? 1 : 0);
}

/*
 * Convert int32 to int8 with stochastic rounding
 * Pipeline: int32 -> fp32 (with scale) -> stochastic round -> clamp -> int8
 *
 * scale: quantization scale factor (output = input * scale)
 */
static inline int8_t int32_to_s8_sr(int32_t val, float scale, xoroshiro128plus_t* rng) {
    float scaled = (float)val * scale;
    int32_t rounded = stochastic_round_f32(scaled, rng);
    /* Clamp to int8 range */
    if (rounded < S8_MIN) return S8_MIN;
    if (rounded > S8_MAX) return S8_MAX;
    return (int8_t)rounded;
}

static inline uint8_t int32_to_u8_sr(int32_t val, float scale, xoroshiro128plus_t* rng) {
    float scaled = (float)val * scale;
    int32_t rounded = stochastic_round_f32(scaled, rng);
    /* Clamp to uint8 range */
    if (rounded < U8_MIN) return U8_MIN;
    if (rounded > U8_MAX) return U8_MAX;
    return (uint8_t)rounded;
}

/*
 * Batch stochastic rounding for output arrays
 * Converts int32 array to int8 array with stochastic rounding
 */
static inline void batch_sr_s8(const int32_t* src, int8_t* dst, size_t n,
                                float scale, xoroshiro128plus_t* rng) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = int32_to_s8_sr(src[i], scale, rng);
    }
}

static inline void batch_sr_u8(const int32_t* src, uint8_t* dst, size_t n,
                                float scale, xoroshiro128plus_t* rng) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = int32_to_u8_sr(src[i], scale, rng);
    }
}

/*
 * Compute quantization scale from int32 output range to fit in int8
 * Returns scale factor such that max(abs(val)) * scale fits in [-127, 127]
 */
static inline float compute_scale_s8(const int32_t* data, size_t n) {
    int32_t max_abs = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t abs_val = data[i] < 0 ? -data[i] : data[i];
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs == 0) return 1.0f;
    return 127.0f / (float)max_abs;
}

static inline float compute_scale_u8(const int32_t* data, size_t n) {
    int32_t max_val = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    if (max_val == 0) return 1.0f;
    return 255.0f / (float)max_val;
}

/* Clamp value to signed INT8 range */
static inline int8_t clamp_s8(int32_t val) {
    if (val < S8_MIN) return S8_MIN;
    if (val > S8_MAX) return S8_MAX;
    return (int8_t)val;
}

/* Clamp value to unsigned INT8 range */
static inline uint8_t clamp_u8(int32_t val) {
    if (val < U8_MIN) return U8_MIN;
    if (val > U8_MAX) return U8_MAX;
    return (uint8_t)val;
}

/* Generate random signed INT8 value */
static inline int8_t random_s8(void) {
    return (int8_t)(rand() % 256 - 128);
}

/* Generate random unsigned INT8 value */
static inline uint8_t random_u8(void) {
    return (uint8_t)(rand() % 256);
}

/* Generate random signed INT8 in range [-range, range] */
static inline int8_t random_s8_range(int range) {
    if (range <= 0) return 0;
    if (range > 127) range = 127;
    return (int8_t)(rand() % (2 * range + 1) - range);
}

/* Generate random unsigned INT8 in range [0, range] */
static inline uint8_t random_u8_range(int range) {
    if (range <= 0) return 0;
    if (range > 255) range = 255;
    return (uint8_t)(rand() % (range + 1));
}

/* Generate random signed INT8 using xoroshiro */
static inline int8_t xoro_random_s8(xoroshiro128plus_t* rng) {
    return (int8_t)(xoro_next(rng) & 0xFF) - 128;
}

/* Generate random unsigned INT8 using xoroshiro */
static inline uint8_t xoro_random_u8(xoroshiro128plus_t* rng) {
    return (uint8_t)(xoro_next(rng) & 0xFF);
}

/* Generate random signed INT8 in range using xoroshiro */
static inline int8_t xoro_random_s8_range(xoroshiro128plus_t* rng, int range) {
    if (range <= 0) return 0;
    if (range > 127) range = 127;
    uint64_t r = xoro_next(rng);
    return (int8_t)((r % (2 * range + 1)) - range);
}

/* Generate random unsigned INT8 in range using xoroshiro */
static inline uint8_t xoro_random_u8_range(xoroshiro128plus_t* rng, int range) {
    if (range <= 0) return 0;
    if (range > 255) range = 255;
    uint64_t r = xoro_next(rng);
    return (uint8_t)(r % (range + 1));
}

#endif /* INT8_TYPES_H */
