#ifndef PCG32_H
#define PCG32_H

#include <stdint.h>

// PCG32 random number generator
// Minimal implementation based on https://www.pcg-random.org/

static uint64_t pcg32_state = 0x853c49e6748fea9bULL;
static uint64_t pcg32_inc   = 0xda3e39cb94b95bdbULL;

static inline uint32_t pcg32_random(void) {
    uint64_t oldstate = pcg32_state;
    pcg32_state = oldstate * 6364136223846793005ULL + pcg32_inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline void pcg32_seed(uint64_t seed, uint64_t seq) {
    pcg32_state = 0U;
    pcg32_inc = (seq << 1u) | 1u;
    pcg32_random();
    pcg32_state += seed;
    pcg32_random();
}

// Returns float in [0, 1)
static inline float pcg32_float(void) {
    return (float)(pcg32_random() >> 8) * 0x1.0p-24f;
}

// Returns float in [-1, 1)
static inline float pcg32_float_signed(void) {
    return pcg32_float() * 2.0f - 1.0f;
}

// Returns float in [min, max)
static inline float pcg32_float_range(float min, float max) {
    return min + pcg32_float() * (max - min);
}

// Returns double in [0, 1)
static inline double pcg32_double(void) {
    return (double)(pcg32_random() >> 11) * 0x1.0p-21;
}

// Returns double in [min, max)
static inline double pcg32_double_range(double min, double max) {
    return min + pcg32_double() * (max - min);
}

#endif // PCG32_H
