// philox.h
// SVE-optimized Philox Counter-Based RNG for A64FX
//
// Philox-4x32-10: 4 words, 10 rounds
// - High quality randomness (passes BigCrush)
// - Counter-based (stateless, reproducible, parallelizable)
// - Suitable for ML applications (dropout, initialization, augmentation)

#ifndef PHILOX_H
#define PHILOX_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Philox Constants
//=============================================================================
#define PHILOX_M0 0xD2511F53U
#define PHILOX_M1 0xCD9E8D57U
#define PHILOX_W0 0x9E3779B9U  // golden ratio
#define PHILOX_W1 0xBB67AE85U  // sqrt(3) - 1

//=============================================================================
// Assembly Implementations (philox_asm.S)
//=============================================================================

// Scalar Philox-4x32-10 (for testing)
void philox4x32_scalar(uint32_t ctr[4], uint32_t key[2], uint32_t out[4]);

// Generate N random uint32 values
void philox4x32_sve_u32(uint64_t counter_base, uint64_t key,
                        uint32_t* output, size_t count);

// Generate N random floats in [0, 1)
void philox4x32_sve_f32(uint64_t counter_base, uint64_t key,
                        float* output, size_t count);

// Generate N random floats for Box-Muller (uniform, caller applies transform)
void philox4x32_sve_normal_f32(uint64_t counter_base, uint64_t key,
                               float* output, size_t count);

// Generate N random bytes
void philox4x32_sve_bytes(uint64_t counter_base, uint64_t key,
                          uint8_t* output, size_t count);

// Generate dropout mask: mask[i] = 1 if rand < prob, else 0
void philox_dropout_mask_sve(uint64_t counter_base, uint64_t key,
                             uint8_t* mask, size_t count, float prob);

//=============================================================================
// Philox State Structure
//=============================================================================
typedef struct {
    uint64_t counter;       // 64-bit counter (split into ctr0, ctr1)
    uint64_t key;           // 64-bit key (split into key0, key1)
    uint32_t stream_id;     // Stream identifier (goes into ctr2)
} philox_state_t;

static inline void philox_init(philox_state_t* state, uint64_t seed, uint32_t stream)
{
    state->counter = 0;
    state->key = seed;
    state->stream_id = stream;
}

static inline void philox_skip(philox_state_t* state, uint64_t count)
{
    state->counter += count;
}

//=============================================================================
// C Reference Implementations
//=============================================================================

// Multiply high/low for 32-bit
static inline void mulhilo32(uint32_t a, uint32_t b, uint32_t* hi, uint32_t* lo)
{
    uint64_t product = (uint64_t)a * (uint64_t)b;
    *lo = (uint32_t)product;
    *hi = (uint32_t)(product >> 32);
}

// Single Philox-4x32-10 round
static inline void philox4x32_round(uint32_t* ctr, const uint32_t* key)
{
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(PHILOX_M0, ctr[0], &hi0, &lo0);
    mulhilo32(PHILOX_M1, ctr[2], &hi1, &lo1);

    uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t new1 = lo1;
    uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t new3 = lo0;

    ctr[0] = new0;
    ctr[1] = new1;
    ctr[2] = new2;
    ctr[3] = new3;
}

// Philox-4x32-10 reference implementation
static inline void philox4x32_10_ref(const uint32_t ctr_in[4],
                                     const uint32_t key_in[2],
                                     uint32_t out[4])
{
    uint32_t ctr[4] = {ctr_in[0], ctr_in[1], ctr_in[2], ctr_in[3]};
    uint32_t key[2] = {key_in[0], key_in[1]};

    for (int i = 0; i < 10; i++) {
        philox4x32_round(ctr, key);
        key[0] += PHILOX_W0;
        key[1] += PHILOX_W1;
    }

    out[0] = ctr[0];
    out[1] = ctr[1];
    out[2] = ctr[2];
    out[3] = ctr[3];
}

// Convert uint32 to float in [0, 1)
static inline float u32_to_f32_01(uint32_t u)
{
    // Use upper 24 bits for mantissa
    return (float)(u >> 8) * (1.0f / 16777216.0f);
}

// Generate N uniform floats [0, 1) using reference implementation
static inline void philox_f32_ref(uint64_t counter_base, uint64_t key,
                                  float* output, size_t count)
{
    uint32_t key32[2] = {(uint32_t)key, (uint32_t)(key >> 32)};
    uint32_t ctr[4], out[4];

    size_t i = 0;
    while (i < count) {
        ctr[0] = (uint32_t)(counter_base + i / 4);
        ctr[1] = (uint32_t)((counter_base + i / 4) >> 32);
        ctr[2] = 0;
        ctr[3] = 0;

        philox4x32_10_ref(ctr, key32, out);

        for (int j = 0; j < 4 && i < count; j++, i++) {
            output[i] = u32_to_f32_01(out[j]);
        }
    }
}

// Generate N uint32 values using reference implementation
static inline void philox_u32_ref(uint64_t counter_base, uint64_t key,
                                  uint32_t* output, size_t count)
{
    uint32_t key32[2] = {(uint32_t)key, (uint32_t)(key >> 32)};
    uint32_t ctr[4], out[4];

    size_t i = 0;
    while (i < count) {
        ctr[0] = (uint32_t)(counter_base + i / 4);
        ctr[1] = (uint32_t)((counter_base + i / 4) >> 32);
        ctr[2] = 0;
        ctr[3] = 0;

        philox4x32_10_ref(ctr, key32, out);

        for (int j = 0; j < 4 && i < count; j++, i++) {
            output[i] = out[j];
        }
    }
}

//=============================================================================
// Box-Muller Transform (for normal distribution)
//=============================================================================

// Transform pairs of uniform randoms to normal distribution
// Uses Box-Muller: z0 = sqrt(-2*ln(u1)) * cos(2*pi*u2)
//                  z1 = sqrt(-2*ln(u1)) * sin(2*pi*u2)
static inline void box_muller_transform(const float* uniform, float* normal, size_t count)
{
    const float two_pi = 6.283185307179586f;

    size_t pairs = count / 2;
    for (size_t i = 0; i < pairs; i++) {
        float u1 = uniform[2*i];
        float u2 = uniform[2*i + 1];

        // Avoid log(0)
        if (u1 < 1e-10f) u1 = 1e-10f;

        float r = sqrtf(-2.0f * logf(u1));
        float theta = two_pi * u2;

        normal[2*i] = r * cosf(theta);
        normal[2*i + 1] = r * sinf(theta);
    }

    // Handle odd count
    if (count % 2 == 1) {
        // Generate one more pair and discard second
        float u1 = uniform[count - 1];
        float u2 = 0.5f;  // arbitrary
        if (u1 < 1e-10f) u1 = 1e-10f;
        float r = sqrtf(-2.0f * logf(u1));
        float theta = two_pi * u2;
        normal[count - 1] = r * cosf(theta);
    }
}

// Generate N normal floats (mean=0, std=1)
static inline void philox_normal_f32_ref(uint64_t counter_base, uint64_t key,
                                         float* output, size_t count)
{
    // Generate uniforms
    float* uniform = (float*)output;  // Reuse buffer
    philox_f32_ref(counter_base, key, uniform, count);

    // Transform to normal
    box_muller_transform(uniform, output, count);
}

//=============================================================================
// Convenience Functions for ML
//=============================================================================

// Initialize tensor with uniform random values in [low, high)
static inline void philox_uniform_range_f32(philox_state_t* state,
                                            float* output, size_t count,
                                            float low, float high)
{
    philox4x32_sve_f32(state->counter, state->key, output, count);
    state->counter += (count + 3) / 4;

    float range = high - low;
    for (size_t i = 0; i < count; i++) {
        output[i] = output[i] * range + low;
    }
}

// Initialize tensor with normal random values (mean, std)
static inline void philox_normal_range_f32(philox_state_t* state,
                                           float* output, size_t count,
                                           float mean, float std)
{
    philox4x32_sve_normal_f32(state->counter, state->key, output, count);
    state->counter += (count + 3) / 4;

    box_muller_transform(output, output, count);

    for (size_t i = 0; i < count; i++) {
        output[i] = output[i] * std + mean;
    }
}

// Kaiming/He initialization (for ReLU networks)
static inline void philox_kaiming_f32(philox_state_t* state,
                                      float* output, size_t fan_in, size_t count)
{
    float std = sqrtf(2.0f / (float)fan_in);
    philox_normal_range_f32(state, output, count, 0.0f, std);
}

// Xavier/Glorot initialization
static inline void philox_xavier_f32(philox_state_t* state,
                                     float* output, size_t fan_in, size_t fan_out,
                                     size_t count)
{
    float std = sqrtf(2.0f / (float)(fan_in + fan_out));
    philox_normal_range_f32(state, output, count, 0.0f, std);
}

// Generate dropout mask with keep probability
static inline void philox_dropout_f32(philox_state_t* state,
                                      float* output, size_t count,
                                      float keep_prob, float scale)
{
    philox4x32_sve_f32(state->counter, state->key, output, count);
    state->counter += (count + 3) / 4;

    for (size_t i = 0; i < count; i++) {
        output[i] = (output[i] < keep_prob) ? scale : 0.0f;
    }
}

#ifdef __cplusplus
}
#endif

#endif // PHILOX_H
