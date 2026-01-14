// softmax_exp2.h
// Fast softmax using exp2 approximation for fused attention
// Two variants: int8 and uint8 quantization

#ifndef SOFTMAX_EXP2_H
#define SOFTMAX_EXP2_H

#include <stdint.h>
#include <stddef.h>

// Softmax configuration
typedef struct {
    float scale;           // 1/sqrt(d) scaling factor
    float temperature;     // Temperature for softmax (usually 1.0)
    int use_uint8;         // 0 = int8, 1 = uint8 variant
} softmax_config_t;

// ============================================================================
// INT8 Softmax Variant
// ============================================================================
// 1. Convert int32 S to float32
// 2. Apply scaling: S_scaled = S * scale
// 3. Row-wise max for stability: S_stable = S_scaled - row_max
// 4. exp2(S_stable * log2(e)): fast approximation
// 5. Normalize: P = exp_vals / row_sum
// 6. Quantize to int8: P_int8 = round(P * 127)
// 7. SDOT with V (int8)

void softmax_int8(const int32_t* S_tile,    // [rows, cols] input scores
                  int8_t* P_tile,            // [rows, cols] output probabilities
                  int rows, int cols,
                  float scale,
                  float* row_sums);          // [rows] for online normalization

// ============================================================================
// UINT8 Softmax Variant
// ============================================================================
// Uses uint8 quantization with bias correction:
// 1. Same exp2 computation as int8
// 2. Quantize to uint8: P_uint8 = round(P * 255)
// 3. V matrix has +128 offset: V_biased = V_orig + 128
// 4. UDOT: result = sum(P_uint8 * V_biased)
// 5. Correct bias: result -= sum(P_uint8) * 128
//
// Advantage: Full [0, 255] range for probabilities (vs [-127, 127])

void softmax_uint8(const int32_t* S_tile,   // [rows, cols] input scores
                   uint8_t* P_tile,          // [rows, cols] output probabilities
                   int rows, int cols,
                   float scale,
                   uint32_t* row_sums);      // [rows] sum of P for bias correction

// ============================================================================
// Fast exp2 approximation using integer bit manipulation
// ============================================================================
// exp2(x) = 2^x
// For x in [-126, 127]: reinterpret((int)(x + 127) << 23) as float
// More accurate: polynomial refinement

// Scalar version
static inline float fast_exp2f(float x) {
    // Clamp to valid range
    if (x < -126.0f) x = -126.0f;
    if (x > 127.0f) x = 127.0f;

    // Integer approximation: 2^x = 2^floor(x) * 2^frac(x)
    // 2^floor(x) via exponent manipulation
    // 2^frac(x) via polynomial

    int xi = (int)(x + 126.0f);  // Shift to positive
    float frac = x - (float)(xi - 126);  // Fractional part

    // 2^frac polynomial (for frac in [0, 1])
    // p(f) = 1 + f*(0.6931472 + f*(0.2402265 + f*0.0554913))
    float p = 1.0f + frac * (0.6931472f + frac * (0.2402265f + frac * 0.0554913f));

    // Combine with integer exponent
    union { float f; uint32_t u; } val;
    val.u = (uint32_t)(xi) << 23;

    return val.f * p;
}

// log2(e) constant for converting natural log to log base 2
#define LOG2_E 1.4426950408889634f

// ============================================================================
// SVE-optimized versions (declared, implemented in .c)
// ============================================================================

// Process a tile with SVE vectorization
void softmax_int8_sve(const int32_t* S_tile, int8_t* P_tile,
                       int rows, int cols, float scale);

void softmax_uint8_sve(const int32_t* S_tile, uint8_t* P_tile,
                        int rows, int cols, float scale,
                        uint32_t* row_sums);

// ============================================================================
// Pack V matrix for UDOT (add +128 offset)
// ============================================================================
void pack_V_uint8_biased(const int8_t* V, uint8_t* V_biased,
                          int L, int d);

#endif // SOFTMAX_EXP2_H
