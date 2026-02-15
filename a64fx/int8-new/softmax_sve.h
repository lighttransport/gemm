// softmax_sve.h - SVE Optimized Softmax
// Fast vectorized softmax for INT32 scores -> INT8 probabilities

#ifndef SOFTMAX_SVE_H
#define SOFTMAX_SVE_H

#include <stdint.h>

// Online softmax state per row
typedef struct {
    float max;      // Running max
    float sum;      // Running sum of exp values
} softmax_state_t;

// Initialize softmax state
static inline void softmax_state_init(softmax_state_t* state, int num_rows) {
    for (int r = 0; r < num_rows; r++) {
        state[r].max = -1e30f;
        state[r].sum = 0.0f;
    }
}

// SVE-optimized online softmax for one row chunk
// Computes exp(score - max), updates running state, outputs INT8
// Returns max_exp for dequantization
float softmax_chunk_sve(
    const int32_t* scores,  // Input scores [64]
    float scale,            // Score scale (1/sqrt(D))
    softmax_state_t* state, // Running state (updated)
    float* O_accum,         // FP32 O accumulator [D] (rescaled)
    int D,                  // Head dimension
    int8_t* P_out           // Output INT8 probabilities [64]
);

// Process all 6 rows of a tile
void softmax_tile_sve(
    const int32_t* scores,  // Input scores [6][64]
    float scale,
    softmax_state_t* state, // [6]
    float* O_accum,         // [6][D]
    int D,
    int8_t* P_out,          // [6][64]
    float* max_exp_out      // [6] for dequantization
);

// Final normalization after all chunks
void softmax_finalize_sve(
    float* O,               // Output [6][D]
    const float* O_accum,   // Accumulated O [6][D]
    const softmax_state_t* state, // Final state [6]
    int D
);

#endif // SOFTMAX_SVE_H
