// online_softmax.h
// Online softmax normalization for FlashAttention-style tiled computation
// Maintains running max and sum to correctly compute global softmax

#ifndef ONLINE_SOFTMAX_H
#define ONLINE_SOFTMAX_H

#include <stdint.h>
#include <float.h>

// Per-row state for online softmax
// Tracks running statistics across tiles
typedef struct {
    float* row_max;     // [MR] Running maximum per row
    float* row_sum;     // [MR] Running sum of exp(x - max) per row
    float* O_acc;       // [MR, d] Running output accumulator (float for precision)
    int mr;             // Number of active rows
    int d;              // Output dimension
} online_softmax_state_t;

// Initialize online softmax state
static inline void online_softmax_init(online_softmax_state_t* state,
                                        float* row_max, float* row_sum,
                                        float* O_acc, int mr, int d) {
    state->row_max = row_max;
    state->row_sum = row_sum;
    state->O_acc = O_acc;
    state->mr = mr;
    state->d = d;

    // Initialize to identity values
    for (int m = 0; m < mr; m++) {
        row_max[m] = -FLT_MAX;  // Will be updated on first tile
        row_sum[m] = 0.0f;
    }

    // Zero output accumulator
    for (int i = 0; i < mr * d; i++) {
        O_acc[i] = 0.0f;
    }
}

// Update online softmax with new tile of scores
// S_tile: [mr, lb] int32 scores from Q @ K^T
// V_tile: [lb, d] int8 value matrix
// scale: softmax temperature scale (1/sqrt(d))
//
// Algorithm:
// 1. Compute local max per row: m_local = max(S_tile[row, :])
// 2. Update global max: m_new = max(m_old, m_local)
// 3. Compute correction factor: alpha = exp2((m_old - m_new) * log2(e))
// 4. Scale existing sums and outputs: l *= alpha, O *= alpha
// 5. Compute exp2((S - m_new) * scale * log2(e)) for new tile
// 6. Accumulate: l += sum(exp_tile), O += exp_tile @ V
void online_softmax_update(online_softmax_state_t* state,
                            const int32_t* S_tile,
                            const int8_t* V_tile,
                            int lb, float scale);

// SVE-optimized version
void online_softmax_update_sve(online_softmax_state_t* state,
                                const int32_t* S_tile,
                                const int8_t* V_tile,
                                int lb, float scale);

// Finalize: divide output by row sums
// O_final[m, :] = O_acc[m, :] / row_sum[m]
void online_softmax_finalize(const online_softmax_state_t* state,
                              int32_t* O_out, int ldo);

// Finalize with int8 quantization for further SDOT
void online_softmax_finalize_int8(const online_softmax_state_t* state,
                                   int8_t* O_out, int ldo, float out_scale);

#endif // ONLINE_SOFTMAX_H
