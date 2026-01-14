// fused_attention.c
// Fused Attention: O = softmax(Q @ K^T / sqrt(d)) @ V

#include <stdio.h>
#include "fused_attention.h"
#include "online_softmax.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_sve.h>

// Declare assembly kernels
extern void kernel_6x4_unroll(const int8_t* Apack, const int8_t* Bpack,
                               int32_t* C, int ldc);
extern void kernel_stage2_k64(const int8_t* S_tile, const int8_t* Cpack,
                               int32_t* O_acc, int ldo);

// ============================================================================
// Pack P_tile (softmax output) for stage 2 SDOT
// ============================================================================
// Input: P_tile [MR, LB] row-major int8
// Output: P_pack [K/4][MR][4] where K=LB=64
static inline void pack_P_tile_int8(const int8_t* P_tile, int8_t* P_pack,
                                     int mr, int lb) {
    for (int k = 0; k < lb; k += 4) {
        for (int m = 0; m < FUSED_MR; m++) {
            if (m < mr && (k + 3) < lb) {
                P_pack[0] = P_tile[m * lb + k + 0];
                P_pack[1] = P_tile[m * lb + k + 1];
                P_pack[2] = P_tile[m * lb + k + 2];
                P_pack[3] = P_tile[m * lb + k + 3];
            } else if (m < mr) {
                // Handle edge case
                for (int i = 0; i < 4; i++) {
                    P_pack[i] = (k + i < lb) ? P_tile[m * lb + k + i] : 0;
                }
            } else {
                P_pack[0] = P_pack[1] = P_pack[2] = P_pack[3] = 0;
            }
            P_pack += 4;
        }
    }
}

// ============================================================================
// INT8 Fused Attention
// ============================================================================
void fused_attention_int8(const fused_matrix_t* Qpack,
                           const fused_matrix_t* Kpack,
                           const fused_matrix_t* Vpack,
                           int32_t* O, int ldo,
                           float scale) {
    int L = Qpack->L;
    int d = Qpack->d;

    // Temporary buffers
    int32_t S_tile_i32[FUSED_MR * FUSED_LB] __attribute__((aligned(256)));
    int8_t P_tile_i8[FUSED_MR * FUSED_LB] __attribute__((aligned(256)));
    int8_t P_pack[FUSED_MR * FUSED_LB] __attribute__((aligned(256)));

    // Per-output-tile accumulator - use compile-time constant
    int32_t O_tile[FUSED_MR * FUSED_D] __attribute__((aligned(256)));

    int Q_tile_stride = FUSED_MR * d;  // Bytes per Q tile
    int K_tile_stride = FUSED_LB * d;  // Bytes per K tile
    int V_tile_stride = FUSED_LB * d;  // Bytes per V L-tile
    int V_d_stride = FUSED_LB * 64;    // Bytes per V d-tile within L-tile

    // Outer loop over M dimension (output rows)
    for (int m0 = 0; m0 < L; m0 += FUSED_MR) {
        int mr = (m0 + FUSED_MR <= L) ? FUSED_MR : (L - m0);
        const int8_t* Qptr = Qpack->data + (m0 / FUSED_MR) * Q_tile_stride;

        // Zero output accumulator for this M-tile
        memset(O_tile, 0, FUSED_MR * d * sizeof(int32_t));

        // Loop over L dimension (attention span)
        for (int l0 = 0; l0 < L; l0 += FUSED_LB) {
            int lb = (l0 + FUSED_LB <= L) ? FUSED_LB : (L - l0);

            const int8_t* Kptr = Kpack->data + (l0 / FUSED_LB) * K_tile_stride;
            const int8_t* Vptr = Vpack->data + (l0 / FUSED_LB) * V_tile_stride;

            // Stage 1: S_tile = Q[m0:m0+MR, :] @ K[l0:l0+LB, :]^T
            // Result: [MR, LB] from [MR, d] @ [LB, d]^T, K=d=256
            memset(S_tile_i32, 0, sizeof(S_tile_i32));
            kernel_6x4_unroll(Qptr, Kptr, S_tile_i32, FUSED_LB * sizeof(int32_t));

            // Apply softmax with exp2: P = softmax(S * scale)
            // Quantize to int8
            softmax_int8_sve(S_tile_i32, P_tile_i8, mr, lb, scale);

            // Pack P_tile for stage 2
            pack_P_tile_int8(P_tile_i8, P_pack, mr, lb);

            // Stage 2: O_tile += P_tile @ V[l0:l0+LB, :]
            // For each d-tile of 64 columns
            for (int n0 = 0; n0 < d; n0 += 64) {
                const int8_t* V_block = Vptr + (n0 / 64) * V_d_stride;
                int32_t* O_block = O_tile + n0;

                // [MR, 64] += [MR, LB] @ [LB, 64], K=LB=64
                kernel_stage2_k64(P_pack, V_block, O_block, d * sizeof(int32_t));
            }
        }

        // Store O_tile to output
        for (int m = 0; m < mr; m++) {
            for (int n = 0; n < d; n++) {
                O[(m0 + m) * ldo + n] = O_tile[m * d + n];
            }
        }
    }
}

// ============================================================================
// UINT8 Fused Attention with bias correction
// ============================================================================
// Uses UDOT with P in [0, 255] and V_biased = V + 128
// Result needs correction: O -= sum(P) * 128

// UDOT kernel for uint8 (same structure as SDOT but different instruction)
// For now, we'll use a scalar implementation since A64FX UDOT is similar
static void udot_kernel_scalar(const uint8_t* P_pack, const uint8_t* V_pack,
                                int32_t* O_acc, int mr, int nr, int k) {
    // Simple scalar UDOT emulation
    for (int m = 0; m < mr; m++) {
        for (int n = 0; n < nr; n++) {
            int32_t acc = O_acc[m * nr + n];
            for (int kk = 0; kk < k; kk++) {
                // P_pack layout: [K/4][MR][4]
                // V_pack layout: [K/4][4][16][4]
                int k_group = kk / 4;
                int k_off = kk % 4;
                int v_vec = n / 16;
                int v_lane = n % 16;

                uint8_t p_val = P_pack[k_group * FUSED_MR * 4 + m * 4 + k_off];
                uint8_t v_val = V_pack[k_group * 4 * 64 + v_vec * 64 + v_lane * 4 + k_off];

                acc += (int32_t)p_val * (int32_t)v_val;
            }
            O_acc[m * nr + n] = acc;
        }
    }
}

void fused_attention_uint8(const fused_matrix_t* Qpack,
                            const fused_matrix_t* Kpack,
                            const fused_matrix_t* Vpack,
                            int32_t* O, int ldo,
                            float scale) {
    int L = Qpack->L;
    int d = Qpack->d;

    // Prepare V with +128 bias
    // V_biased needs to be pre-packed with bias added
    // For now, we'll do this inline (could be precomputed)

    // Temporary buffers
    int32_t S_tile_i32[FUSED_MR * FUSED_LB] __attribute__((aligned(256)));
    uint8_t P_tile_u8[FUSED_MR * FUSED_LB] __attribute__((aligned(256)));
    uint8_t P_pack[FUSED_MR * FUSED_LB] __attribute__((aligned(256)));
    uint32_t row_sums[FUSED_MR];

    // Per-output-tile accumulator - use compile-time constant
    int32_t O_tile[FUSED_MR * FUSED_D] __attribute__((aligned(256)));
    int32_t bias_acc[FUSED_MR];  // Accumulated bias correction per row

    int Q_tile_stride = FUSED_MR * d;
    int K_tile_stride = FUSED_LB * d;
    int V_tile_stride = FUSED_LB * d;
    int V_d_stride = FUSED_LB * 64;

    // Create biased V (add 128 to convert int8 to uint8 range)
    uint8_t* V_biased = malloc(Vpack->size);
    if (!V_biased) return;

    pack_V_uint8_biased((const int8_t*)Vpack->data, V_biased, L, d);

    for (int m0 = 0; m0 < L; m0 += FUSED_MR) {
        int mr = (m0 + FUSED_MR <= L) ? FUSED_MR : (L - m0);
        const int8_t* Qptr = Qpack->data + (m0 / FUSED_MR) * Q_tile_stride;

        memset(O_tile, 0, FUSED_MR * d * sizeof(int32_t));
        memset(bias_acc, 0, sizeof(bias_acc));

        for (int l0 = 0; l0 < L; l0 += FUSED_LB) {
            int lb = (l0 + FUSED_LB <= L) ? FUSED_LB : (L - l0);

            const int8_t* Kptr = Kpack->data + (l0 / FUSED_LB) * K_tile_stride;
            const uint8_t* Vptr = V_biased + (l0 / FUSED_LB) * V_tile_stride;

            // Stage 1: S_tile = Q @ K^T
            memset(S_tile_i32, 0, sizeof(S_tile_i32));
            kernel_6x4_unroll(Qptr, Kptr, S_tile_i32, FUSED_LB * sizeof(int32_t));

            // Softmax with uint8 output
            softmax_uint8_sve(S_tile_i32, P_tile_u8, mr, lb, scale, row_sums);

            // Accumulate row sums for bias correction
            for (int m = 0; m < mr; m++) {
                bias_acc[m] += row_sums[m];
            }

            // Pack P_tile for stage 2
            // Same layout as int8, just with uint8 values
            for (int k = 0; k < lb; k += 4) {
                for (int m = 0; m < FUSED_MR; m++) {
                    if (m < mr && (k + 3) < lb) {
                        P_pack[k / 4 * FUSED_MR * 4 + m * 4 + 0] = P_tile_u8[m * lb + k + 0];
                        P_pack[k / 4 * FUSED_MR * 4 + m * 4 + 1] = P_tile_u8[m * lb + k + 1];
                        P_pack[k / 4 * FUSED_MR * 4 + m * 4 + 2] = P_tile_u8[m * lb + k + 2];
                        P_pack[k / 4 * FUSED_MR * 4 + m * 4 + 3] = P_tile_u8[m * lb + k + 3];
                    }
                }
            }

            // Stage 2: O_tile += P @ V_biased (using UDOT / scalar)
            for (int n0 = 0; n0 < d; n0 += 64) {
                const uint8_t* V_block = Vptr + (n0 / 64) * V_d_stride;
                int32_t* O_block = O_tile + n0;

                // Scalar UDOT for now
                udot_kernel_scalar(P_pack, V_block, O_block, mr, 64, lb);
            }
        }

        // Apply bias correction: O -= sum(P) * 128
        // sum(P) is accumulated in bias_acc
        for (int m = 0; m < mr; m++) {
            int32_t correction = (int32_t)bias_acc[m] * 128;
            for (int n = 0; n < d; n++) {
                O[(m0 + m) * ldo + n] = O_tile[m * d + n] - correction;
            }
        }
    }

    free(V_biased);
}

// ============================================================================
// Reference implementation
// ============================================================================
void ref_attention(const int8_t* Q, const int8_t* K, const int8_t* V,
                    int32_t* O, int L, int d, float scale) {
    // Allocate intermediate buffers
    float* S = malloc((size_t)L * L * sizeof(float));
    float* P = malloc((size_t)L * L * sizeof(float));
    if (!S || !P) {
        free(S);
        free(P);
        return;
    }

    float scale_log2e = scale * LOG2_E;

    // Stage 1: S = Q @ K^T (scaled)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int32_t sum = 0;
            for (int k = 0; k < d; k++) {
                sum += (int32_t)Q[i * d + k] * (int32_t)K[j * d + k];
            }
            S[i * L + j] = (float)sum * scale_log2e;
        }
    }

    // Stage 2: P = softmax(S) row-wise
    for (int i = 0; i < L; i++) {
        // Find max for stability
        float row_max = S[i * L];
        for (int j = 1; j < L; j++) {
            if (S[i * L + j] > row_max) row_max = S[i * L + j];
        }

        // Compute exp2 and sum
        float row_sum = 0.0f;
        for (int j = 0; j < L; j++) {
            P[i * L + j] = fast_exp2f(S[i * L + j] - row_max);
            row_sum += P[i * L + j];
        }

        // Normalize
        float inv_sum = 1.0f / row_sum;
        for (int j = 0; j < L; j++) {
            P[i * L + j] *= inv_sum;
        }
    }

    // Stage 3: O = P @ V
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < L; k++) {
                sum += P[i * L + k] * (float)V[k * d + j];
            }
            O[i * d + j] = (int32_t)sum;
        }
    }

    free(S);
    free(P);
}

// ============================================================================
// Online Softmax Attention (Correct FlashAttention-style)
// ============================================================================
// Uses online softmax normalization to correctly compute global softmax
// while processing in tiles.
//
// Algorithm:
// For each Q tile (m0 to m0+MR rows):
//   Initialize: row_max = -inf, row_sum = 0, O_acc = 0
//   For each K,V tile (l0 to l0+LB cols):
//     1. Compute S_tile = Q_tile @ K_tile^T
//     2. Update running max: m_new = max(m_old, max(S_tile))
//     3. Rescale old values: alpha = exp2(m_old - m_new)
//     4. O_acc *= alpha, row_sum *= alpha
//     5. Compute P_tile = exp2(S_tile - m_new)
//     6. row_sum += sum(P_tile)
//     7. O_acc += P_tile @ V_tile
//   Finalize: O = O_acc / row_sum

void fused_attention_online(const fused_matrix_t* Qpack,
                             const fused_matrix_t* Kpack,
                             const int8_t* V,           // Unpacked V [L, d]
                             int32_t* O,
                             int ldo,
                             float scale) {
    // CRITICAL: Copy scale to local variable immediately to prevent stack corruption
    // from aligned stack arrays overwriting the parameter register/slot
    volatile float local_scale = scale;

    int L = Qpack->L;
    int d = Qpack->d;

    // Temporary buffers - use compile-time constants to avoid VLA issues
    int32_t S_tile_i32[FUSED_MR * FUSED_LB] __attribute__((aligned(256)));

    // Online softmax state per M-tile
    float row_max[FUSED_MR];
    float row_sum[FUSED_MR];
    float O_acc[FUSED_MR * FUSED_D] __attribute__((aligned(256)));

    int Q_tile_stride = FUSED_MR * d;
    int K_tile_stride = FUSED_LB * d;

    // Outer loop over M dimension (output rows)
    for (int m0 = 0; m0 < L; m0 += FUSED_MR) {
        int mr = (m0 + FUSED_MR <= L) ? FUSED_MR : (L - m0);
        const int8_t* Qptr = Qpack->data + (m0 / FUSED_MR) * Q_tile_stride;

        // Initialize online softmax state
        online_softmax_state_t state;
        online_softmax_init(&state, row_max, row_sum, O_acc, mr, d);

        // Loop over L dimension (K tiles)
        for (int l0 = 0; l0 < L; l0 += FUSED_LB) {
            int lb = (l0 + FUSED_LB <= L) ? FUSED_LB : (L - l0);

            const int8_t* Kptr = Kpack->data + (l0 / FUSED_LB) * K_tile_stride;
            const int8_t* Vptr = V + l0 * d;  // V[l0:l0+lb, :]

            // Stage 1: S_tile = Q_tile @ K_tile^T
            memset(S_tile_i32, 0, sizeof(S_tile_i32));
            kernel_6x4_unroll(Qptr, Kptr, S_tile_i32, FUSED_LB * sizeof(int32_t));

            // Update online softmax with this tile
            // This handles: max update, rescaling, exp2, accumulation
            online_softmax_update_sve(&state, S_tile_i32, Vptr, lb, local_scale);
        }

        // Finalize: O = O_acc / row_sum
        // Store to output
        for (int m = 0; m < mr; m++) {
            // Use double precision division directly to avoid compiler optimization issues
            double row_sum_val = (double)row_sum[m];
            for (int n = 0; n < d; n++) {
                double val = (double)O_acc[m * d + n] / row_sum_val;
                O[(m0 + m) * ldo + n] = (int32_t)(val + (val >= 0 ? 0.5 : -0.5));
            }
        }
    }
}
