#include "flash_attn_2pass.h"
#include <math.h>
#include <float.h>

// ============================================
// C Wrappers
// ============================================

// C implementation of pass2 for debugging
static void pass2_softmax_pv_c(
    const float* S,     // [4, 64]
    const float* V,     // [64, 64]
    const float* m,     // [4] row max
    float* O,           // [4, 64] output
    float* l            // [4] row sum output
) {
    // Initialize O = 0 and l = 0
    for (int i = 0; i < BR * D; i++) O[i] = 0.0f;
    for (int i = 0; i < BR; i++) l[i] = 0.0f;

    // For each column j in S (BC columns)
    for (int j = 0; j < BC; j++) {
        // Compute P[i, j] = exp(S[i, j] - m[i]) for all rows
        float P[BR];
        for (int i = 0; i < BR; i++) {
            P[i] = expf(S[i * BC + j] - m[i]);
            l[i] += P[i];
        }

        // O[i, :] += P[i, j] * V[j, :]
        for (int i = 0; i < BR; i++) {
            for (int d = 0; d < D; d++) {
                O[i * D + d] += P[i] * V[j * D + d];
            }
        }
    }
}

void flash_attention_tile(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* S_scratch,
    float* m,
    float* l
) {
    // Pass 1: S = Q @ K^T, compute row max
    pass1_qkt_rowmax(Q, K, S_scratch, m);

    // Pass 2: O = softmax(S) @ V
    // Use C implementation for correctness
    pass2_softmax_pv_c(S_scratch, V, m, O, l);

    // Final normalization: O /= l
    normalize_output(O, l);
}

void normalize_output(float* O, const float* l) {
    for (int i = 0; i < BR; i++) {
        float inv_l = 1.0f / l[i];
        for (int d = 0; d < D; d++) {
            O[i * D + d] *= inv_l;
        }
    }
}

// ============================================
// Reference Implementation
// ============================================

void flash_attention_ref(
    const float* Q,
    const float* K,
    const float* V,
    float* O
) {
    float S[BR * BC];
    float P[BR * BC];
    float m[BR];
    float l[BR];

    // S = Q @ K^T
    for (int i = 0; i < BR; i++) {
        for (int j = 0; j < BC; j++) {
            float sum = 0.0f;
            for (int k = 0; k < D; k++) {
                sum += Q[i * D + k] * K[j * D + k];
            }
            S[i * BC + j] = sum;
        }
    }

    // Row max
    for (int i = 0; i < BR; i++) {
        m[i] = -FLT_MAX;
        for (int j = 0; j < BC; j++) {
            if (S[i * BC + j] > m[i]) {
                m[i] = S[i * BC + j];
            }
        }
    }

    // P = exp(S - m), l = rowsum(P)
    for (int i = 0; i < BR; i++) {
        l[i] = 0.0f;
        for (int j = 0; j < BC; j++) {
            P[i * BC + j] = expf(S[i * BC + j] - m[i]);
            l[i] += P[i * BC + j];
        }
    }

    // O = P @ V
    for (int i = 0; i < BR; i++) {
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int j = 0; j < BC; j++) {
                sum += P[i * BC + j] * V[j * D + d];
            }
            O[i * D + d] = sum;
        }
    }

    // Normalize: O /= l
    for (int i = 0; i < BR; i++) {
        float inv_l = 1.0f / l[i];
        for (int d = 0; d < D; d++) {
            O[i * D + d] *= inv_l;
        }
    }
}
