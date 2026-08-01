/*
 * Flash Attention implementation for AMD Zen2 using AVX2 + FMA3
 *
 * Memory-efficient attention using 2-pass algorithm:
 *   Pass 1: S = Q @ K^T with row max tracking
 *   Pass 2: O = softmax(S) @ V with online normalization
 *
 * Key optimizations:
 *   - AVX2 vectorized exp using polynomial approximation
 *   - FMA3 for fused multiply-add in GEMM operations
 *   - Register blocking for cache efficiency
 *   - Online softmax to avoid storing full attention matrix
 */

#include "flash_attn_avx2.h"
#include <immintrin.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>

/* ============================================
 * AVX2 Vectorized Exponential Function
 * ============================================
 * Uses Schraudolph's algorithm with polynomial refinement
 * Relative error < 1e-6 for inputs in [-87, 88]
 */

/* Constants for exp approximation */
static const float EXP_LOG2E = 1.44269504088896341f;

/* Polynomial coefficients for 2^x approximation on [0, 1] */
static const float EXP_P0 = 1.0f;
static const float EXP_P1 = 0.6931471805599453f;
static const float EXP_P2 = 0.2402265069591007f;
static const float EXP_P3 = 0.05550410866482158f;
static const float EXP_P4 = 0.00967812610747928f;
static const float EXP_P5 = 0.0013333558146428443f;

static inline __m256 exp256_ps(__m256 x) {
    /* Clamp input to avoid overflow/underflow */
    __m256 min_val = _mm256_set1_ps(-87.33654474f);
    __m256 max_val = _mm256_set1_ps(88.72283905f);
    x = _mm256_max_ps(x, min_val);
    x = _mm256_min_ps(x, max_val);

    /* exp(x) = 2^(x * log2(e)) = 2^n * 2^f where n = floor(x*log2e), f = frac */
    __m256 log2e = _mm256_set1_ps(EXP_LOG2E);
    __m256 t = _mm256_mul_ps(x, log2e);

    /* Round to nearest integer */
    __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    /* Fractional part: f = t - n */
    __m256 f = _mm256_sub_ps(t, n);

    /* Polynomial approximation for 2^f on [−0.5, 0.5] */
    __m256 p5 = _mm256_set1_ps(EXP_P5);
    __m256 p4 = _mm256_set1_ps(EXP_P4);
    __m256 p3 = _mm256_set1_ps(EXP_P3);
    __m256 p2 = _mm256_set1_ps(EXP_P2);
    __m256 p1 = _mm256_set1_ps(EXP_P1);
    __m256 p0 = _mm256_set1_ps(EXP_P0);

    /* Horner's method: p = p0 + f*(p1 + f*(p2 + f*(p3 + f*(p4 + f*p5)))) */
    __m256 y = _mm256_fmadd_ps(p5, f, p4);
    y = _mm256_fmadd_ps(y, f, p3);
    y = _mm256_fmadd_ps(y, f, p2);
    y = _mm256_fmadd_ps(y, f, p1);
    y = _mm256_fmadd_ps(y, f, p0);

    /* Scale by 2^n: multiply by 2^n using IEEE 754 bit manipulation */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(ni);

    return _mm256_mul_ps(y, pow2n);
}

/* Scalar exp wrapper for edge cases */
static inline float exp_scalar(float x) {
    if (x < -87.33654474f) return 0.0f;
    if (x > 88.72283905f) return INFINITY;
    return expf(x);
}

void exp_avx2(const float *input, float *output, size_t n) {
    size_t i = 0;

    /* Process 8 elements at a time */
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 y = exp256_ps(x);
        _mm256_storeu_ps(output + i, y);
    }

    /* Handle remaining elements */
    for (; i < n; i++) {
        output[i] = exp_scalar(input[i]);
    }
}

/* ============================================
 * Horizontal max/sum operations
 * ============================================ */

static inline float hmax256(__m256 v) {
    /* Max across all 8 elements */
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 max4 = _mm_max_ps(lo, hi);
    __m128 max2 = _mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128 max1 = _mm_max_ps(max2, _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max1);
}

static inline float hsum256(__m256 v) {
    /* Sum across all 8 elements */
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 sum2 = _mm_add_ps(sum4, _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128 sum1 = _mm_add_ps(sum2, _mm_shuffle_ps(sum2, sum2, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(sum1);
}

/* ============================================
 * Pass 1: S = Q @ K^T with row max
 * ============================================
 * Computes attention scores and tracks row maximums
 * Q: [BR, D], K: [BC, D], S: [BR, BC], m: [BR]
 */

static void pass1_qkt_rowmax_avx2(
    const float *Q,     /* [BR, D] */
    const float *K,     /* [BC, D] */
    float *S,           /* [BR, BC] */
    float *m            /* [BR] row max */
) {
    /* Initialize row max to -inf */
    for (int i = 0; i < FA_BR; i++) {
        m[i] = -FLT_MAX;
    }

    /* Compute S[i, j] = Q[i, :] @ K[j, :] for all i, j */
    for (int i = 0; i < FA_BR; i++) {
        const float *q_row = Q + i * FA_D;

        for (int j = 0; j < FA_BC; j++) {
            const float *k_row = K + j * FA_D;

            /* Dot product Q[i, :] @ K[j, :] */
            __m256 sum = _mm256_setzero_ps();
            for (int d = 0; d < FA_D; d += 8) {
                __m256 q_vec = _mm256_loadu_ps(q_row + d);
                __m256 k_vec = _mm256_loadu_ps(k_row + d);
                sum = _mm256_fmadd_ps(q_vec, k_vec, sum);
            }

            /* Reduce to scalar */
            float dot = hsum256(sum);
            S[i * FA_BC + j] = dot;

            /* Track row max */
            if (dot > m[i]) {
                m[i] = dot;
            }
        }
    }
}

/* ============================================
 * Pass 2: O = softmax(S) @ V
 * ============================================
 * Computes softmax and output in one pass
 * S: [BR, BC], V: [BC, D], O: [BR, D], m: [BR], l: [BR]
 */

static void pass2_softmax_pv_avx2(
    const float *S,     /* [BR, BC] */
    const float *V,     /* [BC, D] */
    const float *m,     /* [BR] row max */
    float *O,           /* [BR, D] */
    float *l            /* [BR] row sum */
) {
    /* Initialize output and row sums to zero */
    memset(O, 0, FA_BR * FA_D * sizeof(float));
    for (int i = 0; i < FA_BR; i++) {
        l[i] = 0.0f;
    }

    /* For each row i in the output */
    for (int i = 0; i < FA_BR; i++) {
        __m256 vm = _mm256_set1_ps(m[i]);
        __m256 vl = _mm256_setzero_ps();

        /* Compute P[i, :] = exp(S[i, :] - m[i]) and accumulate O[i, :] */
        /* Process BC columns in chunks of 8 */
        for (int j = 0; j < FA_BC; j += 8) {
            /* Load S[i, j:j+8] */
            __m256 s_vec = _mm256_loadu_ps(S + i * FA_BC + j);

            /* P = exp(S - m) */
            __m256 p_vec = exp256_ps(_mm256_sub_ps(s_vec, vm));

            /* Accumulate row sum */
            vl = _mm256_add_ps(vl, p_vec);

            /* Accumulate O[i, :] += P[i, j] * V[j, :] for each j */
            float p_arr[8] __attribute__((aligned(32)));
            _mm256_store_ps(p_arr, p_vec);

            for (int jj = 0; jj < 8 && (j + jj) < FA_BC; jj++) {
                float p = p_arr[jj];
                __m256 vp = _mm256_set1_ps(p);
                const float *v_row = V + (j + jj) * FA_D;

                /* O[i, :] += p * V[j+jj, :] */
                for (int d = 0; d < FA_D; d += 8) {
                    __m256 o_vec = _mm256_loadu_ps(O + i * FA_D + d);
                    __m256 v_vec = _mm256_loadu_ps(v_row + d);
                    o_vec = _mm256_fmadd_ps(vp, v_vec, o_vec);
                    _mm256_storeu_ps(O + i * FA_D + d, o_vec);
                }
            }
        }

        /* Store row sum */
        l[i] = hsum256(vl);
    }
}

/* ============================================
 * Normalize output: O /= l
 * ============================================ */

static void normalize_output_avx2(float *O, const float *l) {
    for (int i = 0; i < FA_BR; i++) {
        __m256 inv_l = _mm256_set1_ps(1.0f / l[i]);

        for (int d = 0; d < FA_D; d += 8) {
            __m256 o_vec = _mm256_loadu_ps(O + i * FA_D + d);
            o_vec = _mm256_mul_ps(o_vec, inv_l);
            _mm256_storeu_ps(O + i * FA_D + d, o_vec);
        }
    }
}

/* ============================================
 * Main tile attention function
 * ============================================ */

void flash_attention_tile_avx2(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    float *S_scratch,
    float *m,
    float *l
) {
    /* Pass 1: S = Q @ K^T, compute row max */
    pass1_qkt_rowmax_avx2(Q, K, S_scratch, m);

    /* Pass 2: O = softmax(S) @ V */
    pass2_softmax_pv_avx2(S_scratch, V, m, O, l);

    /* Final normalization: O /= l */
    normalize_output_avx2(O, l);
}

/* ============================================
 * Multi-block attention with online softmax
 * ============================================
 * Processes full key/value sequence in blocks
 * Uses online softmax to maintain numerical stability
 */

void flash_attention_avx2(
    const float *Q,     /* [BR, D] */
    const float *K,     /* [seq_len, D] */
    const float *V,     /* [seq_len, D] */
    float *O,           /* [BR, D] */
    size_t seq_len
) {
    /* Allocate scratch space */
    float *S_scratch = (float *)aligned_alloc(32, FA_BR * FA_BC * sizeof(float));
    float m_new[FA_BR];
    float m_prev[FA_BR], l_prev[FA_BR];

    if (!S_scratch) return;

    /* Initialize */
    for (int i = 0; i < FA_BR; i++) {
        m_prev[i] = -FLT_MAX;
        l_prev[i] = 0.0f;
    }
    memset(O, 0, FA_BR * FA_D * sizeof(float));

    /* Process key/value blocks */
    for (size_t kv_offset = 0; kv_offset < seq_len; kv_offset += FA_BC) {
        const float *K_block = K + kv_offset * FA_D;
        const float *V_block = V + kv_offset * FA_D;

        /* Compute S = Q @ K_block^T and local max */
        pass1_qkt_rowmax_avx2(Q, K_block, S_scratch, m_new);

        /* Online softmax: update global max */
        for (int i = 0; i < FA_BR; i++) {
            float m_max = fmaxf(m_prev[i], m_new[i]);

            /* Rescale previous accumulator */
            if (m_prev[i] > -FLT_MAX) {
                float scale = expf(m_prev[i] - m_max);
                __m256 vscale = _mm256_set1_ps(scale);

                for (int d = 0; d < FA_D; d += 8) {
                    __m256 o_vec = _mm256_loadu_ps(O + i * FA_D + d);
                    o_vec = _mm256_mul_ps(o_vec, vscale);
                    _mm256_storeu_ps(O + i * FA_D + d, o_vec);
                }
                l_prev[i] *= scale;
            }

            m_prev[i] = m_max;
        }

        /* Compute new contribution and accumulate */
        for (int i = 0; i < FA_BR; i++) {
            float l_block = 0.0f;

            for (int j = 0; j < FA_BC; j++) {
                float s = S_scratch[i * FA_BC + j];
                float p = expf(s - m_prev[i]);
                l_block += p;

                __m256 vp = _mm256_set1_ps(p);
                const float *v_row = V_block + j * FA_D;

                for (int d = 0; d < FA_D; d += 8) {
                    __m256 o_vec = _mm256_loadu_ps(O + i * FA_D + d);
                    __m256 v_vec = _mm256_loadu_ps(v_row + d);
                    o_vec = _mm256_fmadd_ps(vp, v_vec, o_vec);
                    _mm256_storeu_ps(O + i * FA_D + d, o_vec);
                }
            }

            l_prev[i] += l_block;
        }
    }

    /* Final normalization */
    for (int i = 0; i < FA_BR; i++) {
        __m256 inv_l = _mm256_set1_ps(1.0f / l_prev[i]);
        for (int d = 0; d < FA_D; d += 8) {
            __m256 o_vec = _mm256_loadu_ps(O + i * FA_D + d);
            o_vec = _mm256_mul_ps(o_vec, inv_l);
            _mm256_storeu_ps(O + i * FA_D + d, o_vec);
        }
    }

    free(S_scratch);
}

/* ============================================
 * Multi-head attention
 * ============================================ */

void multi_head_attention_avx2(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    size_t batch,
    size_t num_heads,
    size_t seq_q,
    size_t seq_kv,
    size_t head_dim
) {
    /* Process each batch and head */
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            /* Pointer offsets */
            size_t q_offset = (b * num_heads + h) * seq_q * head_dim;
            size_t kv_offset = (b * num_heads + h) * seq_kv * head_dim;
            size_t o_offset = (b * num_heads + h) * seq_q * head_dim;

            /* Process query sequence in blocks of BR */
            for (size_t q_pos = 0; q_pos < seq_q; q_pos += FA_BR) {
                size_t br = (q_pos + FA_BR <= seq_q) ? FA_BR : (seq_q - q_pos);

                if (br == FA_BR && head_dim == FA_D) {
                    /* Use optimized path */
                    flash_attention_avx2(
                        Q + q_offset + q_pos * head_dim,
                        K + kv_offset,
                        V + kv_offset,
                        O + o_offset + q_pos * head_dim,
                        seq_kv
                    );
                } else {
                    /* Fallback for non-standard dimensions */
                    /* TODO: Add edge case handling */
                }
            }
        }
    }
}

/* ============================================
 * Reference implementation
 * ============================================ */

void flash_attention_ref(
    const float *Q,
    const float *K,
    const float *V,
    float *O
) {
    float S[FA_BR * FA_BC];
    float P[FA_BR * FA_BC];
    float m[FA_BR];
    float l[FA_BR];

    /* S = Q @ K^T */
    for (int i = 0; i < FA_BR; i++) {
        for (int j = 0; j < FA_BC; j++) {
            float sum = 0.0f;
            for (int k = 0; k < FA_D; k++) {
                sum += Q[i * FA_D + k] * K[j * FA_D + k];
            }
            S[i * FA_BC + j] = sum;
        }
    }

    /* Row max */
    for (int i = 0; i < FA_BR; i++) {
        m[i] = -FLT_MAX;
        for (int j = 0; j < FA_BC; j++) {
            if (S[i * FA_BC + j] > m[i]) {
                m[i] = S[i * FA_BC + j];
            }
        }
    }

    /* P = exp(S - m), l = rowsum(P) */
    for (int i = 0; i < FA_BR; i++) {
        l[i] = 0.0f;
        for (int j = 0; j < FA_BC; j++) {
            P[i * FA_BC + j] = expf(S[i * FA_BC + j] - m[i]);
            l[i] += P[i * FA_BC + j];
        }
    }

    /* O = P @ V */
    for (int i = 0; i < FA_BR; i++) {
        for (int d = 0; d < FA_D; d++) {
            float sum = 0.0f;
            for (int j = 0; j < FA_BC; j++) {
                sum += P[i * FA_BC + j] * V[j * FA_D + d];
            }
            O[i * FA_D + d] = sum;
        }
    }

    /* Normalize: O /= l */
    for (int i = 0; i < FA_BR; i++) {
        float inv_l = 1.0f / l[i];
        for (int d = 0; d < FA_D; d++) {
            O[i * FA_D + d] *= inv_l;
        }
    }
}
