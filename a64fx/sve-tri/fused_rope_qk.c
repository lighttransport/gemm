#include "fused_rope_qk.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Pack weight matrix from row-major [d_model × d_head]
 * to blocked [d_head/NR × d_model × NR] layout.
 */
/*
 * Packed size includes FUSED_NR padding for SW-pipelined kernel's
 * speculative loads past the last K step.
 */
size_t packed_weight_size(int d_model, int d_head)
{
    int nb = d_head / FUSED_NR;
    return (size_t)nb * (d_model * FUSED_NR + FUSED_NR) * sizeof(float);
}

void pack_weight_f32(const float *W_rm, float *W_packed,
                     int d_model, int d_head)
{
    int nb = d_head / FUSED_NR;
    for (int jb = 0; jb < nb; jb++) {
        float *dst = W_packed + (size_t)jb * (d_model * FUSED_NR + FUSED_NR);
        for (int k = 0; k < d_model; k++) {
            for (int jr = 0; jr < FUSED_NR; jr++) {
                int j = jb * FUSED_NR + jr;
                dst[k * FUSED_NR + jr] = W_rm[k * d_head + j];
            }
        }
        /* Zero padding for speculative loads */
        memset(dst + d_model * FUSED_NR, 0, FUSED_NR * sizeof(float));
    }
}

/*
 * Pack X from row-major [n_tokens × d_model]
 * to [d_model × MR] layout for a block of MR tokens starting at m_start.
 */
/*
 * Pack X with FUSED_MR padding at end for speculative loads.
 * Allocate at least (d_model + 1) * FUSED_MR floats.
 */
static void pack_x_f32(const float *X, float *X_packed,
                        int d_model, int n_tokens, int m_start)
{
    for (int k = 0; k < d_model; k++) {
        for (int m = 0; m < FUSED_MR; m++) {
            int idx = m_start + m;
            if (idx < n_tokens)
                X_packed[k * FUSED_MR + m] = X[idx * d_model + k];
            else
                X_packed[k * FUSED_MR + m] = 0.0f;
        }
    }
    /* Zero padding for speculative loads */
    memset(X_packed + d_model * FUSED_MR, 0, FUSED_MR * sizeof(float));
}

/*
 * Optimized path using the assembly micro-kernel.
 *
 * Constraint: the micro-kernel's RoPE epilogue applies the SAME angles
 * to all MR=4 token rows. This is correct when all tokens in a block
 * share the same RoPE position (e.g., decode with batch of tokens at
 * the same position, or when pos=0 for all).
 *
 * For the general case (prefill with different positions per token),
 * use fused_proj_rope_f32 which handles per-token positions.
 */
void fused_proj_rope_asm_f32(
    const float *X,
    const float *Wq_packed,
    const float *Wk_packed,
    float *Q, float *K,
    const float *theta_scaled,  /* [d_head/2] pre-scaled angles */
    int n_tokens, int d_model, int d_head)
{
    int nb = d_head / FUSED_NR;
    int64_t ldc_bytes = (int64_t)d_head * sizeof(float);
    size_t w_block_stride = (size_t)(d_model * FUSED_NR + FUSED_NR);  /* with padding */

    /* Extra FUSED_MR floats for SW-pipeline speculative loads */
    float *X_packed = (float *)aligned_alloc(64,
        ((size_t)d_model + 1) * FUSED_MR * sizeof(float));

    /* Temp buffers for tail block where n_tokens % MR != 0.
     * The kernel always writes MR=4 rows, so we need padding. */
    float *Q_tmp = NULL, *K_tmp = NULL;
    if (n_tokens % FUSED_MR != 0) {
        Q_tmp = (float *)aligned_alloc(64, (size_t)FUSED_MR * d_head * sizeof(float));
        K_tmp = (float *)aligned_alloc(64, (size_t)FUSED_MR * d_head * sizeof(float));
    }

    for (int mb = 0; mb < n_tokens; mb += FUSED_MR) {
        int m_count = n_tokens - mb;
        if (m_count > FUSED_MR) m_count = FUSED_MR;

        pack_x_f32(X, X_packed, d_model, n_tokens, mb);

        /* Use temp buffer for partial tail block */
        int use_tmp = (m_count < FUSED_MR);
        float *Q_dst = use_tmp ? Q_tmp : Q + (size_t)mb * d_head;
        float *K_dst = use_tmp ? K_tmp : K + (size_t)mb * d_head;

        for (int jb = 0; jb < nb; jb++) {
            int dim_start = jb * FUSED_NR;
            const float *Wq_blk = Wq_packed + (size_t)jb * w_block_stride;
            const float *Wk_blk = Wk_packed + (size_t)jb * w_block_stride;
            const float *th_blk = theta_scaled + dim_start / 2;

            float *Q_ptr = Q_dst + dim_start;
            float *K_ptr = K_dst + dim_start;
            int64_t stride = use_tmp ? (int64_t)d_head * (int64_t)sizeof(float) : ldc_bytes;

            micro_fused_proj_rope_4x2(
                X_packed, Wq_blk, Wk_blk,
                Q_ptr, K_ptr,
                (int64_t)d_model,
                th_blk,
                stride
            );
        }

        /* Copy valid rows from temp buffer */
        if (use_tmp) {
            for (int m = 0; m < m_count; m++) {
                memcpy(Q + ((size_t)mb + m) * d_head,
                       Q_tmp + (size_t)m * d_head,
                       d_head * sizeof(float));
                memcpy(K + ((size_t)mb + m) * d_head,
                       K_tmp + (size_t)m * d_head,
                       d_head * sizeof(float));
            }
        }
    }

    free(X_packed);
    free(Q_tmp);
    free(K_tmp);
}

/* ================================================================
 * FP16 path
 * ================================================================ */

size_t packed_weight_size_f16(int d_model, int d_head)
{
    int nb = d_head / FUSED_NR_F16;
    return (size_t)nb * (d_model * FUSED_NR_F16 + FUSED_NR_F16) * sizeof(_Float16);
}

void pack_weight_f16(const _Float16 *W_rm, _Float16 *W_packed,
                     int d_model, int d_head)
{
    int nb = d_head / FUSED_NR_F16;
    for (int jb = 0; jb < nb; jb++) {
        _Float16 *dst = W_packed + (size_t)jb * (d_model * FUSED_NR_F16 + FUSED_NR_F16);
        for (int k = 0; k < d_model; k++) {
            for (int jr = 0; jr < FUSED_NR_F16; jr++) {
                int j = jb * FUSED_NR_F16 + jr;
                dst[k * FUSED_NR_F16 + jr] = W_rm[k * d_head + j];
            }
        }
        /* Zero padding for speculative loads */
        memset(dst + d_model * FUSED_NR_F16, 0, FUSED_NR_F16 * sizeof(_Float16));
    }
}

static void pack_x_f16(const _Float16 *X, _Float16 *X_packed,
                        int d_model, int n_tokens, int m_start)
{
    for (int k = 0; k < d_model; k++) {
        for (int m = 0; m < FUSED_MR_F16; m++) {
            int idx = m_start + m;
            if (idx < n_tokens)
                X_packed[k * FUSED_MR_F16 + m] = X[idx * d_model + k];
            else
                X_packed[k * FUSED_MR_F16 + m] = (_Float16)0;
        }
    }
    /* Zero padding for speculative loads */
    memset(X_packed + d_model * FUSED_MR_F16, 0, FUSED_MR_F16 * sizeof(_Float16));
}

/*
 * Compute sin/cos lookup table for fp16 RoPE.
 * Output: sin_cos_all[jb * FUSED_NR_F16 + 0..31] = sin,
 *         sin_cos_all[jb * FUSED_NR_F16 + 32..63] = cos
 * Caller allocates: nb * FUSED_NR_F16 * sizeof(_Float16), 64-byte aligned.
 */
void compute_sin_cos_f16(const float *theta_scaled, _Float16 *sin_cos_all,
                         int d_head)
{
    int nb = d_head / FUSED_NR_F16;
    for (int jb = 0; jb < nb; jb++) {
        int dim_start = jb * FUSED_NR_F16;
        const float *th_blk = theta_scaled + dim_start / 2;
        _Float16 *sc = sin_cos_all + (size_t)jb * FUSED_NR_F16;
        for (int p = 0; p < FUSED_NR_F16 / 2; p++) {
            sc[p]                    = (_Float16)sinf(th_blk[p]);
            sc[p + FUSED_NR_F16 / 2] = (_Float16)cosf(th_blk[p]);
        }
    }
}

/*
 * Set FPCR.FZ16 (bit 19) to flush fp16 denormals to zero.
 * Without this, fp16 subnormals trap to microcode (~100+ cy/op).
 */
static inline void set_fpcr_fz16(void)
{
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1UL << 19);
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
}

/*
 * Core tiling loop: assumes all buffers pre-allocated, sin/cos pre-computed.
 * This is the hot path suitable for benchmarking.
 */
void fused_proj_rope_core_f16(
    const _Float16 *X,
    const _Float16 *Wq_packed,
    const _Float16 *Wk_packed,
    _Float16 *Q, _Float16 *K,
    const _Float16 *sin_cos_all,
    _Float16 *X_packed,
    _Float16 *Q_tmp, _Float16 *K_tmp,
    int n_tokens, int d_model, int d_head)
{
    set_fpcr_fz16();
    int nb = d_head / FUSED_NR_F16;
    int64_t ldc_bytes = (int64_t)d_head * sizeof(_Float16);
    size_t w_block_stride = (size_t)(d_model * FUSED_NR_F16 + FUSED_NR_F16);

    for (int mb = 0; mb < n_tokens; mb += FUSED_MR_F16) {
        int m_count = n_tokens - mb;
        if (m_count > FUSED_MR_F16) m_count = FUSED_MR_F16;

        pack_x_f16(X, X_packed, d_model, n_tokens, mb);

        int use_tmp = (m_count < FUSED_MR_F16);
        _Float16 *Q_dst = use_tmp ? Q_tmp : Q + (size_t)mb * d_head;
        _Float16 *K_dst = use_tmp ? K_tmp : K + (size_t)mb * d_head;

        for (int jb = 0; jb < nb; jb++) {
            int dim_start = jb * FUSED_NR_F16;
            const _Float16 *Wq_blk = Wq_packed + (size_t)jb * w_block_stride;
            const _Float16 *Wk_blk = Wk_packed + (size_t)jb * w_block_stride;
            const _Float16 *sc = sin_cos_all + (size_t)jb * FUSED_NR_F16;

            _Float16 *Q_ptr = Q_dst + dim_start;
            _Float16 *K_ptr = K_dst + dim_start;
            int64_t stride = use_tmp ? (int64_t)d_head * (int64_t)sizeof(_Float16) : ldc_bytes;

            micro_fused_proj_rope_f16_4x2(
                X_packed, Wq_blk, Wk_blk,
                Q_ptr, K_ptr,
                (int64_t)d_model,
                sc,
                stride
            );
        }

        if (use_tmp) {
            for (int m = 0; m < m_count; m++) {
                memcpy(Q + ((size_t)mb + m) * d_head,
                       Q_tmp + (size_t)m * d_head,
                       d_head * sizeof(_Float16));
                memcpy(K + ((size_t)mb + m) * d_head,
                       K_tmp + (size_t)m * d_head,
                       d_head * sizeof(_Float16));
            }
        }
    }
}

/*
 * Convenience wrapper: allocates workspace, computes sin/cos, and calls core.
 */
void fused_proj_rope_asm_f16(
    const _Float16 *X,
    const _Float16 *Wq_packed,
    const _Float16 *Wk_packed,
    _Float16 *Q, _Float16 *K,
    const float *theta_scaled,
    int n_tokens, int d_model, int d_head)
{
    int nb = d_head / FUSED_NR_F16;

    _Float16 *X_packed = (_Float16 *)aligned_alloc(64,
        ((size_t)d_model + 1) * FUSED_MR_F16 * sizeof(_Float16));
    _Float16 *sin_cos_all = (_Float16 *)aligned_alloc(64,
        (size_t)nb * FUSED_NR_F16 * sizeof(_Float16));

    compute_sin_cos_f16(theta_scaled, sin_cos_all, d_head);

    _Float16 *Q_tmp = NULL, *K_tmp = NULL;
    if (n_tokens % FUSED_MR_F16 != 0) {
        Q_tmp = (_Float16 *)aligned_alloc(64, (size_t)FUSED_MR_F16 * d_head * sizeof(_Float16));
        K_tmp = (_Float16 *)aligned_alloc(64, (size_t)FUSED_MR_F16 * d_head * sizeof(_Float16));
    }

    fused_proj_rope_core_f16(X, Wq_packed, Wk_packed, Q, K,
                             sin_cos_all, X_packed, Q_tmp, K_tmp,
                             n_tokens, d_model, d_head);

    free(X_packed);
    free(sin_cos_all);
    free(Q_tmp);
    free(K_tmp);
}

/*
 * FP16 reference: fp32 compute with fp16-compatible results.
 * Same interface as fused_proj_rope_f32 but computes what the fp16
 * kernel should produce (for correctness testing).
 */
void fused_proj_rope_f16(
    const float *X,
    const float *Wq,
    const float *Wk,
    float *Q, float *K,
    const float *theta,
    const int *pos,
    int n_tokens, int d_model, int d_head)
{
    for (int t = 0; t < n_tokens; t++) {
        const float *x_row = X + t * d_model;
        float *q_row = Q + t * d_head;
        float *k_row = K + t * d_head;

        /* GEMM: Q[t,:] = X[t,:] * Wq, K[t,:] = X[t,:] * Wk */
        for (int j = 0; j < d_head; j++) {
            float sq = 0.0f, sk = 0.0f;
            for (int kk = 0; kk < d_model; kk++) {
                sq += x_row[kk] * Wq[kk * d_head + j];
                sk += x_row[kk] * Wk[kk * d_head + j];
            }
            q_row[j] = sq;
            k_row[j] = sk;
        }

        /* RoPE */
        for (int p = 0; p < d_head / 2; p++) {
            float angle = theta[p] * (float)pos[t];
            float c = cosf(angle);
            float s = sinf(angle);
            float qe = q_row[2*p], qo = q_row[2*p+1];
            q_row[2*p]   = qe * c - qo * s;
            q_row[2*p+1] = qe * s + qo * c;
            float ke = k_row[2*p], ko = k_row[2*p+1];
            k_row[2*p]   = ke * c - ko * s;
            k_row[2*p+1] = ke * s + ko * c;
        }
    }
}

/*
 * Reference C implementation: naive matmul + RoPE per token.
 * Supports arbitrary per-token positions.
 */
void fused_proj_rope_f32(
    const float *X,
    const float *Wq,
    const float *Wk,
    float *Q, float *K,
    const float *theta,
    const int *pos,
    int n_tokens, int d_model, int d_head)
{
    for (int t = 0; t < n_tokens; t++) {
        const float *x_row = X + t * d_model;
        float *q_row = Q + t * d_head;
        float *k_row = K + t * d_head;

        /* GEMM: Q[t,:] = X[t,:] · Wq, K[t,:] = X[t,:] · Wk */
        for (int j = 0; j < d_head; j++) {
            float sq = 0.0f, sk = 0.0f;
            for (int kk = 0; kk < d_model; kk++) {
                sq += x_row[kk] * Wq[kk * d_head + j];
                sk += x_row[kk] * Wk[kk * d_head + j];
            }
            q_row[j] = sq;
            k_row[j] = sk;
        }

        /* RoPE */
        for (int p = 0; p < d_head / 2; p++) {
            float angle = theta[p] * (float)pos[t];
            float c = cosf(angle);
            float s = sinf(angle);
            float qe = q_row[2*p], qo = q_row[2*p+1];
            q_row[2*p]   = qe * c - qo * s;
            q_row[2*p+1] = qe * s + qo * c;
            float ke = k_row[2*p], ko = k_row[2*p+1];
            k_row[2*p]   = ke * c - ko * s;
            k_row[2*p+1] = ke * s + ko * c;
        }
    }
}
