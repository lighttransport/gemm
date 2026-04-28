/*
 * test_ssdit_block_forward — Phase 2c.9 standalone microbench.
 *
 * Validates one full SS Flow DiT MOT block forward end-to-end against
 * host reference (mirrors `ssdit_block_forward`-equivalent body in
 * sam3d_ss_flow_dit.h). Composes every Phase 2c kernel:
 *
 *   AdaLN modulation params (shift/scale/gate × msa/mlp) are passed in
 *   directly as random vectors — Phase 2c.2 already validates the
 *   silu+gemm path that produces them, so feeding them as inputs keeps
 *   this microbench focused on the block composition itself.
 *
 *   per stream s ∈ {shape, pose}:
 *     h_s  = modulated_ln_f32(x_s, shift_msa, scale_msa)
 *   self-attn (segmented):
 *     t_s  = sa(h_shape) using qkv_split + multi_head_rmsnorm + sdpa over shape only
 *     t_p  = sa(h_pose)  using sdpa over concat([pose; shape])
 *     out_proj per stream
 *   gated residual:
 *     x_s += t_s * gate_msa     (NEW: gated_residual_add_f32)
 *
 *   norm2 (affine):
 *     h_s  = layernorm_token_f32(x_s, norm2_w, norm2_b, affine=1)
 *   cross-attn:
 *     t_s  = xa(h_s, cond) via xa_q + xa_kv + kv_split + sdpa + xa_out
 *   residual:
 *     x_s += t_s
 *
 *   norm3 (no affine) + AdaLN_mlp:
 *     h_s  = modulated_ln_f32(x_s, shift_mlp, scale_mlp)
 *   FFN:
 *     t_s  = mlp(h_s) via fc1 + gelu_tanh_inplace + fc2
 *   gated residual:
 *     x_s += t_s * gate_mlp
 *
 * Geometry: D=1024, H=16, D_h=64, mlp_h=4D, N_s=512, N_p=4, N_c=512.
 * Random weights ~ 1/sqrt(fan_in); modulation params drawn small
 * (shift ±0.05, scale ±0.1, gate ±0.5) so the gated paths don't
 * dominate the residual stream. Threshold 5e-4 covers drift propagated
 * through every Phase 2c kernel.
 *
 * Usage:
 *   ./test_ssdit_block_forward [--ns 512] [--np 4] [--nc 512]
 *                              [--threshold 5e-4] [-v]
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"
#include "cuda_sam3d_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float urand(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return (float)((*s) >> 8) / (float)(1u << 24);
}
static float max_abs(const float *a, const float *b, size_t n, double *mean_out) {
    double sum = 0.0; float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    if (mean_out) *mean_out = sum / (n > 0 ? n : 1);
    return mx;
}
static void hgemm(float *Y, const float *X, const float *W, const float *b,
                  int N, int D_out, int D_in) {
    for (int n = 0; n < N; n++)
        for (int d = 0; d < D_out; d++) {
            float acc = b ? b[d] : 0.0f;
            const float *xr = X + (size_t)n * D_in;
            const float *wr = W + (size_t)d * D_in;
            for (int k = 0; k < D_in; k++) acc += wr[k] * xr[k];
            Y[(size_t)n * D_out + d] = acc;
        }
}
static void hmod_ln(float *out, const float *in, const float *shift, const float *scale,
                    int N, int dim, float eps) {
    for (int t = 0; t < N; t++) {
        const float *x = in  + (size_t)t * dim;
        float       *y = out + (size_t)t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= (float)dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= (float)dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++) y[i] = (x[i] - mean) * inv * (1.0f + scale[i]) + shift[i];
    }
}
static void hln_affine(float *out, const float *in, const float *gamma, const float *beta,
                       int N, int dim, float eps) {
    for (int t = 0; t < N; t++) {
        const float *x = in  + (size_t)t * dim;
        float       *y = out + (size_t)t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= (float)dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= (float)dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
    }
}
static void hmhrms(float *v, int N, int H, int D_h, int stride, const float *gamma) {
    float sc = sqrtf((float)D_h);
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < H; h++) {
            float *x = v + (size_t)t * stride + h * D_h;
            const float *g = gamma + (size_t)h * D_h;
            double ss = 0.0;
            for (int i = 0; i < D_h; i++) ss += (double)x[i] * x[i];
            float inv = 1.0f / (sqrtf((float)ss) + 1e-12f);
            for (int i = 0; i < D_h; i++) x[i] = x[i] * inv * g[i] * sc;
        }
    }
}
static void hsdpa(float *out, const float *Q, const float *K, const float *V,
                  int N_q, int N_k, int H, int D_h, float scale) {
    int E = H * D_h;
    float *scores = (float *)malloc((size_t)N_k * sizeof(float));
    for (int q = 0; q < N_q; q++) {
        for (int h = 0; h < H; h++) {
            const float *qv = Q + (size_t)q * E + h * D_h;
            float mx = -1e38f;
            for (int k = 0; k < N_k; k++) {
                const float *kv = K + (size_t)k * E + h * D_h;
                float s = 0.0f;
                for (int d = 0; d < D_h; d++) s += qv[d] * kv[d];
                s *= scale;
                scores[k] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            for (int k = 0; k < N_k; k++) {
                scores[k] = expf(scores[k] - mx);
                sum += scores[k];
            }
            float inv = 1.0f / sum;
            for (int d = 0; d < D_h; d++) {
                float acc = 0.0f;
                for (int k = 0; k < N_k; k++)
                    acc += scores[k] * V[(size_t)k * E + h * D_h + d];
                out[(size_t)q * E + h * D_h + d] = acc * inv;
            }
        }
    }
    free(scores);
}
static void hgelu_tanh(float *x, int n) {
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float u = k * (v + c * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(u));
    }
}

/* Per-stream weights. */
typedef struct {
    float *qkv_w, *qkv_b;       /* [3D, D], [3D] */
    float *gamma_q, *gamma_k;   /* [D] */
    float *sa_out_w, *sa_out_b; /* [D, D], [D] */
    float *n2_w, *n2_b;         /* [D] */
    float *xq_w, *xq_b;         /* [D, D], [D] */
    float *xkv_w, *xkv_b;       /* [2D, D], [2D] */
    float *xo_w, *xo_b;         /* [D, D], [D] */
    float *fc1_w, *fc1_b;       /* [4D, D], [4D] */
    float *fc2_w, *fc2_b;       /* [D, 4D], [D] */
} sw_t;
static void alloc_sw(sw_t *w, int dim, int H_dim, uint32_t *rng) {
    int D2 = 2 * dim, D3 = 3 * dim;
    w->qkv_w = (float *)malloc((size_t)D3 * dim * sizeof(float));
    w->qkv_b = (float *)malloc((size_t)D3 * sizeof(float));
    w->gamma_q = (float *)malloc((size_t)dim * sizeof(float));
    w->gamma_k = (float *)malloc((size_t)dim * sizeof(float));
    w->sa_out_w = (float *)malloc((size_t)dim * dim * sizeof(float));
    w->sa_out_b = (float *)malloc((size_t)dim * sizeof(float));
    w->n2_w = (float *)malloc((size_t)dim * sizeof(float));
    w->n2_b = (float *)malloc((size_t)dim * sizeof(float));
    w->xq_w = (float *)malloc((size_t)dim * dim * sizeof(float));
    w->xq_b = (float *)malloc((size_t)dim * sizeof(float));
    w->xkv_w = (float *)malloc((size_t)D2 * dim * sizeof(float));
    w->xkv_b = (float *)malloc((size_t)D2 * sizeof(float));
    w->xo_w = (float *)malloc((size_t)dim * dim * sizeof(float));
    w->xo_b = (float *)malloc((size_t)dim * sizeof(float));
    w->fc1_w = (float *)malloc((size_t)H_dim * dim * sizeof(float));
    w->fc1_b = (float *)malloc((size_t)H_dim * sizeof(float));
    w->fc2_w = (float *)malloc((size_t)dim * H_dim * sizeof(float));
    w->fc2_b = (float *)malloc((size_t)dim * sizeof(float));
    float sd = 1.0f / sqrtf((float)dim);
    float sh = 1.0f / sqrtf((float)H_dim);
    for (int i = 0; i < D3 * dim;     i++) w->qkv_w[i]    = (urand(rng) * 2.f - 1.f) * sd;
    for (int i = 0; i < D3;           i++) w->qkv_b[i]    = (urand(rng) * 2.f - 1.f) * 0.01f;
    for (int i = 0; i < dim;          i++) w->gamma_q[i]  = (urand(rng) * 2.f - 1.f) * 0.5f;
    for (int i = 0; i < dim;          i++) w->gamma_k[i]  = (urand(rng) * 2.f - 1.f) * 0.5f;
    for (int i = 0; i < dim * dim;    i++) w->sa_out_w[i] = (urand(rng) * 2.f - 1.f) * sd;
    for (int i = 0; i < dim;          i++) w->sa_out_b[i] = (urand(rng) * 2.f - 1.f) * 0.01f;
    for (int i = 0; i < dim;          i++) w->n2_w[i]     = 1.0f + (urand(rng) * 2.f - 1.f) * 0.05f;
    for (int i = 0; i < dim;          i++) w->n2_b[i]     = (urand(rng) * 2.f - 1.f) * 0.02f;
    for (int i = 0; i < dim * dim;    i++) w->xq_w[i]     = (urand(rng) * 2.f - 1.f) * sd;
    for (int i = 0; i < dim;          i++) w->xq_b[i]     = (urand(rng) * 2.f - 1.f) * 0.01f;
    for (int i = 0; i < D2 * dim;     i++) w->xkv_w[i]    = (urand(rng) * 2.f - 1.f) * sd;
    for (int i = 0; i < D2;           i++) w->xkv_b[i]    = (urand(rng) * 2.f - 1.f) * 0.01f;
    for (int i = 0; i < dim * dim;    i++) w->xo_w[i]     = (urand(rng) * 2.f - 1.f) * sd;
    for (int i = 0; i < dim;          i++) w->xo_b[i]     = (urand(rng) * 2.f - 1.f) * 0.01f;
    for (int i = 0; i < H_dim * dim;  i++) w->fc1_w[i]    = (urand(rng) * 2.f - 1.f) * sd;
    for (int i = 0; i < H_dim;        i++) w->fc1_b[i]    = (urand(rng) * 2.f - 1.f) * 0.01f;
    for (int i = 0; i < dim * H_dim;  i++) w->fc2_w[i]    = (urand(rng) * 2.f - 1.f) * sh;
    for (int i = 0; i < dim;          i++) w->fc2_b[i]    = (urand(rng) * 2.f - 1.f) * 0.01f;
}
static void free_sw(sw_t *w) {
    free(w->qkv_w); free(w->qkv_b); free(w->gamma_q); free(w->gamma_k);
    free(w->sa_out_w); free(w->sa_out_b); free(w->n2_w); free(w->n2_b);
    free(w->xq_w); free(w->xq_b); free(w->xkv_w); free(w->xkv_b);
    free(w->xo_w); free(w->xo_b);
    free(w->fc1_w); free(w->fc1_b); free(w->fc2_w); free(w->fc2_b);
}

/* Compute per-stream qkv → split → q/k rms norm. */
static void host_sa_qkv(const sw_t *w, const float *x, int N, int dim, int H, int D_h,
                        float *q, float *k, float *v) {
    int D3 = 3 * dim;
    float *qkv = (float *)malloc((size_t)N * D3 * sizeof(float));
    hgemm(qkv, x, w->qkv_w, w->qkv_b, N, D3, dim);
    int E = H * D_h;
    for (int n = 0; n < N; n++) {
        const float *src = qkv + (size_t)n * D3;
        memcpy(q + (size_t)n * E, src + 0,        (size_t)dim * sizeof(float));
        memcpy(k + (size_t)n * E, src + dim,      (size_t)dim * sizeof(float));
        memcpy(v + (size_t)n * E, src + 2 * dim,  (size_t)dim * sizeof(float));
    }
    free(qkv);
    hmhrms(q, N, H, D_h, E, w->gamma_q);
    hmhrms(k, N, H, D_h, E, w->gamma_k);
}

/* Host: full block forward. Mirrors ssdit_mot_block_forward in
 * sam3d_ss_flow_dit.h. x_shape, x_pose are updated in place. */
static void host_block_forward(const sw_t *ws, const sw_t *wp,
                               float *x_s, int N_s, float *x_p, int N_p,
                               const float *cond, int N_c,
                               const float *shift_msa, const float *scale_msa, const float *gate_msa,
                               const float *shift_mlp, const float *scale_mlp, const float *gate_mlp,
                               int dim, int H, int D_h, int H_dim, float eps)
{
    float scale = 1.0f / sqrtf((float)D_h);
    int D2 = 2 * dim;

    /* === norm1 + adaLN_msa === */
    float *h_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *h_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    hmod_ln(h_s, x_s, shift_msa, scale_msa, N_s, dim, eps);
    hmod_ln(h_p, x_p, shift_msa, scale_msa, N_p, dim, eps);

    /* === MOT self-attn === */
    float *q_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *k_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *v_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *q_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    float *k_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    float *v_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    host_sa_qkv(ws, h_s, N_s, dim, H, D_h, q_s, k_s, v_s);
    host_sa_qkv(wp, h_p, N_p, dim, H, D_h, q_p, k_p, v_p);
    /* shape: attends self only. */
    float *sdpa_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    hsdpa(sdpa_s, q_s, k_s, v_s, N_s, N_s, H, D_h, scale);
    /* pose: K/V = concat(pose, shape). */
    int N_kv = N_p + N_s;
    float *k_pkv = (float *)malloc((size_t)N_kv * dim * sizeof(float));
    float *v_pkv = (float *)malloc((size_t)N_kv * dim * sizeof(float));
    memcpy(k_pkv,                         k_p, (size_t)N_p * dim * sizeof(float));
    memcpy(k_pkv + (size_t)N_p * dim,     k_s, (size_t)N_s * dim * sizeof(float));
    memcpy(v_pkv,                         v_p, (size_t)N_p * dim * sizeof(float));
    memcpy(v_pkv + (size_t)N_p * dim,     v_s, (size_t)N_s * dim * sizeof(float));
    float *sdpa_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    hsdpa(sdpa_p, q_p, k_pkv, v_pkv, N_p, N_kv, H, D_h, scale);
    free(q_s); free(k_s); free(v_s); free(q_p); free(k_p); free(v_p);
    free(k_pkv); free(v_pkv);
    /* out_proj per stream. */
    float *t_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *t_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    hgemm(t_s, sdpa_s, ws->sa_out_w, ws->sa_out_b, N_s, dim, dim);
    hgemm(t_p, sdpa_p, wp->sa_out_w, wp->sa_out_b, N_p, dim, dim);
    free(sdpa_s); free(sdpa_p);
    /* gated residual. */
    for (int n = 0; n < N_s; n++)
        for (int i = 0; i < dim; i++) x_s[n * dim + i] += t_s[n * dim + i] * gate_msa[i];
    for (int n = 0; n < N_p; n++)
        for (int i = 0; i < dim; i++) x_p[n * dim + i] += t_p[n * dim + i] * gate_msa[i];

    /* === norm2 (affine) === */
    hln_affine(h_s, x_s, ws->n2_w, ws->n2_b, N_s, dim, eps);
    hln_affine(h_p, x_p, wp->n2_w, wp->n2_b, N_p, dim, eps);

    /* === cross-attn (Q from h, KV from cond) === */
    float *qx_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *qx_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    hgemm(qx_s, h_s, ws->xq_w, ws->xq_b, N_s, dim, dim);
    hgemm(qx_p, h_p, wp->xq_w, wp->xq_b, N_p, dim, dim);
    float *kvs = (float *)malloc((size_t)N_c * D2 * sizeof(float));
    float *kvp = (float *)malloc((size_t)N_c * D2 * sizeof(float));
    hgemm(kvs, cond, ws->xkv_w, ws->xkv_b, N_c, D2, dim);
    hgemm(kvp, cond, wp->xkv_w, wp->xkv_b, N_c, D2, dim);
    float *Ks_x = (float *)malloc((size_t)N_c * dim * sizeof(float));
    float *Vs_x = (float *)malloc((size_t)N_c * dim * sizeof(float));
    float *Kp_x = (float *)malloc((size_t)N_c * dim * sizeof(float));
    float *Vp_x = (float *)malloc((size_t)N_c * dim * sizeof(float));
    for (int n = 0; n < N_c; n++) {
        memcpy(Ks_x + (size_t)n * dim, kvs + (size_t)n * D2,         (size_t)dim * sizeof(float));
        memcpy(Vs_x + (size_t)n * dim, kvs + (size_t)n * D2 + dim,   (size_t)dim * sizeof(float));
        memcpy(Kp_x + (size_t)n * dim, kvp + (size_t)n * D2,         (size_t)dim * sizeof(float));
        memcpy(Vp_x + (size_t)n * dim, kvp + (size_t)n * D2 + dim,   (size_t)dim * sizeof(float));
    }
    free(kvs); free(kvp);
    float *o_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *o_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    hsdpa(o_s, qx_s, Ks_x, Vs_x, N_s, N_c, H, D_h, scale);
    hsdpa(o_p, qx_p, Kp_x, Vp_x, N_p, N_c, H, D_h, scale);
    free(qx_s); free(qx_p); free(Ks_x); free(Vs_x); free(Kp_x); free(Vp_x);
    hgemm(t_s, o_s, ws->xo_w, ws->xo_b, N_s, dim, dim);
    hgemm(t_p, o_p, wp->xo_w, wp->xo_b, N_p, dim, dim);
    free(o_s); free(o_p);
    /* residual (no gate). */
    for (size_t i = 0; i < (size_t)N_s * dim; i++) x_s[i] += t_s[i];
    for (size_t i = 0; i < (size_t)N_p * dim; i++) x_p[i] += t_p[i];

    /* === norm3 + adaLN_mlp + FFN === */
    hmod_ln(h_s, x_s, shift_mlp, scale_mlp, N_s, dim, eps);
    hmod_ln(h_p, x_p, shift_mlp, scale_mlp, N_p, dim, eps);
    float *m1_s = (float *)malloc((size_t)N_s * H_dim * sizeof(float));
    float *m1_p = (float *)malloc((size_t)N_p * H_dim * sizeof(float));
    hgemm(m1_s, h_s, ws->fc1_w, ws->fc1_b, N_s, H_dim, dim);
    hgemm(m1_p, h_p, wp->fc1_w, wp->fc1_b, N_p, H_dim, dim);
    hgelu_tanh(m1_s, N_s * H_dim);
    hgelu_tanh(m1_p, N_p * H_dim);
    hgemm(t_s, m1_s, ws->fc2_w, ws->fc2_b, N_s, dim, H_dim);
    hgemm(t_p, m1_p, wp->fc2_w, wp->fc2_b, N_p, dim, H_dim);
    free(m1_s); free(m1_p);
    for (int n = 0; n < N_s; n++)
        for (int i = 0; i < dim; i++) x_s[n * dim + i] += t_s[n * dim + i] * gate_mlp[i];
    for (int n = 0; n < N_p; n++)
        for (int i = 0; i < dim; i++) x_p[n * dim + i] += t_p[n * dim + i] * gate_mlp[i];

    free(h_s); free(h_p); free(t_s); free(t_p);
}

/* GPU kernel function-pointer pack. */
typedef struct {
    CUfunction gemm, mod_ln, ln, qkv_split, kv_split, mhrms, sdpa, gelu, gated, resadd;
} fn_t;

static int upload_sw(CUdeviceptr *bufs, const sw_t *w, int dim, int H_dim) {
    int i = 0, D2 = 2 * dim, D3 = 3 * dim;
    bufs[i++] = cu_upload_raw(w->qkv_w,    (size_t)D3 * dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->qkv_b,    (size_t)D3 * sizeof(float));
    bufs[i++] = cu_upload_raw(w->gamma_q,  (size_t)dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->gamma_k,  (size_t)dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->sa_out_w, (size_t)dim * dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->sa_out_b, (size_t)dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->n2_w,     (size_t)dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->n2_b,     (size_t)dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->xq_w,     (size_t)dim * dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->xq_b,     (size_t)dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->xkv_w,    (size_t)D2 * dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->xkv_b,    (size_t)D2 * sizeof(float));
    bufs[i++] = cu_upload_raw(w->xo_w,     (size_t)dim * dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->xo_b,     (size_t)dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->fc1_w,    (size_t)H_dim * dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->fc1_b,    (size_t)H_dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->fc2_w,    (size_t)dim * H_dim * sizeof(float));
    bufs[i++] = cu_upload_raw(w->fc2_b,    (size_t)dim * sizeof(float));
    return i;
}
enum { SWB_QKVW, SWB_QKVB, SWB_GQ, SWB_GK, SWB_SAOW, SWB_SAOB, SWB_N2W, SWB_N2B,
       SWB_XQW, SWB_XQB, SWB_XKVW, SWB_XKVB, SWB_XOW, SWB_XOB,
       SWB_FC1W, SWB_FC1B, SWB_FC2W, SWB_FC2B, SWB_N };

/* Run one stream's QKV-split-rms path. q, k, v output buffers must be allocated. */
static int gpu_sa_qkv(const fn_t *fn, const CUdeviceptr *swb,
                      CUdeviceptr d_x, int N, int dim, int H, int D_h,
                      CUdeviceptr d_qkv, CUdeviceptr d_q, CUdeviceptr d_k, CUdeviceptr d_v)
{
    int D3 = 3 * dim;
    {
        unsigned gx = (N + 15) / 16, gy = (D3 + 15) / 16;
        void *args[] = { &d_qkv, &d_x, (void *)&swb[SWB_QKVW], (void *)&swb[SWB_QKVB], &N, &dim, &D3 };
        if (cuLaunchKernel(fn->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 1;
    }
    {
        unsigned grid = (unsigned)((N * dim + 255) / 256);
        void *args[] = { &d_q, &d_k, &d_v, &d_qkv, &N, &dim };
        if (cuLaunchKernel(fn->qkv_split, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 1;
    }
    unsigned threads = 64;
    size_t smem = threads * sizeof(float);
    int stride = dim;
    {
        void *args[] = { &d_q, (void *)&swb[SWB_GQ], &N, &H, &D_h, &stride };
        if (cuLaunchKernel(fn->mhrms, (unsigned)H, (unsigned)N, 1, threads, 1, 1,
                           (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return 1;
    }
    {
        void *args[] = { &d_k, (void *)&swb[SWB_GK], &N, &H, &D_h, &stride };
        if (cuLaunchKernel(fn->mhrms, (unsigned)H, (unsigned)N, 1, threads, 1, 1,
                           (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    int   N_s = 512, N_p = 4, N_c = 512;
    int   H = 16, D_h = 64;
    int   ratio = 4;
    float threshold = 5e-4f;
    int   verbose   = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--ns")        && i+1 < argc) N_s = atoi(argv[++i]);
        else if (!strcmp(a, "--np")        && i+1 < argc) N_p = atoi(argv[++i]);
        else if (!strcmp(a, "--nc")        && i+1 < argc) N_c = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                        verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    int dim = H * D_h;
    int H_dim = ratio * dim;
    int D2 = 2 * dim, D3 = 3 * dim;
    float eps = 1e-6f;

    uint32_t rng = 0xC0FFEEu;
    float *x_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *x_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    float *cond = (float *)malloc((size_t)N_c * dim * sizeof(float));
    for (int i = 0; i < N_s * dim; i++) x_s[i]  = urand(&rng) * 2.f - 1.f;
    for (int i = 0; i < N_p * dim; i++) x_p[i]  = urand(&rng) * 2.f - 1.f;
    for (int i = 0; i < N_c * dim; i++) cond[i] = urand(&rng) * 2.f - 1.f;
    float *shift_msa = (float *)malloc((size_t)dim * sizeof(float));
    float *scale_msa = (float *)malloc((size_t)dim * sizeof(float));
    float *gate_msa  = (float *)malloc((size_t)dim * sizeof(float));
    float *shift_mlp = (float *)malloc((size_t)dim * sizeof(float));
    float *scale_mlp = (float *)malloc((size_t)dim * sizeof(float));
    float *gate_mlp  = (float *)malloc((size_t)dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        shift_msa[i] = (urand(&rng) * 2.f - 1.f) * 0.05f;
        scale_msa[i] = (urand(&rng) * 2.f - 1.f) * 0.10f;
        gate_msa[i]  = (urand(&rng) * 2.f - 1.f) * 0.50f;
        shift_mlp[i] = (urand(&rng) * 2.f - 1.f) * 0.05f;
        scale_mlp[i] = (urand(&rng) * 2.f - 1.f) * 0.10f;
        gate_mlp[i]  = (urand(&rng) * 2.f - 1.f) * 0.50f;
    }
    sw_t ws, wp;
    alloc_sw(&ws, dim, H_dim, &rng);
    alloc_sw(&wp, dim, H_dim, &rng);

    /* Host ref. Deep-copy x_s/x_p so device side gets the originals. */
    float *xs_ref = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *xp_ref = (float *)malloc((size_t)N_p * dim * sizeof(float));
    memcpy(xs_ref, x_s, (size_t)N_s * dim * sizeof(float));
    memcpy(xp_ref, x_p, (size_t)N_p * dim * sizeof(float));
    host_block_forward(&ws, &wp, xs_ref, N_s, xp_ref, N_p,
                       cond, N_c,
                       shift_msa, scale_msa, gate_msa,
                       shift_mlp, scale_mlp, gate_mlp,
                       dim, H, D_h, H_dim, eps);

    /* Device. */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0; if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;
    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose, "test_ssdit_block_forward") < 0) return 4;
    fn_t fn;
    if (cuModuleGetFunction(&fn.gemm,      mod, "gemm_f32_bias")           != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn.mod_ln,    mod, "modulated_ln_f32")        != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn.ln,        mod, "layernorm_token_f32")     != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn.qkv_split, mod, "qkv_split_f32")           != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn.kv_split,  mod, "kv_split_f32")            != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn.mhrms,     mod, "multi_head_rmsnorm_f32")  != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn.sdpa,      mod, "sdpa_f32")                != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn.gelu,      mod, "gelu_tanh_inplace_f32")   != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn.gated,     mod, "gated_residual_add_f32")  != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn.resadd,    mod, "residual_add_f32")        != CUDA_SUCCESS) return 4;

    /* Upload everything. */
    CUdeviceptr d_xs = cu_upload_raw(x_s,  (size_t)N_s * dim * sizeof(float));
    CUdeviceptr d_xp = cu_upload_raw(x_p,  (size_t)N_p * dim * sizeof(float));
    CUdeviceptr d_cond = cu_upload_raw(cond, (size_t)N_c * dim * sizeof(float));
    CUdeviceptr d_smsa = cu_upload_raw(shift_msa, (size_t)dim * sizeof(float));
    CUdeviceptr d_cmsa = cu_upload_raw(scale_msa, (size_t)dim * sizeof(float));
    CUdeviceptr d_gmsa = cu_upload_raw(gate_msa,  (size_t)dim * sizeof(float));
    CUdeviceptr d_smlp = cu_upload_raw(shift_mlp, (size_t)dim * sizeof(float));
    CUdeviceptr d_cmlp = cu_upload_raw(scale_mlp, (size_t)dim * sizeof(float));
    CUdeviceptr d_gmlp = cu_upload_raw(gate_mlp,  (size_t)dim * sizeof(float));
    CUdeviceptr swsb[SWB_N], swpb[SWB_N];
    upload_sw(swsb, &ws, dim, H_dim);
    upload_sw(swpb, &wp, dim, H_dim);

    /* Scratch buffers. */
    CUdeviceptr d_hs = 0, d_hp = 0;
    CUdeviceptr d_qkvs = 0, d_qkvp = 0;
    CUdeviceptr d_qs = 0, d_ks = 0, d_vs = 0, d_qp = 0, d_kp = 0, d_vp = 0;
    CUdeviceptr d_kpkv = 0, d_vpkv = 0;
    CUdeviceptr d_sdpa_s = 0, d_sdpa_p = 0;
    CUdeviceptr d_t_s = 0, d_t_p = 0;
    CUdeviceptr d_kvs_x = 0, d_kvp_x = 0;
    CUdeviceptr d_Ks = 0, d_Vs = 0, d_Kp = 0, d_Vp = 0;
    CUdeviceptr d_qx_s = 0, d_qx_p = 0, d_o_s = 0, d_o_p = 0;
    CUdeviceptr d_m1s = 0, d_m1p = 0;
    cuMemAlloc(&d_hs, (size_t)N_s * dim * sizeof(float));
    cuMemAlloc(&d_hp, (size_t)N_p * dim * sizeof(float));
    cuMemAlloc(&d_qkvs, (size_t)N_s * D3 * sizeof(float));
    cuMemAlloc(&d_qkvp, (size_t)N_p * D3 * sizeof(float));
    cuMemAlloc(&d_qs, (size_t)N_s * dim * sizeof(float));
    cuMemAlloc(&d_ks, (size_t)N_s * dim * sizeof(float));
    cuMemAlloc(&d_vs, (size_t)N_s * dim * sizeof(float));
    cuMemAlloc(&d_qp, (size_t)N_p * dim * sizeof(float));
    cuMemAlloc(&d_kp, (size_t)N_p * dim * sizeof(float));
    cuMemAlloc(&d_vp, (size_t)N_p * dim * sizeof(float));
    int N_kv = N_p + N_s;
    cuMemAlloc(&d_kpkv, (size_t)N_kv * dim * sizeof(float));
    cuMemAlloc(&d_vpkv, (size_t)N_kv * dim * sizeof(float));
    cuMemAlloc(&d_sdpa_s, (size_t)N_s * dim * sizeof(float));
    cuMemAlloc(&d_sdpa_p, (size_t)N_p * dim * sizeof(float));
    cuMemAlloc(&d_t_s, (size_t)N_s * dim * sizeof(float));
    cuMemAlloc(&d_t_p, (size_t)N_p * dim * sizeof(float));
    cuMemAlloc(&d_kvs_x, (size_t)N_c * D2 * sizeof(float));
    cuMemAlloc(&d_kvp_x, (size_t)N_c * D2 * sizeof(float));
    cuMemAlloc(&d_Ks, (size_t)N_c * dim * sizeof(float));
    cuMemAlloc(&d_Vs, (size_t)N_c * dim * sizeof(float));
    cuMemAlloc(&d_Kp, (size_t)N_c * dim * sizeof(float));
    cuMemAlloc(&d_Vp, (size_t)N_c * dim * sizeof(float));
    cuMemAlloc(&d_qx_s, (size_t)N_s * dim * sizeof(float));
    cuMemAlloc(&d_qx_p, (size_t)N_p * dim * sizeof(float));
    cuMemAlloc(&d_o_s, (size_t)N_s * dim * sizeof(float));
    cuMemAlloc(&d_o_p, (size_t)N_p * dim * sizeof(float));
    cuMemAlloc(&d_m1s, (size_t)N_s * H_dim * sizeof(float));
    cuMemAlloc(&d_m1p, (size_t)N_p * H_dim * sizeof(float));

    float scale = 1.0f / sqrtf((float)D_h);
    int affine = 1;

    /* === norm1 + adaLN_msa === */
    {
        unsigned threads = 256;
        size_t smem = 2 * threads * sizeof(float);
        void *args[] = { &d_hs, &d_xs, &d_smsa, &d_cmsa, &N_s, &dim, &eps };
        if (cuLaunchKernel(fn.mod_ln, (unsigned)N_s, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned threads = 256;
        size_t smem = 2 * threads * sizeof(float);
        void *args[] = { &d_hp, &d_xp, &d_smsa, &d_cmsa, &N_p, &dim, &eps };
        if (cuLaunchKernel(fn.mod_ln, (unsigned)N_p, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }

    /* === MOT self-attn === */
    if (gpu_sa_qkv(&fn, swsb, d_hs, N_s, dim, H, D_h, d_qkvs, d_qs, d_ks, d_vs) != 0) return 5;
    if (gpu_sa_qkv(&fn, swpb, d_hp, N_p, dim, H, D_h, d_qkvp, d_qp, d_kp, d_vp) != 0) return 5;
    /* shape SDPA self only. */
    {
        unsigned threads = 256;
        size_t sm = (threads + (size_t)N_s) * sizeof(float);
        void *args[] = { &d_sdpa_s, &d_qs, &d_ks, &d_vs, &N_s, &N_s, &H, &D_h, &scale };
        if (cuLaunchKernel(fn.sdpa, (unsigned)N_s, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    /* pose KV concat via DtoD. */
    cuMemcpyDtoD(d_kpkv,                                       d_kp, (size_t)N_p * dim * sizeof(float));
    cuMemcpyDtoD(d_kpkv + (size_t)N_p * dim * sizeof(float),   d_ks, (size_t)N_s * dim * sizeof(float));
    cuMemcpyDtoD(d_vpkv,                                       d_vp, (size_t)N_p * dim * sizeof(float));
    cuMemcpyDtoD(d_vpkv + (size_t)N_p * dim * sizeof(float),   d_vs, (size_t)N_s * dim * sizeof(float));
    {
        unsigned threads = 256;
        size_t sm = (threads + (size_t)N_kv) * sizeof(float);
        void *args[] = { &d_sdpa_p, &d_qp, &d_kpkv, &d_vpkv, &N_p, &N_kv, &H, &D_h, &scale };
        if (cuLaunchKernel(fn.sdpa, (unsigned)N_p, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    /* sa out_proj per stream. */
    {
        unsigned gx = (N_s + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_t_s, &d_sdpa_s, &swsb[SWB_SAOW], &swsb[SWB_SAOB], &N_s, &dim, &dim };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_t_p, &d_sdpa_p, &swpb[SWB_SAOW], &swpb[SWB_SAOB], &N_p, &dim, &dim };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    /* gated residual add. */
    {
        unsigned grid = (unsigned)((N_s * dim + 255) / 256);
        void *args[] = { &d_xs, &d_t_s, &d_gmsa, &N_s, &dim };
        if (cuLaunchKernel(fn.gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned grid = (unsigned)((N_p * dim + 255) / 256);
        void *args[] = { &d_xp, &d_t_p, &d_gmsa, &N_p, &dim };
        if (cuLaunchKernel(fn.gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }

    /* === norm2 (affine) === */
    {
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *args[] = { &d_hs, &d_xs, &swsb[SWB_N2W], &swsb[SWB_N2B], &N_s, &dim, &eps, &affine };
        if (cuLaunchKernel(fn.ln, (unsigned)N_s, 1, 1, threads, 1, 1, (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *args[] = { &d_hp, &d_xp, &swpb[SWB_N2W], &swpb[SWB_N2B], &N_p, &dim, &eps, &affine };
        if (cuLaunchKernel(fn.ln, (unsigned)N_p, 1, 1, threads, 1, 1, (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    /* === cross-attn === */
    {
        unsigned gx = (N_s + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_qx_s, &d_hs, &swsb[SWB_XQW], &swsb[SWB_XQB], &N_s, &dim, &dim };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_qx_p, &d_hp, &swpb[SWB_XQW], &swpb[SWB_XQB], &N_p, &dim, &dim };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned gx = (N_c + 15) / 16, gy = (D2 + 15) / 16;
        void *args[] = { &d_kvs_x, &d_cond, &swsb[SWB_XKVW], &swsb[SWB_XKVB], &N_c, &dim, &D2 };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned gx = (N_c + 15) / 16, gy = (D2 + 15) / 16;
        void *args[] = { &d_kvp_x, &d_cond, &swpb[SWB_XKVW], &swpb[SWB_XKVB], &N_c, &dim, &D2 };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned grid = (unsigned)((N_c * dim + 255) / 256);
        void *args1[] = { &d_Ks, &d_Vs, &d_kvs_x, &N_c, &dim };
        if (cuLaunchKernel(fn.kv_split, grid, 1, 1, 256, 1, 1, 0, 0, args1, NULL) != CUDA_SUCCESS) return 5;
        void *args2[] = { &d_Kp, &d_Vp, &d_kvp_x, &N_c, &dim };
        if (cuLaunchKernel(fn.kv_split, grid, 1, 1, 256, 1, 1, 0, 0, args2, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned threads = 256; size_t sm = (threads + (size_t)N_c) * sizeof(float);
        void *args[] = { &d_o_s, &d_qx_s, &d_Ks, &d_Vs, &N_s, &N_c, &H, &D_h, &scale };
        if (cuLaunchKernel(fn.sdpa, (unsigned)N_s, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned threads = 256; size_t sm = (threads + (size_t)N_c) * sizeof(float);
        void *args[] = { &d_o_p, &d_qx_p, &d_Kp, &d_Vp, &N_p, &N_c, &H, &D_h, &scale };
        if (cuLaunchKernel(fn.sdpa, (unsigned)N_p, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned gx = (N_s + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_t_s, &d_o_s, &swsb[SWB_XOW], &swsb[SWB_XOB], &N_s, &dim, &dim };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_t_p, &d_o_p, &swpb[SWB_XOW], &swpb[SWB_XOB], &N_p, &dim, &dim };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    /* residual (no gate). */
    {
        int total = N_s * dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &d_xs, &d_t_s, &total };
        if (cuLaunchKernel(fn.resadd, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        int total = N_p * dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &d_xp, &d_t_p, &total };
        if (cuLaunchKernel(fn.resadd, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }

    /* === norm3 + adaLN_mlp + FFN + gated residual === */
    {
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *args[] = { &d_hs, &d_xs, &d_smlp, &d_cmlp, &N_s, &dim, &eps };
        if (cuLaunchKernel(fn.mod_ln, (unsigned)N_s, 1, 1, threads, 1, 1, (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *args[] = { &d_hp, &d_xp, &d_smlp, &d_cmlp, &N_p, &dim, &eps };
        if (cuLaunchKernel(fn.mod_ln, (unsigned)N_p, 1, 1, threads, 1, 1, (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    /* fc1 */
    {
        unsigned gx = (N_s + 15) / 16, gy = (H_dim + 15) / 16;
        void *args[] = { &d_m1s, &d_hs, &swsb[SWB_FC1W], &swsb[SWB_FC1B], &N_s, &dim, &H_dim };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (H_dim + 15) / 16;
        void *args[] = { &d_m1p, &d_hp, &swpb[SWB_FC1W], &swpb[SWB_FC1B], &N_p, &dim, &H_dim };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    /* gelu */
    {
        int total = N_s * H_dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &d_m1s, &total };
        if (cuLaunchKernel(fn.gelu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        int total = N_p * H_dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &d_m1p, &total };
        if (cuLaunchKernel(fn.gelu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    /* fc2 */
    {
        unsigned gx = (N_s + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_t_s, &d_m1s, &swsb[SWB_FC2W], &swsb[SWB_FC2B], &N_s, &H_dim, &dim };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_t_p, &d_m1p, &swpb[SWB_FC2W], &swpb[SWB_FC2B], &N_p, &H_dim, &dim };
        if (cuLaunchKernel(fn.gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    /* gated residual add. */
    {
        unsigned grid = (unsigned)((N_s * dim + 255) / 256);
        void *args[] = { &d_xs, &d_t_s, &d_gmlp, &N_s, &dim };
        if (cuLaunchKernel(fn.gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned grid = (unsigned)((N_p * dim + 255) / 256);
        void *args[] = { &d_xp, &d_t_p, &d_gmlp, &N_p, &dim };
        if (cuLaunchKernel(fn.gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    cuCtxSynchronize();

    float *xs_dst = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *xp_dst = (float *)malloc((size_t)N_p * dim * sizeof(float));
    cuMemcpyDtoH(xs_dst, d_xs, (size_t)N_s * dim * sizeof(float));
    cuMemcpyDtoH(xp_dst, d_xp, (size_t)N_p * dim * sizeof(float));

    double mean_s = 0.0, mean_p = 0.0;
    float mx_s = max_abs(xs_dst, xs_ref, (size_t)N_s * dim, &mean_s);
    float mx_p = max_abs(xp_dst, xp_ref, (size_t)N_p * dim, &mean_p);
    int ok = (mx_s <= threshold) && (mx_p <= threshold);

    fprintf(stderr,
        "[test_ssdit_block_forward] N_s=%d N_p=%d N_c=%d D=%d H=%d D_h=%d  "
        "shape max_abs=%.4g (mean %.4g)  pose max_abs=%.4g (mean %.4g)  %s (threshold %.1g)\n",
        N_s, N_p, N_c, dim, H, D_h, (double)mx_s, mean_s, (double)mx_p, mean_p,
        ok ? "OK" : "FAIL", (double)threshold);

    free(x_s); free(x_p); free(cond);
    free(shift_msa); free(scale_msa); free(gate_msa);
    free(shift_mlp); free(scale_mlp); free(gate_mlp);
    free(xs_ref); free(xp_ref); free(xs_dst); free(xp_dst);
    free_sw(&ws); free_sw(&wp);
    /* Memfree everything (best-effort; process exit will clean up too). */
    cuMemFree(d_xs); cuMemFree(d_xp); cuMemFree(d_cond);
    cuMemFree(d_smsa); cuMemFree(d_cmsa); cuMemFree(d_gmsa);
    cuMemFree(d_smlp); cuMemFree(d_cmlp); cuMemFree(d_gmlp);
    for (int i = 0; i < SWB_N; i++) { cuMemFree(swsb[i]); cuMemFree(swpb[i]); }
    cuMemFree(d_hs); cuMemFree(d_hp);
    cuMemFree(d_qkvs); cuMemFree(d_qkvp);
    cuMemFree(d_qs); cuMemFree(d_ks); cuMemFree(d_vs);
    cuMemFree(d_qp); cuMemFree(d_kp); cuMemFree(d_vp);
    cuMemFree(d_kpkv); cuMemFree(d_vpkv);
    cuMemFree(d_sdpa_s); cuMemFree(d_sdpa_p);
    cuMemFree(d_t_s); cuMemFree(d_t_p);
    cuMemFree(d_kvs_x); cuMemFree(d_kvp_x);
    cuMemFree(d_Ks); cuMemFree(d_Vs); cuMemFree(d_Kp); cuMemFree(d_Vp);
    cuMemFree(d_qx_s); cuMemFree(d_qx_p); cuMemFree(d_o_s); cuMemFree(d_o_p);
    cuMemFree(d_m1s); cuMemFree(d_m1p);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
