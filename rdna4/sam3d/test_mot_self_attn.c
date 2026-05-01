/*
 * test_mot_self_attn — Phase 2c.5 standalone microbench.
 *
 * Validates SS DiT MOT self-attention end-to-end on-device against the
 * host reference `ssdit_mot_self_attn` (sam3d_ss_flow_dit.h):
 *
 *   per-stream: x  → qkv = gemm_f32_bias(D, 3D) + bias
 *                  → qkv_split_f32 → q, k, v   [N, H*D_h]
 *                  → multi_head_rmsnorm_f32(q, gamma_q)
 *                  → multi_head_rmsnorm_f32(k, gamma_k)
 *
 *   shape:    h_s = sdpa_f32(q_s, k_s, v_s, scale)
 *   pose:     h_p = sdpa_f32(q_p, [k_p; k_s], [v_p; v_s], scale)
 *
 *   per-stream: out = gemm_f32_bias(h, D, D) + bias_out
 *
 * The pose stream's KV is the concat of pose+shape KV; we materialize
 * concat buffers via hipMemcpyDtoD rather than adding a concat kernel —
 * this is the production composition the runner will use.
 *
 * Random weights scaled by 1/sqrt(fan_in); RMS gammas drawn ~ ±0.5;
 * threshold accounts for SDPA expf vs libm + reduction-order drift.
 *
 * Usage:
 *   ./test_mot_self_attn [--ns 512] [--np 4] [--heads 16] [--hd 64] [--threshold 5e-4] [-v]
 */

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include "hip_sam3d_kernels.h"

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

/* Per-stream weights. */
typedef struct {
    float *qkv_w, *qkv_b;     /* [3D, D], [3D] */
    float *gamma_q, *gamma_k; /* [H, D_h] */
    float *out_w, *out_b;     /* [D, D], [D] */
} stream_w;

static void alloc_stream(stream_w *s, int dim, int H, int D_h, uint32_t *rng) {
    int qkv = 3 * dim;
    s->qkv_w   = (float *)malloc((size_t)qkv * dim * sizeof(float));
    s->qkv_b   = (float *)malloc((size_t)qkv      * sizeof(float));
    s->gamma_q = (float *)malloc((size_t)H * D_h * sizeof(float));
    s->gamma_k = (float *)malloc((size_t)H * D_h * sizeof(float));
    s->out_w   = (float *)malloc((size_t)dim * dim * sizeof(float));
    s->out_b   = (float *)malloc((size_t)dim       * sizeof(float));
    float sw = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < qkv * dim; i++)  s->qkv_w[i] = (urand(rng) * 2.0f - 1.0f) * sw;
    for (int i = 0; i < qkv;       i++)  s->qkv_b[i] = (urand(rng) * 2.0f - 1.0f) * 0.01f;
    for (int i = 0; i < H * D_h;   i++)  s->gamma_q[i] = (urand(rng) * 2.0f - 1.0f) * 0.5f;
    for (int i = 0; i < H * D_h;   i++)  s->gamma_k[i] = (urand(rng) * 2.0f - 1.0f) * 0.5f;
    for (int i = 0; i < dim * dim; i++)  s->out_w[i] = (urand(rng) * 2.0f - 1.0f) * sw;
    for (int i = 0; i < dim;       i++)  s->out_b[i] = (urand(rng) * 2.0f - 1.0f) * 0.01f;
}
static void free_stream(stream_w *s) {
    free(s->qkv_w); free(s->qkv_b);
    free(s->gamma_q); free(s->gamma_k);
    free(s->out_w); free(s->out_b);
}

/* host: gemm Y = X @ W^T + b, where W is [D_out, D_in]. */
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
/* SDPA: out (N_q, H*D_h) = softmax(Q K^T * scale) V, per head. */
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

/* Run one stream's qkv path on host: x → qkv, split, rms-norm. */
static void hstream_qkv(const stream_w *s, const float *x, int N, int dim,
                        int H, int D_h, float *q, float *k, float *v) {
    float *qkv = (float *)malloc((size_t)N * 3 * dim * sizeof(float));
    hgemm(qkv, x, s->qkv_w, s->qkv_b, N, 3 * dim, dim);
    int E = H * D_h;
    for (int n = 0; n < N; n++) {
        const float *src = qkv + (size_t)n * 3 * dim;
        memcpy(q + (size_t)n * E, src + 0,        (size_t)dim * sizeof(float));
        memcpy(k + (size_t)n * E, src + dim,      (size_t)dim * sizeof(float));
        memcpy(v + (size_t)n * E, src + 2 * dim,  (size_t)dim * sizeof(float));
    }
    free(qkv);
    hmhrms(q, N, H, D_h, E, s->gamma_q);
    hmhrms(k, N, H, D_h, E, s->gamma_k);
}

int main(int argc, char **argv)
{
    int   N_s = 512, N_p = 4, H = 16, D_h = 64;
    float threshold = 5e-4f;
    int   verbose   = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--ns")        && i+1 < argc) N_s = atoi(argv[++i]);
        else if (!strcmp(a, "--np")        && i+1 < argc) N_p = atoi(argv[++i]);
        else if (!strcmp(a, "--heads")     && i+1 < argc) H   = atoi(argv[++i]);
        else if (!strcmp(a, "--hd")        && i+1 < argc) D_h = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                        verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    int dim = H * D_h;
    int N_kv = N_p + N_s;

    uint32_t rng = 0xC0FFEEu;
    float *x_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *x_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    for (int i = 0; i < N_s * dim; i++) x_s[i] = (urand(&rng) * 2.0f - 1.0f);
    for (int i = 0; i < N_p * dim; i++) x_p[i] = (urand(&rng) * 2.0f - 1.0f);
    stream_w sw_s, sw_p;
    alloc_stream(&sw_s, dim, H, D_h, &rng);
    alloc_stream(&sw_p, dim, H, D_h, &rng);

    /* Host reference. */
    float *q_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *k_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *v_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *q_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    float *k_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    float *v_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    hstream_qkv(&sw_s, x_s, N_s, dim, H, D_h, q_s, k_s, v_s);
    hstream_qkv(&sw_p, x_p, N_p, dim, H, D_h, q_p, k_p, v_p);
    float *k_pkv = (float *)malloc((size_t)N_kv * dim * sizeof(float));
    float *v_pkv = (float *)malloc((size_t)N_kv * dim * sizeof(float));
    memcpy(k_pkv,                          k_p, (size_t)N_p * dim * sizeof(float));
    memcpy(k_pkv + (size_t)N_p * dim,      k_s, (size_t)N_s * dim * sizeof(float));
    memcpy(v_pkv,                          v_p, (size_t)N_p * dim * sizeof(float));
    memcpy(v_pkv + (size_t)N_p * dim,      v_s, (size_t)N_s * dim * sizeof(float));
    float scale = 1.0f / sqrtf((float)D_h);
    float *h_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *h_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    hsdpa(h_s, q_s, k_s,  v_s,  N_s, N_s,  H, D_h, scale);
    hsdpa(h_p, q_p, k_pkv, v_pkv, N_p, N_kv, H, D_h, scale);
    float *o_s_ref = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *o_p_ref = (float *)malloc((size_t)N_p * dim * sizeof(float));
    hgemm(o_s_ref, h_s, sw_s.out_w, sw_s.out_b, N_s, dim, dim);
    hgemm(o_p_ref, h_p, sw_p.out_w, sw_p.out_b, N_p, dim, dim);
    free(q_s); free(k_s); free(v_s); free(q_p); free(k_p); free(v_p);
    free(k_pkv); free(v_pkv); free(h_s); free(h_p);

    /* Device. */
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0; if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    if (hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose, "test_mot_self_attn") < 0) return 4;
    hipFunction_t fn_gemm = 0, fn_split = 0, fn_rms = 0, fn_sdpa = 0;
    if (hipModuleGetFunction(&fn_gemm,  mod, "gemm_f32_bias")          != hipSuccess) return 4;
    if (hipModuleGetFunction(&fn_split, mod, "qkv_split_f32")          != hipSuccess) return 4;
    if (hipModuleGetFunction(&fn_rms,   mod, "multi_head_rmsnorm_f32") != hipSuccess) return 4;
    if (hipModuleGetFunction(&fn_sdpa,  mod, "sdpa_f32")               != hipSuccess) return 4;

    /* Upload inputs+weights. */
    int qkv_dim = 3 * dim;
    hipDeviceptr_t d_xs = hip_upload_raw(x_s, (size_t)N_s * dim * sizeof(float));
    hipDeviceptr_t d_xp = hip_upload_raw(x_p, (size_t)N_p * dim * sizeof(float));
    hipDeviceptr_t d_sqkvw = hip_upload_raw(sw_s.qkv_w, (size_t)qkv_dim * dim * sizeof(float));
    hipDeviceptr_t d_sqkvb = hip_upload_raw(sw_s.qkv_b, (size_t)qkv_dim * sizeof(float));
    hipDeviceptr_t d_pqkvw = hip_upload_raw(sw_p.qkv_w, (size_t)qkv_dim * dim * sizeof(float));
    hipDeviceptr_t d_pqkvb = hip_upload_raw(sw_p.qkv_b, (size_t)qkv_dim * sizeof(float));
    hipDeviceptr_t d_sgq   = hip_upload_raw(sw_s.gamma_q, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_sgk   = hip_upload_raw(sw_s.gamma_k, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_pgq   = hip_upload_raw(sw_p.gamma_q, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_pgk   = hip_upload_raw(sw_p.gamma_k, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_soutw = hip_upload_raw(sw_s.out_w, (size_t)dim * dim * sizeof(float));
    hipDeviceptr_t d_soutb = hip_upload_raw(sw_s.out_b, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_poutw = hip_upload_raw(sw_p.out_w, (size_t)dim * dim * sizeof(float));
    hipDeviceptr_t d_poutb = hip_upload_raw(sw_p.out_b, (size_t)dim * sizeof(float));

    hipDeviceptr_t d_qkvs = 0, d_qkvp = 0;
    hipDeviceptr_t d_qs = 0, d_ks = 0, d_vs = 0;
    hipDeviceptr_t d_qp = 0, d_kp = 0, d_vp = 0;
    hipDeviceptr_t d_kpkv = 0, d_vpkv = 0;
    hipDeviceptr_t d_hs = 0, d_hp = 0, d_os = 0, d_op = 0;
    hipMalloc(&d_qkvs, (size_t)N_s * qkv_dim * sizeof(float));
    hipMalloc(&d_qkvp, (size_t)N_p * qkv_dim * sizeof(float));
    hipMalloc(&d_qs, (size_t)N_s * dim * sizeof(float));
    hipMalloc(&d_ks, (size_t)N_s * dim * sizeof(float));
    hipMalloc(&d_vs, (size_t)N_s * dim * sizeof(float));
    hipMalloc(&d_qp, (size_t)N_p * dim * sizeof(float));
    hipMalloc(&d_kp, (size_t)N_p * dim * sizeof(float));
    hipMalloc(&d_vp, (size_t)N_p * dim * sizeof(float));
    hipMalloc(&d_kpkv, (size_t)N_kv * dim * sizeof(float));
    hipMalloc(&d_vpkv, (size_t)N_kv * dim * sizeof(float));
    hipMalloc(&d_hs, (size_t)N_s * dim * sizeof(float));
    hipMalloc(&d_hp, (size_t)N_p * dim * sizeof(float));
    hipMalloc(&d_os, (size_t)N_s * dim * sizeof(float));
    hipMalloc(&d_op, (size_t)N_p * dim * sizeof(float));

    /* qkv gemms. */
    {
        unsigned gx = (N_s + 15) / 16, gy = (qkv_dim + 15) / 16;
        void *args[] = { &d_qkvs, &d_xs, &d_sqkvw, &d_sqkvb, &N_s, &dim, &qkv_dim };
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return 5;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (qkv_dim + 15) / 16;
        void *args[] = { &d_qkvp, &d_xp, &d_pqkvw, &d_pqkvb, &N_p, &dim, &qkv_dim };
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return 5;
    }
    /* split. */
    {
        unsigned grid = (unsigned)((N_s * dim + 255) / 256);
        void *args[] = { &d_qs, &d_ks, &d_vs, &d_qkvs, &N_s, &dim };
        if (hipModuleLaunchKernel(fn_split, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return 5;
    }
    {
        unsigned grid = (unsigned)((N_p * dim + 255) / 256);
        void *args[] = { &d_qp, &d_kp, &d_vp, &d_qkvp, &N_p, &dim };
        if (hipModuleLaunchKernel(fn_split, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return 5;
    }
    /* multi-head RMS norm on q and k for both streams. */
    unsigned threads = 64;
    size_t smem = threads * sizeof(float);
    int stride = dim;
    {
        void *args[] = { &d_qs, &d_sgq, &N_s, &H, &D_h, &stride };
        if (hipModuleLaunchKernel(fn_rms, (unsigned)H, (unsigned)N_s, 1, threads, 1, 1, (unsigned)smem, 0, args, NULL) != hipSuccess) return 5;
    }
    {
        void *args[] = { &d_ks, &d_sgk, &N_s, &H, &D_h, &stride };
        if (hipModuleLaunchKernel(fn_rms, (unsigned)H, (unsigned)N_s, 1, threads, 1, 1, (unsigned)smem, 0, args, NULL) != hipSuccess) return 5;
    }
    {
        void *args[] = { &d_qp, &d_pgq, &N_p, &H, &D_h, &stride };
        if (hipModuleLaunchKernel(fn_rms, (unsigned)H, (unsigned)N_p, 1, threads, 1, 1, (unsigned)smem, 0, args, NULL) != hipSuccess) return 5;
    }
    {
        void *args[] = { &d_kp, &d_pgk, &N_p, &H, &D_h, &stride };
        if (hipModuleLaunchKernel(fn_rms, (unsigned)H, (unsigned)N_p, 1, threads, 1, 1, (unsigned)smem, 0, args, NULL) != hipSuccess) return 5;
    }
    /* concat pose+shape K and V via DtoD. */
    hipMemcpyDtoD(d_kpkv,                                       d_kp, (size_t)N_p * dim * sizeof(float));
    hipMemcpyDtoD(d_kpkv + (size_t)N_p * dim * sizeof(float),   d_ks, (size_t)N_s * dim * sizeof(float));
    hipMemcpyDtoD(d_vpkv,                                       d_vp, (size_t)N_p * dim * sizeof(float));
    hipMemcpyDtoD(d_vpkv + (size_t)N_p * dim * sizeof(float),   d_vs, (size_t)N_s * dim * sizeof(float));

    /* SDPA: shape (Q=q_s, KV=k_s/v_s, N_q=N_s, N_k=N_s); pose (Q=q_p, KV=concat). */
    unsigned sdpa_threads = 256;
    {
        size_t sm = (sdpa_threads + (size_t)N_s) * sizeof(float);
        void *args[] = { &d_hs, &d_qs, &d_ks, &d_vs, &N_s, &N_s, &H, &D_h, &scale };
        if (hipModuleLaunchKernel(fn_sdpa, (unsigned)N_s, (unsigned)H, 1,
                           sdpa_threads, 1, 1, (unsigned)sm, 0, args, NULL) != hipSuccess) return 5;
    }
    {
        size_t sm = (sdpa_threads + (size_t)N_kv) * sizeof(float);
        void *args[] = { &d_hp, &d_qp, &d_kpkv, &d_vpkv, &N_p, &N_kv, &H, &D_h, &scale };
        if (hipModuleLaunchKernel(fn_sdpa, (unsigned)N_p, (unsigned)H, 1,
                           sdpa_threads, 1, 1, (unsigned)sm, 0, args, NULL) != hipSuccess) return 5;
    }
    /* out_proj. */
    {
        unsigned gx = (N_s + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_os, &d_hs, &d_soutw, &d_soutb, &N_s, &dim, &dim };
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return 5;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_op, &d_hp, &d_poutw, &d_poutb, &N_p, &dim, &dim };
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return 5;
    }
    hipDeviceSynchronize();

    float *o_s_dst = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *o_p_dst = (float *)malloc((size_t)N_p * dim * sizeof(float));
    hipMemcpyDtoH(o_s_dst, d_os, (size_t)N_s * dim * sizeof(float));
    hipMemcpyDtoH(o_p_dst, d_op, (size_t)N_p * dim * sizeof(float));

    double mean_s = 0.0, mean_p = 0.0;
    float mx_s = max_abs(o_s_dst, o_s_ref, (size_t)N_s * dim, &mean_s);
    float mx_p = max_abs(o_p_dst, o_p_ref, (size_t)N_p * dim, &mean_p);
    int ok = (mx_s <= threshold) && (mx_p <= threshold);

    fprintf(stderr,
        "[test_mot_self_attn] N_s=%d N_p=%d H=%d D_h=%d  shape max_abs=%.4g (mean %.4g)  pose max_abs=%.4g (mean %.4g)  %s (threshold %.1g)\n",
        N_s, N_p, H, D_h, (double)mx_s, mean_s, (double)mx_p, mean_p,
        ok ? "OK" : "FAIL", (double)threshold);

    free(x_s); free(x_p);
    free(o_s_ref); free(o_p_ref); free(o_s_dst); free(o_p_dst);
    free_stream(&sw_s); free_stream(&sw_p);
    hipFree(d_xs); hipFree(d_xp);
    hipFree(d_sqkvw); hipFree(d_sqkvb); hipFree(d_pqkvw); hipFree(d_pqkvb);
    hipFree(d_sgq); hipFree(d_sgk); hipFree(d_pgq); hipFree(d_pgk);
    hipFree(d_soutw); hipFree(d_soutb); hipFree(d_poutw); hipFree(d_poutb);
    hipFree(d_qkvs); hipFree(d_qkvp);
    hipFree(d_qs); hipFree(d_ks); hipFree(d_vs);
    hipFree(d_qp); hipFree(d_kp); hipFree(d_vp);
    hipFree(d_kpkv); hipFree(d_vpkv);
    hipFree(d_hs); hipFree(d_hp); hipFree(d_os); hipFree(d_op);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
