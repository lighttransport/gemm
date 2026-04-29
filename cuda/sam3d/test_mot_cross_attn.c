/*
 * test_mot_cross_attn — Phase 2c.6 standalone microbench.
 *
 * Validates SS DiT MOT cross-attention on-device against host reference
 * `ssdit_cross_attn` (sam3d_ss_flow_dit.h). Per stream:
 *
 *   q  = gemm_f32_bias(h,    xa_q_w,  xa_q_b)        [N, dim]
 *   kv = gemm_f32_bias(cond, xa_kv_w, xa_kv_b)       [N_c, 2*dim]
 *   K, V = kv_split_f32(kv)                          [N_c, dim] each
 *   O  = sdpa_f32(q, K, V, scale=1/sqrt(D_h))        [N, dim]
 *   out = gemm_f32_bias(O, xa_out_w, xa_out_b)       [N, dim]
 *
 * Cross-attn has NO q/k RMS norm (unlike self-attn). Tests both stream
 * sizes (shape N=512, pose N=4) sharing a single cond buffer (N_c=2740
 * — the production dino-only token count from gen_image_ref) — exactly
 * the production runner's call pattern.
 *
 * Random weights ~ N(0, 1/D); cond ~ unit; threshold accounts for
 * SDPA expf + reduction-order drift propagated through 3 gemms.
 *
 * Usage:
 *   ./test_mot_cross_attn [--ns 512] [--np 4] [--nc 2740]
 *                         [--heads 16] [--hd 64] [--threshold 5e-4] [-v]
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

typedef struct {
    float *q_w, *q_b;       /* [D, D], [D] */
    float *kv_w, *kv_b;     /* [2D, D], [2D] */
    float *out_w, *out_b;   /* [D, D], [D] */
} xa_w;
static void alloc_xa(xa_w *w, int dim, uint32_t *rng) {
    w->q_w   = (float *)malloc((size_t)dim * dim * sizeof(float));
    w->q_b   = (float *)malloc((size_t)dim * sizeof(float));
    w->kv_w  = (float *)malloc((size_t)2 * dim * dim * sizeof(float));
    w->kv_b  = (float *)malloc((size_t)2 * dim * sizeof(float));
    w->out_w = (float *)malloc((size_t)dim * dim * sizeof(float));
    w->out_b = (float *)malloc((size_t)dim * sizeof(float));
    float sw = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < dim * dim;     i++) w->q_w[i]   = (urand(rng) * 2.0f - 1.0f) * sw;
    for (int i = 0; i < dim;           i++) w->q_b[i]   = (urand(rng) * 2.0f - 1.0f) * 0.01f;
    for (int i = 0; i < 2 * dim * dim; i++) w->kv_w[i]  = (urand(rng) * 2.0f - 1.0f) * sw;
    for (int i = 0; i < 2 * dim;       i++) w->kv_b[i]  = (urand(rng) * 2.0f - 1.0f) * 0.01f;
    for (int i = 0; i < dim * dim;     i++) w->out_w[i] = (urand(rng) * 2.0f - 1.0f) * sw;
    for (int i = 0; i < dim;           i++) w->out_b[i] = (urand(rng) * 2.0f - 1.0f) * 0.01f;
}
static void free_xa(xa_w *w) {
    free(w->q_w); free(w->q_b); free(w->kv_w); free(w->kv_b); free(w->out_w); free(w->out_b);
}

/* Run one stream's cross-attn on host, returns out [N, dim]. */
static void hxattn(const xa_w *w, const float *h, int N,
                   const float *cond, int N_c,
                   int dim, int H, int D_h, float *out)
{
    float *q  = (float *)malloc((size_t)N * dim * sizeof(float));
    float *kv = (float *)malloc((size_t)N_c * 2 * dim * sizeof(float));
    float *K  = (float *)malloc((size_t)N_c * dim * sizeof(float));
    float *V  = (float *)malloc((size_t)N_c * dim * sizeof(float));
    float *O  = (float *)malloc((size_t)N * dim * sizeof(float));
    hgemm(q,  h,    w->q_w,  w->q_b,  N,   dim,     dim);
    hgemm(kv, cond, w->kv_w, w->kv_b, N_c, 2 * dim, dim);
    /* split interleaved [N_c, 2*dim] into K, V [N_c, dim] each. */
    for (int n = 0; n < N_c; n++) {
        memcpy(K + (size_t)n * dim, kv + (size_t)n * 2 * dim,         (size_t)dim * sizeof(float));
        memcpy(V + (size_t)n * dim, kv + (size_t)n * 2 * dim + dim,   (size_t)dim * sizeof(float));
    }
    float scale = 1.0f / sqrtf((float)D_h);
    hsdpa(O, q, K, V, N, N_c, H, D_h, scale);
    hgemm(out, O, w->out_w, w->out_b, N, dim, dim);
    free(q); free(kv); free(K); free(V); free(O);
}

/* GPU side per-stream cross-attn helper. */
static int gxattn(CUfunction fn_gemm, CUfunction fn_split, CUfunction fn_sdpa,
                  const float *h, int N,
                  CUdeviceptr d_kv_pre,    /* nullable: if nonzero, reuse this kv buffer (already gemmed for this stream) */
                  CUdeviceptr d_qw, CUdeviceptr d_qb,
                  CUdeviceptr d_kvw, CUdeviceptr d_kvb,
                  CUdeviceptr d_outw, CUdeviceptr d_outb,
                  CUdeviceptr d_cond, int N_c,
                  int dim, int H, int D_h, float scale,
                  float *out)
{
    (void)d_kv_pre;
    int kv_dim = 2 * dim;
    CUdeviceptr d_h = cu_upload_raw(h, (size_t)N * dim * sizeof(float));
    CUdeviceptr d_q = 0, d_kv = 0, d_K = 0, d_V = 0, d_O = 0, d_out = 0;
    cuMemAlloc(&d_q,   (size_t)N   * dim    * sizeof(float));
    cuMemAlloc(&d_kv,  (size_t)N_c * kv_dim * sizeof(float));
    cuMemAlloc(&d_K,   (size_t)N_c * dim    * sizeof(float));
    cuMemAlloc(&d_V,   (size_t)N_c * dim    * sizeof(float));
    cuMemAlloc(&d_O,   (size_t)N   * dim    * sizeof(float));
    cuMemAlloc(&d_out, (size_t)N   * dim    * sizeof(float));

    /* xa_q gemm. */
    {
        unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_q, &d_h, &d_qw, &d_qb, &N, &dim, &dim };
        if (cuLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 1;
    }
    /* xa_kv gemm. */
    {
        unsigned gx = (N_c + 15) / 16, gy = (kv_dim + 15) / 16;
        void *args[] = { &d_kv, &d_cond, &d_kvw, &d_kvb, &N_c, &dim, &kv_dim };
        if (cuLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 1;
    }
    /* kv_split. */
    {
        unsigned grid = (unsigned)((N_c * dim + 255) / 256);
        void *args[] = { &d_K, &d_V, &d_kv, &N_c, &dim };
        if (cuLaunchKernel(fn_split, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 1;
    }
    /* sdpa. */
    {
        unsigned threads = 256;
        size_t sm = (threads + (size_t)N_c) * sizeof(float);
        void *args[] = { &d_O, &d_q, &d_K, &d_V, &N, &N_c, &H, &D_h, &scale };
        if (cuLaunchKernel(fn_sdpa, (unsigned)N, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, args, NULL) != CUDA_SUCCESS) return 1;
    }
    /* xa_out. */
    {
        unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_out, &d_O, &d_outw, &d_outb, &N, &dim, &dim };
        if (cuLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 1;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(out, d_out, (size_t)N * dim * sizeof(float));
    cuMemFree(d_h); cuMemFree(d_q); cuMemFree(d_kv);
    cuMemFree(d_K); cuMemFree(d_V); cuMemFree(d_O); cuMemFree(d_out);
    return 0;
}

int main(int argc, char **argv)
{
    int   N_s = 512, N_p = 4, N_c = 2740, H = 16, D_h = 64;
    float threshold = 5e-4f;
    int   verbose   = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--ns")        && i+1 < argc) N_s = atoi(argv[++i]);
        else if (!strcmp(a, "--np")        && i+1 < argc) N_p = atoi(argv[++i]);
        else if (!strcmp(a, "--nc")        && i+1 < argc) N_c = atoi(argv[++i]);
        else if (!strcmp(a, "--heads")     && i+1 < argc) H   = atoi(argv[++i]);
        else if (!strcmp(a, "--hd")        && i+1 < argc) D_h = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                        verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    int dim = H * D_h;
    float scale = 1.0f / sqrtf((float)D_h);

    uint32_t rng = 0xC0FFEEu;
    float *h_s  = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *h_p  = (float *)malloc((size_t)N_p * dim * sizeof(float));
    float *cond = (float *)malloc((size_t)N_c * dim * sizeof(float));
    for (int i = 0; i < N_s * dim; i++) h_s[i]  = urand(&rng) * 2.0f - 1.0f;
    for (int i = 0; i < N_p * dim; i++) h_p[i]  = urand(&rng) * 2.0f - 1.0f;
    for (int i = 0; i < N_c * dim; i++) cond[i] = urand(&rng) * 2.0f - 1.0f;
    xa_w xs, xp;
    alloc_xa(&xs, dim, &rng);
    alloc_xa(&xp, dim, &rng);

    /* Host ref. */
    float *ref_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *ref_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    hxattn(&xs, h_s, N_s, cond, N_c, dim, H, D_h, ref_s);
    hxattn(&xp, h_p, N_p, cond, N_c, dim, H, D_h, ref_p);

    /* Device. */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0; if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose, "test_mot_cross_attn") < 0) return 4;
    CUfunction fn_gemm = 0, fn_split = 0, fn_sdpa = 0;
    if (cuModuleGetFunction(&fn_gemm,  mod, "gemm_f32_bias") != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn_split, mod, "kv_split_f32")  != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn_sdpa,  mod, "sdpa_f32")      != CUDA_SUCCESS) return 4;

    CUdeviceptr d_cond = cu_upload_raw(cond, (size_t)N_c * dim * sizeof(float));
    CUdeviceptr d_sqw  = cu_upload_raw(xs.q_w,   (size_t)dim * dim * sizeof(float));
    CUdeviceptr d_sqb  = cu_upload_raw(xs.q_b,   (size_t)dim * sizeof(float));
    CUdeviceptr d_skvw = cu_upload_raw(xs.kv_w,  (size_t)2 * dim * dim * sizeof(float));
    CUdeviceptr d_skvb = cu_upload_raw(xs.kv_b,  (size_t)2 * dim * sizeof(float));
    CUdeviceptr d_sow  = cu_upload_raw(xs.out_w, (size_t)dim * dim * sizeof(float));
    CUdeviceptr d_sob  = cu_upload_raw(xs.out_b, (size_t)dim * sizeof(float));
    CUdeviceptr d_pqw  = cu_upload_raw(xp.q_w,   (size_t)dim * dim * sizeof(float));
    CUdeviceptr d_pqb  = cu_upload_raw(xp.q_b,   (size_t)dim * sizeof(float));
    CUdeviceptr d_pkvw = cu_upload_raw(xp.kv_w,  (size_t)2 * dim * dim * sizeof(float));
    CUdeviceptr d_pkvb = cu_upload_raw(xp.kv_b,  (size_t)2 * dim * sizeof(float));
    CUdeviceptr d_pow  = cu_upload_raw(xp.out_w, (size_t)dim * dim * sizeof(float));
    CUdeviceptr d_pob  = cu_upload_raw(xp.out_b, (size_t)dim * sizeof(float));

    float *dst_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *dst_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    if (gxattn(fn_gemm, fn_split, fn_sdpa, h_s, N_s, 0,
               d_sqw, d_sqb, d_skvw, d_skvb, d_sow, d_sob,
               d_cond, N_c, dim, H, D_h, scale, dst_s) != 0) return 5;
    if (gxattn(fn_gemm, fn_split, fn_sdpa, h_p, N_p, 0,
               d_pqw, d_pqb, d_pkvw, d_pkvb, d_pow, d_pob,
               d_cond, N_c, dim, H, D_h, scale, dst_p) != 0) return 5;

    double mean_s = 0.0, mean_p = 0.0;
    float mx_s = max_abs(dst_s, ref_s, (size_t)N_s * dim, &mean_s);
    float mx_p = max_abs(dst_p, ref_p, (size_t)N_p * dim, &mean_p);
    int ok = (mx_s <= threshold) && (mx_p <= threshold);

    fprintf(stderr,
        "[test_mot_cross_attn] N_s=%d N_p=%d N_c=%d H=%d D_h=%d  "
        "shape max_abs=%.4g (mean %.4g)  pose max_abs=%.4g (mean %.4g)  %s (threshold %.1g)\n",
        N_s, N_p, N_c, H, D_h, (double)mx_s, mean_s, (double)mx_p, mean_p,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_s); free(h_p); free(cond);
    free(ref_s); free(ref_p); free(dst_s); free(dst_p);
    free_xa(&xs); free_xa(&xp);
    cuMemFree(d_cond);
    cuMemFree(d_sqw); cuMemFree(d_sqb); cuMemFree(d_skvw); cuMemFree(d_skvb);
    cuMemFree(d_sow); cuMemFree(d_sob);
    cuMemFree(d_pqw); cuMemFree(d_pqb); cuMemFree(d_pkvw); cuMemFree(d_pkvb);
    cuMemFree(d_pow); cuMemFree(d_pob);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
