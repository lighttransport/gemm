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
static int gxattn(hipFunction_t fn_gemm, hipFunction_t fn_split, hipFunction_t fn_sdpa,
                  const float *h, int N,
                  hipDeviceptr_t d_kv_pre,    /* nullable: if nonzero, reuse this kv buffer (already gemmed for this stream) */
                  hipDeviceptr_t d_qw, hipDeviceptr_t d_qb,
                  hipDeviceptr_t d_kvw, hipDeviceptr_t d_kvb,
                  hipDeviceptr_t d_outw, hipDeviceptr_t d_outb,
                  hipDeviceptr_t d_cond, int N_c,
                  int dim, int H, int D_h, float scale,
                  float *out)
{
    (void)d_kv_pre;
    int kv_dim = 2 * dim;
    hipDeviceptr_t d_h = hip_upload_raw(h, (size_t)N * dim * sizeof(float));
    hipDeviceptr_t d_q = 0, d_kv = 0, d_K = 0, d_V = 0, d_O = 0, d_out = 0;
    hipMalloc(&d_q,   (size_t)N   * dim    * sizeof(float));
    hipMalloc(&d_kv,  (size_t)N_c * kv_dim * sizeof(float));
    hipMalloc(&d_K,   (size_t)N_c * dim    * sizeof(float));
    hipMalloc(&d_V,   (size_t)N_c * dim    * sizeof(float));
    hipMalloc(&d_O,   (size_t)N   * dim    * sizeof(float));
    hipMalloc(&d_out, (size_t)N   * dim    * sizeof(float));

    /* xa_q gemm. */
    {
        unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_q, &d_h, &d_qw, &d_qb, &N, &dim, &dim };
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return 1;
    }
    /* xa_kv gemm. */
    {
        unsigned gx = (N_c + 15) / 16, gy = (kv_dim + 15) / 16;
        void *args[] = { &d_kv, &d_cond, &d_kvw, &d_kvb, &N_c, &dim, &kv_dim };
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return 1;
    }
    /* kv_split. */
    {
        unsigned grid = (unsigned)((N_c * dim + 255) / 256);
        void *args[] = { &d_K, &d_V, &d_kv, &N_c, &dim };
        if (hipModuleLaunchKernel(fn_split, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return 1;
    }
    /* sdpa. */
    {
        unsigned threads = 256;
        size_t sm = (threads + (size_t)N_c) * sizeof(float);
        void *args[] = { &d_O, &d_q, &d_K, &d_V, &N, &N_c, &H, &D_h, &scale };
        if (hipModuleLaunchKernel(fn_sdpa, (unsigned)N, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, args, NULL) != hipSuccess) return 1;
    }
    /* xa_out. */
    {
        unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_out, &d_O, &d_outw, &d_outb, &N, &dim, &dim };
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return 1;
    }
    hipDeviceSynchronize();
    hipMemcpyDtoH(out, d_out, (size_t)N * dim * sizeof(float));
    hipFree(d_h); hipFree(d_q); hipFree(d_kv);
    hipFree(d_K); hipFree(d_V); hipFree(d_O); hipFree(d_out);
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
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0; if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    if (hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose, "test_mot_cross_attn") < 0) return 4;
    hipFunction_t fn_gemm = 0, fn_split = 0, fn_sdpa = 0;
    if (hipModuleGetFunction(&fn_gemm,  mod, "gemm_f32_bias") != hipSuccess) return 4;
    if (hipModuleGetFunction(&fn_split, mod, "kv_split_f32")  != hipSuccess) return 4;
    if (hipModuleGetFunction(&fn_sdpa,  mod, "sdpa_f32")      != hipSuccess) return 4;

    hipDeviceptr_t d_cond = hip_upload_raw(cond, (size_t)N_c * dim * sizeof(float));
    hipDeviceptr_t d_sqw  = hip_upload_raw(xs.q_w,   (size_t)dim * dim * sizeof(float));
    hipDeviceptr_t d_sqb  = hip_upload_raw(xs.q_b,   (size_t)dim * sizeof(float));
    hipDeviceptr_t d_skvw = hip_upload_raw(xs.kv_w,  (size_t)2 * dim * dim * sizeof(float));
    hipDeviceptr_t d_skvb = hip_upload_raw(xs.kv_b,  (size_t)2 * dim * sizeof(float));
    hipDeviceptr_t d_sow  = hip_upload_raw(xs.out_w, (size_t)dim * dim * sizeof(float));
    hipDeviceptr_t d_sob  = hip_upload_raw(xs.out_b, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_pqw  = hip_upload_raw(xp.q_w,   (size_t)dim * dim * sizeof(float));
    hipDeviceptr_t d_pqb  = hip_upload_raw(xp.q_b,   (size_t)dim * sizeof(float));
    hipDeviceptr_t d_pkvw = hip_upload_raw(xp.kv_w,  (size_t)2 * dim * dim * sizeof(float));
    hipDeviceptr_t d_pkvb = hip_upload_raw(xp.kv_b,  (size_t)2 * dim * sizeof(float));
    hipDeviceptr_t d_pow  = hip_upload_raw(xp.out_w, (size_t)dim * dim * sizeof(float));
    hipDeviceptr_t d_pob  = hip_upload_raw(xp.out_b, (size_t)dim * sizeof(float));

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
    hipFree(d_cond);
    hipFree(d_sqw); hipFree(d_sqb); hipFree(d_skvw); hipFree(d_skvb);
    hipFree(d_sow); hipFree(d_sob);
    hipFree(d_pqw); hipFree(d_pqb); hipFree(d_pkvw); hipFree(d_pkvb);
    hipFree(d_pow); hipFree(d_pob);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
