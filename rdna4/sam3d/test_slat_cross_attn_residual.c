/*
 * test_slat_cross_attn_residual — Phase 5b.10 standalone microbench.
 *
 * Composes the SLAT transformer cross-attention residual path:
 *   affine LN -> Q projection, cond KV projection -> KV split ->
 *   cross SDPA -> output projection -> residual add.
 *
 * Usage:
 *   ./test_slat_cross_attn_residual [--N 128] [--Nc 512] [--dim 1024] [--H 16] [--threshold 1e-3] [--repeat 10] [-v]
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
#include <time.h>

static float urand(uint32_t *state)
{
    *state = (*state) * 1664525u + 1013904223u;
    return (float)((*state) >> 8) / (float)(1u << 24);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static float max_abs(const float *a, const float *b, size_t n, double *mean_out)
{
    double sum = 0.0;
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    if (mean_out) *mean_out = sum / (n > 0 ? n : 1);
    return mx;
}

static void hln_affine(float *out, const float *in,
                       const float *gamma, const float *beta,
                       int N, int dim, float eps)
{
    for (int n = 0; n < N; n++) {
        const float *x = in + (size_t)n * dim;
        float *y = out + (size_t)n * dim;
        float mean = 0.0f;
        for (int c = 0; c < dim; c++) mean += x[c];
        mean /= (float)dim;
        float var = 0.0f;
        for (int c = 0; c < dim; c++) { float d = x[c] - mean; var += d * d; }
        var /= (float)dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < dim; c++) y[c] = (x[c] - mean) * inv * gamma[c] + beta[c];
    }
}

static void hgemm(float *Y, const float *X, const float *W, const float *b,
                  int N, int D_out, int D_in)
{
    for (int n = 0; n < N; n++) {
        const float *xr = X + (size_t)n * D_in;
        for (int d = 0; d < D_out; d++) {
            const float *wr = W + (size_t)d * D_in;
            float acc = b ? b[d] : 0.0f;
            for (int k = 0; k < D_in; k++) acc += wr[k] * xr[k];
            Y[(size_t)n * D_out + d] = acc;
        }
    }
}

static void hkv_split(const float *kv, int Nc, int dim, float *K, float *V)
{
    for (int n = 0; n < Nc; n++) {
        const float *row = kv + (size_t)n * 2 * dim;
        memcpy(K + (size_t)n * dim, row, (size_t)dim * sizeof(float));
        memcpy(V + (size_t)n * dim, row + dim, (size_t)dim * sizeof(float));
    }
}

static void hsdpa(float *out,
                  const float *q, const float *K, const float *V,
                  int N, int Nc, int H, int D_h, float scale)
{
    int E = H * D_h;
    double *scores = (double *)malloc((size_t)Nc * sizeof(double));
    for (int nq = 0; nq < N; nq++) {
        for (int h = 0; h < H; h++) {
            const float *qv = q + (size_t)nq * E + (size_t)h * D_h;
            double mx = -1.0e300;
            for (int nk = 0; nk < Nc; nk++) {
                const float *kv = K + (size_t)nk * E + (size_t)h * D_h;
                double s = 0.0;
                for (int d = 0; d < D_h; d++) s += (double)qv[d] * (double)kv[d];
                s *= (double)scale;
                scores[nk] = s;
                if (s > mx) mx = s;
            }
            double sum = 0.0;
            for (int nk = 0; nk < Nc; nk++) {
                scores[nk] = exp(scores[nk] - mx);
                sum += scores[nk];
            }
            double inv = 1.0 / sum;
            for (int d = 0; d < D_h; d++) {
                double acc = 0.0;
                for (int nk = 0; nk < Nc; nk++)
                    acc += scores[nk] * (double)V[(size_t)nk * E + (size_t)h * D_h + d];
                out[(size_t)nq * E + (size_t)h * D_h + d] = (float)(acc * inv);
            }
        }
    }
    free(scores);
}

int main(int argc, char **argv)
{
    int N = 128;
    int Nc = 512;
    int dim = 1024;
    int H = 16;
    int repeat = 10;
    float threshold = 1e-3f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(a, "--Nc")        && i + 1 < argc) Nc = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(a, "--H")         && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat")    && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (N <= 0 || Nc <= 0 || dim <= 0 || H <= 0 || dim % H != 0) return 2;
    int D_h = dim / H;
    int kv_dim = 2 * dim;
    float attn_scale = 1.0f / sqrtf((float)D_h);

    size_t XD = (size_t)N * dim;
    size_t CD = (size_t)Nc * dim;
    size_t CKV = (size_t)Nc * kv_dim;
    float *x = (float *)malloc(XD * sizeof(float));
    float *cond = (float *)malloc(CD * sizeof(float));
    float *ln_w = (float *)malloc((size_t)dim * sizeof(float));
    float *ln_b = (float *)malloc((size_t)dim * sizeof(float));
    float *q_w = (float *)malloc((size_t)dim * dim * sizeof(float));
    float *q_b = (float *)malloc((size_t)dim * sizeof(float));
    float *kv_w = (float *)malloc((size_t)kv_dim * dim * sizeof(float));
    float *kv_b = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *out_w = (float *)malloc((size_t)dim * dim * sizeof(float));
    float *out_b = (float *)malloc((size_t)dim * sizeof(float));
    float *h = (float *)malloc(XD * sizeof(float));
    float *q = (float *)malloc(XD * sizeof(float));
    float *kv = (float *)malloc(CKV * sizeof(float));
    float *K = (float *)malloc(CD * sizeof(float));
    float *V = (float *)malloc(CD * sizeof(float));
    float *xa = (float *)malloc(XD * sizeof(float));
    float *proj = (float *)malloc(XD * sizeof(float));
    float *ref = (float *)malloc(XD * sizeof(float));
    float *dst = (float *)malloc(XD * sizeof(float));
    if (!x || !cond || !ln_w || !ln_b || !q_w || !q_b || !kv_w || !kv_b ||
        !out_w || !out_b || !h || !q || !kv || !K || !V || !xa || !proj || !ref || !dst)
        return 5;

    uint32_t rng = 0xC4055A10u;
    for (size_t i = 0; i < XD; i++) x[i] = (urand(&rng) * 2.0f - 1.0f);
    for (size_t i = 0; i < CD; i++) cond[i] = (urand(&rng) * 2.0f - 1.0f);
    for (int c = 0; c < dim; c++) {
        ln_w[c] = 0.9f + 0.2f * urand(&rng);
        ln_b[c] = (urand(&rng) * 2.0f - 1.0f) * 0.05f;
        q_b[c] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
        out_b[c] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    }
    float sw = 1.0f / sqrtf((float)dim);
    for (size_t i = 0; i < (size_t)dim * dim; i++) {
        q_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;
        out_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;
    }
    for (size_t i = 0; i < (size_t)kv_dim * dim; i++)
        kv_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;
    for (int i = 0; i < kv_dim; i++)
        kv_b[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;

    hln_affine(h, x, ln_w, ln_b, N, dim, 1e-6f);
    hgemm(q, h, q_w, q_b, N, dim, dim);
    hgemm(kv, cond, kv_w, kv_b, Nc, kv_dim, dim);
    hkv_split(kv, Nc, dim, K, V);
    hsdpa(xa, q, K, V, N, Nc, H, D_h, attn_scale);
    hgemm(proj, xa, out_w, out_b, N, dim, dim);
    for (size_t i = 0; i < XD; i++) ref[i] = x[i] + proj[i];

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0;
    if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    if (hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_slat_cross_attn_residual") < 0) return 4;
    hipFunction_t fn_ln = NULL, fn_gemm = NULL, fn_split = NULL, fn_sdpa = NULL, fn_resadd = NULL;
    if (hipModuleGetFunction(&fn_ln, mod, "layernorm_token_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&fn_split, mod, "kv_split_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_sdpa, mod, "sdpa_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_resadd, mod, "residual_add_f32") != hipSuccess)
        return 4;

    hipDeviceptr_t d_x0 = hip_upload_raw(x, XD * sizeof(float));
    hipDeviceptr_t d_x = hip_upload_raw(x, XD * sizeof(float));
    hipDeviceptr_t d_cond = hip_upload_raw(cond, CD * sizeof(float));
    hipDeviceptr_t d_ln_w = hip_upload_raw(ln_w, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_ln_b = hip_upload_raw(ln_b, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_q_w = hip_upload_raw(q_w, (size_t)dim * dim * sizeof(float));
    hipDeviceptr_t d_q_b = hip_upload_raw(q_b, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_kv_w = hip_upload_raw(kv_w, (size_t)kv_dim * dim * sizeof(float));
    hipDeviceptr_t d_kv_b = hip_upload_raw(kv_b, (size_t)kv_dim * sizeof(float));
    hipDeviceptr_t d_out_w = hip_upload_raw(out_w, (size_t)dim * dim * sizeof(float));
    hipDeviceptr_t d_out_b = hip_upload_raw(out_b, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_h = 0, d_q = 0, d_kv = 0, d_K = 0, d_V = 0, d_xa = 0, d_proj = 0;
    if (!d_x0 || !d_x || !d_cond || !d_ln_w || !d_ln_b || !d_q_w || !d_q_b ||
        !d_kv_w || !d_kv_b || !d_out_w || !d_out_b ||
        hipMalloc(&d_h, XD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_q, XD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_kv, CKV * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_K, CD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_V, CD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_xa, XD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_proj, XD * sizeof(float)) != hipSuccess)
        return 5;

    float eps = 1e-6f;
    int affine = 1;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    size_t sdpa_smem = (size_t)(256 + Nc) * sizeof(float);
    int n_elem = (int)XD;

#define RUN_XA() do { \
    if (hipMemcpyDtoD(d_x, d_x0, XD * sizeof(float)) != hipSuccess) return 5; \
    void *a_ln[] = { &d_h, &d_x, &d_ln_w, &d_ln_b, &N, &dim, &eps, &affine }; \
    if (hipModuleLaunchKernel(fn_ln, N, 1, 1, threads, 1, 1, (unsigned)ln_smem, 0, a_ln, NULL) != hipSuccess) return 6; \
    unsigned gxq = (N + 15) / 16, gyq = (dim + 15) / 16; \
    void *a_q[] = { &d_q, &d_h, &d_q_w, &d_q_b, &N, &dim, &dim }; \
    if (hipModuleLaunchKernel(fn_gemm, gxq, gyq, 1, 16, 16, 1, 0, 0, a_q, NULL) != hipSuccess) return 6; \
    unsigned gxkv = (Nc + 15) / 16, gykv = (kv_dim + 15) / 16; \
    void *a_kv[] = { &d_kv, &d_cond, &d_kv_w, &d_kv_b, &Nc, &dim, &kv_dim }; \
    if (hipModuleLaunchKernel(fn_gemm, gxkv, gykv, 1, 16, 16, 1, 0, 0, a_kv, NULL) != hipSuccess) return 6; \
    void *a_split[] = { &d_K, &d_V, &d_kv, &Nc, &dim }; \
    if (hipModuleLaunchKernel(fn_split, ((int)CD + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_split, NULL) != hipSuccess) return 6; \
    void *a_sdpa[] = { &d_xa, &d_q, &d_K, &d_V, &N, &Nc, &H, &D_h, &attn_scale }; \
    if (hipModuleLaunchKernel(fn_sdpa, N, H, 1, 256, 1, 1, (unsigned)sdpa_smem, 0, a_sdpa, NULL) != hipSuccess) return 6; \
    unsigned gxo = (N + 15) / 16, gyo = (dim + 15) / 16; \
    void *a_out[] = { &d_proj, &d_xa, &d_out_w, &d_out_b, &N, &dim, &dim }; \
    if (hipModuleLaunchKernel(fn_gemm, gxo, gyo, 1, 16, 16, 1, 0, 0, a_out, NULL) != hipSuccess) return 6; \
    void *a_res[] = { &d_x, &d_proj, &n_elem }; \
    if (hipModuleLaunchKernel(fn_resadd, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_res, NULL) != hipSuccess) return 6; \
} while (0)

    RUN_XA();
    hipDeviceSynchronize();
    hipMemcpyDtoH(dst, d_x, XD * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(dst, ref, XD, &mean);
    int ok = (mx <= threshold);

    double avg_ms = 0.0;
    if (repeat > 0) {
        hipDeviceSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) RUN_XA();
        hipDeviceSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }
#undef RUN_XA

    fprintf(stderr,
            "[test_slat_cross_attn_residual] N=%d Nc=%d dim=%d H=%d D_h=%d max_abs=%.4g mean_abs=%.4g avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, Nc, dim, H, D_h, (double)mx, mean, avg_ms, repeat,
            ok ? "OK" : "FAIL", (double)threshold);

    free(x); free(cond); free(ln_w); free(ln_b); free(q_w); free(q_b);
    free(kv_w); free(kv_b); free(out_w); free(out_b); free(h); free(q);
    free(kv); free(K); free(V); free(xa); free(proj); free(ref); free(dst);
    hipFree(d_x0); hipFree(d_x); hipFree(d_cond); hipFree(d_ln_w);
    hipFree(d_ln_b); hipFree(d_q_w); hipFree(d_q_b); hipFree(d_kv_w);
    hipFree(d_kv_b); hipFree(d_out_w); hipFree(d_out_b); hipFree(d_h);
    hipFree(d_q); hipFree(d_kv); hipFree(d_K); hipFree(d_V);
    hipFree(d_xa); hipFree(d_proj);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
