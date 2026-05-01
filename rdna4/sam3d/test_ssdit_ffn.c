/*
 * test_ssdit_ffn — Phase 2c.7 standalone microbench.
 *
 * Validates the SS Flow DiT per-stream FFN end-to-end on-device against
 * host reference `ssdit_mlp_stream` (sam3d_ss_flow_dit.h):
 *
 *   h1 = gemm_f32_bias(x,  fc1_w, fc1_b)        [N, mlp_hidden]
 *        gelu_tanh_inplace_f32(h1)              (NEW: tanh-approx, NOT exact erf)
 *   y  = gemm_f32_bias(h1, fc2_w, fc2_b)        [N, dim]
 *
 * SS DiT uses tanh-approx GELU (`ssdit_gelu_tanh_inplace`). The
 * existing `gelu_inplace_f32` is the exact erf variant — the new
 * `gelu_tanh_inplace_f32` mirrors the host reference bit-for-bit.
 *
 * Random weights ~ 1/sqrt(D); inputs ~ unit variance; mlp_hidden=4*D
 * (production SS DiT ratio); threshold accounts for tanhf intrinsic
 * drift propagated through 2 gemms.
 *
 * Usage:
 *   ./test_ssdit_ffn [--n 512] [--dim 1024] [--ratio 4] [--threshold 5e-5] [-v]
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
static void hgelu_tanh(float *x, int n) {
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float u = k * (v + c * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(u));
    }
}

int main(int argc, char **argv)
{
    int   N         = 512;
    int   dim       = 1024;
    int   ratio     = 4;
    float threshold = 5e-5f;
    int   verbose   = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--n")         && i+1 < argc) N         = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--ratio")     && i+1 < argc) ratio     = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                        verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    int H_dim = ratio * dim;

    uint32_t rng = 0xC0FFEEu;
    float *x      = (float *)malloc((size_t)N * dim * sizeof(float));
    float *fc1_w  = (float *)malloc((size_t)H_dim * dim * sizeof(float));
    float *fc1_b  = (float *)malloc((size_t)H_dim * sizeof(float));
    float *fc2_w  = (float *)malloc((size_t)dim * H_dim * sizeof(float));
    float *fc2_b  = (float *)malloc((size_t)dim * sizeof(float));
    for (int i = 0; i < N * dim; i++) x[i] = urand(&rng) * 2.0f - 1.0f;
    float s1 = 1.0f / sqrtf((float)dim);
    float s2 = 1.0f / sqrtf((float)H_dim);
    for (int i = 0; i < H_dim * dim; i++) fc1_w[i] = (urand(&rng) * 2.0f - 1.0f) * s1;
    for (int i = 0; i < H_dim;       i++) fc1_b[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    for (int i = 0; i < dim * H_dim; i++) fc2_w[i] = (urand(&rng) * 2.0f - 1.0f) * s2;
    for (int i = 0; i < dim;         i++) fc2_b[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;

    /* Host ref. */
    float *h1   = (float *)malloc((size_t)N * H_dim * sizeof(float));
    float *href = (float *)malloc((size_t)N * dim   * sizeof(float));
    hgemm(h1, x, fc1_w, fc1_b, N, H_dim, dim);
    hgelu_tanh(h1, N * H_dim);
    hgemm(href, h1, fc2_w, fc2_b, N, dim, H_dim);

    /* Device. */
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0; if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    if (hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose, "test_ssdit_ffn") < 0) return 4;
    hipFunction_t fn_gemm = 0, fn_gelu = 0;
    if (hipModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias")          != hipSuccess) return 4;
    if (hipModuleGetFunction(&fn_gelu, mod, "gelu_tanh_inplace_f32")  != hipSuccess) return 4;

    hipDeviceptr_t d_x   = hip_upload_raw(x,     (size_t)N * dim * sizeof(float));
    hipDeviceptr_t d_w1  = hip_upload_raw(fc1_w, (size_t)H_dim * dim * sizeof(float));
    hipDeviceptr_t d_b1  = hip_upload_raw(fc1_b, (size_t)H_dim * sizeof(float));
    hipDeviceptr_t d_w2  = hip_upload_raw(fc2_w, (size_t)dim * H_dim * sizeof(float));
    hipDeviceptr_t d_b2  = hip_upload_raw(fc2_b, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_h1 = 0, d_out = 0;
    hipMalloc(&d_h1,  (size_t)N * H_dim * sizeof(float));
    hipMalloc(&d_out, (size_t)N * dim   * sizeof(float));

    /* fc1. */
    {
        unsigned gx = (N + 15) / 16, gy = (H_dim + 15) / 16;
        void *args[] = { &d_h1, &d_x, &d_w1, &d_b1, &N, &dim, &H_dim };
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return 5;
    }
    /* gelu. */
    {
        int total = N * H_dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &d_h1, &total };
        if (hipModuleLaunchKernel(fn_gelu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return 5;
    }
    /* fc2. */
    {
        unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_out, &d_h1, &d_w2, &d_b2, &N, &H_dim, &dim };
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return 5;
    }
    hipDeviceSynchronize();

    float *hdst = (float *)malloc((size_t)N * dim * sizeof(float));
    hipMemcpyDtoH(hdst, d_out, (size_t)N * dim * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(hdst, href, (size_t)N * dim, &mean);
    int ok = (mx <= threshold);

    fprintf(stderr,
        "[test_ssdit_ffn] N=%d dim=%d mlp_hidden=%d  max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        N, dim, H_dim, (double)mx, mean, ok ? "OK" : "FAIL", (double)threshold);

    free(x); free(fc1_w); free(fc1_b); free(fc2_w); free(fc2_b);
    free(h1); free(href); free(hdst);
    hipFree(d_x); hipFree(d_w1); hipFree(d_b1); hipFree(d_w2); hipFree(d_b2);
    hipFree(d_h1); hipFree(d_out);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
