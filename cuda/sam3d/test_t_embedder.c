/*
 * test_t_embedder — Phase 2c.1 standalone microbench.
 *
 * Composes the SS Flow DiT TimestepEmbedder MLP on-device:
 *   sinusoidal_embed(t) → gemm_f32_bias(freq→D) → silu_inplace
 *                       → gemm_f32_bias(D→D)
 *
 * Random weights (LCG, seed 0xC0FFEE), host ref mirrors `ssdit_time_mlp`
 * (sam3d_ss_flow_dit.h). Threshold accounts for `__expf`/`__sincosf` vs
 * libm drift propagated through 2 gemms + silu near the [0, 1000]
 * post-`time_scale` t range.
 *
 * Usage:
 *   ./test_t_embedder [--freq 256] [--dim 1024] [--threshold 1e-3] [-v]
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

/* host: sinusoidal embed + (W1,b1) gemm + silu + (W2,b2) gemm. */
static void host_t_emb(float *out, float t,
                       const float *W1, const float *b1,
                       const float *W2, const float *b2,
                       int freq_dim, int dim)
{
    int half = freq_dim / 2;
    float *emb = (float *)malloc((size_t)freq_dim * sizeof(float));
    float *h1  = (float *)malloc((size_t)dim      * sizeof(float));
    float neg_log10k = -logf(10000.0f);
    for (int j = 0; j < half; j++) {
        float freq = expf(neg_log10k * (float)j / (float)half);
        float arg = t * freq;
        emb[j] = cosf(arg);
        emb[half + j] = sinf(arg);
    }
    for (int d = 0; d < dim; d++) {
        float acc = b1[d];
        const float *wr = W1 + (size_t)d * freq_dim;
        for (int k = 0; k < freq_dim; k++) acc += wr[k] * emb[k];
        h1[d] = acc / (1.0f + expf(-acc)) * 1.0f;  /* silu inline */
    }
    /* silu already applied above */
    for (int d = 0; d < dim; d++) {
        float acc = b2[d];
        const float *wr = W2 + (size_t)d * dim;
        for (int k = 0; k < dim; k++) acc += wr[k] * h1[k];
        out[d] = acc;
    }
    free(emb); free(h1);
}

int main(int argc, char **argv)
{
    int   freq_dim  = 256;
    int   dim       = 1024;
    float threshold = 1e-3f;
    int   verbose   = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--freq")      && i+1 < argc) freq_dim  = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                        verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    /* Random weights, scaled small to keep silu inputs reasonable. */
    uint32_t rng = 0xC0FFEEu;
    float *W1 = (float *)malloc((size_t)dim * freq_dim * sizeof(float));
    float *b1 = (float *)calloc((size_t)dim, sizeof(float));
    float *W2 = (float *)malloc((size_t)dim * dim * sizeof(float));
    float *b2 = (float *)calloc((size_t)dim, sizeof(float));
    float scale1 = 1.0f / sqrtf((float)freq_dim);
    float scale2 = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < dim * freq_dim; i++) W1[i] = (urand(&rng) * 2.0f - 1.0f) * scale1;
    for (int i = 0; i < dim;            i++) b1[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    for (int i = 0; i < dim * dim;      i++) W2[i] = (urand(&rng) * 2.0f - 1.0f) * scale2;
    for (int i = 0; i < dim;            i++) b2[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0; if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose, "test_t_embedder") < 0) return 4;
    CUfunction fn_ts = 0, fn_gemm = 0, fn_silu = 0;
    if (cuModuleGetFunction(&fn_ts,   mod, "timestep_embed_cossin_f32") != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias")             != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn_silu, mod, "silu_inplace_f32")          != CUDA_SUCCESS) return 4;

    CUdeviceptr d_W1 = cu_upload_raw(W1, (size_t)dim * freq_dim * sizeof(float));
    CUdeviceptr d_b1 = cu_upload_raw(b1, (size_t)dim * sizeof(float));
    CUdeviceptr d_W2 = cu_upload_raw(W2, (size_t)dim * dim * sizeof(float));
    CUdeviceptr d_b2 = cu_upload_raw(b2, (size_t)dim * sizeof(float));
    CUdeviceptr d_emb = 0, d_h1 = 0, d_out = 0;
    cuMemAlloc(&d_emb, (size_t)freq_dim * sizeof(float));
    cuMemAlloc(&d_h1,  (size_t)dim      * sizeof(float));
    cuMemAlloc(&d_out, (size_t)dim      * sizeof(float));

    static const float ts[] = { 0.0f, 1.0f, 17.5f, 125.0f, 500.0f, 999.99f };
    int n_ts = (int)(sizeof(ts) / sizeof(ts[0]));
    int half = freq_dim / 2;
    int N = 1;

    float *h_dst = (float *)malloc((size_t)dim * sizeof(float));
    float *h_ref = (float *)malloc((size_t)dim * sizeof(float));
    float worst_mx = 0.0f; double worst_mean = 0.0;
    int ok = 1;
    for (int k = 0; k < n_ts; k++) {
        float t = ts[k];
        /* Step 1: sinusoidal embed → d_emb [freq_dim]. */
        {
            unsigned grid = (unsigned)((half + 255) / 256);
            void *args[] = { &d_emb, &t, &freq_dim };
            if (cuLaunchKernel(fn_ts, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) { ok = 0; break; }
        }
        /* Step 2: h1 = W1 @ emb + b1, gemm grid (1, ceil(dim/16)). */
        {
            unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
            void *args[] = { &d_h1, &d_emb, &d_W1, &d_b1, &N, &freq_dim, &dim };
            if (cuLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) { ok = 0; break; }
        }
        /* Step 3: silu_inplace(h1). */
        {
            unsigned grid = (unsigned)((dim + 255) / 256);
            void *args[] = { &d_h1, &dim };
            if (cuLaunchKernel(fn_silu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) { ok = 0; break; }
        }
        /* Step 4: out = W2 @ h1 + b2. */
        {
            unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
            void *args[] = { &d_out, &d_h1, &d_W2, &d_b2, &N, &dim, &dim };
            if (cuLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) { ok = 0; break; }
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(h_dst, d_out, (size_t)dim * sizeof(float));
        host_t_emb(h_ref, t, W1, b1, W2, b2, freq_dim, dim);
        double mean = 0.0;
        float mx = max_abs(h_dst, h_ref, dim, &mean);
        if (verbose) fprintf(stderr, "  t=%g  max_abs=%.4g mean_abs=%.4g\n", t, (double)mx, mean);
        if (mx > worst_mx) { worst_mx = mx; worst_mean = mean; }
        if (mx > threshold) ok = 0;
    }
    fprintf(stderr,
        "[test_t_embedder] freq=%d dim=%d  worst over %d t-values: max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        freq_dim, dim, n_ts, (double)worst_mx, worst_mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(W1); free(b1); free(W2); free(b2);
    free(h_dst); free(h_ref);
    cuMemFree(d_W1); cuMemFree(d_b1); cuMemFree(d_W2); cuMemFree(d_b2);
    cuMemFree(d_emb); cuMemFree(d_h1); cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
