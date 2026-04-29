/*
 * test_modulated_ln — Phase 2c.3 standalone microbench.
 *
 * Validates `modulated_ln_f32` against the host reference
 * `ssdit_layernorm` (sam3d_ss_flow_dit.h) with w/b=NULL and non-NULL
 * shift/scale — i.e. SS DiT's norm1/norm3 sites:
 *
 *   y[t, c] = (x[t, c] - mean_t) * rsqrt(var_t + eps) * (1 + scale[c])
 *           + shift[c]
 *
 * Random x with unit variance, shift small, scale ~ 0 ± 0.1 to keep
 * the (1 + scale) factor in a realistic AdaLN range. Threshold accounts
 * for warp-reduce vs sequential mean/var summation order.
 *
 * Usage:
 *   ./test_modulated_ln [--ntok 4096] [--dim 1024] [--threshold 5e-5] [-v]
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

static void host_modulated_ln(float *out, const float *in,
                              const float *shift, const float *scale,
                              int n_tok, int dim, float eps)
{
    for (int t = 0; t < n_tok; t++) {
        const float *x = in  + (size_t)t * dim;
        float       *y = out + (size_t)t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= (float)dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= (float)dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++) {
            float v = (x[i] - mean) * inv;
            y[i] = v * (1.0f + scale[i]) + shift[i];
        }
    }
}

int main(int argc, char **argv)
{
    int   n_tok     = 4096;
    int   dim       = 1024;
    float threshold = 5e-5f;
    int   verbose   = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--ntok")      && i+1 < argc) n_tok     = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                        verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    uint32_t rng = 0xC0FFEEu;
    size_t Nx = (size_t)n_tok * dim;
    float *x     = (float *)malloc(Nx * sizeof(float));
    float *shift = (float *)malloc((size_t)dim * sizeof(float));
    float *scale = (float *)malloc((size_t)dim * sizeof(float));
    for (size_t i = 0; i < Nx; i++) x[i] = urand(&rng) * 2.0f - 1.0f;
    for (int i = 0; i < dim; i++) shift[i] = (urand(&rng) * 2.0f - 1.0f) * 0.05f;
    for (int i = 0; i < dim; i++) scale[i] = (urand(&rng) * 2.0f - 1.0f) * 0.10f;

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0; if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose, "test_modulated_ln") < 0) return 4;
    CUfunction fn = 0;
    if (cuModuleGetFunction(&fn, mod, "modulated_ln_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup modulated_ln_f32 failed\n"); return 4;
    }

    CUdeviceptr d_in    = cu_upload_raw(x,     Nx * sizeof(float));
    CUdeviceptr d_shift = cu_upload_raw(shift, (size_t)dim * sizeof(float));
    CUdeviceptr d_scale = cu_upload_raw(scale, (size_t)dim * sizeof(float));
    CUdeviceptr d_out = 0;
    cuMemAlloc(&d_out, Nx * sizeof(float));

    float eps = 1e-6f;
    unsigned threads = 256;
    void *args[] = { &d_out, &d_in, &d_shift, &d_scale, &n_tok, &dim, &eps };
    size_t smem = 2 * threads * sizeof(float);
    if (cuLaunchKernel(fn, (unsigned)n_tok, 1, 1, threads, 1, 1,
                       (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "launch failed\n"); return 5;
    }
    cuCtxSynchronize();

    float *h_dst = (float *)malloc(Nx * sizeof(float));
    cuMemcpyDtoH(h_dst, d_out, Nx * sizeof(float));

    float *h_ref = (float *)malloc(Nx * sizeof(float));
    host_modulated_ln(h_ref, x, shift, scale, n_tok, dim, eps);

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, Nx, &mean);
    int ok = (mx <= threshold);

    fprintf(stderr,
        "[test_modulated_ln] n_tok=%d dim=%d  max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        n_tok, dim, (double)mx, mean, ok ? "OK" : "FAIL", (double)threshold);

    free(x); free(shift); free(scale); free(h_dst); free(h_ref);
    cuMemFree(d_in); cuMemFree(d_shift); cuMemFree(d_scale); cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
