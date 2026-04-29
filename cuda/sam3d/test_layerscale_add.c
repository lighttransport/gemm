/*
 * test_layerscale_add — Phase 1b.5 standalone microbench.
 *
 * Validates layerscale_add_f32 on DINOv2-L geometry [n_tok=1374, dim=1024].
 * Pure F32 mul-add, expected bit-exact (max_abs == 0).
 *
 * Usage:
 *   ./test_layerscale_add [--n_tok 1374] [--dim 1024]
 *                         [--threshold 1e-6] [-v]
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

static float urand(uint32_t *state) {
    *state = (*state) * 1664525u + 1013904223u;
    return (float)((*state) >> 8) / (float)(1u << 24);
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

int main(int argc, char **argv)
{
    int   n_tok     = 1374;
    int   dim       = 1024;
    float threshold = 1e-6f;
    int   verbose   = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--n_tok")     && i+1 < argc) n_tok     = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 3;
    }
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 3; }
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) { fprintf(stderr, "cuDeviceGet failed\n"); return 3; }
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed\n"); return 3;
    }

    CUmodule mod = 0;
    int sm = cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose,
                                "test_layerscale_add");
    if (sm < 0) { fprintf(stderr, "compile failed\n"); return 4; }
    CUfunction fn = NULL;
    if (cuModuleGetFunction(&fn, mod, "layerscale_add_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup layerscale_add_f32 failed\n"); return 4;
    }

    size_t n_act = (size_t)n_tok * dim;
    float *h_hidden_init = (float *)malloc(n_act * sizeof(float));
    float *h_proj        = (float *)malloc(n_act * sizeof(float));
    float *h_gamma       = (float *)malloc((size_t)dim * sizeof(float));
    float *h_ref         = (float *)malloc(n_act * sizeof(float));
    float *h_dst         = (float *)malloc(n_act * sizeof(float));
    if (!h_hidden_init || !h_proj || !h_gamma || !h_ref || !h_dst) {
        fprintf(stderr, "host alloc failed\n"); return 5;
    }
    uint32_t rng = 0xC0FFEEu;
    for (size_t i = 0; i < n_act; i++) h_hidden_init[i] = urand(&rng) * 2.0f - 1.0f;
    for (size_t i = 0; i < n_act; i++) h_proj[i]        = urand(&rng) * 2.0f - 1.0f;
    /* gamma init ~1.0 like real ls1/ls2 (small init, layers initialize at 1e-5). */
    for (int c = 0; c < dim; c++)      h_gamma[c]       = 1e-5f * (urand(&rng) * 2.0f - 1.0f) + 1.0f;

    /* Host reference: hidden_ref[t, c] = hidden_init[t, c] + proj[t, c] * gamma[c]. */
    for (int t = 0; t < n_tok; t++) {
        for (int c = 0; c < dim; c++) {
            size_t i = (size_t)t * dim + c;
            h_ref[i] = h_hidden_init[i] + h_proj[i] * h_gamma[c];
        }
    }

    CUdeviceptr d_hidden = cu_upload_raw(h_hidden_init, n_act * sizeof(float));
    CUdeviceptr d_proj   = cu_upload_raw(h_proj,        n_act * sizeof(float));
    CUdeviceptr d_gamma  = cu_upload_raw(h_gamma,       (size_t)dim * sizeof(float));

    int total = n_tok * dim;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    void *args[] = { &d_hidden, &d_proj, &d_gamma, &n_tok, &dim };
    if (cuLaunchKernel(fn,
                       blocks, 1, 1,
                       threads, 1, 1,
                       0, 0, args, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "launch failed\n"); return 6;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(h_dst, d_hidden, n_act * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_act, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_layerscale_add] n_tok=%d dim=%d  n_out=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        n_tok, dim, n_act, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_hidden_init); free(h_proj); free(h_gamma); free(h_ref); free(h_dst);
    cuMemFree(d_hidden); cuMemFree(d_proj); cuMemFree(d_gamma);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
