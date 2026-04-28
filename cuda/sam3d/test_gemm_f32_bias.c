/*
 * test_gemm_f32_bias — Phase 1b.4 (foundation) standalone microbench.
 *
 * Validates gemm_f32_bias on a DINOv2-L QKV shape: N=1374 (n_tokens),
 * D_in=1024 (dim), D_out=3072 (3 * dim). Diffs against a host
 * double-precision reference. Threshold accounts for the 1024-muladd
 * float-accumulation drift in the kernel.
 *
 * Usage:
 *   ./test_gemm_f32_bias [--N 1374] [--Din 1024] [--Dout 3072]
 *                        [--threshold 1e-3] [-v]
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

static void host_gemm(float *Y, const float *X, const float *W, const float *b,
                      int N, int D_in, int D_out)
{
    for (int n = 0; n < N; n++) {
        const float *xr = X + (size_t)n * D_in;
        for (int d = 0; d < D_out; d++) {
            const float *wr = W + (size_t)d * D_in;
            double acc = b ? b[d] : 0.0;
            for (int k = 0; k < D_in; k++) acc += (double)wr[k] * (double)xr[k];
            Y[(size_t)n * D_out + d] = (float)acc;
        }
    }
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
    int   N         = 1374;
    int   D_in      = 1024;
    int   D_out     = 3 * 1024;
    float threshold = 1e-3f;
    int   verbose   = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i+1 < argc) N         = atoi(argv[++i]);
        else if (!strcmp(a, "--Din")       && i+1 < argc) D_in      = atoi(argv[++i]);
        else if (!strcmp(a, "--Dout")      && i+1 < argc) D_out     = atoi(argv[++i]);
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
                                "test_gemm_f32_bias");
    if (sm < 0) { fprintf(stderr, "compile failed\n"); return 4; }
    CUfunction fn = NULL;
    if (cuModuleGetFunction(&fn, mod, "gemm_f32_bias") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup gemm_f32_bias failed\n"); return 4;
    }

    /* Inputs. He init scale ~ 1/sqrt(D_in) keeps activations bounded. */
    size_t n_X = (size_t)N * D_in;
    size_t n_W = (size_t)D_out * D_in;
    size_t n_Y = (size_t)N * D_out;
    float *h_X = (float *)malloc(n_X * sizeof(float));
    float *h_W = (float *)malloc(n_W * sizeof(float));
    float *h_b = (float *)malloc((size_t)D_out * sizeof(float));
    float *h_ref = (float *)malloc(n_Y * sizeof(float));
    float *h_dst = (float *)malloc(n_Y * sizeof(float));
    if (!h_X || !h_W || !h_b || !h_ref || !h_dst) {
        fprintf(stderr, "host alloc failed\n"); return 5;
    }
    uint32_t rng = 0xC0FFEEu;
    float w_scale = 1.0f / sqrtf((float)D_in);
    for (size_t i = 0; i < n_X; i++) h_X[i] = (urand(&rng) * 2.0f - 1.0f);
    for (size_t i = 0; i < n_W; i++) h_W[i] = (urand(&rng) * 2.0f - 1.0f) * w_scale;
    for (int d = 0; d < D_out; d++)  h_b[d] = (urand(&rng) * 2.0f - 1.0f) * 0.1f;

    host_gemm(h_ref, h_X, h_W, h_b, N, D_in, D_out);

    CUdeviceptr d_X = cu_upload_raw(h_X, n_X * sizeof(float));
    CUdeviceptr d_W = cu_upload_raw(h_W, n_W * sizeof(float));
    CUdeviceptr d_b = cu_upload_raw(h_b, (size_t)D_out * sizeof(float));
    CUdeviceptr d_Y = 0;
    if (cuMemAlloc(&d_Y, n_Y * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAlloc d_Y failed\n"); return 5;
    }

    void *args[] = { &d_Y, &d_X, &d_W, &d_b, &N, &D_in, &D_out };
    int gx = (N     + 15) / 16;
    int gy = (D_out + 15) / 16;
    if (cuLaunchKernel(fn,
                       gx, gy, 1,
                       16, 16, 1,
                       0, 0, args, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "launch failed\n"); return 6;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(h_dst, d_Y, n_Y * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_Y, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_gemm_f32_bias] N=%d D_in=%d D_out=%d  n_out=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        N, D_in, D_out, n_Y, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_X); free(h_W); free(h_b); free(h_ref); free(h_dst);
    cuMemFree(d_X); cuMemFree(d_W); cuMemFree(d_b); cuMemFree(d_Y);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
