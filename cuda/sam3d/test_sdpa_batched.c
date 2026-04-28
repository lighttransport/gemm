/*
 * test_sdpa_batched — Phase 2b.6a standalone microbench.
 *
 * Validates sdpa_batched_f32 at PointPatchEmbed window-attention
 * geometry: B=1024 windows × N_q=N_k=65 tokens × H=16 heads × D_h=32.
 * Cross-checks against the existing sdpa_f32 kernel by running each
 * batch independently and comparing.
 *
 * Usage:
 *   ./test_sdpa_batched [--B 1024] [--N 65] [--H 16] [--D_h 32]
 *                       [--threshold 5e-5] [-v]
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
    int B = 1024, N = 65, H = 16, D_h = 32;
    float threshold = 5e-5f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--B")         && i+1 < argc) B         = atoi(argv[++i]);
        else if (!strcmp(a, "--N")         && i+1 < argc) N         = atoi(argv[++i]);
        else if (!strcmp(a, "--H")         && i+1 < argc) H         = atoi(argv[++i]);
        else if (!strcmp(a, "--D_h")       && i+1 < argc) D_h       = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    int E = H * D_h;
    float scale = 1.0f / sqrtf((float)D_h);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    int sm = cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose, "test_sdpa_batched");
    if (sm < 0) return 4;
    CUfunction fn_b = 0, fn_s = 0;
    if (cuModuleGetFunction(&fn_b, mod, "sdpa_batched_f32") != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn_s, mod, "sdpa_f32")         != CUDA_SUCCESS) return 4;

    size_t n_per   = (size_t)N * E;
    size_t n_total = (size_t)B * N * E;
    float *h_q = (float *)malloc(n_total * sizeof(float));
    float *h_k = (float *)malloc(n_total * sizeof(float));
    float *h_v = (float *)malloc(n_total * sizeof(float));
    float *h_o_b = (float *)malloc(n_total * sizeof(float));
    float *h_o_s = (float *)malloc(n_total * sizeof(float));
    if (!h_q || !h_k || !h_v || !h_o_b || !h_o_s) return 5;
    uint32_t rng = 0xC0FFEEu;
    for (size_t i = 0; i < n_total; i++) h_q[i] = urand(&rng) * 2.0f - 1.0f;
    for (size_t i = 0; i < n_total; i++) h_k[i] = urand(&rng) * 2.0f - 1.0f;
    for (size_t i = 0; i < n_total; i++) h_v[i] = urand(&rng) * 2.0f - 1.0f;

    CUdeviceptr d_q = cu_upload_raw(h_q, n_total * sizeof(float));
    CUdeviceptr d_k = cu_upload_raw(h_k, n_total * sizeof(float));
    CUdeviceptr d_v = cu_upload_raw(h_v, n_total * sizeof(float));
    CUdeviceptr d_o = 0;
    if (cuMemAlloc(&d_o, n_total * sizeof(float)) != CUDA_SUCCESS) return 5;

    /* Batched run. */
    {
        int threads = 256;
        size_t shmem = (threads + N) * sizeof(float);
        void *args[] = { &d_o, &d_q, &d_k, &d_v, &N, &N, &H, &D_h, &scale };
        if (cuLaunchKernel(fn_b, N, H, B, threads, 1, 1, shmem, 0, args, NULL) != CUDA_SUCCESS) {
            fprintf(stderr, "batched launch failed\n"); return 6;
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(h_o_b, d_o, n_total * sizeof(float));
    }

    /* Reference: per-batch sdpa_f32. */
    {
        int threads = 256;
        size_t shmem = (threads + N) * sizeof(float);
        for (int b = 0; b < B; b++) {
            CUdeviceptr q_b = d_q + b * n_per * sizeof(float);
            CUdeviceptr k_b = d_k + b * n_per * sizeof(float);
            CUdeviceptr v_b = d_v + b * n_per * sizeof(float);
            CUdeviceptr o_b = d_o + b * n_per * sizeof(float);
            void *args[] = { &o_b, &q_b, &k_b, &v_b, &N, &N, &H, &D_h, &scale };
            if (cuLaunchKernel(fn_s, N, H, 1, threads, 1, 1, shmem, 0, args, NULL) != CUDA_SUCCESS) {
                fprintf(stderr, "sdpa_f32 launch (batch %d) failed\n", b); return 6;
            }
        }
        cuCtxSynchronize();
        cuMemcpyDtoH(h_o_s, d_o, n_total * sizeof(float));
    }

    double mean = 0.0;
    float mx = max_abs(h_o_b, h_o_s, n_total, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_sdpa_batched] B=%d N=%d H=%d D_h=%d  n_out=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        B, N, H, D_h, n_total, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_q); free(h_k); free(h_v); free(h_o_b); free(h_o_s);
    cuMemFree(d_q); cuMemFree(d_k); cuMemFree(d_v); cuMemFree(d_o);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
