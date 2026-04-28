/*
 * test_slat_ape — Phase 5b.1 standalone microbench.
 *
 * Validates `slat_ape_add_f32` against the host reference used by
 * common/sam3d_slat_dit.h::slat_apply_ape.
 *
 * Usage:
 *   ./test_slat_ape [--N 1188] [--dim 1024] [--threshold 5e-5] [-v]
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
#include <time.h>

static float urand(uint32_t *state)
{
    *state = (*state) * 1664525u + 1013904223u;
    return (float)((*state) >> 8) / (float)(1u << 24);
}

static void host_slat_ape_add(float *feats, const int32_t *coords, int N, int dim)
{
    int freq_dim = dim / 3 / 2;
    int per_axis = freq_dim * 2;
    float *freqs = (float *)malloc((size_t)freq_dim * sizeof(float));
    for (int j = 0; j < freq_dim; j++)
        freqs[j] = 1.0f / powf(10000.0f, (float)j / (float)freq_dim);

    for (int i = 0; i < N; i++) {
        float *row = feats + (size_t)i * dim;
        for (int axis = 0; axis < 3; axis++) {
            float v = (float)coords[i * 4 + 1 + axis];
            for (int j = 0; j < freq_dim; j++) {
                float arg = v * freqs[j];
                row[axis * per_axis + j] += sinf(arg);
                row[axis * per_axis + freq_dim + j] += cosf(arg);
            }
        }
    }
    free(freqs);
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

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

int main(int argc, char **argv)
{
    int N = 1188;
    int dim = 1024;
    float threshold = 5e-5f;
    int repeat = 1;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--repeat")    && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (N <= 0 || dim < 6) {
        fprintf(stderr, "invalid N/dim\n");
        return 2;
    }

    size_t n_feat = (size_t)N * dim;
    float *h_in  = (float *)malloc(n_feat * sizeof(float));
    float *h_ref = (float *)malloc(n_feat * sizeof(float));
    float *h_dst = (float *)malloc(n_feat * sizeof(float));
    int32_t *coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    if (!h_in || !h_ref || !h_dst || !coords) {
        fprintf(stderr, "host alloc failed\n");
        return 5;
    }

    uint32_t rng = 0x51A7001u;
    for (size_t i = 0; i < n_feat; i++)
        h_in[i] = (urand(&rng) * 2.0f - 1.0f) * 0.1f;
    for (int i = 0; i < N; i++) {
        coords[i * 4 + 0] = 0;
        coords[i * 4 + 1] = (int32_t)((i * 17 + 3) & 63);
        coords[i * 4 + 2] = (int32_t)((i * 29 + 5) & 63);
        coords[i * 4 + 3] = (int32_t)((i * 43 + 7) & 63);
    }
    memcpy(h_ref, h_in, n_feat * sizeof(float));
    host_slat_ape_add(h_ref, coords, N, dim);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_slat_ape") < 0) return 4;
    CUfunction fn = NULL;
    if (cuModuleGetFunction(&fn, mod, "slat_ape_add_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup slat_ape_add_f32 failed\n");
        return 4;
    }

    CUdeviceptr d_feat = cu_upload_raw(h_in, n_feat * sizeof(float));
    CUdeviceptr d_coords = cu_upload_raw(coords, (size_t)N * 4 * sizeof(int32_t));
    if (!d_feat || !d_coords) {
        fprintf(stderr, "device upload failed\n");
        return 5;
    }

    int freq_dim = dim / 3 / 2;
    int filled = freq_dim * 2 * 3;
    long long total = (long long)N * filled;
    unsigned grid = (unsigned)((total + 255) / 256);
    void *args[] = { &d_feat, &d_coords, &N, &dim };
    if (cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "launch failed\n");
        return 6;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(h_dst, d_feat, n_feat * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_feat, &mean);
    int ok = (mx <= threshold);

    double avg_ms = 0.0;
    if (repeat > 0) {
        cuMemcpyHtoD(d_feat, h_in, n_feat * sizeof(float));
        cuCtxSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) {
            if (cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) {
                fprintf(stderr, "timing launch failed\n");
                return 6;
            }
        }
        cuCtxSynchronize();
        double t1 = now_ms();
        avg_ms = (t1 - t0) / (double)repeat;
    }

    fprintf(stderr,
            "[test_slat_ape] N=%d dim=%d filled=%d max_abs=%.4g mean_abs=%.4g avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, dim, filled, (double)mx, mean, avg_ms, repeat,
            ok ? "OK" : "FAIL", (double)threshold);

    free(h_in);
    free(h_ref);
    free(h_dst);
    free(coords);
    cuMemFree(d_feat);
    cuMemFree(d_coords);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
