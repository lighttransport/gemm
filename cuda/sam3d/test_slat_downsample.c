/*
 * test_slat_downsample — Phase 5b.2 standalone microbench.
 *
 * Validates `slat_downsample2_mean_include_self_serial_f32` against the
 * host reference semantics of sp3d_downsample(..., factor=2, pool_mode=2).
 *
 * Usage:
 *   ./test_slat_downsample [--N 1188] [--C 128] [--threshold 0] [--repeat 200] [-v]
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

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static int host_downsample2_mean_include_self(const int32_t *coords,
                                              const float *feats,
                                              int N, int C,
                                              int32_t *out_coords,
                                              float *out_feats,
                                              int *out_counts)
{
    memset(out_coords, 0, (size_t)N * 4 * sizeof(int32_t));
    memset(out_feats, 0, (size_t)N * C * sizeof(float));
    memset(out_counts, 0, (size_t)N * sizeof(int));

    int M = 0;
    for (int i = 0; i < N; i++) {
        int32_t b = coords[i * 4 + 0];
        int32_t z = coords[i * 4 + 1] / 2;
        int32_t y = coords[i * 4 + 2] / 2;
        int32_t x = coords[i * 4 + 3] / 2;
        int oi = -1;
        for (int j = 0; j < M; j++) {
            if (out_coords[j * 4 + 0] == b && out_coords[j * 4 + 1] == z &&
                out_coords[j * 4 + 2] == y && out_coords[j * 4 + 3] == x) {
                oi = j;
                break;
            }
        }
        if (oi < 0) {
            oi = M++;
            out_coords[oi * 4 + 0] = b;
            out_coords[oi * 4 + 1] = z;
            out_coords[oi * 4 + 2] = y;
            out_coords[oi * 4 + 3] = x;
        }
        for (int c = 0; c < C; c++)
            out_feats[(size_t)oi * C + c] += feats[(size_t)i * C + c];
        out_counts[oi] += 1;
    }
    for (int i = 0; i < M; i++) {
        float inv = 1.0f / (float)(out_counts[i] + 1);
        for (int c = 0; c < C; c++) out_feats[(size_t)i * C + c] *= inv;
    }
    return M;
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

int main(int argc, char **argv)
{
    int N = 1188;
    int C = 128;
    int repeat = 200;
    float threshold = 0.0f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(a, "--C")         && i + 1 < argc) C = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat")    && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (N <= 0 || C <= 0) return 2;

    int32_t *coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    int32_t *ref_coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    int32_t *dst_coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    float *feats = (float *)malloc((size_t)N * C * sizeof(float));
    float *ref_feats = (float *)malloc((size_t)N * C * sizeof(float));
    float *dst_feats = (float *)malloc((size_t)N * C * sizeof(float));
    int *ref_counts = (int *)malloc((size_t)N * sizeof(int));
    int *dst_counts = (int *)malloc((size_t)N * sizeof(int));
    if (!coords || !ref_coords || !dst_coords || !feats || !ref_feats ||
        !dst_feats || !ref_counts || !dst_counts) {
        fprintf(stderr, "host alloc failed\n");
        return 5;
    }

    uint32_t rng = 0x5D005A6u;
    unsigned char *used = (unsigned char *)calloc((size_t)64 * 64 * 64, 1);
    if (!used) return 5;
    int filled = 0;
    while (filled < N) {
        uint32_t r = (uint32_t)(urand(&rng) * (float)(64 * 64 * 64));
        if (r >= 64u * 64u * 64u) r = 64u * 64u * 64u - 1u;
        if (used[r]) continue;
        used[r] = 1;
        int z = (int)(r / (64u * 64u));
        int rem = (int)(r - (uint32_t)z * 64u * 64u);
        int y = rem / 64;
        int x = rem - y * 64;
        coords[filled * 4 + 0] = 0;
        coords[filled * 4 + 1] = z;
        coords[filled * 4 + 2] = y;
        coords[filled * 4 + 3] = x;
        filled++;
    }
    free(used);
    for (size_t i = 0; i < (size_t)N * C; i++)
        feats[i] = (urand(&rng) * 2.0f - 1.0f);

    int ref_N = host_downsample2_mean_include_self(coords, feats, N, C,
                                                   ref_coords, ref_feats, ref_counts);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_slat_downsample") < 0) return 4;
    CUfunction fn = NULL;
    if (cuModuleGetFunction(&fn, mod,
                            "slat_downsample2_mean_include_self_serial_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup downsample kernel failed\n");
        return 4;
    }

    CUdeviceptr d_coords = cu_upload_raw(coords, (size_t)N * 4 * sizeof(int32_t));
    CUdeviceptr d_feats = cu_upload_raw(feats, (size_t)N * C * sizeof(float));
    CUdeviceptr d_out_coords = 0, d_out_feats = 0, d_counts = 0, d_out_N = 0;
    if (!d_coords || !d_feats ||
        cuMemAlloc(&d_out_coords, (size_t)N * 4 * sizeof(int32_t)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_out_feats, (size_t)N * C * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_counts, (size_t)N * sizeof(int)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_out_N, sizeof(int)) != CUDA_SUCCESS) {
        fprintf(stderr, "device alloc failed\n");
        return 5;
    }

    void *args[] = {
        &d_coords, &d_feats, &N, &C,
        &d_out_coords, &d_out_feats, &d_counts, &d_out_N
    };
    if (cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "launch failed\n");
        return 6;
    }
    cuCtxSynchronize();
    int dst_N = 0;
    cuMemcpyDtoH(&dst_N, d_out_N, sizeof(int));
    cuMemcpyDtoH(dst_coords, d_out_coords, (size_t)N * 4 * sizeof(int32_t));
    cuMemcpyDtoH(dst_feats, d_out_feats, (size_t)N * C * sizeof(float));
    cuMemcpyDtoH(dst_counts, d_counts, (size_t)N * sizeof(int));

    int coords_ok = (dst_N == ref_N) &&
                    (memcmp(dst_coords, ref_coords, (size_t)ref_N * 4 * sizeof(int32_t)) == 0) &&
                    (memcmp(dst_counts, ref_counts, (size_t)ref_N * sizeof(int)) == 0);
    double mean = 0.0;
    float mx = max_abs(dst_feats, ref_feats, (size_t)ref_N * C, &mean);
    int ok = coords_ok && (mx <= threshold);

    double avg_ms = 0.0;
    if (repeat > 0) {
        cuCtxSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) {
            if (cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) {
                fprintf(stderr, "timing launch failed\n");
                return 6;
            }
        }
        cuCtxSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }

    fprintf(stderr,
            "[test_slat_downsample] N=%d C=%d out_N=%d max_abs=%.4g mean_abs=%.4g coords=%s avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, C, ref_N, (double)mx, mean, coords_ok ? "OK" : "FAIL",
            avg_ms, repeat, ok ? "OK" : "FAIL", (double)threshold);

    free(coords); free(ref_coords); free(dst_coords);
    free(feats); free(ref_feats); free(dst_feats);
    free(ref_counts); free(dst_counts);
    cuMemFree(d_coords); cuMemFree(d_feats); cuMemFree(d_out_coords);
    cuMemFree(d_out_feats); cuMemFree(d_counts); cuMemFree(d_out_N);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
