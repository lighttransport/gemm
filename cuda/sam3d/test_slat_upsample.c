/*
 * test_slat_upsample — Phase 5b.3 standalone microbench.
 *
 * Validates `slat_upsample2_nearest_f32` against the host reference
 * semantics of sp3d_upsample(..., factor=2, target_coords, target_N).
 *
 * Usage:
 *   ./test_slat_upsample [--target-N 1188] [--C 2048] [--repeat 200] [-v]
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

static int make_coarse_first_occurrence(const int32_t *target_coords, int target_N,
                                        int32_t *src_coords)
{
    int src_N = 0;
    for (int i = 0; i < target_N; i++) {
        int32_t b = target_coords[i * 4 + 0];
        int32_t z = target_coords[i * 4 + 1] / 2;
        int32_t y = target_coords[i * 4 + 2] / 2;
        int32_t x = target_coords[i * 4 + 3] / 2;
        int found = 0;
        for (int j = 0; j < src_N; j++) {
            if (src_coords[j * 4 + 0] == b && src_coords[j * 4 + 1] == z &&
                src_coords[j * 4 + 2] == y && src_coords[j * 4 + 3] == x) {
                found = 1;
                break;
            }
        }
        if (!found) {
            src_coords[src_N * 4 + 0] = b;
            src_coords[src_N * 4 + 1] = z;
            src_coords[src_N * 4 + 2] = y;
            src_coords[src_N * 4 + 3] = x;
            src_N++;
        }
    }
    return src_N;
}

static void host_upsample2(const int32_t *src_coords, const float *src_feats,
                           int src_N, int C,
                           const int32_t *target_coords, int target_N,
                           float *out_feats)
{
    memset(out_feats, 0, (size_t)target_N * C * sizeof(float));
    for (int i = 0; i < target_N; i++) {
        int32_t b = target_coords[i * 4 + 0];
        int32_t z = target_coords[i * 4 + 1] / 2;
        int32_t y = target_coords[i * 4 + 2] / 2;
        int32_t x = target_coords[i * 4 + 3] / 2;
        int src = -1;
        for (int j = 0; j < src_N; j++) {
            if (src_coords[j * 4 + 0] == b && src_coords[j * 4 + 1] == z &&
                src_coords[j * 4 + 2] == y && src_coords[j * 4 + 3] == x) {
                src = j;
                break;
            }
        }
        if (src >= 0) {
            memcpy(out_feats + (size_t)i * C,
                   src_feats + (size_t)src * C,
                   (size_t)C * sizeof(float));
        }
    }
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
    int target_N = 1188;
    int C = 2048;
    int repeat = 200;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--target-N") && i + 1 < argc) target_N = atoi(argv[++i]);
        else if (!strcmp(a, "--C")        && i + 1 < argc) C = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat")   && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (target_N <= 0 || C <= 0) return 2;

    int32_t *target_coords = (int32_t *)malloc((size_t)target_N * 4 * sizeof(int32_t));
    int32_t *src_coords = (int32_t *)malloc((size_t)target_N * 4 * sizeof(int32_t));
    if (!target_coords || !src_coords) return 5;

    uint32_t rng = 0x0F515EEDu;
    unsigned char *used = (unsigned char *)calloc((size_t)64 * 64 * 64, 1);
    if (!used) return 5;
    int filled = 0;
    while (filled < target_N) {
        uint32_t r = (uint32_t)(urand(&rng) * (float)(64 * 64 * 64));
        if (r >= 64u * 64u * 64u) r = 64u * 64u * 64u - 1u;
        if (used[r]) continue;
        used[r] = 1;
        int z = (int)(r / (64u * 64u));
        int rem = (int)(r - (uint32_t)z * 64u * 64u);
        int y = rem / 64;
        int x = rem - y * 64;
        target_coords[filled * 4 + 0] = 0;
        target_coords[filled * 4 + 1] = z;
        target_coords[filled * 4 + 2] = y;
        target_coords[filled * 4 + 3] = x;
        filled++;
    }
    free(used);
    int src_N = make_coarse_first_occurrence(target_coords, target_N, src_coords);

    float *src_feats = (float *)malloc((size_t)src_N * C * sizeof(float));
    float *ref_feats = (float *)malloc((size_t)target_N * C * sizeof(float));
    float *dst_feats = (float *)malloc((size_t)target_N * C * sizeof(float));
    if (!src_feats || !ref_feats || !dst_feats) return 5;
    for (size_t i = 0; i < (size_t)src_N * C; i++)
        src_feats[i] = (urand(&rng) * 2.0f - 1.0f);

    host_upsample2(src_coords, src_feats, src_N, C,
                   target_coords, target_N, ref_feats);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_slat_upsample") < 0) return 4;
    CUfunction fn = NULL;
    if (cuModuleGetFunction(&fn, mod, "slat_upsample2_nearest_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup slat_upsample2_nearest_f32 failed\n");
        return 4;
    }

    CUdeviceptr d_src_coords = cu_upload_raw(src_coords, (size_t)src_N * 4 * sizeof(int32_t));
    CUdeviceptr d_src_feats = cu_upload_raw(src_feats, (size_t)src_N * C * sizeof(float));
    CUdeviceptr d_target_coords = cu_upload_raw(target_coords, (size_t)target_N * 4 * sizeof(int32_t));
    CUdeviceptr d_out = 0;
    if (!d_src_coords || !d_src_feats || !d_target_coords ||
        cuMemAlloc(&d_out, (size_t)target_N * C * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "device alloc/upload failed\n");
        return 5;
    }

    unsigned grid = (unsigned)((target_N + 127) / 128);
    void *args[] = {
        &d_src_coords, &d_src_feats, &src_N, &C,
        &d_target_coords, &target_N, &d_out
    };
    if (cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "launch failed\n");
        return 6;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(dst_feats, d_out, (size_t)target_N * C * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(dst_feats, ref_feats, (size_t)target_N * C, &mean);
    int ok = (mx == 0.0f);

    double avg_ms = 0.0;
    if (repeat > 0) {
        cuCtxSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) {
            if (cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) {
                fprintf(stderr, "timing launch failed\n");
                return 6;
            }
        }
        cuCtxSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }

    fprintf(stderr,
            "[test_slat_upsample] src_N=%d target_N=%d C=%d max_abs=%.4g mean_abs=%.4g avg=%.4f ms x%d %s\n",
            src_N, target_N, C, (double)mx, mean, avg_ms, repeat, ok ? "OK" : "FAIL");

    free(target_coords); free(src_coords); free(src_feats); free(ref_feats); free(dst_feats);
    cuMemFree(d_src_coords); cuMemFree(d_src_feats); cuMemFree(d_target_coords); cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
