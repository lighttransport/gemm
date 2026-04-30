/*
 * test_slat_submconv3d — Phase 5b.4 standalone microbench.
 *
 * Validates `slat_submconv3x3_f32` against a scalar host reference for
 * sp3d_conv3d_forward's submanifold 3x3x3 semantics.
 *
 * Usage:
 *   ./test_slat_submconv3d [--N 1188] [--Cin 128] [--Cout 128] [--threshold 1e-6] [--repeat 20] [-v]
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

static void host_build_index(const int32_t *coords, int N, int *index_grid)
{
    for (int i = 0; i < 64 * 64 * 64; i++) index_grid[i] = -1;
    for (int i = 0; i < N; i++) {
        int z = coords[i * 4 + 1];
        int y = coords[i * 4 + 2];
        int x = coords[i * 4 + 3];
        index_grid[(z * 64 + y) * 64 + x] = i;
    }
}

static void host_submconv3x3(float *out,
                             const int32_t *coords,
                             const float *feats,
                             const int *index_grid,
                             const float *weight,
                             const float *bias,
                             int N, int in_C, int out_C)
{
    for (int i = 0; i < N; i++) {
        int z0 = coords[i * 4 + 1];
        int y0 = coords[i * 4 + 2];
        int x0 = coords[i * 4 + 3];
        for (int oc = 0; oc < out_C; oc++) {
            float acc = bias ? bias[oc] : 0.0f;
            for (int dz = -1; dz <= 1; dz++) {
                int z = z0 + dz;
                if ((unsigned)z >= 64u) continue;
                for (int dy = -1; dy <= 1; dy++) {
                    int y = y0 + dy;
                    if ((unsigned)y >= 64u) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        int x = x0 + dx;
                        if ((unsigned)x >= 64u) continue;
                        int src = index_grid[(z * 64 + y) * 64 + x];
                        if (src < 0) continue;
                        int k = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
                        const float *w = weight + ((size_t)oc * 27 + k) * in_C;
                        const float *f = feats + (size_t)src * in_C;
                        for (int ic = 0; ic < in_C; ic++) acc += w[ic] * f[ic];
                    }
                }
            }
            out[(size_t)i * out_C + oc] = acc;
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
    int N = 1188;
    int in_C = 128;
    int out_C = 128;
    int repeat = 20;
    float threshold = 1e-6f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(a, "--Cin")       && i + 1 < argc) in_C = atoi(argv[++i]);
        else if (!strcmp(a, "--Cout")      && i + 1 < argc) out_C = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat")    && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (N <= 0 || in_C <= 0 || out_C <= 0) return 2;

    int32_t *coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    float *feats = (float *)malloc((size_t)N * in_C * sizeof(float));
    float *weight = (float *)malloc((size_t)out_C * 27 * in_C * sizeof(float));
    float *bias = (float *)malloc((size_t)out_C * sizeof(float));
    float *ref = (float *)malloc((size_t)N * out_C * sizeof(float));
    float *dst = (float *)malloc((size_t)N * out_C * sizeof(float));
    int *index_grid = (int *)malloc((size_t)64 * 64 * 64 * sizeof(int));
    if (!coords || !feats || !weight || !bias || !ref || !dst || !index_grid) return 5;

    uint32_t rng = 0x5B3DC0A7u;
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
    for (size_t i = 0; i < (size_t)N * in_C; i++)
        feats[i] = (urand(&rng) * 2.0f - 1.0f);
    float w_scale = 1.0f / sqrtf((float)(27 * in_C));
    for (size_t i = 0; i < (size_t)out_C * 27 * in_C; i++)
        weight[i] = (urand(&rng) * 2.0f - 1.0f) * w_scale;
    for (int i = 0; i < out_C; i++)
        bias[i] = (urand(&rng) * 2.0f - 1.0f) * 0.1f;

    host_build_index(coords, N, index_grid);
    host_submconv3x3(ref, coords, feats, index_grid, weight, bias, N, in_C, out_C);

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0;
    if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    if (hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_slat_submconv3d") < 0) return 4;
    hipFunction_t fn_index = NULL, fn_conv = NULL;
    if (hipModuleGetFunction(&fn_index, mod, "slat_build_coord_index64_i32") != hipSuccess ||
        hipModuleGetFunction(&fn_conv, mod, "slat_submconv3x3_f32") != hipSuccess) {
        fprintf(stderr, "kernel lookup failed\n");
        return 4;
    }

    hipDeviceptr_t d_coords = hip_upload_raw(coords, (size_t)N * 4 * sizeof(int32_t));
    hipDeviceptr_t d_feats = hip_upload_raw(feats, (size_t)N * in_C * sizeof(float));
    hipDeviceptr_t d_weight = hip_upload_raw(weight, (size_t)out_C * 27 * in_C * sizeof(float));
    hipDeviceptr_t d_bias = hip_upload_raw(bias, (size_t)out_C * sizeof(float));
    hipDeviceptr_t d_index = 0, d_out = 0;
    if (!d_coords || !d_feats || !d_weight || !d_bias ||
        hipMalloc(&d_index, (size_t)64 * 64 * 64 * sizeof(int)) != hipSuccess ||
        hipMalloc(&d_out, (size_t)N * out_C * sizeof(float)) != hipSuccess) {
        fprintf(stderr, "device alloc/upload failed\n");
        return 5;
    }

    if (hipMemsetD8(d_index, 0xff, (size_t)64 * 64 * 64 * sizeof(int)) != hipSuccess) return 5;
    void *idx_args[] = { &d_coords, &N, &d_index };
    if (hipModuleLaunchKernel(fn_index, (N + 255) / 256, 1, 1, 256, 1, 1, 0, 0, idx_args, NULL) != hipSuccess) {
        fprintf(stderr, "index launch failed\n");
        return 6;
    }
    void *conv_args[] = {
        &d_coords, &d_feats, &d_index, &d_weight, &d_bias,
        &N, &in_C, &out_C, &d_out
    };
    int total = N * out_C;
    if (hipModuleLaunchKernel(fn_conv, (total + 255) / 256, 1, 1, 256, 1, 1, 0, 0, conv_args, NULL) != hipSuccess) {
        fprintf(stderr, "conv launch failed\n");
        return 6;
    }
    hipDeviceSynchronize();
    hipMemcpyDtoH(dst, d_out, (size_t)N * out_C * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(dst, ref, (size_t)N * out_C, &mean);
    int ok = (mx <= threshold);

    double avg_ms = 0.0;
    if (repeat > 0) {
        hipDeviceSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) {
            if (hipMemsetD8(d_index, 0xff, (size_t)64 * 64 * 64 * sizeof(int)) != hipSuccess) return 5;
            if (hipModuleLaunchKernel(fn_index, (N + 255) / 256, 1, 1, 256, 1, 1, 0, 0, idx_args, NULL) != hipSuccess) return 6;
            if (hipModuleLaunchKernel(fn_conv, (total + 255) / 256, 1, 1, 256, 1, 1, 0, 0, conv_args, NULL) != hipSuccess) return 6;
        }
        hipDeviceSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }

    fprintf(stderr,
            "[test_slat_submconv3d] N=%d Cin=%d Cout=%d max_abs=%.4g mean_abs=%.4g avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, in_C, out_C, (double)mx, mean, avg_ms, repeat,
            ok ? "OK" : "FAIL", (double)threshold);

    free(coords); free(feats); free(weight); free(bias); free(ref); free(dst); free(index_grid);
    hipFree(d_coords); hipFree(d_feats); hipFree(d_weight); hipFree(d_bias);
    hipFree(d_index); hipFree(d_out);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
