/*
 * test_slat_resblock — Phase 5b.7 standalone microbench.
 *
 * Composes the CUDA primitives for a SLAT IO SparseResBlock3d with
 * no up/down and identity skip (input_blocks[0] shape):
 *   LN(affine) -> SiLU -> SubMConv3d -> modulated LN -> SiLU ->
 *   SubMConv3d -> residual add.
 *
 * Usage:
 *   ./test_slat_resblock [--N 1188] [--C 128] [--threshold 5e-4] [--repeat 20] [-v]
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

static void host_ln_affine(float *out, const float *in,
                           const float *gamma, const float *beta,
                           int N, int C, float eps)
{
    for (int n = 0; n < N; n++) {
        const float *x = in + (size_t)n * C;
        float *y = out + (size_t)n * C;
        float mean = 0.0f;
        for (int c = 0; c < C; c++) mean += x[c];
        mean /= (float)C;
        float var = 0.0f;
        for (int c = 0; c < C; c++) { float d = x[c] - mean; var += d * d; }
        var /= (float)C;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < C; c++) y[c] = (x[c] - mean) * inv * gamma[c] + beta[c];
    }
}

static void host_mod_ln(float *out, const float *in,
                        const float *shift, const float *scale,
                        int N, int C, float eps)
{
    for (int n = 0; n < N; n++) {
        const float *x = in + (size_t)n * C;
        float *y = out + (size_t)n * C;
        float mean = 0.0f;
        for (int c = 0; c < C; c++) mean += x[c];
        mean /= (float)C;
        float var = 0.0f;
        for (int c = 0; c < C; c++) { float d = x[c] - mean; var += d * d; }
        var /= (float)C;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < C; c++) {
            float v = (x[c] - mean) * inv;
            y[c] = v * (1.0f + scale[c]) + shift[c];
        }
    }
}

static void host_silu(float *x, int n)
{
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

static void host_submconv3x3(float *out,
                             const int32_t *coords,
                             const float *feats,
                             const int *index_grid,
                             const float *weight,
                             const float *bias,
                             int N, int C)
{
    for (int i = 0; i < N; i++) {
        int z0 = coords[i * 4 + 1];
        int y0 = coords[i * 4 + 2];
        int x0 = coords[i * 4 + 3];
        for (int oc = 0; oc < C; oc++) {
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
                        const float *w = weight + ((size_t)oc * 27 + k) * C;
                        const float *f = feats + (size_t)src * C;
                        for (int ic = 0; ic < C; ic++) acc += w[ic] * f[ic];
                    }
                }
            }
            out[(size_t)i * C + oc] = acc;
        }
    }
}

int main(int argc, char **argv)
{
    int N = 1188;
    int C = 128;
    int repeat = 20;
    float threshold = 5e-4f;
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

    size_t XC = (size_t)N * C;
    size_t WC = (size_t)C * 27 * C;
    int32_t *coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    int *index_grid = (int *)malloc((size_t)64 * 64 * 64 * sizeof(int));
    float *x = (float *)malloc(XC * sizeof(float));
    float *nw = (float *)malloc((size_t)C * sizeof(float));
    float *nb = (float *)malloc((size_t)C * sizeof(float));
    float *shift = (float *)malloc((size_t)C * sizeof(float));
    float *scale = (float *)malloc((size_t)C * sizeof(float));
    float *w1 = (float *)malloc(WC * sizeof(float));
    float *b1 = (float *)malloc((size_t)C * sizeof(float));
    float *w2 = (float *)malloc(WC * sizeof(float));
    float *b2 = (float *)malloc((size_t)C * sizeof(float));
    float *h1 = (float *)malloc(XC * sizeof(float));
    float *h2 = (float *)malloc(XC * sizeof(float));
    float *h3 = (float *)malloc(XC * sizeof(float));
    float *h4 = (float *)malloc(XC * sizeof(float));
    float *ref = (float *)malloc(XC * sizeof(float));
    float *dst = (float *)malloc(XC * sizeof(float));
    if (!coords || !index_grid || !x || !nw || !nb || !shift || !scale ||
        !w1 || !b1 || !w2 || !b2 || !h1 || !h2 || !h3 || !h4 || !ref || !dst)
        return 5;

    uint32_t rng = 0x5B10C700u;
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
        int xx = rem - y * 64;
        coords[filled * 4 + 0] = 0;
        coords[filled * 4 + 1] = z;
        coords[filled * 4 + 2] = y;
        coords[filled * 4 + 3] = xx;
        filled++;
    }
    free(used);

    for (size_t i = 0; i < XC; i++) x[i] = (urand(&rng) * 2.0f - 1.0f);
    for (int c = 0; c < C; c++) {
        nw[c] = 0.9f + 0.2f * urand(&rng);
        nb[c] = (urand(&rng) * 2.0f - 1.0f) * 0.05f;
        shift[c] = (urand(&rng) * 2.0f - 1.0f) * 0.05f;
        scale[c] = (urand(&rng) * 2.0f - 1.0f) * 0.10f;
        b1[c] = (urand(&rng) * 2.0f - 1.0f) * 0.02f;
        b2[c] = (urand(&rng) * 2.0f - 1.0f) * 0.02f;
    }
    float w_scale = 1.0f / sqrtf((float)(27 * C));
    for (size_t i = 0; i < WC; i++) {
        w1[i] = (urand(&rng) * 2.0f - 1.0f) * w_scale;
        w2[i] = (urand(&rng) * 2.0f - 1.0f) * w_scale;
    }

    host_build_index(coords, N, index_grid);
    host_ln_affine(h1, x, nw, nb, N, C, 1e-6f);
    host_silu(h1, (int)XC);
    host_submconv3x3(h2, coords, h1, index_grid, w1, b1, N, C);
    host_mod_ln(h3, h2, shift, scale, N, C, 1e-6f);
    host_silu(h3, (int)XC);
    host_submconv3x3(h4, coords, h3, index_grid, w2, b2, N, C);
    for (size_t i = 0; i < XC; i++) ref[i] = h4[i] + x[i];

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0;
    if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    if (hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_slat_resblock") < 0) return 4;
    hipFunction_t fn_idx = NULL, fn_ln = NULL, fn_silu = NULL, fn_conv = NULL;
    hipFunction_t fn_modln = NULL, fn_resadd = NULL;
    if (hipModuleGetFunction(&fn_idx, mod, "slat_build_coord_index64_i32") != hipSuccess ||
        hipModuleGetFunction(&fn_ln, mod, "layernorm_token_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_silu, mod, "silu_inplace_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_conv, mod, "slat_submconv3x3_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_modln, mod, "modulated_ln_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_resadd, mod, "residual_add_f32") != hipSuccess)
        return 4;

    hipDeviceptr_t d_coords = hip_upload_raw(coords, (size_t)N * 4 * sizeof(int32_t));
    hipDeviceptr_t d_x = hip_upload_raw(x, XC * sizeof(float));
    hipDeviceptr_t d_nw = hip_upload_raw(nw, (size_t)C * sizeof(float));
    hipDeviceptr_t d_nb = hip_upload_raw(nb, (size_t)C * sizeof(float));
    hipDeviceptr_t d_shift = hip_upload_raw(shift, (size_t)C * sizeof(float));
    hipDeviceptr_t d_scale = hip_upload_raw(scale, (size_t)C * sizeof(float));
    hipDeviceptr_t d_w1 = hip_upload_raw(w1, WC * sizeof(float));
    hipDeviceptr_t d_b1 = hip_upload_raw(b1, (size_t)C * sizeof(float));
    hipDeviceptr_t d_w2 = hip_upload_raw(w2, WC * sizeof(float));
    hipDeviceptr_t d_b2 = hip_upload_raw(b2, (size_t)C * sizeof(float));
    hipDeviceptr_t d_index = 0, d_h1 = 0, d_h2 = 0, d_h3 = 0, d_h4 = 0;
    if (!d_coords || !d_x || !d_nw || !d_nb || !d_shift || !d_scale ||
        !d_w1 || !d_b1 || !d_w2 || !d_b2 ||
        hipMalloc(&d_index, (size_t)64 * 64 * 64 * sizeof(int)) != hipSuccess ||
        hipMalloc(&d_h1, XC * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_h2, XC * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_h3, XC * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_h4, XC * sizeof(float)) != hipSuccess) {
        fprintf(stderr, "device alloc/upload failed\n");
        return 5;
    }

    float eps = 1e-6f;
    int affine = 1;
    int n_elem = (int)XC;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    size_t m_smem = 2 * threads * sizeof(float);
    int total_conv = N * C;

#define RUN_PIPELINE() do { \
    if (hipMemsetD8(d_index, 0xff, (size_t)64 * 64 * 64 * sizeof(int)) != hipSuccess) return 5; \
    void *a_idx[] = { &d_coords, &N, &d_index }; \
    if (hipModuleLaunchKernel(fn_idx, (N + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_idx, NULL) != hipSuccess) return 6; \
    void *a_ln[] = { &d_h1, &d_x, &d_nw, &d_nb, &N, &C, &eps, &affine }; \
    if (hipModuleLaunchKernel(fn_ln, N, 1, 1, threads, 1, 1, (unsigned)ln_smem, 0, a_ln, NULL) != hipSuccess) return 6; \
    void *a_s1[] = { &d_h1, &n_elem }; \
    if (hipModuleLaunchKernel(fn_silu, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_s1, NULL) != hipSuccess) return 6; \
    void *a_c1[] = { &d_coords, &d_h1, &d_index, &d_w1, &d_b1, &N, &C, &C, &d_h2 }; \
    if (hipModuleLaunchKernel(fn_conv, (total_conv + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_c1, NULL) != hipSuccess) return 6; \
    void *a_ml[] = { &d_h3, &d_h2, &d_shift, &d_scale, &N, &C, &eps }; \
    if (hipModuleLaunchKernel(fn_modln, N, 1, 1, threads, 1, 1, (unsigned)m_smem, 0, a_ml, NULL) != hipSuccess) return 6; \
    void *a_s2[] = { &d_h3, &n_elem }; \
    if (hipModuleLaunchKernel(fn_silu, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_s2, NULL) != hipSuccess) return 6; \
    void *a_c2[] = { &d_coords, &d_h3, &d_index, &d_w2, &d_b2, &N, &C, &C, &d_h4 }; \
    if (hipModuleLaunchKernel(fn_conv, (total_conv + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_c2, NULL) != hipSuccess) return 6; \
    void *a_ra[] = { &d_h4, &d_x, &n_elem }; \
    if (hipModuleLaunchKernel(fn_resadd, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_ra, NULL) != hipSuccess) return 6; \
} while (0)

    RUN_PIPELINE();
    hipDeviceSynchronize();
    hipMemcpyDtoH(dst, d_h4, XC * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(dst, ref, XC, &mean);
    int ok = (mx <= threshold);

    double avg_ms = 0.0;
    if (repeat > 0) {
        hipDeviceSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) RUN_PIPELINE();
        hipDeviceSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }
#undef RUN_PIPELINE

    fprintf(stderr,
            "[test_slat_resblock] N=%d C=%d max_abs=%.4g mean_abs=%.4g avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, C, (double)mx, mean, avg_ms, repeat,
            ok ? "OK" : "FAIL", (double)threshold);

    free(coords); free(index_grid); free(x); free(nw); free(nb); free(shift); free(scale);
    free(w1); free(b1); free(w2); free(b2); free(h1); free(h2); free(h3); free(h4); free(ref); free(dst);
    hipFree(d_coords); hipFree(d_x); hipFree(d_nw); hipFree(d_nb);
    hipFree(d_shift); hipFree(d_scale); hipFree(d_w1); hipFree(d_b1);
    hipFree(d_w2); hipFree(d_b2); hipFree(d_index);
    hipFree(d_h1); hipFree(d_h2); hipFree(d_h3); hipFree(d_h4);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
