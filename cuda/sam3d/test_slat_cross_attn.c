/*
 * test_slat_cross_attn — Phase 5b.6 standalone microbench.
 *
 * Validates the SLAT cross-attention core at production geometry:
 *   q[N,dim], fused kv_c[N_cond,2*dim] -> kv_split_f32 -> sdpa_f32(q,K,V).
 * This pins the CUDA layout used by sam3d_slat_dit.h::cpu_cross_attention
 * before composing a full SLAT transformer block.
 *
 * Usage:
 *   ./test_slat_cross_attn [--N 1188] [--Nc 2740] [--dim 1024] [--H 16] [--threshold 1e-3] [--repeat 10] [-v]
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

static void host_kv_split(const float *kv, int N_c, int dim, float *K, float *V)
{
    for (int n = 0; n < N_c; n++) {
        const float *row = kv + (size_t)n * 2 * dim;
        memcpy(K + (size_t)n * dim, row, (size_t)dim * sizeof(float));
        memcpy(V + (size_t)n * dim, row + dim, (size_t)dim * sizeof(float));
    }
}

static void host_sdpa(float *out,
                      const float *q, const float *K, const float *V,
                      int N, int N_c, int H, int D_h, float scale)
{
    int E = H * D_h;
    double *scores = (double *)malloc((size_t)N_c * sizeof(double));
    for (int nq = 0; nq < N; nq++) {
        for (int h = 0; h < H; h++) {
            const float *qv = q + (size_t)nq * E + (size_t)h * D_h;
            double mx = -1.0e300;
            for (int nk = 0; nk < N_c; nk++) {
                const float *kv = K + (size_t)nk * E + (size_t)h * D_h;
                double s = 0.0;
                for (int d = 0; d < D_h; d++) s += (double)qv[d] * (double)kv[d];
                s *= (double)scale;
                scores[nk] = s;
                if (s > mx) mx = s;
            }
            double sum = 0.0;
            for (int nk = 0; nk < N_c; nk++) {
                scores[nk] = exp(scores[nk] - mx);
                sum += scores[nk];
            }
            double inv = 1.0 / sum;
            for (int d = 0; d < D_h; d++) {
                double acc = 0.0;
                for (int nk = 0; nk < N_c; nk++)
                    acc += scores[nk] * (double)V[(size_t)nk * E + (size_t)h * D_h + d];
                out[(size_t)nq * E + (size_t)h * D_h + d] = (float)(acc * inv);
            }
        }
    }
    free(scores);
}

int main(int argc, char **argv)
{
    int N = 1188;
    int N_c = 2740;
    int dim = 1024;
    int H = 16;
    int repeat = 10;
    float threshold = 1e-3f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(a, "--Nc")        && i + 1 < argc) N_c = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(a, "--H")         && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat")    && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (N <= 0 || N_c <= 0 || dim <= 0 || H <= 0 || dim % H != 0) return 2;
    int D_h = dim / H;
    float scale = 1.0f / sqrtf((float)D_h);

    size_t q_n = (size_t)N * dim;
    size_t kv_n = (size_t)N_c * 2 * dim;
    size_t c_n = (size_t)N_c * dim;
    float *q = (float *)malloc(q_n * sizeof(float));
    float *kv = (float *)malloc(kv_n * sizeof(float));
    float *K = (float *)malloc(c_n * sizeof(float));
    float *V = (float *)malloc(c_n * sizeof(float));
    float *ref = (float *)malloc(q_n * sizeof(float));
    float *dst = (float *)malloc(q_n * sizeof(float));
    if (!q || !kv || !K || !V || !ref || !dst) return 5;

    uint32_t rng = 0xC4055A77u;
    for (size_t i = 0; i < q_n; i++) q[i] = (urand(&rng) * 2.0f - 1.0f);
    for (size_t i = 0; i < kv_n; i++) kv[i] = (urand(&rng) * 2.0f - 1.0f);
    host_kv_split(kv, N_c, dim, K, V);
    host_sdpa(ref, q, K, V, N, N_c, H, D_h, scale);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_slat_cross_attn") < 0) return 4;
    CUfunction fn_split = NULL, fn_sdpa = NULL;
    if (cuModuleGetFunction(&fn_split, mod, "kv_split_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_sdpa, mod, "sdpa_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "kernel lookup failed\n");
        return 4;
    }

    CUdeviceptr d_q = cu_upload_raw(q, q_n * sizeof(float));
    CUdeviceptr d_kv = cu_upload_raw(kv, kv_n * sizeof(float));
    CUdeviceptr d_K = 0, d_V = 0, d_out = 0;
    if (!d_q || !d_kv ||
        cuMemAlloc(&d_K, c_n * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_V, c_n * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_out, q_n * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "device alloc/upload failed\n");
        return 5;
    }

    void *split_args[] = { &d_K, &d_V, &d_kv, &N_c, &dim };
    if (cuLaunchKernel(fn_split, (int)((c_n + 255) / 256), 1, 1,
                       256, 1, 1, 0, 0, split_args, NULL) != CUDA_SUCCESS) return 6;
    void *sdpa_args[] = { &d_out, &d_q, &d_K, &d_V, &N, &N_c, &H, &D_h, &scale };
    size_t smem = (size_t)(256 + N_c) * sizeof(float);
    if (cuLaunchKernel(fn_sdpa, N, H, 1, 256, 1, 1,
                       (unsigned)smem, 0, sdpa_args, NULL) != CUDA_SUCCESS) return 6;
    cuCtxSynchronize();
    cuMemcpyDtoH(dst, d_out, q_n * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(dst, ref, q_n, &mean);
    int ok = mx <= threshold;

    double avg_ms = 0.0;
    if (repeat > 0) {
        cuCtxSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) {
            if (cuLaunchKernel(fn_sdpa, N, H, 1, 256, 1, 1,
                               (unsigned)smem, 0, sdpa_args, NULL) != CUDA_SUCCESS) return 6;
        }
        cuCtxSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }

    fprintf(stderr,
            "[test_slat_cross_attn] N=%d Nc=%d dim=%d H=%d D_h=%d max_abs=%.4g mean_abs=%.4g sdpa_avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, N_c, dim, H, D_h, (double)mx, mean, avg_ms, repeat,
            ok ? "OK" : "FAIL", (double)threshold);

    free(q); free(kv); free(K); free(V); free(ref); free(dst);
    cuMemFree(d_q); cuMemFree(d_kv); cuMemFree(d_K); cuMemFree(d_V); cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
