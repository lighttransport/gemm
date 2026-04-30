/*
 * test_slat_self_attn — Phase 5b.5 standalone microbench.
 *
 * Validates SLAT full sparse self-attention for the batch=1 case:
 * fused qkv[N,3*dim] -> qkv_split_f32 -> sdpa_f32(q,k,v).
 * This is the CUDA equivalent of common/sparse3d.h::sp3d_attention for
 * the SLAT DiT transformer blocks when all active voxels belong to one
 * batch element.
 *
 * Usage:
 *   ./test_slat_self_attn [--N 1188] [--dim 1024] [--H 16] [--threshold 1e-3] [--repeat 20] [-v]
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

static void host_qkv_split(const float *qkv, int N, int dim,
                           float *q, float *k, float *v)
{
    for (int n = 0; n < N; n++) {
        const float *row = qkv + (size_t)n * 3 * dim;
        memcpy(q + (size_t)n * dim, row, (size_t)dim * sizeof(float));
        memcpy(k + (size_t)n * dim, row + dim, (size_t)dim * sizeof(float));
        memcpy(v + (size_t)n * dim, row + 2 * dim, (size_t)dim * sizeof(float));
    }
}

static void host_sdpa(float *out, const float *q, const float *k, const float *v,
                      int N, int H, int D_h, float scale)
{
    int E = H * D_h;
    double *scores = (double *)malloc((size_t)N * sizeof(double));
    for (int nq = 0; nq < N; nq++) {
        for (int h = 0; h < H; h++) {
            const float *qv = q + (size_t)nq * E + (size_t)h * D_h;
            double mx = -1.0e300;
            for (int nk = 0; nk < N; nk++) {
                const float *kv = k + (size_t)nk * E + (size_t)h * D_h;
                double s = 0.0;
                for (int d = 0; d < D_h; d++) s += (double)qv[d] * (double)kv[d];
                s *= (double)scale;
                scores[nk] = s;
                if (s > mx) mx = s;
            }
            double sum = 0.0;
            for (int nk = 0; nk < N; nk++) {
                scores[nk] = exp(scores[nk] - mx);
                sum += scores[nk];
            }
            double inv = 1.0 / sum;
            for (int d = 0; d < D_h; d++) {
                double acc = 0.0;
                for (int nk = 0; nk < N; nk++)
                    acc += scores[nk] * (double)v[(size_t)nk * E + (size_t)h * D_h + d];
                out[(size_t)nq * E + (size_t)h * D_h + d] = (float)(acc * inv);
            }
        }
    }
    free(scores);
}

int main(int argc, char **argv)
{
    int N = 1188;
    int dim = 1024;
    int H = 16;
    int repeat = 20;
    float threshold = 1e-3f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(a, "--H")         && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat")    && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (N <= 0 || dim <= 0 || H <= 0 || dim % H != 0) return 2;
    int D_h = dim / H;
    float scale = 1.0f / sqrtf((float)D_h);

    size_t qkv_n = (size_t)N * 3 * dim;
    size_t x_n = (size_t)N * dim;
    float *qkv = (float *)malloc(qkv_n * sizeof(float));
    float *q = (float *)malloc(x_n * sizeof(float));
    float *k = (float *)malloc(x_n * sizeof(float));
    float *v = (float *)malloc(x_n * sizeof(float));
    float *ref = (float *)malloc(x_n * sizeof(float));
    float *dst = (float *)malloc(x_n * sizeof(float));
    if (!qkv || !q || !k || !v || !ref || !dst) return 5;

    uint32_t rng = 0x5A7A77E7u;
    for (size_t i = 0; i < qkv_n; i++)
        qkv[i] = (urand(&rng) * 2.0f - 1.0f);
    host_qkv_split(qkv, N, dim, q, k, v);
    host_sdpa(ref, q, k, v, N, H, D_h, scale);

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0;
    if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    if (hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_slat_self_attn") < 0) return 4;
    hipFunction_t fn_split = NULL, fn_sdpa = NULL;
    if (hipModuleGetFunction(&fn_split, mod, "qkv_split_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_sdpa, mod, "sdpa_f32") != hipSuccess) {
        fprintf(stderr, "kernel lookup failed\n");
        return 4;
    }

    hipDeviceptr_t d_qkv = hip_upload_raw(qkv, qkv_n * sizeof(float));
    hipDeviceptr_t d_q = 0, d_k = 0, d_v = 0, d_out = 0;
    if (!d_qkv ||
        hipMalloc(&d_q, x_n * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_k, x_n * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_v, x_n * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_out, x_n * sizeof(float)) != hipSuccess) {
        fprintf(stderr, "device alloc/upload failed\n");
        return 5;
    }

    void *split_args[] = { &d_q, &d_k, &d_v, &d_qkv, &N, &dim };
    if (hipModuleLaunchKernel(fn_split, (int)((x_n + 255) / 256), 1, 1,
                       256, 1, 1, 0, 0, split_args, NULL) != hipSuccess) return 6;
    void *sdpa_args[] = { &d_out, &d_q, &d_k, &d_v, &N, &N, &H, &D_h, &scale };
    size_t smem = (size_t)(256 + N) * sizeof(float);
    if (hipModuleLaunchKernel(fn_sdpa, N, H, 1, 256, 1, 1,
                       (unsigned)smem, 0, sdpa_args, NULL) != hipSuccess) return 6;
    hipDeviceSynchronize();
    hipMemcpyDtoH(dst, d_out, x_n * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(dst, ref, x_n, &mean);
    int ok = mx <= threshold;

    double avg_ms = 0.0;
    if (repeat > 0) {
        hipDeviceSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) {
            if (hipModuleLaunchKernel(fn_sdpa, N, H, 1, 256, 1, 1,
                               (unsigned)smem, 0, sdpa_args, NULL) != hipSuccess) return 6;
        }
        hipDeviceSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }

    fprintf(stderr,
            "[test_slat_self_attn] N=%d dim=%d H=%d D_h=%d max_abs=%.4g mean_abs=%.4g sdpa_avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, dim, H, D_h, (double)mx, mean, avg_ms, repeat,
            ok ? "OK" : "FAIL", (double)threshold);

    free(qkv); free(q); free(k); free(v); free(ref); free(dst);
    hipFree(d_qkv); hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
