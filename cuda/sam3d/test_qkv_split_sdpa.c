/*
 * test_qkv_split_sdpa — Phase 1b.4b standalone microbench.
 *
 * Exercises qkv_split_f32 + sdpa_f32 end-to-end on the DINOv2-L
 * geometry: N=1374 (n_tokens), dim=1024, H=16 heads, D_h=64.
 *
 * Builds a random fused-QKV [N, 3*dim] buffer, splits it via
 * qkv_split_f32, runs sdpa_f32 (scale = 1/sqrt(D_h)), and diffs the
 * output against a host softmax-attention reference. Two thresholds:
 *   - split bit-exact (max_abs == 0)
 *   - sdpa  ≤ threshold (default 1e-3, accounts for D_h=64 muladd
 *     drift + N_k=1374 softmax accumulation)
 *
 * Usage:
 *   ./test_qkv_split_sdpa [--N 1374] [--dim 1024] [--H 16]
 *                         [--threshold 1e-3] [-v]
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

/* Host SDPA reference (double accumulator). q/k/v laid out [N, H*D_h]. */
static void host_sdpa(float *out, const float *q, const float *k, const float *v,
                      int N_q, int N_k, int H, int D_h, float scale)
{
    int E = H * D_h;
    double *scores = (double *)malloc((size_t)N_k * sizeof(double));
    for (int nq = 0; nq < N_q; nq++) {
        for (int h = 0; h < H; h++) {
            const float *qv = q + (size_t)nq * E + (size_t)h * D_h;
            double mx = -1e300;
            for (int nk = 0; nk < N_k; nk++) {
                const float *kv = k + (size_t)nk * E + (size_t)h * D_h;
                double s = 0.0;
                for (int d = 0; d < D_h; d++) s += (double)qv[d] * (double)kv[d];
                s *= (double)scale;
                scores[nk] = s;
                if (s > mx) mx = s;
            }
            double sum = 0.0;
            for (int nk = 0; nk < N_k; nk++) {
                scores[nk] = exp(scores[nk] - mx);
                sum += scores[nk];
            }
            double inv = 1.0 / sum;
            for (int d = 0; d < D_h; d++) {
                double acc = 0.0;
                for (int nk = 0; nk < N_k; nk++)
                    acc += scores[nk] * (double)v[(size_t)nk * E + (size_t)h * D_h + d];
                out[(size_t)nq * E + (size_t)h * D_h + d] = (float)(acc * inv);
            }
        }
    }
    free(scores);
}

int main(int argc, char **argv)
{
    int   N         = 1374;
    int   dim       = 1024;
    int   H         = 16;
    float threshold = 1e-3f;
    int   verbose   = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i+1 < argc) N         = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--H")         && i+1 < argc) H         = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    if (dim % H != 0) { fprintf(stderr, "dim (%d) must be divisible by H (%d)\n", dim, H); return 2; }
    int D_h = dim / H;

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
                                "test_qkv_split_sdpa");
    if (sm < 0) { fprintf(stderr, "compile failed\n"); return 4; }
    CUfunction fn_split = NULL, fn_sdpa = NULL;
    if (cuModuleGetFunction(&fn_split, mod, "qkv_split_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup qkv_split_f32 failed\n"); return 4;
    }
    if (cuModuleGetFunction(&fn_sdpa, mod, "sdpa_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup sdpa_f32 failed\n"); return 4;
    }

    /* Inputs. Fused-QKV [N, 3*dim], He-init scale 1/sqrt(D_h) so attention
     * scores stay in a sane range. */
    size_t n_qkv = (size_t)N * 3 * dim;
    size_t n_act = (size_t)N * dim;
    float *h_qkv  = (float *)malloc(n_qkv * sizeof(float));
    float *h_Q    = (float *)malloc(n_act * sizeof(float));
    float *h_K    = (float *)malloc(n_act * sizeof(float));
    float *h_V    = (float *)malloc(n_act * sizeof(float));
    float *h_Q_d  = (float *)malloc(n_act * sizeof(float));
    float *h_K_d  = (float *)malloc(n_act * sizeof(float));
    float *h_V_d  = (float *)malloc(n_act * sizeof(float));
    float *h_ref  = (float *)malloc(n_act * sizeof(float));
    float *h_dst  = (float *)malloc(n_act * sizeof(float));
    if (!h_qkv || !h_Q || !h_K || !h_V || !h_Q_d || !h_K_d || !h_V_d || !h_ref || !h_dst) {
        fprintf(stderr, "host alloc failed\n"); return 5;
    }
    uint32_t rng = 0xC0FFEEu;
    float scl = 1.0f / sqrtf((float)D_h);
    for (size_t i = 0; i < n_qkv; i++) h_qkv[i] = (urand(&rng) * 2.0f - 1.0f) * scl;

    /* Host split: stride-3 over rows. */
    for (int t = 0; t < N; t++) {
        const float *row = h_qkv + (size_t)t * 3 * dim;
        memcpy(h_Q + (size_t)t * dim, row,             (size_t)dim * sizeof(float));
        memcpy(h_K + (size_t)t * dim, row + dim,       (size_t)dim * sizeof(float));
        memcpy(h_V + (size_t)t * dim, row + 2 * dim,   (size_t)dim * sizeof(float));
    }
    host_sdpa(h_ref, h_Q, h_K, h_V, N, N, H, D_h, scl);

    CUdeviceptr d_qkv = cu_upload_raw(h_qkv, n_qkv * sizeof(float));
    CUdeviceptr d_Q = 0, d_K = 0, d_V = 0, d_out = 0;
    if (cuMemAlloc(&d_Q,   n_act * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_K,   n_act * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_V,   n_act * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_out, n_act * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAlloc failed\n"); return 5;
    }

    /* qkv_split: 1D grid over N*dim. */
    {
        int total = N * dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        void *args[] = { &d_Q, &d_K, &d_V, &d_qkv, &N, &dim };
        if (cuLaunchKernel(fn_split,
                           blocks, 1, 1,
                           threads, 1, 1,
                           0, 0, args, NULL) != CUDA_SUCCESS) {
            fprintf(stderr, "launch qkv_split failed\n"); return 6;
        }
    }
    cuCtxSynchronize();

    /* Verify split bit-exact before SDPA so a split bug doesn't masquerade
     * as an SDPA drift. */
    cuMemcpyDtoH(h_Q_d, d_Q, n_act * sizeof(float));
    cuMemcpyDtoH(h_K_d, d_K, n_act * sizeof(float));
    cuMemcpyDtoH(h_V_d, d_V, n_act * sizeof(float));
    double dummy = 0.0;
    float mx_q = max_abs(h_Q_d, h_Q, n_act, &dummy);
    float mx_k = max_abs(h_K_d, h_K, n_act, &dummy);
    float mx_v = max_abs(h_V_d, h_V, n_act, &dummy);
    int split_ok = (mx_q == 0.0f && mx_k == 0.0f && mx_v == 0.0f);

    /* sdpa: grid (N, H), block 256, shmem (256 + N_k) * 4. */
    {
        int N_q = N, N_k = N;
        int threads = 256;
        unsigned shmem = (unsigned)((threads + N_k) * sizeof(float));
        void *args[] = { &d_out, &d_Q, &d_K, &d_V, &N_q, &N_k, &H, &D_h, &scl };
        if (cuLaunchKernel(fn_sdpa,
                           N_q, H, 1,
                           threads, 1, 1,
                           shmem, 0, args, NULL) != CUDA_SUCCESS) {
            fprintf(stderr, "launch sdpa failed\n"); return 6;
        }
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(h_dst, d_out, n_act * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_act, &mean);
    int sdpa_ok = (mx <= threshold);
    int ok = split_ok && sdpa_ok;
    fprintf(stderr,
        "[test_qkv_split_sdpa] N=%d dim=%d H=%d D_h=%d  "
        "split max_abs=(Q %.0f K %.0f V %.0f) %s  "
        "sdpa max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        N, dim, H, D_h,
        (double)mx_q, (double)mx_k, (double)mx_v,
        split_ok ? "BIT-EXACT" : "FAIL",
        (double)mx, mean,
        sdpa_ok ? "OK" : "FAIL", (double)threshold);

    free(h_qkv); free(h_Q); free(h_K); free(h_V);
    free(h_Q_d); free(h_K_d); free(h_V_d); free(h_ref); free(h_dst);
    cuMemFree(d_qkv); cuMemFree(d_Q); cuMemFree(d_K); cuMemFree(d_V); cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
