/*
 * test_fuser_proj — Phase 2b.1 standalone microbench.
 *
 * Composes LN → gemm(w1) + gemm(w3) → silu_mul → gemm(w2) on
 * random-init weights at CondFuser projection geometry
 * (n_tok=1370, D_in=1024, ffn=2816, D_out=1024).
 *
 * Validates that the four kernels chain correctly end-to-end against a
 * host reference (double-precision accumulators on the gemms, libm exp
 * on the silu). No bias on w1/w2/w3 (Llama3 SwiGLU FFN convention).
 *
 * Usage:
 *   ./test_fuser_proj [--n_tok 1370] [--d_in 1024] [--ffn 2816]
 *                     [--d_out 1024] [--threshold 1e-3] [-v]
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
static float nrand(uint32_t *s) {
    /* Crude Box-Muller for unit-variance noise. */
    float u1 = urand(s); if (u1 < 1e-7f) u1 = 1e-7f;
    float u2 = urand(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
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
    int n_tok = 1370, D_in = 1024, ffn = 2816, D_out = 1024;
    float threshold = 1e-3f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--n_tok")     && i+1 < argc) n_tok     = atoi(argv[++i]);
        else if (!strcmp(a, "--d_in")      && i+1 < argc) D_in      = atoi(argv[++i]);
        else if (!strcmp(a, "--ffn")       && i+1 < argc) ffn       = atoi(argv[++i]);
        else if (!strcmp(a, "--d_out")     && i+1 < argc) D_out     = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    int sm = cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose, "test_fuser_proj");
    if (sm < 0) return 4;
    CUfunction fn_ln = 0, fn_gemm = 0, fn_silu = 0;
    if (cuModuleGetFunction(&fn_ln,   mod, "layernorm_token_f32") != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias")        != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn_silu, mod, "silu_mul_f32")         != CUDA_SUCCESS) return 4;

    /* ---- Host inputs (x), weights (gamma, beta, w1, w2, w3). ---- */
    size_t n_x = (size_t)n_tok * D_in;
    size_t n_y = (size_t)n_tok * D_out;
    size_t n_h = (size_t)n_tok * ffn;
    float *h_x   = (float *)malloc(n_x * sizeof(float));
    float *h_g   = (float *)malloc((size_t)D_in * sizeof(float));
    float *h_b   = (float *)malloc((size_t)D_in * sizeof(float));
    float *h_w1  = (float *)malloc((size_t)ffn  * D_in  * sizeof(float));
    float *h_w3  = (float *)malloc((size_t)ffn  * D_in  * sizeof(float));
    float *h_w2  = (float *)malloc((size_t)D_out* ffn   * sizeof(float));
    float *h_ref = (float *)malloc(n_y * sizeof(float));
    float *h_dst = (float *)malloc(n_y * sizeof(float));
    if (!h_x || !h_g || !h_b || !h_w1 || !h_w3 || !h_w2 || !h_ref || !h_dst) return 5;

    uint32_t rng = 0xC0FFEEu;
    /* Inputs ~ unit variance. */
    for (size_t i = 0; i < n_x; i++) h_x[i] = nrand(&rng);
    /* LN weights ~ ones / zeros (typical init). */
    for (int c = 0; c < D_in; c++) { h_g[c] = 1.0f + 0.01f * nrand(&rng); h_b[c] = 0.01f * nrand(&rng); }
    /* He-init scaling for the gemms. */
    float s1 = sqrtf(2.0f / (float)D_in);
    float s2 = sqrtf(2.0f / (float)ffn);
    for (size_t i = 0; i < (size_t)ffn  * D_in;  i++) h_w1[i] = nrand(&rng) * s1;
    for (size_t i = 0; i < (size_t)ffn  * D_in;  i++) h_w3[i] = nrand(&rng) * s1;
    for (size_t i = 0; i < (size_t)D_out* ffn;   i++) h_w2[i] = nrand(&rng) * s2;

    /* ---- Host reference: LN, then a@w1^T, a@w3^T, silu_mul, then @w2^T. ---- */
    {
        float *ln  = (float *)malloc(n_x * sizeof(float));
        float *t1  = (float *)malloc(n_h * sizeof(float));
        float *t3  = (float *)malloc(n_h * sizeof(float));
        if (!ln || !t1 || !t3) return 5;
        for (int t = 0; t < n_tok; t++) {
            const float *xr = h_x + (size_t)t * D_in;
            double s = 0.0, ss = 0.0;
            for (int c = 0; c < D_in; c++) { s += xr[c]; ss += (double)xr[c] * xr[c]; }
            double m = s / D_in;
            double v = ss / D_in - m * m;
            double inv = 1.0 / sqrt(v + 1e-5);
            float *lr = ln + (size_t)t * D_in;
            for (int c = 0; c < D_in; c++)
                lr[c] = (float)(((double)xr[c] - m) * inv * (double)h_g[c] + (double)h_b[c]);
        }
        for (int t = 0; t < n_tok; t++) {
            const float *lr = ln + (size_t)t * D_in;
            for (int d = 0; d < ffn; d++) {
                const float *wr1 = h_w1 + (size_t)d * D_in;
                const float *wr3 = h_w3 + (size_t)d * D_in;
                double a1 = 0.0, a3 = 0.0;
                for (int k = 0; k < D_in; k++) { a1 += (double)wr1[k] * lr[k]; a3 += (double)wr3[k] * lr[k]; }
                t1[(size_t)t * ffn + d] = (float)a1;
                t3[(size_t)t * ffn + d] = (float)a3;
            }
        }
        for (size_t i = 0; i < n_h; i++) {
            double a = t1[i];
            double sig = 1.0 / (1.0 + exp(-a));
            t1[i] = (float)(a * sig * (double)t3[i]);
        }
        for (int t = 0; t < n_tok; t++) {
            const float *hr = t1 + (size_t)t * ffn;
            for (int d = 0; d < D_out; d++) {
                const float *wr = h_w2 + (size_t)d * ffn;
                double a = 0.0;
                for (int k = 0; k < ffn; k++) a += (double)wr[k] * hr[k];
                h_ref[(size_t)t * D_out + d] = (float)a;
            }
        }
        free(ln); free(t1); free(t3);
    }

    /* ---- Device. ---- */
    CUdeviceptr d_x  = cu_upload_raw(h_x,  n_x * sizeof(float));
    CUdeviceptr d_g  = cu_upload_raw(h_g,  (size_t)D_in * sizeof(float));
    CUdeviceptr d_b  = cu_upload_raw(h_b,  (size_t)D_in * sizeof(float));
    CUdeviceptr d_w1 = cu_upload_raw(h_w1, (size_t)ffn  * D_in  * sizeof(float));
    CUdeviceptr d_w3 = cu_upload_raw(h_w3, (size_t)ffn  * D_in  * sizeof(float));
    CUdeviceptr d_w2 = cu_upload_raw(h_w2, (size_t)D_out* ffn   * sizeof(float));
    CUdeviceptr d_ln = 0, d_t1 = 0, d_t3 = 0, d_y = 0;
    cuMemAlloc(&d_ln, n_x * sizeof(float));
    cuMemAlloc(&d_t1, n_h * sizeof(float));
    cuMemAlloc(&d_t3, n_h * sizeof(float));
    cuMemAlloc(&d_y,  n_y * sizeof(float));

    /* layernorm_token_f32 */
    {
        int affine = 1; float eps = 1e-5f;
        void *args[] = { &d_ln, &d_x, &d_g, &d_b, &n_tok, &D_in, &eps, &affine };
        int threads = 256;
        size_t shmem = 2 * threads * sizeof(float);
        if (cuLaunchKernel(fn_ln, n_tok, 1, 1, threads, 1, 1, shmem, 0, args, NULL) != CUDA_SUCCESS) return 6;
    }
    /* gemm w1 → t1 */
    CUdeviceptr d_null = 0;
    {
        void *args[] = { &d_t1, &d_ln, &d_w1, &d_null, &n_tok, &D_in, &ffn };
        if (cuLaunchKernel(fn_gemm, (n_tok+15)/16, (ffn+15)/16, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 6;
    }
    /* gemm w3 → t3 */
    {
        void *args[] = { &d_t3, &d_ln, &d_w3, &d_null, &n_tok, &D_in, &ffn };
        if (cuLaunchKernel(fn_gemm, (n_tok+15)/16, (ffn+15)/16, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 6;
    }
    /* silu_mul: t1 = silu(t1) * t3 */
    {
        int total = (int)n_h;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        void *args[] = { &d_t1, &d_t3, &d_t1, &total };
        if (cuLaunchKernel(fn_silu, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 6;
    }
    /* gemm w2 → y */
    {
        void *args[] = { &d_y, &d_t1, &d_w2, &d_null, &n_tok, &ffn, &D_out };
        if (cuLaunchKernel(fn_gemm, (n_tok+15)/16, (D_out+15)/16, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 6;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(h_dst, d_y, n_y * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_y, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_fuser_proj] n_tok=%d D_in=%d ffn=%d D_out=%d  n_out=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        n_tok, D_in, ffn, D_out, n_y, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_x); free(h_g); free(h_b); free(h_w1); free(h_w3); free(h_w2);
    free(h_ref); free(h_dst);
    cuMemFree(d_x); cuMemFree(d_g); cuMemFree(d_b);
    cuMemFree(d_w1); cuMemFree(d_w3); cuMemFree(d_w2);
    cuMemFree(d_ln); cuMemFree(d_t1); cuMemFree(d_t3); cuMemFree(d_y);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
