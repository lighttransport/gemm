/*
 * test_layernorm_token — standalone NVRTC test for the
 * `layernorm_token_f32` kernel in hip_sam3d_kernels.h.
 *
 * Phase 1b.0: validates the foundational LN kernel that DINOv2,
 * CondFuser, SS-DiT, SLAT-DiT, and SLAT-GS-decoder will all consume
 * once per-stage GPU forwards land.
 *
 * Generates a deterministic random input [n_tokens, dim] f32, runs the
 * kernel + a host reference, diffs.
 *
 * Usage:
 *   ./test_layernorm_token [--n N] [--dim D] [--eps F] [--threshold F] [-v]
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

static void host_layernorm_token(float *dst, const float *src,
                                 const float *gamma, const float *beta,
                                 int n_tokens, int dim, float eps, int affine)
{
    for (int t = 0; t < n_tokens; t++) {
        const float *row_in  = src + (size_t)t * dim;
        float       *row_out = dst + (size_t)t * dim;
        double sum = 0.0, sq = 0.0;
        for (int c = 0; c < dim; c++) { float v = row_in[c]; sum += v; sq += (double)v * v; }
        float mean = (float)(sum / dim);
        float var  = (float)(sq / dim) - mean * mean;
        float inv  = 1.0f / sqrtf(var + eps);
        if (affine) {
            for (int c = 0; c < dim; c++)
                row_out[c] = (row_in[c] - mean) * inv * gamma[c] + beta[c];
        } else {
            for (int c = 0; c < dim; c++)
                row_out[c] = (row_in[c] - mean) * inv;
        }
    }
}

static float urand(uint32_t *state) {
    *state = (*state) * 1664525u + 1013904223u;
    return (float)((*state) >> 8) / (float)(1u << 24);
}

static float max_abs(const float *a, const float *b, int n, double *mean_out) {
    double sum = 0.0; float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    if (mean_out) *mean_out = sum / (n > 0 ? n : 1);
    return mx;
}

static int launch_ln(hipFunction_t fn, hipDeviceptr_t d_dst, hipDeviceptr_t d_src,
                     hipDeviceptr_t d_gamma, hipDeviceptr_t d_beta,
                     int n_tokens, int dim, float eps, int affine, int threads)
{
    void *args[] = { &d_dst, &d_src, &d_gamma, &d_beta,
                     &n_tokens, &dim, &eps, &affine };
    size_t shmem_bytes = (size_t)threads * 2 * sizeof(float);
    return hipModuleLaunchKernel(fn,
                          n_tokens, 1, 1,
                          threads, 1, 1,
                          (unsigned int)shmem_bytes, 0, args, NULL);
}

int main(int argc, char **argv) {
    int    n_tokens = 1374;          /* DINOv2-L/14+reg token count */
    int    dim      = 1024;          /* DINOv2-L embed dim          */
    float  eps      = 1e-6f;
    float  threshold = 5e-5f;
    int    verbose  = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--n")        && i+1 < argc) n_tokens = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")      && i+1 < argc) dim      = atoi(argv[++i]);
        else if (!strcmp(a, "--eps")      && i+1 < argc) eps      = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--threshold")&& i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose  = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 3;
    }
    if (hipInit(0) != hipSuccess) { fprintf(stderr, "hipInit failed\n"); return 3; }
    hipDevice_t dev = 0;
    if ((dev = (0), hipSetDevice(0)) != hipSuccess) { fprintf(stderr, "cuDeviceGet failed\n"); return 3; }
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) {
        fprintf(stderr, "hipCtxCreate failed\n"); return 3;
    }

    hipModule_t mod = 0;
    int sm = hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose,
                                "test_layernorm_token");
    if (sm < 0) { fprintf(stderr, "compile failed\n"); return 4; }

    hipFunction_t fn = NULL;
    if (hipModuleGetFunction(&fn, mod, "layernorm_token_f32") != hipSuccess) {
        fprintf(stderr, "lookup layernorm_token_f32 failed\n"); return 4;
    }

    size_t n_elt = (size_t)n_tokens * dim;
    float *h_src    = (float *)malloc(n_elt * sizeof(float));
    float *h_gamma  = (float *)malloc((size_t)dim * sizeof(float));
    float *h_beta   = (float *)malloc((size_t)dim * sizeof(float));
    float *h_ref    = (float *)malloc(n_elt * sizeof(float));
    float *h_dst    = (float *)malloc(n_elt * sizeof(float));
    uint32_t rng = 0xC0FFEEu;
    for (size_t i = 0; i < n_elt; i++)        h_src[i]   = urand(&rng) * 2.0f - 1.0f;
    for (int c = 0; c < dim; c++) {
        h_gamma[c] = 0.5f + urand(&rng);
        h_beta [c] = urand(&rng) * 0.1f - 0.05f;
    }

    host_layernorm_token(h_ref, h_src, h_gamma, h_beta, n_tokens, dim, eps, 1);

    hipDeviceptr_t d_src   = hip_upload_raw(h_src,   n_elt    * sizeof(float));
    hipDeviceptr_t d_gamma = hip_upload_raw(h_gamma, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_beta  = hip_upload_raw(h_beta,  (size_t)dim * sizeof(float));
    hipDeviceptr_t d_dst   = 0;
    if (hipMalloc(&d_dst, n_elt * sizeof(float)) != hipSuccess) {
        fprintf(stderr, "hipMalloc d_dst failed\n"); return 5;
    }

    int threads = 256;
    if (launch_ln(fn, d_dst, d_src, d_gamma, d_beta,
                  n_tokens, dim, eps, 1, threads) != hipSuccess) {
        fprintf(stderr, "launch (affine=1) failed\n"); return 6;
    }
    hipDeviceSynchronize();
    hipMemcpyDtoH(h_dst, d_dst, n_elt * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, (int)n_elt, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
            "[test_layernorm_token] affine=1 n=%d dim=%d eps=%g  "
            "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
            n_tokens, dim, (double)eps, (double)mx, mean,
            ok ? "OK" : "FAIL", (double)threshold);

    /* affine=0 sanity: per-token mean ≈ 0, var ≈ 1. */
    if (launch_ln(fn, d_dst, d_src, d_gamma, d_beta,
                  n_tokens, dim, eps, 0, threads) == hipSuccess) {
        hipDeviceSynchronize();
        hipMemcpyDtoH(h_dst, d_dst, n_elt * sizeof(float));
        double sum_mu = 0.0, sum_var = 0.0;
        for (int t = 0; t < n_tokens; t++) {
            const float *r = h_dst + (size_t)t * dim;
            double mu = 0.0;
            for (int c = 0; c < dim; c++) mu += r[c];
            mu /= dim;
            double v = 0.0;
            for (int c = 0; c < dim; c++) v += (r[c] - mu) * (r[c] - mu);
            v /= dim;
            sum_mu  += fabs(mu);
            sum_var += fabs(v - 1.0);
        }
        fprintf(stderr,
                "[test_layernorm_token] affine=0 stats: |mean|/tok=%.4g  "
                "|var-1|/tok=%.4g\n",
                sum_mu / n_tokens, sum_var / n_tokens);
    }

    free(h_src); free(h_gamma); free(h_beta); free(h_ref); free(h_dst);
    hipFree(d_src); hipFree(d_gamma); hipFree(d_beta); hipFree(d_dst);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
