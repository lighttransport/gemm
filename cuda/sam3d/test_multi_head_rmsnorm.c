/*
 * test_multi_head_rmsnorm — Phase 2c.4 standalone microbench.
 *
 * Validates `multi_head_rmsnorm_f32` against the host reference
 * `ssdit_mhrmsnorm` (sam3d_ss_flow_dit.h). Note: NOT classic RMSNorm —
 * upstream uses F.normalize (L2 norm over head_dim) then multiplies by
 * γ[H, head_dim] AND sqrt(head_dim).
 *
 * Tests both contiguous layout (stride = H*D) and packed-QKV layout
 * (stride = 3*H*D) to exercise the stride parameter — the latter is the
 * intended use site, applying RMS norm to Q and K rows of a fused QKV
 * activation buffer without extracting them first.
 *
 * Host ref uses double accumulation; device uses float — the resulting
 * f32 sum-of-squares drift sets the threshold floor.
 *
 * Usage:
 *   ./test_multi_head_rmsnorm [--ntok 4096] [--heads 16] [--hd 64] [--threshold 5e-5] [-v]
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

static float urand(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return (float)((*s) >> 8) / (float)(1u << 24);
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

static void host_mhrmsnorm(float *v, int n_tok, int n_heads, int head_dim,
                           int stride, const float *gamma) {
    float scale = sqrtf((float)head_dim);
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *x = v + (size_t)t * stride + h * head_dim;
            const float *g = gamma + h * head_dim;
            double ss = 0.0;
            for (int i = 0; i < head_dim; i++) ss += (double)x[i] * x[i];
            float inv = 1.0f / (sqrtf((float)ss) + 1e-12f);
            for (int i = 0; i < head_dim; i++) x[i] = x[i] * inv * g[i] * scale;
        }
    }
}

static int run_case(CUfunction fn, const char *label, int n_tok, int n_heads,
                    int head_dim, int stride, float threshold, int verbose)
{
    size_t Nx = (size_t)n_tok * stride;
    uint32_t rng = 0xC0FFEEu;
    float *x_init = (float *)malloc(Nx * sizeof(float));
    float *gamma  = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
    for (size_t i = 0; i < Nx; i++) x_init[i] = urand(&rng) * 2.0f - 1.0f;
    for (int i = 0; i < n_heads * head_dim; i++) gamma[i] = (urand(&rng) * 2.0f - 1.0f) * 0.5f;

    /* Host ref: full buffer copy so unused stride lanes pass through. */
    float *h_ref = (float *)malloc(Nx * sizeof(float));
    memcpy(h_ref, x_init, Nx * sizeof(float));
    host_mhrmsnorm(h_ref, n_tok, n_heads, head_dim, stride, gamma);

    CUdeviceptr d_x = cu_upload_raw(x_init, Nx * sizeof(float));
    CUdeviceptr d_g = cu_upload_raw(gamma,  (size_t)n_heads * head_dim * sizeof(float));

    unsigned threads = 64;
    if (head_dim < 64) threads = 32;
    void *args[] = { &d_x, &d_g, &n_tok, &n_heads, &head_dim, &stride };
    size_t smem = threads * sizeof(float);
    if (cuLaunchKernel(fn, (unsigned)n_heads, (unsigned)n_tok, 1,
                       threads, 1, 1, (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "launch failed (%s)\n", label); return 1;
    }
    cuCtxSynchronize();

    float *h_dst = (float *)malloc(Nx * sizeof(float));
    cuMemcpyDtoH(h_dst, d_x, Nx * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, Nx, &mean);
    int ok = (mx <= threshold);

    if (verbose || !ok) {
        fprintf(stderr,
            "  [%s] n_tok=%d H=%d D_h=%d stride=%d  max_abs=%.4g mean_abs=%.4g  %s\n",
            label, n_tok, n_heads, head_dim, stride, (double)mx, mean, ok ? "OK" : "FAIL");
    }
    free(x_init); free(gamma); free(h_ref); free(h_dst);
    cuMemFree(d_x); cuMemFree(d_g);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    int   n_tok     = 4096;
    int   n_heads   = 16;
    int   head_dim  = 64;
    float threshold = 5e-5f;
    int   verbose   = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--ntok")      && i+1 < argc) n_tok     = atoi(argv[++i]);
        else if (!strcmp(a, "--heads")     && i+1 < argc) n_heads   = atoi(argv[++i]);
        else if (!strcmp(a, "--hd")        && i+1 < argc) head_dim  = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                        verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0; if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose, "test_multi_head_rmsnorm") < 0) return 4;
    CUfunction fn = 0;
    if (cuModuleGetFunction(&fn, mod, "multi_head_rmsnorm_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup multi_head_rmsnorm_f32 failed\n"); return 4;
    }

    int hd = n_heads * head_dim;
    int fail = 0;
    fail |= run_case(fn, "contig",     n_tok, n_heads, head_dim, hd,     threshold, verbose);
    fail |= run_case(fn, "qkv-packed", n_tok, n_heads, head_dim, 3 * hd, threshold, verbose);

    fprintf(stderr,
        "[test_multi_head_rmsnorm] H=%d D_h=%d  %s (threshold %.1g)\n",
        n_heads, head_dim, fail ? "FAIL" : "OK", (double)threshold);

    cuCtxDestroy(cu_ctx);
    return fail ? 9 : 0;
}
