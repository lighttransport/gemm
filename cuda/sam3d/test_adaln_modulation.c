/*
 * test_adaln_modulation — Phase 2c.2 standalone microbench.
 *
 * Validates the SS Flow DiT AdaLN_modulation block on-device:
 *   silu_inplace(t_emb_copy) → gemm_f32_bias(D → 6*D) → split into
 *   6 [D] modulation vectors (msa_shift/scale/gate, mlp_shift/scale/gate).
 *
 * The split is just contiguous slicing of the 6*D output — no new
 * kernel — so this test exercises the shared device kernels and
 * confirms callers can pull out modulation params via pointer offsets.
 *
 * Random weights (LCG seed 0xC0FFEE), host ref mirrors the same
 * silu→gemm composition. Threshold accounts for `__expf`/silu drift
 * propagated through one D→6*D gemm.
 *
 * Usage:
 *   ./test_adaln_modulation [--dim 1024] [--threshold 5e-5] [-v]
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

int main(int argc, char **argv)
{
    int   dim       = 1024;
    float threshold = 5e-5f;
    int   verbose   = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                        verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    int D6 = 6 * dim;

    /* Random t_emb in roughly the post-MLP range (~ unit variance). */
    uint32_t rng = 0xC0FFEEu;
    float *t_emb = (float *)malloc((size_t)dim * sizeof(float));
    for (int i = 0; i < dim; i++) t_emb[i] = (urand(&rng) * 2.0f - 1.0f);
    /* AdaLN weight initialized small (zero-init in pytorch); use small-ish here
     * to keep modulation in a realistic range without nuking signal. */
    float *W = (float *)malloc((size_t)D6 * dim * sizeof(float));
    float *b = (float *)calloc((size_t)D6, sizeof(float));
    float scale = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < D6 * dim; i++) W[i] = (urand(&rng) * 2.0f - 1.0f) * scale;
    for (int i = 0; i < D6;       i++) b[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0; if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose, "test_adaln_modulation") < 0) return 4;
    CUfunction fn_silu = 0, fn_gemm = 0;
    if (cuModuleGetFunction(&fn_silu, mod, "silu_inplace_f32") != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias")    != CUDA_SUCCESS) return 4;

    CUdeviceptr d_t_emb = cu_upload_raw(t_emb, (size_t)dim * sizeof(float));
    CUdeviceptr d_W     = cu_upload_raw(W,     (size_t)D6 * dim * sizeof(float));
    CUdeviceptr d_b     = cu_upload_raw(b,     (size_t)D6 * sizeof(float));
    CUdeviceptr d_scratch = 0, d_out = 0;
    cuMemAlloc(&d_scratch, (size_t)dim * sizeof(float));
    cuMemAlloc(&d_out,     (size_t)D6  * sizeof(float));

    /* AdaLN preserves the original t_emb for reuse across blocks; copy
     * to scratch before the in-place silu. */
    cuMemcpyDtoD(d_scratch, d_t_emb, (size_t)dim * sizeof(float));
    {
        unsigned grid = (unsigned)((dim + 255) / 256);
        void *args[] = { &d_scratch, &dim };
        if (cuLaunchKernel(fn_silu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    int N = 1;
    {
        unsigned gx = (N + 15) / 16, gy = (D6 + 15) / 16;
        void *args[] = { &d_out, &d_scratch, &d_W, &d_b, &N, &dim, &D6 };
        if (cuLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    cuCtxSynchronize();

    float *h_dst = (float *)malloc((size_t)D6 * sizeof(float));
    cuMemcpyDtoH(h_dst, d_out, (size_t)D6 * sizeof(float));

    /* Host ref. */
    float *silu_t = (float *)malloc((size_t)dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        float v = t_emb[i];
        silu_t[i] = v / (1.0f + expf(-v));
    }
    float *h_ref = (float *)malloc((size_t)D6 * sizeof(float));
    for (int d = 0; d < D6; d++) {
        float acc = b[d];
        const float *wr = W + (size_t)d * dim;
        for (int k = 0; k < dim; k++) acc += wr[k] * silu_t[k];
        h_ref[d] = acc;
    }

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, D6, &mean);
    int ok = (mx <= threshold);

    /* Sanity-check the split semantics: each [D]-slice should be pointer-
     * indexable as msa_shift/scale/gate, mlp_shift/scale/gate without any
     * additional kernel. Just verify slice 0 matches h_ref's first D bytes. */
    double slice_mean = 0.0;
    float slice_mx = max_abs(h_dst, h_ref, dim, &slice_mean);

    fprintf(stderr,
        "[test_adaln_modulation] dim=%d D6=%d  full max_abs=%.4g mean_abs=%.4g "
        "(slice0 max_abs=%.4g)  %s (threshold %.1g)\n",
        dim, D6, (double)mx, mean, (double)slice_mx,
        ok ? "OK" : "FAIL", (double)threshold);

    free(t_emb); free(W); free(b); free(silu_t);
    free(h_dst); free(h_ref);
    cuMemFree(d_t_emb); cuMemFree(d_W); cuMemFree(d_b);
    cuMemFree(d_scratch); cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
