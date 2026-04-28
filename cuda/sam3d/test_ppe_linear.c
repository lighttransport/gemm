/*
 * test_ppe_linear — Phase 2b.3 standalone microbench.
 *
 * Validates ppe_linear3_invalid_f32 at PointPatchEmbed geometry
 * (S=256, D=512). Sprinkles ~10% NaN pixels in the pointmap to
 * exercise the invalid-token path.
 *
 * Usage:
 *   ./test_ppe_linear [--S 256] [--D 512] [--threshold 5e-6] [-v]
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

int main(int argc, char **argv)
{
    int S = 256, D = 512;
    float threshold = 5e-6f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--S")         && i+1 < argc) S         = atoi(argv[++i]);
        else if (!strcmp(a, "--D")         && i+1 < argc) D         = atoi(argv[++i]);
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
                                "sam3d_kernels.cu", verbose, "test_ppe_linear");
    if (sm < 0) return 4;
    CUfunction fn = NULL;
    if (cuModuleGetFunction(&fn, mod, "ppe_linear3_invalid_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup failed\n"); return 4;
    }

    int n_pix = S * S;
    size_t n_in  = (size_t)n_pix * 3;
    size_t n_out = (size_t)n_pix * D;
    float *h_p   = (float *)malloc(n_in  * sizeof(float));
    float *h_w   = (float *)malloc((size_t)D * 3 * sizeof(float));
    float *h_b   = (float *)malloc((size_t)D     * sizeof(float));
    float *h_t   = (float *)malloc((size_t)D     * sizeof(float));
    float *h_ref = (float *)malloc(n_out * sizeof(float));
    float *h_dst = (float *)malloc(n_out * sizeof(float));
    if (!h_p || !h_w || !h_b || !h_t || !h_ref || !h_dst) return 5;

    uint32_t rng = 0xC0FFEEu;
    for (size_t i = 0; i < n_in; i++) h_p[i] = urand(&rng) * 4.0f - 2.0f;
    /* ~10% pixels invalid (NaN x). */
    int n_invalid = 0;
    for (int p = 0; p < n_pix; p++) {
        if (urand(&rng) < 0.10f) { h_p[(size_t)p * 3 + 0] = NAN; n_invalid++; }
    }
    for (size_t i = 0; i < (size_t)D * 3; i++) h_w[i] = urand(&rng) * 0.5f - 0.25f;
    for (int d = 0; d < D; d++) h_b[d] = urand(&rng) * 0.1f - 0.05f;
    for (int d = 0; d < D; d++) h_t[d] = urand(&rng) * 0.5f - 0.25f;

    /* Host reference. */
    for (int p = 0; p < n_pix; p++) {
        const float *src = h_p + (size_t)p * 3;
        float *or_ = h_ref + (size_t)p * D;
        int valid = isfinite(src[0]) && isfinite(src[1]) && isfinite(src[2]);
        if (!valid) {
            memcpy(or_, h_t, (size_t)D * sizeof(float));
        } else {
            for (int d = 0; d < D; d++) {
                const float *wr = h_w + (size_t)d * 3;
                or_[d] = h_b[d] + wr[0] * src[0] + wr[1] * src[1] + wr[2] * src[2];
            }
        }
    }

    CUdeviceptr d_p = cu_upload_raw(h_p, n_in * sizeof(float));
    CUdeviceptr d_w = cu_upload_raw(h_w, (size_t)D * 3 * sizeof(float));
    CUdeviceptr d_b = cu_upload_raw(h_b, (size_t)D     * sizeof(float));
    CUdeviceptr d_t = cu_upload_raw(h_t, (size_t)D     * sizeof(float));
    CUdeviceptr d_o = 0;
    if (cuMemAlloc(&d_o, n_out * sizeof(float)) != CUDA_SUCCESS) return 5;

    int threads = 256;
    void *args[] = { &d_o, &d_p, &d_w, &d_b, &d_t, &n_pix, &D };
    if (cuLaunchKernel(fn, n_pix, 1, 1, threads, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 6;
    cuCtxSynchronize();
    cuMemcpyDtoH(h_dst, d_o, n_out * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_out, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_ppe_linear] S=%d D=%d  n_pix=%d invalid=%d  n_out=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        S, D, n_pix, n_invalid, n_out, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_p); free(h_w); free(h_b); free(h_t); free(h_ref); free(h_dst);
    cuMemFree(d_p); cuMemFree(d_w); cuMemFree(d_b); cuMemFree(d_t); cuMemFree(d_o);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
