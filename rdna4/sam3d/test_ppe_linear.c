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

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include "hip_sam3d_kernels.h"

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

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0;
    if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    int sm = hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose, "test_ppe_linear");
    if (sm < 0) return 4;
    hipFunction_t fn = NULL;
    if (hipModuleGetFunction(&fn, mod, "ppe_linear3_invalid_f32") != hipSuccess) {
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

    hipDeviceptr_t d_p = hip_upload_raw(h_p, n_in * sizeof(float));
    hipDeviceptr_t d_w = hip_upload_raw(h_w, (size_t)D * 3 * sizeof(float));
    hipDeviceptr_t d_b = hip_upload_raw(h_b, (size_t)D     * sizeof(float));
    hipDeviceptr_t d_t = hip_upload_raw(h_t, (size_t)D     * sizeof(float));
    hipDeviceptr_t d_o = 0;
    if (hipMalloc(&d_o, n_out * sizeof(float)) != hipSuccess) return 5;

    int threads = 256;
    void *args[] = { &d_o, &d_p, &d_w, &d_b, &d_t, &n_pix, &D };
    if (hipModuleLaunchKernel(fn, n_pix, 1, 1, threads, 1, 1, 0, 0, args, NULL) != hipSuccess) return 6;
    hipDeviceSynchronize();
    hipMemcpyDtoH(h_dst, d_o, n_out * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_out, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_ppe_linear] S=%d D=%d  n_pix=%d invalid=%d  n_out=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        S, D, n_pix, n_invalid, n_out, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_p); free(h_w); free(h_b); free(h_t); free(h_ref); free(h_dst);
    hipFree(d_p); hipFree(d_w); hipFree(d_b); hipFree(d_t); hipFree(d_o);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
