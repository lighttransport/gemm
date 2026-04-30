/*
 * test_ppe_window_pack — Phase 2b.4 standalone microbench.
 *
 * Validates ppe_window_pack_f32 at PointPatchEmbed geometry
 * (S=256, P=8, Np=32, D=512).  Pure F32 gather + add — bit-exact
 * against host reference.
 *
 * Usage:
 *   ./test_ppe_window_pack [--Np 32] [--P 8] [--D 512] [--threshold 1e-7] [-v]
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
    int Np = 32, P = 8, D = 512;
    float threshold = 1e-7f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--Np")        && i+1 < argc) Np        = atoi(argv[++i]);
        else if (!strcmp(a, "--P")         && i+1 < argc) P         = atoi(argv[++i]);
        else if (!strcmp(a, "--D")         && i+1 < argc) D         = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    int S    = Np * P;
    int WL   = 1 + P * P;
    int Nwin = Np * Np;

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0;
    if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    int sm = hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose, "test_ppe_window_pack");
    if (sm < 0) return 4;
    hipFunction_t fn = NULL;
    if (hipModuleGetFunction(&fn, mod, "ppe_window_pack_f32") != hipSuccess) return 4;

    size_t n_x   = (size_t)S * S * D;
    size_t n_pew = (size_t)WL * D;
    size_t n_out = (size_t)Nwin * WL * D;
    float *h_x   = (float *)malloc(n_x   * sizeof(float));
    float *h_cls = (float *)malloc((size_t)D * sizeof(float));
    float *h_pew = (float *)malloc(n_pew * sizeof(float));
    float *h_ref = (float *)malloc(n_out * sizeof(float));
    float *h_dst = (float *)malloc(n_out * sizeof(float));
    if (!h_x || !h_cls || !h_pew || !h_ref || !h_dst) return 5;

    uint32_t rng = 0xC0FFEEu;
    for (size_t i = 0; i < n_x;   i++) h_x[i]   = urand(&rng) * 2.0f - 1.0f;
    for (int    d = 0; d < D;     d++) h_cls[d] = urand(&rng) * 2.0f - 1.0f;
    for (size_t i = 0; i < n_pew; i++) h_pew[i] = urand(&rng) * 2.0f - 1.0f;

    /* Host reference. */
    for (int wy = 0; wy < Np; wy++) {
        for (int wx = 0; wx < Np; wx++) {
            int w = wy * Np + wx;
            float *wt = h_ref + (size_t)w * WL * D;
            for (int t = 0; t < WL; t++) {
                const float *src;
                if (t == 0) {
                    src = h_cls;
                } else {
                    int py = (t - 1) / P, px = (t - 1) - py * P;
                    src = h_x + (size_t)((wy * P + py) * S + (wx * P + px)) * D;
                }
                const float *pe = h_pew + (size_t)t * D;
                float *dst = wt + (size_t)t * D;
                for (int d = 0; d < D; d++) dst[d] = src[d] + pe[d];
            }
        }
    }

    hipDeviceptr_t d_x   = hip_upload_raw(h_x,   n_x   * sizeof(float));
    hipDeviceptr_t d_cls = hip_upload_raw(h_cls, (size_t)D * sizeof(float));
    hipDeviceptr_t d_pew = hip_upload_raw(h_pew, n_pew * sizeof(float));
    hipDeviceptr_t d_o = 0;
    if (hipMalloc(&d_o, n_out * sizeof(float)) != hipSuccess) return 5;

    int threads = 256;
    int blocks  = Nwin * WL;
    void *args[] = { &d_o, &d_x, &d_cls, &d_pew, &Np, &P, &D };
    if (hipModuleLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != hipSuccess) return 6;
    hipDeviceSynchronize();
    hipMemcpyDtoH(h_dst, d_o, n_out * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_out, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_ppe_window_pack] Np=%d P=%d D=%d  Nwin=%d WL=%d  n_out=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        Np, P, D, Nwin, WL, n_out, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_x); free(h_cls); free(h_pew); free(h_ref); free(h_dst);
    hipFree(d_x); hipFree(d_cls); hipFree(d_pew); hipFree(d_o);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
