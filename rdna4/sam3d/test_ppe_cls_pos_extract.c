/*
 * test_ppe_cls_pos_extract — Phase 2b.7 standalone microbench.
 *
 * Validates ppe_cls_pos_extract_f32 at PointPatchEmbed geometry
 * (Np=32, WL=65, D=512). Pure F32 gather + add — bit-exact against
 * host reference.
 *
 * Usage:
 *   ./test_ppe_cls_pos_extract [--Np 32] [--WL 65] [--D 512]
 *                              [--threshold 1e-7] [-v]
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
    int Np = 32, WL = 65, D = 512;
    float threshold = 1e-7f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--Np")        && i+1 < argc) Np        = atoi(argv[++i]);
        else if (!strcmp(a, "--WL")        && i+1 < argc) WL        = atoi(argv[++i]);
        else if (!strcmp(a, "--D")         && i+1 < argc) D         = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    int Nwin = Np * Np;

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0;
    if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    int sm = hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose, "test_ppe_cls_pos_extract");
    if (sm < 0) return 4;
    hipFunction_t fn = NULL;
    if (hipModuleGetFunction(&fn, mod, "ppe_cls_pos_extract_f32") != hipSuccess) return 4;

    size_t n_tok = (size_t)Nwin * WL * D;
    size_t n_pe  = (size_t)D * Nwin;
    size_t n_out = (size_t)Nwin * D;
    float *h_tok = (float *)malloc(n_tok * sizeof(float));
    float *h_pe  = (float *)malloc(n_pe  * sizeof(float));
    float *h_ref = (float *)malloc(n_out * sizeof(float));
    float *h_dst = (float *)malloc(n_out * sizeof(float));
    if (!h_tok || !h_pe || !h_ref || !h_dst) return 5;
    uint32_t rng = 0xC0FFEEu;
    for (size_t i = 0; i < n_tok; i++) h_tok[i] = urand(&rng) * 2.0f - 1.0f;
    for (size_t i = 0; i < n_pe;  i++) h_pe[i]  = urand(&rng) * 2.0f - 1.0f;

    /* Host reference. */
    int Np2 = Np * Np;
    for (int w = 0; w < Nwin; w++) {
        int wy = w / Np, wx = w - wy * Np;
        const float *cls_row = h_tok + (size_t)w * WL * D;
        float       *dst     = h_ref + (size_t)w * D;
        for (int d = 0; d < D; d++) {
            dst[d] = cls_row[d] + h_pe[(size_t)d * Np2 + wy * Np + wx];
        }
    }

    hipDeviceptr_t d_tok = hip_upload_raw(h_tok, n_tok * sizeof(float));
    hipDeviceptr_t d_pe  = hip_upload_raw(h_pe,  n_pe  * sizeof(float));
    hipDeviceptr_t d_o = 0;
    if (hipMalloc(&d_o, n_out * sizeof(float)) != hipSuccess) return 5;

    int threads = 256;
    void *args[] = { &d_o, &d_tok, &d_pe, &Np, &WL, &D };
    if (hipModuleLaunchKernel(fn, Nwin, 1, 1, threads, 1, 1, 0, 0, args, NULL) != hipSuccess) return 6;
    hipDeviceSynchronize();
    hipMemcpyDtoH(h_dst, d_o, n_out * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_out, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_ppe_cls_pos_extract] Np=%d WL=%d D=%d  Nwin=%d  n_out=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        Np, WL, D, Nwin, n_out, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_tok); free(h_pe); free(h_ref); free(h_dst);
    hipFree(d_tok); hipFree(d_pe); hipFree(d_o);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
