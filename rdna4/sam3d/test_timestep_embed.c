/*
 * test_timestep_embed — Phase 2c.0 standalone microbench.
 *
 * Validates `timestep_embed_cossin_f32` against the host reference
 * `ssdit_freq_embed` (sam3d_ss_flow_dit.h) at SS DiT freq_dim=256
 * across a sweep of `t` values (covering both endpoints of the
 * pre-`time_scale` range and the shortcut `d`-jump magnitude).
 *
 * Usage:
 *   ./test_timestep_embed [--freq 256] [--threshold 1e-5] [-v]
 */

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include "hip_sam3d_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void host_freq_embed(float *out, float t, int freq_dim) {
    int half = freq_dim / 2;
    float neg_log10k = -logf(10000.0f);
    for (int j = 0; j < half; j++) {
        float freq = expf(neg_log10k * (float)j / (float)half);
        float arg = t * freq;
        out[j] = cosf(arg);
        out[half + j] = sinf(arg);
    }
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
    int   freq_dim  = 256;
    /* `__expf` vs libm `expf` drift, amplified by cos/sin at large
     * `arg = t * freq` near the [0, 1000] post-time_scale endpoint. */
    float threshold = 5e-4f;
    int   verbose   = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--freq")      && i+1 < argc) freq_dim  = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                        verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0; if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    if (hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_timestep_embed") < 0) return 4;
    hipFunction_t fn = 0;
    if (hipModuleGetFunction(&fn, mod, "timestep_embed_cossin_f32") != hipSuccess) {
        fprintf(stderr, "lookup timestep_embed_cossin_f32 failed\n"); return 4;
    }

    /* Sweep across timesteps spanning the post-time_scale range [0, 1000]
     * and a few shortcut-jump magnitudes (d ~ a few hundred). */
    static const float ts[] = {
        0.0f, 1.0f, 17.5f, 31.25f, 125.0f, 500.0f, 999.99f, 250.0f, 0.125f
    };
    int n_ts = (int)(sizeof(ts) / sizeof(ts[0]));
    int half = freq_dim / 2;

    hipDeviceptr_t d_out = 0;
    if (hipMalloc(&d_out, (size_t)freq_dim * sizeof(float)) != hipSuccess) return 5;
    float *h_dst = (float *)malloc((size_t)freq_dim * sizeof(float));
    float *h_ref = (float *)malloc((size_t)freq_dim * sizeof(float));

    float worst_mx = 0.0f; double worst_mean = 0.0;
    int ok = 1;
    for (int k = 0; k < n_ts; k++) {
        float t = ts[k];
        unsigned grid = (unsigned)((half + 255) / 256);
        void *args[] = { &d_out, &t, &freq_dim };
        if (hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) {
            fprintf(stderr, "launch failed t=%g\n", t); ok = 0; break;
        }
        hipDeviceSynchronize();
        hipMemcpyDtoH(h_dst, d_out, (size_t)freq_dim * sizeof(float));
        host_freq_embed(h_ref, t, freq_dim);
        double mean = 0.0;
        float mx = max_abs(h_dst, h_ref, freq_dim, &mean);
        if (verbose) fprintf(stderr, "  t=%g  max_abs=%.4g mean_abs=%.4g\n", t, (double)mx, mean);
        if (mx > worst_mx) { worst_mx = mx; worst_mean = mean; }
        if (mx > threshold) ok = 0;
    }
    fprintf(stderr,
        "[test_timestep_embed] freq_dim=%d  worst over %d t-values: max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        freq_dim, n_ts, (double)worst_mx, worst_mean, ok ? "OK" : "FAIL", (double)threshold);

    free(h_dst); free(h_ref);
    hipFree(d_out);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
