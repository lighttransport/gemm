/*
 * test_dinov2_patch_embed — Phase 1b.1 standalone microbench.
 *
 * Generates a random CHW input image + random Conv2d weights/bias at
 * DINOv2-L geometry (ps=14, dim=1024, grid=37 → 518×518), runs the
 * `dinov2_patch_embed_f32` NVRTC kernel, diffs against a host reference
 * (matches the CPU dinov2_encode_core inner loop, double-prec accum
 * for tighter ground truth).
 *
 * Usage:
 *   ./test_dinov2_patch_embed [--ps 14] [--dim 1024] [--grid 37]
 *                             [--threads 256] [--threshold 1e-3] [-v]
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

static void host_patch_embed(float *out, const float *img, const float *w,
                             const float *b, int gw, int gh, int dim,
                             int ps, int img_w)
{
    int Co = dim, Ci = 3;
    for (int py = 0; py < gh; py++) {
        for (int px = 0; px < gw; px++) {
            int tok = py * gw + px;
            float *row = out + (size_t)tok * dim;
            for (int co = 0; co < Co; co++) {
                double sum = b ? b[co] : 0.0;
                for (int ci = 0; ci < Ci; ci++) {
                    for (int kh = 0; kh < ps; kh++) {
                        for (int kw = 0; kw < ps; kw++) {
                            sum += (double)w[((co * Ci + ci) * ps + kh) * ps + kw]
                                 * (double)img[ci * img_w * img_w
                                              + (py * ps + kh) * img_w
                                              + (px * ps + kw)];
                        }
                    }
                }
                row[co] = (float)sum;
            }
        }
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
    int   ps        = 14;
    int   dim       = 1024;
    int   grid      = 37;
    int   threads   = 256;
    float threshold = 1e-3f;
    int   verbose   = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--ps")        && i+1 < argc) ps        = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--grid")      && i+1 < argc) grid      = atoi(argv[++i]);
        else if (!strcmp(a, "--threads")   && i+1 < argc) threads   = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    int gh = grid, gw = grid;
    int img_w = grid * ps;
    int n_patches = gh * gw;

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
                                "test_dinov2_patch_embed");
    if (sm < 0) { fprintf(stderr, "compile failed\n"); return 4; }
    hipFunction_t fn = NULL;
    if (hipModuleGetFunction(&fn, mod, "dinov2_patch_embed_f32") != hipSuccess) {
        fprintf(stderr, "lookup dinov2_patch_embed_f32 failed\n"); return 4;
    }

    /* Inputs. */
    size_t n_img  = (size_t)3 * img_w * img_w;
    size_t n_w    = (size_t)dim * 3 * ps * ps;
    size_t n_out  = (size_t)n_patches * dim;
    float *h_img  = (float *)malloc(n_img  * sizeof(float));
    float *h_w    = (float *)malloc(n_w    * sizeof(float));
    float *h_b    = (float *)malloc((size_t)dim * sizeof(float));
    float *h_ref  = (float *)malloc(n_out  * sizeof(float));
    float *h_dst  = (float *)malloc(n_out  * sizeof(float));
    if (!h_img || !h_w || !h_b || !h_ref || !h_dst) {
        fprintf(stderr, "host alloc failed\n"); return 5;
    }
    uint32_t rng = 0xC0FFEEu;
    /* ImageNet-normalized inputs are roughly in [-2, 2]; mimic that. */
    for (size_t i = 0; i < n_img; i++) h_img[i] = (urand(&rng) * 2.0f - 1.0f) * 2.0f;
    /* Conv weights ~ N(0, sqrt(1/fan_in)) ≈ small. fan_in = 3*14*14 = 588. */
    float w_scale = 1.0f / sqrtf((float)(3 * ps * ps));
    for (size_t i = 0; i < n_w; i++) h_w[i] = (urand(&rng) * 2.0f - 1.0f) * w_scale;
    for (int c = 0; c < dim; c++)    h_b[c] = (urand(&rng) * 2.0f - 1.0f) * 0.1f;

    host_patch_embed(h_ref, h_img, h_w, h_b, gw, gh, dim, ps, img_w);

    hipDeviceptr_t d_img = hip_upload_raw(h_img, n_img * sizeof(float));
    hipDeviceptr_t d_w   = hip_upload_raw(h_w,   n_w   * sizeof(float));
    hipDeviceptr_t d_b   = hip_upload_raw(h_b,   (size_t)dim * sizeof(float));
    hipDeviceptr_t d_out = 0;
    if (hipMalloc(&d_out, n_out * sizeof(float)) != hipSuccess) {
        fprintf(stderr, "hipMalloc d_out failed\n"); return 5;
    }

    int base_tok = 0;
    void *args[] = { &d_out, &d_img, &d_w, &d_b,
                     &gw, &dim, &ps, &img_w, &base_tok };
    if (hipModuleLaunchKernel(fn,
                       n_patches, 1, 1,
                       threads, 1, 1,
                       0, 0, args, NULL) != hipSuccess) {
        fprintf(stderr, "launch failed\n"); return 6;
    }
    hipDeviceSynchronize();
    hipMemcpyDtoH(h_dst, d_out, n_out * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_out, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_dinov2_patch_embed] gh=%d gw=%d dim=%d ps=%d  "
        "n_out=%zu  max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        gh, gw, dim, ps, n_out, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_img); free(h_w); free(h_b); free(h_ref); free(h_dst);
    hipFree(d_img); hipFree(d_w); hipFree(d_b); hipFree(d_out);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
