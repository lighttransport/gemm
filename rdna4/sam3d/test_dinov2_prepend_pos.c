/*
 * test_dinov2_prepend_pos — Phase 1b.2 standalone microbench.
 *
 * Builds a [n_tokens, dim] activation buffer with random patch rows
 * pre-filled, then runs `dinov2_prepend_cls_reg_f32` followed by
 * `dinov2_add_pos_embed_f32`. Diffs against a host reference that does
 * the exact same operations.
 *
 * Usage:
 *   ./test_dinov2_prepend_pos [--dim 1024] [--nr 4] [--np 1369]
 *                             [--threads 256] [--threshold 1e-6] [-v]
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
    int   dim       = 1024;
    int   nr        = 4;
    int   np        = 1369;     /* 37 * 37 */
    int   threads   = 256;
    float threshold = 1e-6f;    /* pure copy/add, F32; bit-exact expected */
    int   verbose   = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--nr")        && i+1 < argc) nr        = atoi(argv[++i]);
        else if (!strcmp(a, "--np")        && i+1 < argc) np        = atoi(argv[++i]);
        else if (!strcmp(a, "--threads")   && i+1 < argc) threads   = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    int nt = 1 + nr + np;

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
                                "test_dinov2_prepend_pos");
    if (sm < 0) { fprintf(stderr, "compile failed\n"); return 4; }
    hipFunction_t fn_prep = NULL, fn_pos = NULL;
    if (hipModuleGetFunction(&fn_prep, mod, "dinov2_prepend_cls_reg_f32") != hipSuccess) {
        fprintf(stderr, "lookup dinov2_prepend_cls_reg_f32 failed\n"); return 4;
    }
    if (hipModuleGetFunction(&fn_pos, mod, "dinov2_add_pos_embed_f32") != hipSuccess) {
        fprintf(stderr, "lookup dinov2_add_pos_embed_f32 failed\n"); return 4;
    }

    /* Inputs. */
    size_t n_buf = (size_t)nt * dim;
    size_t n_pe  = (size_t)(1 + np) * dim;
    float *h_buf_init = (float *)malloc(n_buf * sizeof(float));
    float *h_cls      = (float *)malloc((size_t)dim * sizeof(float));
    float *h_reg      = nr > 0 ? (float *)malloc((size_t)nr * dim * sizeof(float)) : NULL;
    float *h_pe       = (float *)malloc(n_pe  * sizeof(float));
    float *h_ref      = (float *)malloc(n_buf * sizeof(float));
    float *h_dst      = (float *)malloc(n_buf * sizeof(float));
    if (!h_buf_init || !h_cls || (nr > 0 && !h_reg) || !h_pe || !h_ref || !h_dst) {
        fprintf(stderr, "host alloc failed\n"); return 5;
    }
    uint32_t rng = 0xC0FFEEu;
    for (size_t i = 0; i < n_buf; i++) h_buf_init[i] = urand(&rng) * 2.0f - 1.0f;
    for (int c = 0; c < dim; c++)      h_cls[c]      = urand(&rng) * 2.0f - 1.0f;
    if (h_reg) {
        for (int i = 0; i < nr * dim; i++) h_reg[i] = urand(&rng) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < n_pe; i++)  h_pe[i]       = urand(&rng) * 2.0f - 1.0f;

    /* ---- Host reference. ---- */
    memcpy(h_ref, h_buf_init, n_buf * sizeof(float));
    /* prepend CLS + register tokens. */
    memcpy(h_ref, h_cls, (size_t)dim * sizeof(float));
    for (int r = 0; r < nr; r++) {
        memcpy(h_ref + (size_t)(1 + r) * dim, h_reg + (size_t)r * dim,
               (size_t)dim * sizeof(float));
    }
    /* add pos_embed to CLS (row 0). */
    for (int c = 0; c < dim; c++) h_ref[c] += h_pe[c];
    /* add pos_embed[1+p] to patch row (1 + nr + p). */
    for (int p = 0; p < np; p++) {
        float       *dst = h_ref + (size_t)(1 + nr + p) * dim;
        const float *src = h_pe  + (size_t)(1 + p) * dim;
        for (int c = 0; c < dim; c++) dst[c] += src[c];
    }

    /* ---- Device path. ---- */
    hipDeviceptr_t d_buf = hip_upload_raw(h_buf_init, n_buf * sizeof(float));
    hipDeviceptr_t d_cls = hip_upload_raw(h_cls,      (size_t)dim * sizeof(float));
    hipDeviceptr_t d_reg = h_reg ? hip_upload_raw(h_reg, (size_t)nr * dim * sizeof(float)) : 0;
    hipDeviceptr_t d_pe  = hip_upload_raw(h_pe,       n_pe  * sizeof(float));

    /* prepend_cls_reg: grid = 1 + nr blocks, each handles one row. */
    {
        void *args[] = { &d_buf, &d_cls, &d_reg, &nr, &dim };
        if (hipModuleLaunchKernel(fn_prep,
                           1 + nr, 1, 1,
                           threads, 1, 1,
                           0, 0, args, NULL) != hipSuccess) {
            fprintf(stderr, "launch prepend_cls_reg failed\n"); return 6;
        }
    }
    hipDeviceSynchronize();

    /* add_pos_embed: grid = 1 + np blocks. */
    {
        void *args[] = { &d_buf, &d_pe, &nr, &np, &dim };
        if (hipModuleLaunchKernel(fn_pos,
                           1 + np, 1, 1,
                           threads, 1, 1,
                           0, 0, args, NULL) != hipSuccess) {
            fprintf(stderr, "launch add_pos_embed failed\n"); return 6;
        }
    }
    hipDeviceSynchronize();
    hipMemcpyDtoH(h_dst, d_buf, n_buf * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_buf, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_dinov2_prepend_pos] dim=%d nr=%d np=%d  n_tok=%d  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        dim, nr, np, nt, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_buf_init); free(h_cls); free(h_reg); free(h_pe); free(h_ref); free(h_dst);
    hipFree(d_buf); hipFree(d_cls); if (d_reg) hipFree(d_reg); hipFree(d_pe);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
