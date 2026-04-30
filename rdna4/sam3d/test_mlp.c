/*
 * test_mlp — Phase 1b.6 standalone microbench.
 *
 * End-to-end DINOv2 MLP path:
 *     hidden  [n_tok, dim]
 *  →  inter  = gemm_f32_bias(hidden, fc1.W, fc1.b)   [n_tok, ffn]
 *  →  inter  = gelu_inplace_f32(inter)
 *  →  out    = gemm_f32_bias(inter,  fc2.W, fc2.b)   [n_tok, dim]
 *
 * gemm_f32_bias was already validated in test_gemm_f32_bias; this test
 * exercises composition + the new gelu_inplace_f32 kernel against a
 * double-precision host reference.
 *
 * DINOv2-L defaults: n_tok=1374, dim=1024, ffn=4096.
 *
 * Usage:
 *   ./test_mlp [--n_tok 1374] [--dim 1024] [--ffn 4096]
 *              [--threshold 5e-3] [-v]
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

static void host_gemm_bias(float *Y, const float *X, const float *W, const float *b,
                           int N, int D_in, int D_out)
{
    for (int n = 0; n < N; n++) {
        const float *xr = X + (size_t)n * D_in;
        for (int d = 0; d < D_out; d++) {
            const float *wr = W + (size_t)d * D_in;
            double acc = b ? (double)b[d] : 0.0;
            for (int k = 0; k < D_in; k++) acc += (double)wr[k] * (double)xr[k];
            Y[(size_t)n * D_out + d] = (float)acc;
        }
    }
}

static void host_gelu_inplace(float *x, size_t n) {
    const double inv_sqrt2 = 0.70710678118654752440;
    for (size_t i = 0; i < n; i++) {
        double v = x[i];
        x[i] = (float)(v * 0.5 * (1.0 + erf(v * inv_sqrt2)));
    }
}

int main(int argc, char **argv)
{
    int   n_tok     = 1374;
    int   dim       = 1024;
    int   ffn       = 4096;
    float threshold = 5e-3f;
    int   verbose   = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--n_tok")     && i+1 < argc) n_tok     = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--ffn")       && i+1 < argc) ffn       = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
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
                                "sam3d_kernels.cu", verbose, "test_mlp");
    if (sm < 0) { fprintf(stderr, "compile failed\n"); return 4; }
    hipFunction_t fn_gemm = NULL, fn_gelu = NULL;
    if (hipModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias") != hipSuccess) {
        fprintf(stderr, "lookup gemm_f32_bias failed\n"); return 4;
    }
    if (hipModuleGetFunction(&fn_gelu, mod, "gelu_inplace_f32") != hipSuccess) {
        fprintf(stderr, "lookup gelu_inplace_f32 failed\n"); return 4;
    }

    size_t n_h    = (size_t)n_tok * dim;
    size_t n_i    = (size_t)n_tok * ffn;
    size_t n_w1   = (size_t)ffn * dim;
    size_t n_w2   = (size_t)dim * ffn;
    float *h_hidden = (float *)malloc(n_h * sizeof(float));
    float *h_w1     = (float *)malloc(n_w1 * sizeof(float));
    float *h_b1     = (float *)malloc((size_t)ffn * sizeof(float));
    float *h_w2     = (float *)malloc(n_w2 * sizeof(float));
    float *h_b2     = (float *)malloc((size_t)dim * sizeof(float));
    float *h_inter  = (float *)malloc(n_i * sizeof(float));
    float *h_ref    = (float *)malloc(n_h * sizeof(float));
    float *h_dst    = (float *)malloc(n_h * sizeof(float));
    if (!h_hidden || !h_w1 || !h_b1 || !h_w2 || !h_b2 || !h_inter || !h_ref || !h_dst) {
        fprintf(stderr, "host alloc failed\n"); return 5;
    }
    uint32_t rng = 0xC0FFEEu;
    float w1_scale = 1.0f / sqrtf((float)dim);
    float w2_scale = 1.0f / sqrtf((float)ffn);
    for (size_t i = 0; i < n_h; i++)  h_hidden[i] = urand(&rng) * 2.0f - 1.0f;
    for (size_t i = 0; i < n_w1; i++) h_w1[i]     = (urand(&rng) * 2.0f - 1.0f) * w1_scale;
    for (int d = 0; d < ffn; d++)     h_b1[d]     = (urand(&rng) * 2.0f - 1.0f) * 0.1f;
    for (size_t i = 0; i < n_w2; i++) h_w2[i]     = (urand(&rng) * 2.0f - 1.0f) * w2_scale;
    for (int d = 0; d < dim; d++)     h_b2[d]     = (urand(&rng) * 2.0f - 1.0f) * 0.1f;

    /* Host reference. */
    host_gemm_bias(h_inter, h_hidden, h_w1, h_b1, n_tok, dim, ffn);
    host_gelu_inplace(h_inter, n_i);
    host_gemm_bias(h_ref,   h_inter,  h_w2, h_b2, n_tok, ffn, dim);

    /* Device path. */
    hipDeviceptr_t d_hidden = hip_upload_raw(h_hidden, n_h * sizeof(float));
    hipDeviceptr_t d_w1     = hip_upload_raw(h_w1,     n_w1 * sizeof(float));
    hipDeviceptr_t d_b1     = hip_upload_raw(h_b1,     (size_t)ffn * sizeof(float));
    hipDeviceptr_t d_w2     = hip_upload_raw(h_w2,     n_w2 * sizeof(float));
    hipDeviceptr_t d_b2     = hip_upload_raw(h_b2,     (size_t)dim * sizeof(float));
    hipDeviceptr_t d_inter = 0, d_out = 0;
    if (hipMalloc(&d_inter, n_i * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_out,   n_h * sizeof(float)) != hipSuccess) {
        fprintf(stderr, "hipMalloc failed\n"); return 5;
    }

    /* fc1: [n_tok, dim] @ W1^T + b1 → [n_tok, ffn]. */
    {
        int N = n_tok, Din = dim, Dout = ffn;
        void *args[] = { &d_inter, &d_hidden, &d_w1, &d_b1, &N, &Din, &Dout };
        int gx = (N + 15) / 16, gy = (Dout + 15) / 16;
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) {
            fprintf(stderr, "launch fc1 failed\n"); return 6;
        }
    }
    /* gelu inplace on [n_tok, ffn]. */
    {
        int total = (int)n_i;
        int threads = 256, blocks = (total + threads - 1) / threads;
        void *args[] = { &d_inter, &total };
        if (hipModuleLaunchKernel(fn_gelu, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != hipSuccess) {
            fprintf(stderr, "launch gelu failed\n"); return 6;
        }
    }
    /* fc2: [n_tok, ffn] @ W2^T + b2 → [n_tok, dim]. */
    {
        int N = n_tok, Din = ffn, Dout = dim;
        void *args[] = { &d_out, &d_inter, &d_w2, &d_b2, &N, &Din, &Dout };
        int gx = (N + 15) / 16, gy = (Dout + 15) / 16;
        if (hipModuleLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) {
            fprintf(stderr, "launch fc2 failed\n"); return 6;
        }
    }
    hipDeviceSynchronize();
    hipMemcpyDtoH(h_dst, d_out, n_h * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_h, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_mlp] n_tok=%d dim=%d ffn=%d  n_out=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        n_tok, dim, ffn, n_h, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_hidden); free(h_w1); free(h_b1); free(h_w2); free(h_b2);
    free(h_inter); free(h_ref); free(h_dst);
    hipFree(d_hidden); hipFree(d_w1); hipFree(d_b1); hipFree(d_w2); hipFree(d_b2);
    hipFree(d_inter); hipFree(d_out);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
