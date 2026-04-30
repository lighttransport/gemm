/*
 * test_silu_mul — Phase 2b.0 standalone microbench.
 *
 * Validates silu_mul_f32 (SwiGLU activation core) on CondFuser FFN
 * geometry: hidden state shape [n_tok, ffn_h] = [1374, 2816].
 * Host reference uses double-precision accumulator for the silu eval.
 *
 * Usage:
 *   ./test_silu_mul [--n_tok 1374] [--ffn 2816] [--threshold 1e-6] [-v]
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
    int   n_tok     = 1374;
    int   ffn       = 2816;
    float threshold = 5e-6f;  /* __expf vs libm exp drift on |a*b| ~ 16 */
    int   verbose   = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--n_tok")     && i+1 < argc) n_tok     = atoi(argv[++i]);
        else if (!strcmp(a, "--ffn")       && i+1 < argc) ffn       = atoi(argv[++i]);
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
                                "sam3d_kernels.cu", verbose, "test_silu_mul");
    if (sm < 0) return 4;
    hipFunction_t fn = NULL;
    if (hipModuleGetFunction(&fn, mod, "silu_mul_f32") != hipSuccess) {
        fprintf(stderr, "lookup silu_mul_f32 failed\n"); return 4;
    }

    size_t n = (size_t)n_tok * ffn;
    float *h_a   = (float *)malloc(n * sizeof(float));
    float *h_b   = (float *)malloc(n * sizeof(float));
    float *h_ref = (float *)malloc(n * sizeof(float));
    float *h_dst = (float *)malloc(n * sizeof(float));
    if (!h_a || !h_b || !h_ref || !h_dst) return 5;

    uint32_t rng = 0xC0FFEEu;
    /* Llama FFN intermediates typically fall in [-4, 4]; sample widely. */
    for (size_t i = 0; i < n; i++) h_a[i] = urand(&rng) * 8.0f - 4.0f;
    for (size_t i = 0; i < n; i++) h_b[i] = urand(&rng) * 8.0f - 4.0f;

    /* Host reference: out[i] = (a[i] * sigmoid(a[i])) * b[i] in double. */
    for (size_t i = 0; i < n; i++) {
        double av = h_a[i];
        double sig = 1.0 / (1.0 + exp(-av));
        h_ref[i] = (float)((av * sig) * (double)h_b[i]);
    }

    hipDeviceptr_t d_a   = hip_upload_raw(h_a, n * sizeof(float));
    hipDeviceptr_t d_b   = hip_upload_raw(h_b, n * sizeof(float));
    hipDeviceptr_t d_out = 0;
    if (hipMalloc(&d_out, n * sizeof(float)) != hipSuccess) return 5;

    int total = (int)n;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    void *args[] = { &d_a, &d_b, &d_out, &total };
    if (hipModuleLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != hipSuccess) {
        fprintf(stderr, "launch failed\n"); return 6;
    }
    hipDeviceSynchronize();
    hipMemcpyDtoH(h_dst, d_out, n * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_silu_mul] n_tok=%d ffn=%d  n=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        n_tok, ffn, n, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_a); free(h_b); free(h_ref); free(h_dst);
    hipFree(d_a); hipFree(d_b); hipFree(d_out);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
