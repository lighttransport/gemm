/*
 * test_slat_mlp — Phase 5b.8 standalone microbench.
 *
 * Validates the SLAT transformer MLP residual path:
 *   modulated LN -> fc1 -> tanh GELU -> fc2 -> gated residual add.
 *
 * Usage:
 *   ./test_slat_mlp [--N 128] [--dim 1024] [--ratio 4] [--threshold 5e-4] [--repeat 10] [-v]
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
#include <time.h>

static float urand(uint32_t *state)
{
    *state = (*state) * 1664525u + 1013904223u;
    return (float)((*state) >> 8) / (float)(1u << 24);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static float max_abs(const float *a, const float *b, size_t n, double *mean_out)
{
    double sum = 0.0;
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    if (mean_out) *mean_out = sum / (n > 0 ? n : 1);
    return mx;
}

static void hmod_ln(float *out, const float *in,
                    const float *shift, const float *scale,
                    int N, int dim, float eps)
{
    for (int n = 0; n < N; n++) {
        const float *x = in + (size_t)n * dim;
        float *y = out + (size_t)n * dim;
        float mean = 0.0f;
        for (int c = 0; c < dim; c++) mean += x[c];
        mean /= (float)dim;
        float var = 0.0f;
        for (int c = 0; c < dim; c++) { float d = x[c] - mean; var += d * d; }
        var /= (float)dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < dim; c++) {
            float v = (x[c] - mean) * inv;
            y[c] = v * (1.0f + scale[c]) + shift[c];
        }
    }
}

static void hgemm(float *Y, const float *X, const float *W, const float *b,
                  int N, int D_out, int D_in)
{
    for (int n = 0; n < N; n++) {
        const float *xr = X + (size_t)n * D_in;
        for (int d = 0; d < D_out; d++) {
            const float *wr = W + (size_t)d * D_in;
            float acc = b ? b[d] : 0.0f;
            for (int k = 0; k < D_in; k++) acc += wr[k] * xr[k];
            Y[(size_t)n * D_out + d] = acc;
        }
    }
}

static void hgelu_tanh(float *x, int n)
{
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float u = k * (v + c * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(u));
    }
}

int main(int argc, char **argv)
{
    int N = 128;
    int dim = 1024;
    int ratio = 4;
    int repeat = 10;
    float threshold = 5e-4f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(a, "--ratio")     && i + 1 < argc) ratio = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat")    && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    int hidden = ratio * dim;
    if (N <= 0 || dim <= 0 || hidden <= 0) return 2;

    size_t XD = (size_t)N * dim;
    size_t XH = (size_t)N * hidden;
    float *x = (float *)malloc(XD * sizeof(float));
    float *shift = (float *)malloc((size_t)dim * sizeof(float));
    float *scale = (float *)malloc((size_t)dim * sizeof(float));
    float *gate = (float *)malloc((size_t)dim * sizeof(float));
    float *w1 = (float *)malloc((size_t)hidden * dim * sizeof(float));
    float *b1 = (float *)malloc((size_t)hidden * sizeof(float));
    float *w2 = (float *)malloc((size_t)dim * hidden * sizeof(float));
    float *b2 = (float *)malloc((size_t)dim * sizeof(float));
    float *h = (float *)malloc(XD * sizeof(float));
    float *mh = (float *)malloc(XH * sizeof(float));
    float *mh2 = (float *)malloc(XD * sizeof(float));
    float *ref = (float *)malloc(XD * sizeof(float));
    float *dst = (float *)malloc(XD * sizeof(float));
    if (!x || !shift || !scale || !gate || !w1 || !b1 || !w2 || !b2 ||
        !h || !mh || !mh2 || !ref || !dst) return 5;

    uint32_t rng = 0x51A7008u;
    for (size_t i = 0; i < XD; i++) x[i] = (urand(&rng) * 2.0f - 1.0f);
    for (int c = 0; c < dim; c++) {
        shift[c] = (urand(&rng) * 2.0f - 1.0f) * 0.05f;
        scale[c] = (urand(&rng) * 2.0f - 1.0f) * 0.10f;
        gate[c] = (urand(&rng) * 2.0f - 1.0f) * 0.50f;
        b2[c] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    }
    float s1 = 1.0f / sqrtf((float)dim);
    float s2 = 1.0f / sqrtf((float)hidden);
    for (size_t i = 0; i < (size_t)hidden * dim; i++)
        w1[i] = (urand(&rng) * 2.0f - 1.0f) * s1;
    for (int i = 0; i < hidden; i++)
        b1[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    for (size_t i = 0; i < (size_t)dim * hidden; i++)
        w2[i] = (urand(&rng) * 2.0f - 1.0f) * s2;

    hmod_ln(h, x, shift, scale, N, dim, 1e-6f);
    hgemm(mh, h, w1, b1, N, hidden, dim);
    hgelu_tanh(mh, (int)XH);
    hgemm(mh2, mh, w2, b2, N, dim, hidden);
    for (size_t i = 0; i < XD; i++) {
        int c = (int)(i % (size_t)dim);
        ref[i] = x[i] + mh2[i] * gate[c];
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0;
    if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    if (hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_slat_mlp") < 0) return 4;
    hipFunction_t fn_modln = NULL, fn_gemm = NULL, fn_gelu = NULL, fn_gated = NULL;
    if (hipModuleGetFunction(&fn_modln, mod, "modulated_ln_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&fn_gelu, mod, "gelu_tanh_inplace_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_gated, mod, "gated_residual_add_f32") != hipSuccess)
        return 4;

    hipDeviceptr_t d_x0 = hip_upload_raw(x, XD * sizeof(float));
    hipDeviceptr_t d_x = hip_upload_raw(x, XD * sizeof(float));
    hipDeviceptr_t d_shift = hip_upload_raw(shift, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_scale = hip_upload_raw(scale, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_gate = hip_upload_raw(gate, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_w1 = hip_upload_raw(w1, (size_t)hidden * dim * sizeof(float));
    hipDeviceptr_t d_b1 = hip_upload_raw(b1, (size_t)hidden * sizeof(float));
    hipDeviceptr_t d_w2 = hip_upload_raw(w2, (size_t)dim * hidden * sizeof(float));
    hipDeviceptr_t d_b2 = hip_upload_raw(b2, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_h = 0, d_mh = 0, d_mh2 = 0;
    if (!d_x0 || !d_x || !d_shift || !d_scale || !d_gate ||
        !d_w1 || !d_b1 || !d_w2 || !d_b2 ||
        hipMalloc(&d_h, XD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_mh, XH * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_mh2, XD * sizeof(float)) != hipSuccess)
        return 5;

    float eps = 1e-6f;
    unsigned threads = 256;
    size_t smem = 2 * threads * sizeof(float);
    int n_mh = (int)XH;
    int n_elem = (int)XD;

#define RUN_MLP() do { \
    if (hipMemcpyDtoD(d_x, d_x0, XD * sizeof(float)) != hipSuccess) return 5; \
    void *a_ln[] = { &d_h, &d_x, &d_shift, &d_scale, &N, &dim, &eps }; \
    if (hipModuleLaunchKernel(fn_modln, N, 1, 1, threads, 1, 1, (unsigned)smem, 0, a_ln, NULL) != hipSuccess) return 6; \
    unsigned gx1 = (N + 15) / 16, gy1 = (hidden + 15) / 16; \
    void *a_fc1[] = { &d_mh, &d_h, &d_w1, &d_b1, &N, &dim, &hidden }; \
    if (hipModuleLaunchKernel(fn_gemm, gx1, gy1, 1, 16, 16, 1, 0, 0, a_fc1, NULL) != hipSuccess) return 6; \
    void *a_gelu[] = { &d_mh, &n_mh }; \
    if (hipModuleLaunchKernel(fn_gelu, (n_mh + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_gelu, NULL) != hipSuccess) return 6; \
    unsigned gx2 = (N + 15) / 16, gy2 = (dim + 15) / 16; \
    void *a_fc2[] = { &d_mh2, &d_mh, &d_w2, &d_b2, &N, &hidden, &dim }; \
    if (hipModuleLaunchKernel(fn_gemm, gx2, gy2, 1, 16, 16, 1, 0, 0, a_fc2, NULL) != hipSuccess) return 6; \
    void *a_gate[] = { &d_x, &d_mh2, &d_gate, &N, &dim }; \
    if (hipModuleLaunchKernel(fn_gated, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_gate, NULL) != hipSuccess) return 6; \
} while (0)

    RUN_MLP();
    hipDeviceSynchronize();
    hipMemcpyDtoH(dst, d_x, XD * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(dst, ref, XD, &mean);
    int ok = (mx <= threshold);

    double avg_ms = 0.0;
    if (repeat > 0) {
        hipDeviceSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) RUN_MLP();
        hipDeviceSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }
#undef RUN_MLP

    fprintf(stderr,
            "[test_slat_mlp] N=%d dim=%d hidden=%d max_abs=%.4g mean_abs=%.4g avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, dim, hidden, (double)mx, mean, avg_ms, repeat,
            ok ? "OK" : "FAIL", (double)threshold);

    free(x); free(shift); free(scale); free(gate); free(w1); free(b1);
    free(w2); free(b2); free(h); free(mh); free(mh2); free(ref); free(dst);
    hipFree(d_x0); hipFree(d_x); hipFree(d_shift); hipFree(d_scale);
    hipFree(d_gate); hipFree(d_w1); hipFree(d_b1); hipFree(d_w2);
    hipFree(d_b2); hipFree(d_h); hipFree(d_mh); hipFree(d_mh2);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
