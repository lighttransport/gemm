/*
 * test_slat_self_attn_residual — Phase 5b.9 standalone microbench.
 *
 * Composes the SLAT transformer self-attention residual path:
 *   modulated LN -> qkv projection -> Q/K multi-head RMSNorm ->
 *   qkv split -> full self-attn -> output projection -> gated residual.
 *
 * Usage:
 *   ./test_slat_self_attn_residual [--N 128] [--dim 1024] [--H 16] [--threshold 1e-3] [--repeat 10] [-v]
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

static void hmhrmsnorm(float *v, int N, int H, int D_h, int stride,
                       const float *gamma)
{
    float root = sqrtf((float)D_h);
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            float *row = v + (size_t)n * stride + (size_t)h * D_h;
            const float *g = gamma + (size_t)h * D_h;
            double ss = 0.0;
            for (int d = 0; d < D_h; d++) ss += (double)row[d] * row[d];
            float inv = 1.0f / (sqrtf((float)ss) + 1e-12f);
            for (int d = 0; d < D_h; d++) row[d] = row[d] * inv * g[d] * root;
        }
    }
}

static void hqkv_split(const float *qkv, int N, int dim,
                       float *q, float *k, float *v)
{
    for (int n = 0; n < N; n++) {
        const float *row = qkv + (size_t)n * 3 * dim;
        memcpy(q + (size_t)n * dim, row, (size_t)dim * sizeof(float));
        memcpy(k + (size_t)n * dim, row + dim, (size_t)dim * sizeof(float));
        memcpy(v + (size_t)n * dim, row + 2 * dim, (size_t)dim * sizeof(float));
    }
}

static void hsdpa(float *out, const float *q, const float *k, const float *v,
                  int N, int H, int D_h, float scale)
{
    int E = H * D_h;
    double *scores = (double *)malloc((size_t)N * sizeof(double));
    for (int nq = 0; nq < N; nq++) {
        for (int h = 0; h < H; h++) {
            const float *qv = q + (size_t)nq * E + (size_t)h * D_h;
            double mx = -1.0e300;
            for (int nk = 0; nk < N; nk++) {
                const float *kv = k + (size_t)nk * E + (size_t)h * D_h;
                double s = 0.0;
                for (int d = 0; d < D_h; d++) s += (double)qv[d] * (double)kv[d];
                s *= (double)scale;
                scores[nk] = s;
                if (s > mx) mx = s;
            }
            double sum = 0.0;
            for (int nk = 0; nk < N; nk++) {
                scores[nk] = exp(scores[nk] - mx);
                sum += scores[nk];
            }
            double inv = 1.0 / sum;
            for (int d = 0; d < D_h; d++) {
                double acc = 0.0;
                for (int nk = 0; nk < N; nk++)
                    acc += scores[nk] * (double)v[(size_t)nk * E + (size_t)h * D_h + d];
                out[(size_t)nq * E + (size_t)h * D_h + d] = (float)(acc * inv);
            }
        }
    }
    free(scores);
}

int main(int argc, char **argv)
{
    int N = 128;
    int dim = 1024;
    int H = 16;
    int repeat = 10;
    float threshold = 1e-3f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(a, "--H")         && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat")    && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (N <= 0 || dim <= 0 || H <= 0 || dim % H != 0) return 2;
    int D_h = dim / H;
    int qkv_dim = 3 * dim;
    float scale_attn = 1.0f / sqrtf((float)D_h);

    size_t XD = (size_t)N * dim;
    size_t XQ = (size_t)N * qkv_dim;
    float *x = (float *)malloc(XD * sizeof(float));
    float *shift = (float *)malloc((size_t)dim * sizeof(float));
    float *scale = (float *)malloc((size_t)dim * sizeof(float));
    float *gate = (float *)malloc((size_t)dim * sizeof(float));
    float *qkv_w = (float *)malloc((size_t)qkv_dim * dim * sizeof(float));
    float *qkv_b = (float *)malloc((size_t)qkv_dim * sizeof(float));
    float *gq = (float *)malloc((size_t)dim * sizeof(float));
    float *gk = (float *)malloc((size_t)dim * sizeof(float));
    float *out_w = (float *)malloc((size_t)dim * dim * sizeof(float));
    float *out_b = (float *)malloc((size_t)dim * sizeof(float));
    float *h = (float *)malloc(XD * sizeof(float));
    float *qkv = (float *)malloc(XQ * sizeof(float));
    float *q = (float *)malloc(XD * sizeof(float));
    float *k = (float *)malloc(XD * sizeof(float));
    float *v = (float *)malloc(XD * sizeof(float));
    float *sa = (float *)malloc(XD * sizeof(float));
    float *proj = (float *)malloc(XD * sizeof(float));
    float *ref = (float *)malloc(XD * sizeof(float));
    float *dst = (float *)malloc(XD * sizeof(float));
    if (!x || !shift || !scale || !gate || !qkv_w || !qkv_b || !gq || !gk ||
        !out_w || !out_b || !h || !qkv || !q || !k || !v || !sa || !proj || !ref || !dst)
        return 5;

    uint32_t rng = 0x51A77009u;
    for (size_t i = 0; i < XD; i++) x[i] = (urand(&rng) * 2.0f - 1.0f);
    for (int c = 0; c < dim; c++) {
        shift[c] = (urand(&rng) * 2.0f - 1.0f) * 0.05f;
        scale[c] = (urand(&rng) * 2.0f - 1.0f) * 0.10f;
        gate[c] = (urand(&rng) * 2.0f - 1.0f) * 0.50f;
        gq[c] = 0.75f + 0.5f * urand(&rng);
        gk[c] = 0.75f + 0.5f * urand(&rng);
        out_b[c] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    }
    float sw = 1.0f / sqrtf((float)dim);
    for (size_t i = 0; i < (size_t)qkv_dim * dim; i++)
        qkv_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;
    for (int i = 0; i < qkv_dim; i++)
        qkv_b[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    for (size_t i = 0; i < (size_t)dim * dim; i++)
        out_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;

    hmod_ln(h, x, shift, scale, N, dim, 1e-6f);
    hgemm(qkv, h, qkv_w, qkv_b, N, qkv_dim, dim);
    hmhrmsnorm(qkv, N, H, D_h, qkv_dim, gq);
    hmhrmsnorm(qkv + dim, N, H, D_h, qkv_dim, gk);
    hqkv_split(qkv, N, dim, q, k, v);
    hsdpa(sa, q, k, v, N, H, D_h, scale_attn);
    hgemm(proj, sa, out_w, out_b, N, dim, dim);
    for (size_t i = 0; i < XD; i++) {
        int c = (int)(i % (size_t)dim);
        ref[i] = x[i] + proj[i] * gate[c];
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
                           "test_slat_self_attn_residual") < 0) return 4;
    hipFunction_t fn_modln = NULL, fn_gemm = NULL, fn_rms = NULL, fn_split = NULL;
    hipFunction_t fn_sdpa = NULL, fn_gated = NULL;
    if (hipModuleGetFunction(&fn_modln, mod, "modulated_ln_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&fn_rms, mod, "multi_head_rmsnorm_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_split, mod, "qkv_split_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_sdpa, mod, "sdpa_f32") != hipSuccess ||
        hipModuleGetFunction(&fn_gated, mod, "gated_residual_add_f32") != hipSuccess)
        return 4;

    hipDeviceptr_t d_x0 = hip_upload_raw(x, XD * sizeof(float));
    hipDeviceptr_t d_x = hip_upload_raw(x, XD * sizeof(float));
    hipDeviceptr_t d_shift = hip_upload_raw(shift, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_scale = hip_upload_raw(scale, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_gate = hip_upload_raw(gate, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_qkv_w = hip_upload_raw(qkv_w, (size_t)qkv_dim * dim * sizeof(float));
    hipDeviceptr_t d_qkv_b = hip_upload_raw(qkv_b, (size_t)qkv_dim * sizeof(float));
    hipDeviceptr_t d_gq = hip_upload_raw(gq, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_gk = hip_upload_raw(gk, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_out_w = hip_upload_raw(out_w, (size_t)dim * dim * sizeof(float));
    hipDeviceptr_t d_out_b = hip_upload_raw(out_b, (size_t)dim * sizeof(float));
    hipDeviceptr_t d_h = 0, d_qkv = 0, d_q = 0, d_k = 0, d_v = 0, d_sa = 0, d_proj = 0;
    if (!d_x0 || !d_x || !d_shift || !d_scale || !d_gate || !d_qkv_w ||
        !d_qkv_b || !d_gq || !d_gk || !d_out_w || !d_out_b ||
        hipMalloc(&d_h, XD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_qkv, XQ * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_q, XD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_k, XD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_v, XD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_sa, XD * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_proj, XD * sizeof(float)) != hipSuccess)
        return 5;

    float eps = 1e-6f;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    size_t rms_smem = 64 * sizeof(float);
    size_t sdpa_smem = (size_t)(256 + N) * sizeof(float);
    int n_elem = (int)XD;

#define RUN_SA() do { \
    if (hipMemcpyDtoD(d_x, d_x0, XD * sizeof(float)) != hipSuccess) return 5; \
    void *a_ln[] = { &d_h, &d_x, &d_shift, &d_scale, &N, &dim, &eps }; \
    if (hipModuleLaunchKernel(fn_modln, N, 1, 1, threads, 1, 1, (unsigned)ln_smem, 0, a_ln, NULL) != hipSuccess) return 6; \
    unsigned gxq = (N + 15) / 16, gyq = (qkv_dim + 15) / 16; \
    void *a_qkv[] = { &d_qkv, &d_h, &d_qkv_w, &d_qkv_b, &N, &dim, &qkv_dim }; \
    if (hipModuleLaunchKernel(fn_gemm, gxq, gyq, 1, 16, 16, 1, 0, 0, a_qkv, NULL) != hipSuccess) return 6; \
    void *a_rq[] = { &d_qkv, &d_gq, &N, &H, &D_h, &qkv_dim }; \
    if (hipModuleLaunchKernel(fn_rms, H, N, 1, 64, 1, 1, (unsigned)rms_smem, 0, a_rq, NULL) != hipSuccess) return 6; \
    hipDeviceptr_t d_qkv_k = d_qkv + (size_t)dim * sizeof(float); \
    void *a_rk[] = { &d_qkv_k, &d_gk, &N, &H, &D_h, &qkv_dim }; \
    if (hipModuleLaunchKernel(fn_rms, H, N, 1, 64, 1, 1, (unsigned)rms_smem, 0, a_rk, NULL) != hipSuccess) return 6; \
    void *a_split[] = { &d_q, &d_k, &d_v, &d_qkv, &N, &dim }; \
    if (hipModuleLaunchKernel(fn_split, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_split, NULL) != hipSuccess) return 6; \
    void *a_sdpa[] = { &d_sa, &d_q, &d_k, &d_v, &N, &N, &H, &D_h, &scale_attn }; \
    if (hipModuleLaunchKernel(fn_sdpa, N, H, 1, 256, 1, 1, (unsigned)sdpa_smem, 0, a_sdpa, NULL) != hipSuccess) return 6; \
    unsigned gxo = (N + 15) / 16, gyo = (dim + 15) / 16; \
    void *a_out[] = { &d_proj, &d_sa, &d_out_w, &d_out_b, &N, &dim, &dim }; \
    if (hipModuleLaunchKernel(fn_gemm, gxo, gyo, 1, 16, 16, 1, 0, 0, a_out, NULL) != hipSuccess) return 6; \
    void *a_gate[] = { &d_x, &d_proj, &d_gate, &N, &dim }; \
    if (hipModuleLaunchKernel(fn_gated, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_gate, NULL) != hipSuccess) return 6; \
} while (0)

    RUN_SA();
    hipDeviceSynchronize();
    hipMemcpyDtoH(dst, d_x, XD * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(dst, ref, XD, &mean);
    int ok = (mx <= threshold);

    double avg_ms = 0.0;
    if (repeat > 0) {
        hipDeviceSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) RUN_SA();
        hipDeviceSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }
#undef RUN_SA

    fprintf(stderr,
            "[test_slat_self_attn_residual] N=%d dim=%d H=%d D_h=%d max_abs=%.4g mean_abs=%.4g avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, dim, H, D_h, (double)mx, mean, avg_ms, repeat,
            ok ? "OK" : "FAIL", (double)threshold);

    free(x); free(shift); free(scale); free(gate); free(qkv_w); free(qkv_b);
    free(gq); free(gk); free(out_w); free(out_b); free(h); free(qkv);
    free(q); free(k); free(v); free(sa); free(proj); free(ref); free(dst);
    hipFree(d_x0); hipFree(d_x); hipFree(d_shift); hipFree(d_scale);
    hipFree(d_gate); hipFree(d_qkv_w); hipFree(d_qkv_b); hipFree(d_gq);
    hipFree(d_gk); hipFree(d_out_w); hipFree(d_out_b); hipFree(d_h);
    hipFree(d_qkv); hipFree(d_q); hipFree(d_k); hipFree(d_v);
    hipFree(d_sa); hipFree(d_proj);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
