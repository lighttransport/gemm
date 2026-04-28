/*
 * test_slat_transformer_block - Phase 5b.11 standalone microbench.
 *
 * Composes one SLAT transformer block in the production order:
 *   self-attn residual -> cross-attn residual -> MLP residual.
 *
 * Usage:
 *   ./test_slat_transformer_block [--N 64] [--Nc 256] [--dim 1024] [--H 16] [--ratio 4] [--threshold 2e-3] [--repeat 5] [-v]
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"
#include "cuda_sam3d_kernels.h"

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

static void hln_affine(float *out, const float *in,
                       const float *gamma, const float *beta,
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
        for (int c = 0; c < dim; c++) y[c] = (x[c] - mean) * inv * gamma[c] + beta[c];
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

static void hkv_split(const float *kv, int N, int dim, float *K, float *V)
{
    for (int n = 0; n < N; n++) {
        const float *row = kv + (size_t)n * 2 * dim;
        memcpy(K + (size_t)n * dim, row, (size_t)dim * sizeof(float));
        memcpy(V + (size_t)n * dim, row + dim, (size_t)dim * sizeof(float));
    }
}

static void hsdpa(float *out,
                  const float *q, const float *k, const float *v,
                  int Nq, int Nk, int H, int D_h, float scale)
{
    int E = H * D_h;
    double *scores = (double *)malloc((size_t)Nk * sizeof(double));
    if (!scores) abort();
    for (int nq = 0; nq < Nq; nq++) {
        for (int h = 0; h < H; h++) {
            const float *qv = q + (size_t)nq * E + (size_t)h * D_h;
            double mx = -1.0e300;
            for (int nk = 0; nk < Nk; nk++) {
                const float *kv = k + (size_t)nk * E + (size_t)h * D_h;
                double s = 0.0;
                for (int d = 0; d < D_h; d++) s += (double)qv[d] * (double)kv[d];
                s *= (double)scale;
                scores[nk] = s;
                if (s > mx) mx = s;
            }
            double sum = 0.0;
            for (int nk = 0; nk < Nk; nk++) {
                scores[nk] = exp(scores[nk] - mx);
                sum += scores[nk];
            }
            double inv = 1.0 / sum;
            for (int d = 0; d < D_h; d++) {
                double acc = 0.0;
                for (int nk = 0; nk < Nk; nk++)
                    acc += scores[nk] * (double)v[(size_t)nk * E + (size_t)h * D_h + d];
                out[(size_t)nq * E + (size_t)h * D_h + d] = (float)(acc * inv);
            }
        }
    }
    free(scores);
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
    int N = 64;
    int Nc = 256;
    int dim = 1024;
    int H = 16;
    int ratio = 4;
    int repeat = 5;
    float threshold = 2e-3f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--N")         && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(a, "--Nc")        && i + 1 < argc) Nc = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(a, "--H")         && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(a, "--ratio")     && i + 1 < argc) ratio = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat")    && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (N <= 0 || Nc <= 0 || dim <= 0 || H <= 0 || dim % H != 0 || ratio <= 0) return 2;
    int D_h = dim / H;
    int qkv_dim = 3 * dim;
    int kv_dim = 2 * dim;
    int hidden = ratio * dim;
    float attn_scale = 1.0f / sqrtf((float)D_h);

    size_t XD = (size_t)N * dim;
    size_t CD = (size_t)Nc * dim;
    size_t XQKV = (size_t)N * qkv_dim;
    size_t CKV = (size_t)Nc * kv_dim;
    size_t XH = (size_t)N * hidden;

    float *x = (float *)malloc(XD * sizeof(float));
    float *cond = (float *)malloc(CD * sizeof(float));
    float *shift_msa = (float *)malloc((size_t)dim * sizeof(float));
    float *scale_msa = (float *)malloc((size_t)dim * sizeof(float));
    float *gate_msa = (float *)malloc((size_t)dim * sizeof(float));
    float *shift_mlp = (float *)malloc((size_t)dim * sizeof(float));
    float *scale_mlp = (float *)malloc((size_t)dim * sizeof(float));
    float *gate_mlp = (float *)malloc((size_t)dim * sizeof(float));
    float *norm2_w = (float *)malloc((size_t)dim * sizeof(float));
    float *norm2_b = (float *)malloc((size_t)dim * sizeof(float));
    float *sa_qkv_w = (float *)malloc((size_t)qkv_dim * dim * sizeof(float));
    float *sa_qkv_b = (float *)malloc((size_t)qkv_dim * sizeof(float));
    float *q_rms = (float *)malloc((size_t)dim * sizeof(float));
    float *k_rms = (float *)malloc((size_t)dim * sizeof(float));
    float *sa_out_w = (float *)malloc((size_t)dim * dim * sizeof(float));
    float *sa_out_b = (float *)malloc((size_t)dim * sizeof(float));
    float *ca_q_w = (float *)malloc((size_t)dim * dim * sizeof(float));
    float *ca_q_b = (float *)malloc((size_t)dim * sizeof(float));
    float *ca_kv_w = (float *)malloc((size_t)kv_dim * dim * sizeof(float));
    float *ca_kv_b = (float *)malloc((size_t)kv_dim * sizeof(float));
    float *ca_out_w = (float *)malloc((size_t)dim * dim * sizeof(float));
    float *ca_out_b = (float *)malloc((size_t)dim * sizeof(float));
    float *fc1_w = (float *)malloc((size_t)hidden * dim * sizeof(float));
    float *fc1_b = (float *)malloc((size_t)hidden * sizeof(float));
    float *fc2_w = (float *)malloc((size_t)dim * hidden * sizeof(float));
    float *fc2_b = (float *)malloc((size_t)dim * sizeof(float));
    float *h = (float *)malloc(XD * sizeof(float));
    float *qkv = (float *)malloc(XQKV * sizeof(float));
    float *q = (float *)malloc(XD * sizeof(float));
    float *k = (float *)malloc(XD * sizeof(float));
    float *v = (float *)malloc(XD * sizeof(float));
    float *sa = (float *)malloc(XD * sizeof(float));
    float *proj = (float *)malloc(XD * sizeof(float));
    float *kv = (float *)malloc(CKV * sizeof(float));
    float *K = (float *)malloc(CD * sizeof(float));
    float *V = (float *)malloc(CD * sizeof(float));
    float *xa = (float *)malloc(XD * sizeof(float));
    float *mh = (float *)malloc(XH * sizeof(float));
    float *mh2 = (float *)malloc(XD * sizeof(float));
    float *ref = (float *)malloc(XD * sizeof(float));
    float *dst = (float *)malloc(XD * sizeof(float));
    if (!x || !cond || !shift_msa || !scale_msa || !gate_msa ||
        !shift_mlp || !scale_mlp || !gate_mlp || !norm2_w || !norm2_b ||
        !sa_qkv_w || !sa_qkv_b || !q_rms || !k_rms || !sa_out_w || !sa_out_b ||
        !ca_q_w || !ca_q_b || !ca_kv_w || !ca_kv_b || !ca_out_w || !ca_out_b ||
        !fc1_w || !fc1_b || !fc2_w || !fc2_b || !h || !qkv || !q || !k || !v ||
        !sa || !proj || !kv || !K || !V || !xa || !mh || !mh2 || !ref || !dst)
        return 5;

    uint32_t rng = 0x51A7B10Cu;
    for (size_t i = 0; i < XD; i++) x[i] = (urand(&rng) * 2.0f - 1.0f);
    for (size_t i = 0; i < CD; i++) cond[i] = (urand(&rng) * 2.0f - 1.0f);
    for (int c = 0; c < dim; c++) {
        shift_msa[c] = (urand(&rng) * 2.0f - 1.0f) * 0.05f;
        scale_msa[c] = (urand(&rng) * 2.0f - 1.0f) * 0.10f;
        gate_msa[c] = (urand(&rng) * 2.0f - 1.0f) * 0.50f;
        shift_mlp[c] = (urand(&rng) * 2.0f - 1.0f) * 0.05f;
        scale_mlp[c] = (urand(&rng) * 2.0f - 1.0f) * 0.10f;
        gate_mlp[c] = (urand(&rng) * 2.0f - 1.0f) * 0.50f;
        norm2_w[c] = 0.9f + 0.2f * urand(&rng);
        norm2_b[c] = (urand(&rng) * 2.0f - 1.0f) * 0.05f;
        q_rms[c] = 0.75f + 0.5f * urand(&rng);
        k_rms[c] = 0.75f + 0.5f * urand(&rng);
        sa_out_b[c] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
        ca_q_b[c] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
        ca_out_b[c] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
        fc2_b[c] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    }
    float sw = 1.0f / sqrtf((float)dim);
    float sw2 = 1.0f / sqrtf((float)hidden);
    for (size_t i = 0; i < (size_t)qkv_dim * dim; i++)
        sa_qkv_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;
    for (int i = 0; i < qkv_dim; i++)
        sa_qkv_b[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    for (size_t i = 0; i < (size_t)dim * dim; i++) {
        sa_out_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;
        ca_q_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;
        ca_out_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;
    }
    for (size_t i = 0; i < (size_t)kv_dim * dim; i++)
        ca_kv_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;
    for (int i = 0; i < kv_dim; i++)
        ca_kv_b[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    for (size_t i = 0; i < (size_t)hidden * dim; i++)
        fc1_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw;
    for (int i = 0; i < hidden; i++)
        fc1_b[i] = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    for (size_t i = 0; i < (size_t)dim * hidden; i++)
        fc2_w[i] = (urand(&rng) * 2.0f - 1.0f) * sw2;

    memcpy(ref, x, XD * sizeof(float));
    hmod_ln(h, ref, shift_msa, scale_msa, N, dim, 1e-6f);
    hgemm(qkv, h, sa_qkv_w, sa_qkv_b, N, qkv_dim, dim);
    hmhrmsnorm(qkv, N, H, D_h, qkv_dim, q_rms);
    hmhrmsnorm(qkv + dim, N, H, D_h, qkv_dim, k_rms);
    hqkv_split(qkv, N, dim, q, k, v);
    hsdpa(sa, q, k, v, N, N, H, D_h, attn_scale);
    hgemm(proj, sa, sa_out_w, sa_out_b, N, dim, dim);
    for (size_t i = 0; i < XD; i++) ref[i] += proj[i] * gate_msa[i % (size_t)dim];

    hln_affine(h, ref, norm2_w, norm2_b, N, dim, 1e-6f);
    hgemm(q, h, ca_q_w, ca_q_b, N, dim, dim);
    hgemm(kv, cond, ca_kv_w, ca_kv_b, Nc, kv_dim, dim);
    hkv_split(kv, Nc, dim, K, V);
    hsdpa(xa, q, K, V, N, Nc, H, D_h, attn_scale);
    hgemm(proj, xa, ca_out_w, ca_out_b, N, dim, dim);
    for (size_t i = 0; i < XD; i++) ref[i] += proj[i];

    hmod_ln(h, ref, shift_mlp, scale_mlp, N, dim, 1e-6f);
    hgemm(mh, h, fc1_w, fc1_b, N, hidden, dim);
    hgelu_tanh(mh, (int)XH);
    hgemm(mh2, mh, fc2_w, fc2_b, N, dim, hidden);
    for (size_t i = 0; i < XD; i++) ref[i] += mh2[i] * gate_mlp[i % (size_t)dim];

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose,
                           "test_slat_transformer_block") < 0) return 4;
    CUfunction fn_modln = NULL, fn_ln = NULL, fn_gemm = NULL, fn_rms = NULL;
    CUfunction fn_qkv_split = NULL, fn_kv_split = NULL, fn_sdpa = NULL;
    CUfunction fn_gated = NULL, fn_resadd = NULL, fn_gelu = NULL;
    if (cuModuleGetFunction(&fn_modln, mod, "modulated_ln_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_ln, mod, "layernorm_token_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_rms, mod, "multi_head_rmsnorm_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_qkv_split, mod, "qkv_split_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_kv_split, mod, "kv_split_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_sdpa, mod, "sdpa_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_gated, mod, "gated_residual_add_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_resadd, mod, "residual_add_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fn_gelu, mod, "gelu_tanh_inplace_f32") != CUDA_SUCCESS)
        return 4;

    CUdeviceptr d_x0 = cu_upload_raw(x, XD * sizeof(float));
    CUdeviceptr d_x = cu_upload_raw(x, XD * sizeof(float));
    CUdeviceptr d_cond = cu_upload_raw(cond, CD * sizeof(float));
    CUdeviceptr d_shift_msa = cu_upload_raw(shift_msa, (size_t)dim * sizeof(float));
    CUdeviceptr d_scale_msa = cu_upload_raw(scale_msa, (size_t)dim * sizeof(float));
    CUdeviceptr d_gate_msa = cu_upload_raw(gate_msa, (size_t)dim * sizeof(float));
    CUdeviceptr d_shift_mlp = cu_upload_raw(shift_mlp, (size_t)dim * sizeof(float));
    CUdeviceptr d_scale_mlp = cu_upload_raw(scale_mlp, (size_t)dim * sizeof(float));
    CUdeviceptr d_gate_mlp = cu_upload_raw(gate_mlp, (size_t)dim * sizeof(float));
    CUdeviceptr d_norm2_w = cu_upload_raw(norm2_w, (size_t)dim * sizeof(float));
    CUdeviceptr d_norm2_b = cu_upload_raw(norm2_b, (size_t)dim * sizeof(float));
    CUdeviceptr d_sa_qkv_w = cu_upload_raw(sa_qkv_w, (size_t)qkv_dim * dim * sizeof(float));
    CUdeviceptr d_sa_qkv_b = cu_upload_raw(sa_qkv_b, (size_t)qkv_dim * sizeof(float));
    CUdeviceptr d_q_rms = cu_upload_raw(q_rms, (size_t)dim * sizeof(float));
    CUdeviceptr d_k_rms = cu_upload_raw(k_rms, (size_t)dim * sizeof(float));
    CUdeviceptr d_sa_out_w = cu_upload_raw(sa_out_w, (size_t)dim * dim * sizeof(float));
    CUdeviceptr d_sa_out_b = cu_upload_raw(sa_out_b, (size_t)dim * sizeof(float));
    CUdeviceptr d_ca_q_w = cu_upload_raw(ca_q_w, (size_t)dim * dim * sizeof(float));
    CUdeviceptr d_ca_q_b = cu_upload_raw(ca_q_b, (size_t)dim * sizeof(float));
    CUdeviceptr d_ca_kv_w = cu_upload_raw(ca_kv_w, (size_t)kv_dim * dim * sizeof(float));
    CUdeviceptr d_ca_kv_b = cu_upload_raw(ca_kv_b, (size_t)kv_dim * sizeof(float));
    CUdeviceptr d_ca_out_w = cu_upload_raw(ca_out_w, (size_t)dim * dim * sizeof(float));
    CUdeviceptr d_ca_out_b = cu_upload_raw(ca_out_b, (size_t)dim * sizeof(float));
    CUdeviceptr d_fc1_w = cu_upload_raw(fc1_w, (size_t)hidden * dim * sizeof(float));
    CUdeviceptr d_fc1_b = cu_upload_raw(fc1_b, (size_t)hidden * sizeof(float));
    CUdeviceptr d_fc2_w = cu_upload_raw(fc2_w, (size_t)dim * hidden * sizeof(float));
    CUdeviceptr d_fc2_b = cu_upload_raw(fc2_b, (size_t)dim * sizeof(float));
    CUdeviceptr d_h = 0, d_qkv = 0, d_q = 0, d_k = 0, d_v = 0, d_sa = 0, d_proj = 0;
    CUdeviceptr d_kv = 0, d_K = 0, d_V = 0, d_xa = 0, d_mh = 0, d_mh2 = 0;
    if (!d_x0 || !d_x || !d_cond || !d_shift_msa || !d_scale_msa || !d_gate_msa ||
        !d_shift_mlp || !d_scale_mlp || !d_gate_mlp || !d_norm2_w || !d_norm2_b ||
        !d_sa_qkv_w || !d_sa_qkv_b || !d_q_rms || !d_k_rms || !d_sa_out_w || !d_sa_out_b ||
        !d_ca_q_w || !d_ca_q_b || !d_ca_kv_w || !d_ca_kv_b || !d_ca_out_w || !d_ca_out_b ||
        !d_fc1_w || !d_fc1_b || !d_fc2_w || !d_fc2_b ||
        cuMemAlloc(&d_h, XD * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_qkv, XQKV * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_q, XD * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_k, XD * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_v, XD * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_sa, XD * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_proj, XD * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_kv, CKV * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_K, CD * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_V, CD * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_xa, XD * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_mh, XH * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_mh2, XD * sizeof(float)) != CUDA_SUCCESS)
        return 5;

    float eps = 1e-6f;
    int affine = 1;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    size_t rms_smem = 64 * sizeof(float);
    size_t sdpa_self_smem = (size_t)(256 + N) * sizeof(float);
    size_t sdpa_cross_smem = (size_t)(256 + Nc) * sizeof(float);
    int n_elem = (int)XD;
    int n_mh = (int)XH;
    int n_cd = (int)CD;

#define RUN_BLOCK() do { \
    if (cuMemcpyDtoD(d_x, d_x0, XD * sizeof(float)) != CUDA_SUCCESS) return 5; \
    void *a_sa_ln[] = { &d_h, &d_x, &d_shift_msa, &d_scale_msa, &N, &dim, &eps }; \
    if (cuLaunchKernel(fn_modln, N, 1, 1, threads, 1, 1, (unsigned)ln_smem, 0, a_sa_ln, NULL) != CUDA_SUCCESS) return 6; \
    unsigned gxqkv = (N + 15) / 16, gyqkv = (qkv_dim + 15) / 16; \
    void *a_qkv[] = { &d_qkv, &d_h, &d_sa_qkv_w, &d_sa_qkv_b, &N, &dim, &qkv_dim }; \
    if (cuLaunchKernel(fn_gemm, gxqkv, gyqkv, 1, 16, 16, 1, 0, 0, a_qkv, NULL) != CUDA_SUCCESS) return 6; \
    void *a_rq[] = { &d_qkv, &d_q_rms, &N, &H, &D_h, &qkv_dim }; \
    if (cuLaunchKernel(fn_rms, H, N, 1, 64, 1, 1, (unsigned)rms_smem, 0, a_rq, NULL) != CUDA_SUCCESS) return 6; \
    CUdeviceptr d_qkv_k = d_qkv + (size_t)dim * sizeof(float); \
    void *a_rk[] = { &d_qkv_k, &d_k_rms, &N, &H, &D_h, &qkv_dim }; \
    if (cuLaunchKernel(fn_rms, H, N, 1, 64, 1, 1, (unsigned)rms_smem, 0, a_rk, NULL) != CUDA_SUCCESS) return 6; \
    void *a_qkv_split[] = { &d_q, &d_k, &d_v, &d_qkv, &N, &dim }; \
    if (cuLaunchKernel(fn_qkv_split, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_qkv_split, NULL) != CUDA_SUCCESS) return 6; \
    void *a_sa[] = { &d_sa, &d_q, &d_k, &d_v, &N, &N, &H, &D_h, &attn_scale }; \
    if (cuLaunchKernel(fn_sdpa, N, H, 1, 256, 1, 1, (unsigned)sdpa_self_smem, 0, a_sa, NULL) != CUDA_SUCCESS) return 6; \
    unsigned gxo = (N + 15) / 16, gyo = (dim + 15) / 16; \
    void *a_sa_out[] = { &d_proj, &d_sa, &d_sa_out_w, &d_sa_out_b, &N, &dim, &dim }; \
    if (cuLaunchKernel(fn_gemm, gxo, gyo, 1, 16, 16, 1, 0, 0, a_sa_out, NULL) != CUDA_SUCCESS) return 6; \
    void *a_sa_gate[] = { &d_x, &d_proj, &d_gate_msa, &N, &dim }; \
    if (cuLaunchKernel(fn_gated, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_sa_gate, NULL) != CUDA_SUCCESS) return 6; \
    void *a_ca_ln[] = { &d_h, &d_x, &d_norm2_w, &d_norm2_b, &N, &dim, &eps, &affine }; \
    if (cuLaunchKernel(fn_ln, N, 1, 1, threads, 1, 1, (unsigned)ln_smem, 0, a_ca_ln, NULL) != CUDA_SUCCESS) return 6; \
    void *a_cq[] = { &d_q, &d_h, &d_ca_q_w, &d_ca_q_b, &N, &dim, &dim }; \
    if (cuLaunchKernel(fn_gemm, gxo, gyo, 1, 16, 16, 1, 0, 0, a_cq, NULL) != CUDA_SUCCESS) return 6; \
    unsigned gxkv = (Nc + 15) / 16, gykv = (kv_dim + 15) / 16; \
    void *a_ckv[] = { &d_kv, &d_cond, &d_ca_kv_w, &d_ca_kv_b, &Nc, &dim, &kv_dim }; \
    if (cuLaunchKernel(fn_gemm, gxkv, gykv, 1, 16, 16, 1, 0, 0, a_ckv, NULL) != CUDA_SUCCESS) return 6; \
    void *a_kv_split[] = { &d_K, &d_V, &d_kv, &Nc, &dim }; \
    if (cuLaunchKernel(fn_kv_split, (n_cd + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_kv_split, NULL) != CUDA_SUCCESS) return 6; \
    void *a_xa[] = { &d_xa, &d_q, &d_K, &d_V, &N, &Nc, &H, &D_h, &attn_scale }; \
    if (cuLaunchKernel(fn_sdpa, N, H, 1, 256, 1, 1, (unsigned)sdpa_cross_smem, 0, a_xa, NULL) != CUDA_SUCCESS) return 6; \
    void *a_ca_out[] = { &d_proj, &d_xa, &d_ca_out_w, &d_ca_out_b, &N, &dim, &dim }; \
    if (cuLaunchKernel(fn_gemm, gxo, gyo, 1, 16, 16, 1, 0, 0, a_ca_out, NULL) != CUDA_SUCCESS) return 6; \
    void *a_res[] = { &d_x, &d_proj, &n_elem }; \
    if (cuLaunchKernel(fn_resadd, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_res, NULL) != CUDA_SUCCESS) return 6; \
    void *a_mlp_ln[] = { &d_h, &d_x, &d_shift_mlp, &d_scale_mlp, &N, &dim, &eps }; \
    if (cuLaunchKernel(fn_modln, N, 1, 1, threads, 1, 1, (unsigned)ln_smem, 0, a_mlp_ln, NULL) != CUDA_SUCCESS) return 6; \
    unsigned gx1 = (N + 15) / 16, gy1 = (hidden + 15) / 16; \
    void *a_fc1[] = { &d_mh, &d_h, &d_fc1_w, &d_fc1_b, &N, &dim, &hidden }; \
    if (cuLaunchKernel(fn_gemm, gx1, gy1, 1, 16, 16, 1, 0, 0, a_fc1, NULL) != CUDA_SUCCESS) return 6; \
    void *a_gelu[] = { &d_mh, &n_mh }; \
    if (cuLaunchKernel(fn_gelu, (n_mh + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_gelu, NULL) != CUDA_SUCCESS) return 6; \
    void *a_fc2[] = { &d_mh2, &d_mh, &d_fc2_w, &d_fc2_b, &N, &hidden, &dim }; \
    if (cuLaunchKernel(fn_gemm, gxo, gyo, 1, 16, 16, 1, 0, 0, a_fc2, NULL) != CUDA_SUCCESS) return 6; \
    void *a_mlp_gate[] = { &d_x, &d_mh2, &d_gate_mlp, &N, &dim }; \
    if (cuLaunchKernel(fn_gated, (n_elem + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_mlp_gate, NULL) != CUDA_SUCCESS) return 6; \
} while (0)

    RUN_BLOCK();
    cuCtxSynchronize();
    cuMemcpyDtoH(dst, d_x, XD * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(dst, ref, XD, &mean);
    int ok = (mx <= threshold);

    double avg_ms = 0.0;
    if (repeat > 0) {
        cuCtxSynchronize();
        double t0 = now_ms();
        for (int r = 0; r < repeat; r++) RUN_BLOCK();
        cuCtxSynchronize();
        avg_ms = (now_ms() - t0) / (double)repeat;
    }
#undef RUN_BLOCK

    fprintf(stderr,
            "[test_slat_transformer_block] N=%d Nc=%d dim=%d H=%d D_h=%d hidden=%d max_abs=%.4g mean_abs=%.4g avg=%.4f ms x%d %s (threshold %.1g)\n",
            N, Nc, dim, H, D_h, hidden, (double)mx, mean, avg_ms, repeat,
            ok ? "OK" : "FAIL", (double)threshold);

    free(x); free(cond); free(shift_msa); free(scale_msa); free(gate_msa);
    free(shift_mlp); free(scale_mlp); free(gate_mlp); free(norm2_w); free(norm2_b);
    free(sa_qkv_w); free(sa_qkv_b); free(q_rms); free(k_rms); free(sa_out_w); free(sa_out_b);
    free(ca_q_w); free(ca_q_b); free(ca_kv_w); free(ca_kv_b); free(ca_out_w); free(ca_out_b);
    free(fc1_w); free(fc1_b); free(fc2_w); free(fc2_b); free(h); free(qkv); free(q); free(k);
    free(v); free(sa); free(proj); free(kv); free(K); free(V); free(xa); free(mh); free(mh2);
    free(ref); free(dst);
    cuMemFree(d_x0); cuMemFree(d_x); cuMemFree(d_cond); cuMemFree(d_shift_msa); cuMemFree(d_scale_msa);
    cuMemFree(d_gate_msa); cuMemFree(d_shift_mlp); cuMemFree(d_scale_mlp); cuMemFree(d_gate_mlp);
    cuMemFree(d_norm2_w); cuMemFree(d_norm2_b); cuMemFree(d_sa_qkv_w); cuMemFree(d_sa_qkv_b);
    cuMemFree(d_q_rms); cuMemFree(d_k_rms); cuMemFree(d_sa_out_w); cuMemFree(d_sa_out_b);
    cuMemFree(d_ca_q_w); cuMemFree(d_ca_q_b); cuMemFree(d_ca_kv_w); cuMemFree(d_ca_kv_b);
    cuMemFree(d_ca_out_w); cuMemFree(d_ca_out_b); cuMemFree(d_fc1_w); cuMemFree(d_fc1_b);
    cuMemFree(d_fc2_w); cuMemFree(d_fc2_b); cuMemFree(d_h); cuMemFree(d_qkv); cuMemFree(d_q);
    cuMemFree(d_k); cuMemFree(d_v); cuMemFree(d_sa); cuMemFree(d_proj); cuMemFree(d_kv);
    cuMemFree(d_K); cuMemFree(d_V); cuMemFree(d_xa); cuMemFree(d_mh); cuMemFree(d_mh2);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
