/*
 * test_dinov2_block — Phase 1b.7a standalone microbench.
 *
 * Validates `cs3d_dinov2_block_forward` (the composed-kernel block
 * defined in cuda_sam3d_dinov2_forward.h) on DINOv2-L geometry:
 *   n_tokens=1374, dim=1024, ffn=4096, n_heads=16, head_dim=64
 *
 * Random hidden + random block weights (He-init) on both paths; host
 * reference does the same op sequence with double accumulators. The
 * threshold accounts for compounded fp32 drift across the 4 stacked
 * matmuls + softmax + 2 LNs in one block.
 *
 * Usage:
 *   ./test_dinov2_block [--n_tok 1374] [--dim 1024] [--ffn 4096]
 *                       [--H 16] [--threshold 1e-3] [-v]
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"
#include "cuda_sam3d_kernels.h"
#include "cuda_sam3d_dinov2_gpu.h"
#include "cuda_sam3d_dinov2_forward.h"

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

/* Host helpers — double accumulators throughout. */

static void host_layernorm(float *dst, const float *src, const float *w,
                           const float *b, int n_tok, int dim, float eps)
{
    for (int t = 0; t < n_tok; t++) {
        const float *row = src + (size_t)t * dim;
        float       *out = dst + (size_t)t * dim;
        double sum = 0.0, sq = 0.0;
        for (int c = 0; c < dim; c++) { double v = row[c]; sum += v; sq += v * v; }
        double mean = sum / dim;
        double var  = sq / dim - mean * mean;
        double inv  = 1.0 / sqrt(var + (double)eps);
        for (int c = 0; c < dim; c++)
            out[c] = (float)(((double)row[c] - mean) * inv * (double)w[c] + (double)b[c]);
    }
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

static void host_sdpa(float *out, const float *q, const float *k, const float *v,
                      int N_q, int N_k, int H, int D_h, float scale)
{
    int E = H * D_h;
    double *scores = (double *)malloc((size_t)N_k * sizeof(double));
    for (int nq = 0; nq < N_q; nq++) {
        for (int h = 0; h < H; h++) {
            const float *qv = q + (size_t)nq * E + (size_t)h * D_h;
            double mx = -1e300;
            for (int nk = 0; nk < N_k; nk++) {
                const float *kv = k + (size_t)nk * E + (size_t)h * D_h;
                double s = 0.0;
                for (int d = 0; d < D_h; d++) s += (double)qv[d] * (double)kv[d];
                s *= (double)scale;
                scores[nk] = s;
                if (s > mx) mx = s;
            }
            double sum = 0.0;
            for (int nk = 0; nk < N_k; nk++) {
                scores[nk] = exp(scores[nk] - mx);
                sum += scores[nk];
            }
            double inv = 1.0 / sum;
            for (int d = 0; d < D_h; d++) {
                double acc = 0.0;
                for (int nk = 0; nk < N_k; nk++)
                    acc += scores[nk] * (double)v[(size_t)nk * E + (size_t)h * D_h + d];
                out[(size_t)nq * E + (size_t)h * D_h + d] = (float)(acc * inv);
            }
        }
    }
    free(scores);
}

int main(int argc, char **argv)
{
    int   n_tok     = 1374;
    int   dim       = 1024;
    int   ffn       = 4096;
    int   H         = 16;
    float threshold = 1e-3f;
    int   verbose   = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--n_tok")     && i+1 < argc) n_tok     = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--ffn")       && i+1 < argc) ffn       = atoi(argv[++i]);
        else if (!strcmp(a, "--H")         && i+1 < argc) H         = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (dim % H != 0) { fprintf(stderr, "dim must be divisible by H\n"); return 2; }
    int D_h = dim / H;
    float ln_eps = 1e-6f;
    float att_scale = 1.0f / sqrtf((float)D_h);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    int sm = cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose, "test_dinov2_block");
    if (sm < 0) return 4;
    cs3d_dinov2_fns fns;
    if (cs3d_dinov2_fns_lookup(&fns, mod) < 0) return 4;

    /* ---- Random inputs + weights. ---- */
    size_t n_h    = (size_t)n_tok * dim;
    size_t n_qkv  = (size_t)n_tok * 3 * dim;
    size_t n_inter = (size_t)n_tok * ffn;
    size_t n_qkvw = (size_t)(3 * dim) * dim;
    size_t n_projw = (size_t)dim * dim;
    size_t n_fc1w = (size_t)ffn * dim;
    size_t n_fc2w = (size_t)dim * ffn;

    float *h_hidden = (float *)malloc(n_h * sizeof(float));
    float *h_ln1w   = (float *)malloc((size_t)dim * sizeof(float));
    float *h_ln1b   = (float *)malloc((size_t)dim * sizeof(float));
    float *h_qkvw   = (float *)malloc(n_qkvw * sizeof(float));
    float *h_qkvb   = (float *)malloc((size_t)(3 * dim) * sizeof(float));
    float *h_projw  = (float *)malloc(n_projw * sizeof(float));
    float *h_projb  = (float *)malloc((size_t)dim * sizeof(float));
    float *h_ls1    = (float *)malloc((size_t)dim * sizeof(float));
    float *h_ln2w   = (float *)malloc((size_t)dim * sizeof(float));
    float *h_ln2b   = (float *)malloc((size_t)dim * sizeof(float));
    float *h_fc1w   = (float *)malloc(n_fc1w * sizeof(float));
    float *h_fc1b   = (float *)malloc((size_t)ffn * sizeof(float));
    float *h_fc2w   = (float *)malloc(n_fc2w * sizeof(float));
    float *h_fc2b   = (float *)malloc((size_t)dim * sizeof(float));
    float *h_ls2    = (float *)malloc((size_t)dim * sizeof(float));
    float *h_ref    = (float *)malloc(n_h * sizeof(float));
    float *h_dst    = (float *)malloc(n_h * sizeof(float));
    /* host scratch */
    float *t_lnbuf  = (float *)malloc(n_h * sizeof(float));
    float *t_qkv    = (float *)malloc(n_qkv * sizeof(float));
    float *t_Q      = (float *)malloc(n_h * sizeof(float));
    float *t_K      = (float *)malloc(n_h * sizeof(float));
    float *t_V      = (float *)malloc(n_h * sizeof(float));
    float *t_attn   = (float *)malloc(n_h * sizeof(float));
    float *t_proj   = (float *)malloc(n_h * sizeof(float));
    float *t_ffn    = (float *)malloc(n_inter * sizeof(float));
    if (!h_hidden || !t_ffn) { fprintf(stderr, "alloc failed\n"); return 5; }

    uint32_t rng = 0xC0FFEEu;
    float w_qkv_s = 1.0f / sqrtf((float)dim);
    float w_proj_s = 1.0f / sqrtf((float)dim);
    float w_fc1_s = 1.0f / sqrtf((float)dim);
    float w_fc2_s = 1.0f / sqrtf((float)ffn);
    for (size_t i = 0; i < n_h; i++)     h_hidden[i] = urand(&rng) * 2.0f - 1.0f;
    for (int c = 0; c < dim; c++)        { h_ln1w[c] = 1.0f + 0.01f * (urand(&rng) * 2.0f - 1.0f); h_ln1b[c] = 0.01f * (urand(&rng) * 2.0f - 1.0f); }
    for (size_t i = 0; i < n_qkvw; i++)  h_qkvw[i]   = (urand(&rng) * 2.0f - 1.0f) * w_qkv_s;
    for (int c = 0; c < 3*dim; c++)      h_qkvb[c]   = 0.05f * (urand(&rng) * 2.0f - 1.0f);
    for (size_t i = 0; i < n_projw; i++) h_projw[i]  = (urand(&rng) * 2.0f - 1.0f) * w_proj_s;
    for (int c = 0; c < dim; c++)        h_projb[c]  = 0.05f * (urand(&rng) * 2.0f - 1.0f);
    for (int c = 0; c < dim; c++)        h_ls1[c]    = 1e-2f + 1e-4f * (urand(&rng) * 2.0f - 1.0f);
    for (int c = 0; c < dim; c++)        { h_ln2w[c] = 1.0f + 0.01f * (urand(&rng) * 2.0f - 1.0f); h_ln2b[c] = 0.01f * (urand(&rng) * 2.0f - 1.0f); }
    for (size_t i = 0; i < n_fc1w; i++)  h_fc1w[i]   = (urand(&rng) * 2.0f - 1.0f) * w_fc1_s;
    for (int c = 0; c < ffn; c++)        h_fc1b[c]   = 0.05f * (urand(&rng) * 2.0f - 1.0f);
    for (size_t i = 0; i < n_fc2w; i++)  h_fc2w[i]   = (urand(&rng) * 2.0f - 1.0f) * w_fc2_s;
    for (int c = 0; c < dim; c++)        h_fc2b[c]   = 0.05f * (urand(&rng) * 2.0f - 1.0f);
    for (int c = 0; c < dim; c++)        h_ls2[c]    = 1e-2f + 1e-4f * (urand(&rng) * 2.0f - 1.0f);

    /* ---- Host reference: identical op sequence as the device path. ---- */
    memcpy(h_ref, h_hidden, n_h * sizeof(float));
    /* attn branch */
    host_layernorm(t_lnbuf, h_ref, h_ln1w, h_ln1b, n_tok, dim, ln_eps);
    host_gemm_bias(t_qkv, t_lnbuf, h_qkvw, h_qkvb, n_tok, dim, 3 * dim);
    for (int t = 0; t < n_tok; t++) {
        memcpy(t_Q + (size_t)t * dim, t_qkv + (size_t)t * 3 * dim,             (size_t)dim * sizeof(float));
        memcpy(t_K + (size_t)t * dim, t_qkv + (size_t)t * 3 * dim + dim,       (size_t)dim * sizeof(float));
        memcpy(t_V + (size_t)t * dim, t_qkv + (size_t)t * 3 * dim + 2 * dim,   (size_t)dim * sizeof(float));
    }
    host_sdpa(t_attn, t_Q, t_K, t_V, n_tok, n_tok, H, D_h, att_scale);
    host_gemm_bias(t_proj, t_attn, h_projw, h_projb, n_tok, dim, dim);
    for (int t = 0; t < n_tok; t++)
        for (int c = 0; c < dim; c++) {
            size_t i = (size_t)t * dim + c;
            h_ref[i] += t_proj[i] * h_ls1[c];
        }
    /* MLP branch */
    host_layernorm(t_lnbuf, h_ref, h_ln2w, h_ln2b, n_tok, dim, ln_eps);
    host_gemm_bias(t_ffn, t_lnbuf, h_fc1w, h_fc1b, n_tok, dim, ffn);
    for (size_t i = 0; i < n_inter; i++) {
        double v = t_ffn[i];
        t_ffn[i] = (float)(v * 0.5 * (1.0 + erf(v * 0.70710678118654752440)));
    }
    host_gemm_bias(t_proj, t_ffn, h_fc2w, h_fc2b, n_tok, ffn, dim);
    for (int t = 0; t < n_tok; t++)
        for (int c = 0; c < dim; c++) {
            size_t i = (size_t)t * dim + c;
            h_ref[i] += t_proj[i] * h_ls2[c];
        }

    /* ---- Device path. ---- */
    CUdeviceptr d_hidden = cu_upload_raw(h_hidden, n_h * sizeof(float));
    cs3d_dinov2_block_w bw;
    bw.ln1_w  = cu_upload_raw(h_ln1w,  (size_t)dim * sizeof(float));
    bw.ln1_b  = cu_upload_raw(h_ln1b,  (size_t)dim * sizeof(float));
    bw.qkv_w  = cu_upload_raw(h_qkvw,  n_qkvw * sizeof(float));
    bw.qkv_b  = cu_upload_raw(h_qkvb,  (size_t)(3 * dim) * sizeof(float));
    bw.proj_w = cu_upload_raw(h_projw, n_projw * sizeof(float));
    bw.proj_b = cu_upload_raw(h_projb, (size_t)dim * sizeof(float));
    bw.ls1    = cu_upload_raw(h_ls1,   (size_t)dim * sizeof(float));
    bw.ln2_w  = cu_upload_raw(h_ln2w,  (size_t)dim * sizeof(float));
    bw.ln2_b  = cu_upload_raw(h_ln2b,  (size_t)dim * sizeof(float));
    bw.fc1_w  = cu_upload_raw(h_fc1w,  n_fc1w * sizeof(float));
    bw.fc1_b  = cu_upload_raw(h_fc1b,  (size_t)ffn * sizeof(float));
    bw.fc2_w  = cu_upload_raw(h_fc2w,  n_fc2w * sizeof(float));
    bw.fc2_b  = cu_upload_raw(h_fc2b,  (size_t)dim * sizeof(float));
    bw.ls2    = cu_upload_raw(h_ls2,   (size_t)dim * sizeof(float));

    cs3d_dinov2_block_ws ws = {0};
    if (cs3d_dinov2_block_ws_alloc(&ws, n_tok, dim, ffn) < 0) {
        fprintf(stderr, "ws alloc failed\n"); return 5;
    }

    if (cs3d_dinov2_block_forward(&fns, &ws, &bw, d_hidden,
                                  n_tok, dim, ffn, H, D_h, ln_eps) < 0) {
        fprintf(stderr, "block forward launch failed\n"); return 6;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(h_dst, d_hidden, n_h * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_h, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_dinov2_block] n_tok=%d dim=%d ffn=%d H=%d D_h=%d  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        n_tok, dim, ffn, H, D_h, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    cs3d_dinov2_block_ws_free(&ws);
    cuMemFree(bw.ln1_w);  cuMemFree(bw.ln1_b);
    cuMemFree(bw.qkv_w);  cuMemFree(bw.qkv_b);
    cuMemFree(bw.proj_w); cuMemFree(bw.proj_b);
    cuMemFree(bw.ls1);
    cuMemFree(bw.ln2_w);  cuMemFree(bw.ln2_b);
    cuMemFree(bw.fc1_w);  cuMemFree(bw.fc1_b);
    cuMemFree(bw.fc2_w);  cuMemFree(bw.fc2_b);
    cuMemFree(bw.ls2);
    cuMemFree(d_hidden);
    free(h_hidden); free(h_ln1w); free(h_ln1b);
    free(h_qkvw); free(h_qkvb); free(h_projw); free(h_projb); free(h_ls1);
    free(h_ln2w); free(h_ln2b); free(h_fc1w); free(h_fc1b);
    free(h_fc2w); free(h_fc2b); free(h_ls2);
    free(h_ref); free(h_dst);
    free(t_lnbuf); free(t_qkv); free(t_Q); free(t_K); free(t_V);
    free(t_attn); free(t_proj); free(t_ffn);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
