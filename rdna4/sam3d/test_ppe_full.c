/*
 * test_ppe_full — Phase 2b.8a standalone microbench.
 *
 * Composes the full PointPatchEmbed GPU graph end-to-end:
 *   1. ppe_linear3_invalid   pmap [3,S,S] → x [S,S,D]
 *   2. ppe_window_pack       x [S,S,D] + cls + pew → tok [Nwin,WL,D]
 *   3. PPE single ViT block  tok → tok (LN→qkv+bias→split→sdpa_batched→
 *                                       proj+bias→resid→LN→fc1+bias→
 *                                       GELU→fc2+bias→resid)
 *   4. ppe_cls_pos_extract   tok + pe → win_cls [Nwin,D]
 *
 * Random pmap (~10% NaN), random weights (He-init). Host reference uses
 * double accumulators throughout. Default Np=4 (16 windows) keeps the
 * host SDPA tractable; pass --Np 32 to stress at full PPE geometry
 * (skips the host check beyond a much-relaxed threshold).
 *
 * Usage:
 *   ./test_ppe_full [--Np 4] [--P 8] [--D 128] [--ffn 256]
 *                   [--H 16] [--threshold 5e-4] [-v]
 */

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include "hip_sam3d_kernels.h"
#include "hip_sam3d_ppe_forward.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float urand(uint32_t *st) {
    *st = (*st) * 1664525u + 1013904223u;
    return (float)((*st) >> 8) / (float)(1u << 24);
}
static float max_abs(const float *a, const float *b, size_t n, double *mean_out) {
    double s = 0.0; float mx = 0.0f;
    for (size_t i = 0; i < n; i++) { float d = fabsf(a[i] - b[i]); if (d > mx) mx = d; s += d; }
    if (mean_out) *mean_out = s / (n > 0 ? n : 1);
    return mx;
}

static void host_layernorm(float *dst, const float *src, const float *w,
                           const float *b, int n_tok, int dim, float eps) {
    for (int t = 0; t < n_tok; t++) {
        const float *r = src + (size_t)t * dim;
        float       *o = dst + (size_t)t * dim;
        double sum = 0, sq = 0;
        for (int c = 0; c < dim; c++) { double v = r[c]; sum += v; sq += v*v; }
        double mean = sum / dim;
        double var  = sq / dim - mean * mean;
        double inv  = 1.0 / sqrt(var + (double)eps);
        for (int c = 0; c < dim; c++)
            o[c] = (float)(((double)r[c] - mean) * inv * (double)w[c] + (double)b[c]);
    }
}
static void host_gemm_bias(float *Y, const float *X, const float *W, const float *b,
                           int N, int D_in, int D_out) {
    for (int n = 0; n < N; n++) {
        const float *xr = X + (size_t)n * D_in;
        for (int d = 0; d < D_out; d++) {
            const float *wr = W + (size_t)d * D_in;
            double a = b ? (double)b[d] : 0.0;
            for (int k = 0; k < D_in; k++) a += (double)wr[k] * (double)xr[k];
            Y[(size_t)n * D_out + d] = (float)a;
        }
    }
}
static void host_sdpa_batched(float *out, const float *q, const float *k, const float *v,
                              int B, int N_q, int N_k, int H, int D_h, float scale) {
    int E = H * D_h;
    double *sc = (double *)malloc((size_t)N_k * sizeof(double));
    for (int b = 0; b < B; b++) {
        const float *qb = q + (size_t)b * N_q * E;
        const float *kb = k + (size_t)b * N_k * E;
        const float *vb = v + (size_t)b * N_k * E;
        float       *ob = out + (size_t)b * N_q * E;
        for (int nq = 0; nq < N_q; nq++) for (int h = 0; h < H; h++) {
            const float *qv = qb + (size_t)nq * E + (size_t)h * D_h;
            double mx = -1e300;
            for (int nk = 0; nk < N_k; nk++) {
                const float *kv = kb + (size_t)nk * E + (size_t)h * D_h;
                double s = 0;
                for (int d = 0; d < D_h; d++) s += (double)qv[d] * (double)kv[d];
                s *= (double)scale; sc[nk] = s; if (s > mx) mx = s;
            }
            double sum = 0;
            for (int nk = 0; nk < N_k; nk++) { sc[nk] = exp(sc[nk] - mx); sum += sc[nk]; }
            double inv = 1.0 / sum;
            for (int d = 0; d < D_h; d++) {
                double a = 0;
                for (int nk = 0; nk < N_k; nk++)
                    a += sc[nk] * (double)vb[(size_t)nk * E + (size_t)h * D_h + d];
                ob[(size_t)nq * E + (size_t)h * D_h + d] = (float)(a * inv);
            }
        }
    }
    free(sc);
}

int main(int argc, char **argv)
{
    int Np = 4, P = 8, D = 128, ffn = 256, H = 16;
    float threshold = 5e-4f;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--Np")        && i+1 < argc) Np        = atoi(argv[++i]);
        else if (!strcmp(a, "--P")         && i+1 < argc) P         = atoi(argv[++i]);
        else if (!strcmp(a, "--D")         && i+1 < argc) D         = atoi(argv[++i]);
        else if (!strcmp(a, "--ffn")       && i+1 < argc) ffn       = atoi(argv[++i]);
        else if (!strcmp(a, "--H")         && i+1 < argc) H         = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                       verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (D % H != 0) { fprintf(stderr, "D %% H != 0\n"); return 2; }
    int D_h = D / H;
    int S    = Np * P;
    int WL   = 1 + P * P;
    int Nwin = Np * Np;
    int n_tok = Nwin * WL;
    float ln_eps = 1e-6f;
    float att_scale = 1.0f / sqrtf((float)D_h);

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return 3;
    if (hipInit(0) != hipSuccess) return 3;
    hipDevice_t dev = 0; if ((dev = (0), hipSetDevice(0)) != hipSuccess) return 3;
    hipCtx_t cu_ctx = NULL;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) return 3;

    hipModule_t mod = 0;
    int sm = hip_compile_kernels(&mod, dev, hip_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose, "test_ppe_full");
    if (sm < 0) return 4;
    cs3d_ppe_fns fns;
    if (cs3d_ppe_fns_lookup(&fns, mod) < 0) return 4;

    /* ---- Random inputs + weights. ---- */
    size_t n_pmap   = (size_t)3 * S * S;
    size_t n_xflat  = (size_t)S * S * D;
    size_t n_pew    = (size_t)WL * D;
    size_t n_h      = (size_t)n_tok * D;
    size_t n_qkv    = (size_t)n_tok * 3 * D;
    size_t n_inter  = (size_t)n_tok * ffn;
    size_t n_qkvw   = (size_t)(3 * D) * D;
    size_t n_projw  = (size_t)D * D;
    size_t n_fc1w   = (size_t)ffn * D;
    size_t n_fc2w   = (size_t)D * ffn;
    size_t n_pe     = (size_t)D * Nwin;
    size_t n_lin_w  = (size_t)D * 3;
    size_t n_out    = (size_t)Nwin * D;

    float *h_pmap = (float *)malloc(n_pmap  * sizeof(float));
    float *h_lin_w= (float *)malloc(n_lin_w * sizeof(float));
    float *h_lin_b= (float *)malloc((size_t)D * sizeof(float));
    float *h_inv  = (float *)malloc((size_t)D * sizeof(float));
    float *h_cls  = (float *)malloc((size_t)D * sizeof(float));
    float *h_pew  = (float *)malloc(n_pew * sizeof(float));
    float *h_ln1w = (float *)malloc((size_t)D * sizeof(float));
    float *h_ln1b = (float *)malloc((size_t)D * sizeof(float));
    float *h_qkvw = (float *)malloc(n_qkvw * sizeof(float));
    float *h_qkvb = (float *)malloc((size_t)(3*D) * sizeof(float));
    float *h_projw= (float *)malloc(n_projw * sizeof(float));
    float *h_projb= (float *)malloc((size_t)D * sizeof(float));
    float *h_ln2w = (float *)malloc((size_t)D * sizeof(float));
    float *h_ln2b = (float *)malloc((size_t)D * sizeof(float));
    float *h_fc1w = (float *)malloc(n_fc1w * sizeof(float));
    float *h_fc1b = (float *)malloc((size_t)ffn * sizeof(float));
    float *h_fc2w = (float *)malloc(n_fc2w * sizeof(float));
    float *h_fc2b = (float *)malloc((size_t)D * sizeof(float));
    float *h_pe   = (float *)malloc(n_pe * sizeof(float));
    float *h_ref  = (float *)malloc(n_out * sizeof(float));
    float *h_dst  = (float *)malloc(n_out * sizeof(float));
    /* host scratch */
    float *t_xflat= (float *)malloc(n_xflat * sizeof(float));
    float *t_tok  = (float *)malloc(n_h * sizeof(float));
    float *t_lnbuf= (float *)malloc(n_h * sizeof(float));
    float *t_qkv  = (float *)malloc(n_qkv * sizeof(float));
    float *t_Q    = (float *)malloc(n_h * sizeof(float));
    float *t_K    = (float *)malloc(n_h * sizeof(float));
    float *t_V    = (float *)malloc(n_h * sizeof(float));
    float *t_attn = (float *)malloc(n_h * sizeof(float));
    float *t_proj = (float *)malloc(n_h * sizeof(float));
    float *t_ffn  = (float *)malloc(n_inter * sizeof(float));
    if (!t_ffn) return 5;

    uint32_t rng = 0xC0FFEEu;
    for (size_t i = 0; i < n_pmap; i++) {
        h_pmap[i] = urand(&rng) * 2.0f - 1.0f;
        if ((urand(&rng) < 0.03f)) h_pmap[i] = NAN; /* sprinkle invalid */
    }
    float w_lin_s = 1.0f / sqrtf(3.0f);
    for (size_t i = 0; i < n_lin_w; i++) h_lin_w[i] = (urand(&rng) * 2.0f - 1.0f) * w_lin_s;
    for (int c = 0; c < D; c++) h_lin_b[c] = 0.05f * (urand(&rng) * 2.0f - 1.0f);
    for (int c = 0; c < D; c++) h_inv[c]   = (urand(&rng) * 2.0f - 1.0f) * 0.1f;
    for (int c = 0; c < D; c++) h_cls[c]   = 0.1f * (urand(&rng) * 2.0f - 1.0f);
    for (size_t i = 0; i < n_pew; i++) h_pew[i] = 0.05f * (urand(&rng) * 2.0f - 1.0f);
    float w_qkv_s = 1.0f / sqrtf((float)D), w_proj_s = w_qkv_s;
    float w_fc1_s = 1.0f / sqrtf((float)D), w_fc2_s = 1.0f / sqrtf((float)ffn);
    for (int c = 0; c < D; c++)         { h_ln1w[c] = 1.0f + 0.01f*(urand(&rng)*2-1); h_ln1b[c] = 0.01f*(urand(&rng)*2-1); }
    for (size_t i = 0; i < n_qkvw; i++)   h_qkvw[i] = (urand(&rng)*2-1) * w_qkv_s;
    for (int c = 0; c < 3*D; c++)         h_qkvb[c] = 0.05f*(urand(&rng)*2-1);
    for (size_t i = 0; i < n_projw; i++)  h_projw[i] = (urand(&rng)*2-1) * w_proj_s;
    for (int c = 0; c < D; c++)           h_projb[c] = 0.05f*(urand(&rng)*2-1);
    for (int c = 0; c < D; c++)         { h_ln2w[c] = 1.0f + 0.01f*(urand(&rng)*2-1); h_ln2b[c] = 0.01f*(urand(&rng)*2-1); }
    for (size_t i = 0; i < n_fc1w; i++)   h_fc1w[i] = (urand(&rng)*2-1) * w_fc1_s;
    for (int c = 0; c < ffn; c++)         h_fc1b[c] = 0.05f*(urand(&rng)*2-1);
    for (size_t i = 0; i < n_fc2w; i++)   h_fc2w[i] = (urand(&rng)*2-1) * w_fc2_s;
    for (int c = 0; c < D; c++)           h_fc2b[c] = 0.05f*(urand(&rng)*2-1);
    for (size_t i = 0; i < n_pe; i++)     h_pe[i]   = 0.05f*(urand(&rng)*2-1);

    /* ---- Host reference. ---- */
    /* step 1: per-pixel linear; pmap is [S, S, 3] interleaved. */
    for (int p = 0; p < S * S; p++) {
        float xyz[3];
        int bad = 0;
        for (int k = 0; k < 3; k++) {
            xyz[k] = h_pmap[(size_t)p * 3 + k];
            if (!isfinite(xyz[k])) bad = 1;
        }
        float *dst = t_xflat + (size_t)p * D;
        if (bad) {
            memcpy(dst, h_inv, (size_t)D * sizeof(float));
        } else {
            for (int d = 0; d < D; d++) {
                double a = h_lin_b[d];
                for (int k = 0; k < 3; k++) a += (double)h_lin_w[(size_t)d * 3 + k] * (double)xyz[k];
                dst[d] = (float)a;
            }
        }
    }
    /* step 2: window pack + cls + pew. */
    for (int wy = 0; wy < Np; wy++) for (int wx = 0; wx < Np; wx++) {
        int w = wy * Np + wx;
        for (int t = 0; t < WL; t++) {
            const float *src;
            if (t == 0) src = h_cls;
            else {
                int py = (t - 1) / P, px = (t - 1) - py * P;
                src = t_xflat + (size_t)((wy * P + py) * S + (wx * P + px)) * D;
            }
            const float *pe_t = h_pew + (size_t)t * D;
            float *dst = t_tok + ((size_t)w * WL + t) * D;
            for (int d = 0; d < D; d++) dst[d] = src[d] + pe_t[d];
        }
    }
    /* step 3: PPE single block. */
    host_layernorm(t_lnbuf, t_tok, h_ln1w, h_ln1b, n_tok, D, ln_eps);
    host_gemm_bias(t_qkv, t_lnbuf, h_qkvw, h_qkvb, n_tok, D, 3 * D);
    for (int t = 0; t < n_tok; t++) {
        memcpy(t_Q + (size_t)t * D, t_qkv + (size_t)t * 3 * D,         (size_t)D * sizeof(float));
        memcpy(t_K + (size_t)t * D, t_qkv + (size_t)t * 3 * D + D,     (size_t)D * sizeof(float));
        memcpy(t_V + (size_t)t * D, t_qkv + (size_t)t * 3 * D + 2 * D, (size_t)D * sizeof(float));
    }
    host_sdpa_batched(t_attn, t_Q, t_K, t_V, Nwin, WL, WL, H, D_h, att_scale);
    host_gemm_bias(t_proj, t_attn, h_projw, h_projb, n_tok, D, D);
    for (size_t i = 0; i < n_h; i++) t_tok[i] += t_proj[i];
    host_layernorm(t_lnbuf, t_tok, h_ln2w, h_ln2b, n_tok, D, ln_eps);
    host_gemm_bias(t_ffn, t_lnbuf, h_fc1w, h_fc1b, n_tok, D, ffn);
    for (size_t i = 0; i < n_inter; i++) {
        double v = t_ffn[i];
        t_ffn[i] = (float)(v * 0.5 * (1.0 + erf(v * 0.70710678118654752440)));
    }
    host_gemm_bias(t_proj, t_ffn, h_fc2w, h_fc2b, n_tok, ffn, D);
    for (size_t i = 0; i < n_h; i++) t_tok[i] += t_proj[i];
    /* step 4: CLS extract + pos. */
    int Np2 = Np * Np;
    for (int w = 0; w < Nwin; w++) {
        int wy = w / Np, wx = w - wy * Np;
        const float *cls_row = t_tok + (size_t)w * WL * D;
        float       *dst     = h_ref + (size_t)w * D;
        for (int d = 0; d < D; d++)
            dst[d] = cls_row[d] + h_pe[(size_t)d * Np2 + wy * Np + wx];
    }

    /* ---- Device path. ---- */
    hipDeviceptr_t d_pmap  = hip_upload_raw(h_pmap, n_pmap * sizeof(float));
    hipDeviceptr_t d_lin_w = hip_upload_raw(h_lin_w, n_lin_w * sizeof(float));
    hipDeviceptr_t d_lin_b = hip_upload_raw(h_lin_b, (size_t)D * sizeof(float));
    hipDeviceptr_t d_inv   = hip_upload_raw(h_inv, (size_t)D * sizeof(float));
    hipDeviceptr_t d_cls   = hip_upload_raw(h_cls, (size_t)D * sizeof(float));
    hipDeviceptr_t d_pew   = hip_upload_raw(h_pew, n_pew * sizeof(float));
    hipDeviceptr_t d_ln1w  = hip_upload_raw(h_ln1w, (size_t)D * sizeof(float));
    hipDeviceptr_t d_ln1b  = hip_upload_raw(h_ln1b, (size_t)D * sizeof(float));
    hipDeviceptr_t d_qkvw  = hip_upload_raw(h_qkvw, n_qkvw * sizeof(float));
    hipDeviceptr_t d_qkvb  = hip_upload_raw(h_qkvb, (size_t)(3*D) * sizeof(float));
    hipDeviceptr_t d_projw = hip_upload_raw(h_projw, n_projw * sizeof(float));
    hipDeviceptr_t d_projb = hip_upload_raw(h_projb, (size_t)D * sizeof(float));
    hipDeviceptr_t d_ln2w  = hip_upload_raw(h_ln2w, (size_t)D * sizeof(float));
    hipDeviceptr_t d_ln2b  = hip_upload_raw(h_ln2b, (size_t)D * sizeof(float));
    hipDeviceptr_t d_fc1w  = hip_upload_raw(h_fc1w, n_fc1w * sizeof(float));
    hipDeviceptr_t d_fc1b  = hip_upload_raw(h_fc1b, (size_t)ffn * sizeof(float));
    hipDeviceptr_t d_fc2w  = hip_upload_raw(h_fc2w, n_fc2w * sizeof(float));
    hipDeviceptr_t d_fc2b  = hip_upload_raw(h_fc2b, (size_t)D * sizeof(float));
    hipDeviceptr_t d_pe    = hip_upload_raw(h_pe, n_pe * sizeof(float));
    hipDeviceptr_t d_o = 0;
    if (hipMalloc(&d_o, n_out * sizeof(float)) != hipSuccess) return 5;

    cs3d_ppe_w bw;
    bw.point_proj_w      = d_lin_w;
    bw.point_proj_b      = d_lin_b;
    bw.invalid_xyz_token = d_inv;
    bw.cls_token         = d_cls;
    bw.pos_embed_window  = d_pew;
    bw.pos_embed         = d_pe;
    bw.ln1_w = d_ln1w; bw.ln1_b = d_ln1b;
    bw.qkv_w = d_qkvw; bw.qkv_b = d_qkvb;
    bw.proj_w = d_projw; bw.proj_b = d_projb;
    bw.ln2_w = d_ln2w; bw.ln2_b = d_ln2b;
    bw.fc1_w = d_fc1w; bw.fc1_b = d_fc1b;
    bw.fc2_w = d_fc2w; bw.fc2_b = d_fc2b;

    cs3d_ppe_ws ws = {0};
    if (cs3d_ppe_ws_alloc(&ws, Np, P, D, ffn) < 0) {
        fprintf(stderr, "ws alloc failed\n"); return 5;
    }
    if (cs3d_ppe_forward(&fns, &ws, &bw, d_pmap, d_o, H, ln_eps) < 0) {
        fprintf(stderr, "ppe forward launch failed\n"); return 6;
    }
    (void)att_scale; /* now derived inside cs3d_ppe_forward */

    hipDeviceSynchronize();
    hipMemcpyDtoH(h_dst, d_o, n_out * sizeof(float));

    double mean = 0.0;
    float mx = max_abs(h_dst, h_ref, n_out, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[test_ppe_full] Np=%d P=%d D=%d ffn=%d H=%d D_h=%d  S=%d WL=%d Nwin=%d  n_out=%zu  "
        "max_abs=%.4g mean_abs=%.4g  %s (threshold %.1g)\n",
        Np, P, D, ffn, H, D_h, S, WL, Nwin, n_out, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(h_pmap); free(h_lin_w); free(h_lin_b); free(h_inv); free(h_cls); free(h_pew);
    free(h_ln1w); free(h_ln1b); free(h_qkvw); free(h_qkvb); free(h_projw); free(h_projb);
    free(h_ln2w); free(h_ln2b); free(h_fc1w); free(h_fc1b); free(h_fc2w); free(h_fc2b);
    free(h_pe); free(h_ref); free(h_dst);
    free(t_xflat); free(t_tok); free(t_lnbuf); free(t_qkv); free(t_Q); free(t_K); free(t_V);
    free(t_attn); free(t_proj); free(t_ffn);
    hipFree(d_pmap); hipFree(d_lin_w); hipFree(d_lin_b); hipFree(d_inv);
    hipFree(d_cls); hipFree(d_pew);
    hipFree(d_ln1w); hipFree(d_ln1b); hipFree(d_qkvw); hipFree(d_qkvb);
    hipFree(d_projw); hipFree(d_projb);
    hipFree(d_ln2w); hipFree(d_ln2b);
    hipFree(d_fc1w); hipFree(d_fc1b); hipFree(d_fc2w); hipFree(d_fc2b);
    hipFree(d_pe);
    cs3d_ppe_ws_free(&ws);
    hipFree(d_o);
    hipModuleUnload(mod);
    hipCtxDestroy(cu_ctx);
    return ok ? 0 : 7;
}
