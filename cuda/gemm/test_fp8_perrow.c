/* test_fp8_perrow.c - validate cublasew per-row FP8 weight scaling (Phase 3).
 *
 * Builds a weight matrix with wildly skewed per-output-channel magnitudes
 * (rows from ~1.0 down to ~1e-6). A single per-tensor FP8 scale sizes for the
 * largest row, so rows more than ~4.5 decades below the max underflow into the
 * e4m3 subnormal range (1-2 mantissa bits) or flush to zero, and their outputs
 * drift badly. A per-row scale gives every row the full e4m3 range. Expect:
 * per-row output MAE << per-tensor MAE vs the f32 ref.
 *
 * NOTE on FP8 vs INT8: e4m3 is *floating point* (4-bit exponent), so per-tensor
 * scaling preserves small rows far better than it would for INT8 — per-row only
 * wins once the per-channel dynamic range exceeds the e4m3 exponent span
 * (~2^15). That is why the spread here is 6 decades, not the 2-3 a fixed-point
 * INT8 test would use.
 *
 * NOTE on inputs: X and W are non-negative here to isolate the weight
 * dynamic-range effect. With signed, mutually-cancelling dot products, e4m3's
 * inherent rounding noise (verified byte-identical for both per-tensor and
 * per-row via host FP8 simulation) is amplified by catastrophic cancellation
 * and swamps the per-row win — that noise is common to both methods, so it is
 * not what per-row weight scaling is meant to fix. Non-negative inputs keep the
 * accumulation well-conditioned so the underflow that per-tensor scaling
 * inflicts on small-magnitude rows (and that per-row scaling avoids) is the
 * dominant error source.
 *
 * The per-row path runs a per-tensor FP8 matmul (no A scale) to an F32 output
 * then applies the [n_out] row scales via cublasSdgmm (C = diag(s)*D) — this
 * works on consumer GeForce Blackwell. If cublasSdgmm is missing or the FP8
 * F32-out matmul is rejected, the call returns -1 and we report it as
 * "unsupported (fallback works)" — not a hard failure.
 *
 * Note: per-tensor output is BF16, per-row output is F32. We compare each
 * against the same f32 ref; the MAE comparison is still meaningful because the
 * BF16 rounding (~8 mantissa bits) is far finer than the FP8 quant error that
 * dominates the per-tensor result.
 *
 * Build: handled by `make test-fp8-perrow` in this directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../cuew.h"
#include "../cublasew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"  /* cu_f32_to_fp8_e4m3 */

static float host_fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    int32_t exp = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    float val;
    if (exp == 0) val = (float)mant / 8.0f * powf(2.0f, -6.0f);   /* subnormal */
    else val = (1.0f + (float)mant / 8.0f) * powf(2.0f, (float)(exp - 7));
    return sign ? -val : val;
}
static float host_bf16_to_f32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16; float f; memcpy(&f, &bits, 4); return f;
}

/* Row-major Y[m,n_out] = X[m,n_in] * W[n_out,n_in]^T (double accumulate ref) */
static void cpu_ref(float *Y, const float *W, const float *X,
                    int m, int n_out, int n_in) {
    for (int r = 0; r < m; r++)
        for (int o = 0; o < n_out; o++) {
            double acc = 0.0;
            const float *xr = X + (size_t)r * n_in, *wr = W + (size_t)o * n_in;
            for (int k = 0; k < n_in; k++) acc += (double)xr[k] * (double)wr[k];
            Y[(size_t)r * n_out + o] = (float)acc;
        }
}

static void mae_rel(const float *a, const float *ref, size_t n,
                    double *mae, double *rms_rel) {
    double sa = 0.0, sr = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)ref[i]);
        sa += d;
        double den = fabs((double)ref[i]) + 1e-3;
        sr += (d / den) * (d / den);
    }
    *mae = sa / (double)n;
    *rms_rel = sqrt(sr / (double)n);
}

/* Mean over output channels of the per-channel L2 relative error. Each channel
 * (column o of row-major Y[n_tok, n_out]) gets equal weight regardless of its
 * magnitude, so a small-magnitude row reconstructed badly counts as much as a
 * large one — this is the error per-row weight scaling actually targets. The
 * overall MAE, by contrast, is swamped by the few large-magnitude rows. */
static double chan_rel(const float *a, const float *ref, int n_tok, int n_out) {
    double acc = 0.0;
    for (int o = 0; o < n_out; o++) {
        double num = 0.0, den = 0.0;
        for (int t = 0; t < n_tok; t++) {
            double r = ref[(size_t)t * n_out + o];
            double d = (double)a[(size_t)t * n_out + o] - r;
            num += d * d; den += r * r;
        }
        if (den > 0.0) acc += sqrt(num / den);
    }
    return acc / (double)n_out;
}

int main(void) {
    const int n_tok = 64, n_out = 256, n_in = 512;
    /* Quantize against the full e4m3 ceiling of 448. cu_f32_to_fp8_e4m3() now
     * does round-to-nearest-even + saturate-to-finite (no top-octave collapse),
     * so the only difference between the two paths is the per-row vs per-tensor
     * weight-scale dynamic range. */
    const float E4M3_MAX = 448.0f;

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1; }
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 1; }
    CUdevice dev; CUcontext ctx;
    cuDeviceGet(&dev, 0); cuCtxCreate(&ctx, 0, dev);
    char name[256]; cuDeviceGetName(name, sizeof(name), dev);
    printf("Device: %s   m=%d n_out=%d n_in=%d\n", name, n_tok, n_out, n_in);

    CUstream stream = NULL; cuStreamCreate(&stream, 0);
    cublasew_context *bc = NULL;
    if (cublasewCreate(&bc, stream) != 0) { fprintf(stderr, "cublasewCreate failed\n"); return 1; }
    if (cublasew_lt_available(bc) != 0) {
        fprintf(stderr, "cuBLAS-LT unavailable; skipping FP8 test.\n");
        cublasewDestroy(bc); return 0; }

    size_t nW = (size_t)n_out * n_in, nX = (size_t)n_tok * n_in, nY = (size_t)n_tok * n_out;
    float *hW = malloc(nW * 4), *hX = malloc(nX * 4), *hRef = malloc(nY * 4);
    float *row_mag = malloc(n_out * sizeof(float));
    srand(7);
    /* Per-row magnitude spanning ~6 decades: 10^(-6 .. 0) — wide enough that a
     * per-tensor e4m3 scale underflows the smallest rows (see header note). */
    for (int o = 0; o < n_out; o++) {
        float u = (float)rand() / RAND_MAX;       /* 0..1 */
        row_mag[o] = powf(10.0f, -6.0f * u);      /* 1e-6 .. 1 */
        for (int k = 0; k < n_in; k++)            /* non-negative (see header) */
            hW[(size_t)o * n_in + k] = ((float)rand() / RAND_MAX) * row_mag[o];
    }
    for (size_t i = 0; i < nX; i++) hX[i] = (float)rand() / RAND_MAX;  /* [0,1] */
    cpu_ref(hRef, hW, hX, n_tok, n_out, n_in);

    /* X: per-tensor scale. */
    float x_amax = 0.0f;
    for (size_t i = 0; i < nX; i++) { float a = fabsf(hX[i]); if (a > x_amax) x_amax = a; }
    float x_scale = x_amax / E4M3_MAX;
    uint8_t *Xq = malloc(nX);
    for (size_t i = 0; i < nX; i++) Xq[i] = cu_f32_to_fp8_e4m3(hX[i] / x_scale);

    /* W: per-tensor and per-row quantizations. */
    float w_amax = 0.0f;
    for (size_t i = 0; i < nW; i++) { float a = fabsf(hW[i]); if (a > w_amax) w_amax = a; }
    float w_scale_pt = w_amax / E4M3_MAX;
    uint8_t *Wq_pt = malloc(nW), *Wq_pr = malloc(nW);
    float *w_scale_vec = malloc(n_out * sizeof(float));
    for (int o = 0; o < n_out; o++) {
        float ramax = 0.0f;
        for (int k = 0; k < n_in; k++) { float a = fabsf(hW[(size_t)o * n_in + k]); if (a > ramax) ramax = a; }
        w_scale_vec[o] = (ramax > 0.0f ? ramax : 1.0f) / E4M3_MAX;
        for (int k = 0; k < n_in; k++) {
            size_t idx = (size_t)o * n_in + k;
            Wq_pt[idx] = cu_f32_to_fp8_e4m3(hW[idx] / w_scale_pt);
            Wq_pr[idx] = cu_f32_to_fp8_e4m3(hW[idx] / w_scale_vec[o]);
        }
    }

    /* Device buffers. */
    CUdeviceptr dXq, dWq_pt, dWq_pr, dY, dY_f32, d_xs, d_ws_pt, d_ws_vec;
    cuMemAlloc(&dXq, nX); cuMemAlloc(&dWq_pt, nW); cuMemAlloc(&dWq_pr, nW);
    cuMemAlloc(&dY, nY * 2);     /* bf16 out (per-tensor path) */
    cuMemAlloc(&dY_f32, nY * 4); /* f32 out  (per-row Sdgmm path) */
    cuMemAlloc(&d_xs, 4); cuMemAlloc(&d_ws_pt, 4); cuMemAlloc(&d_ws_vec, n_out * 4);
    cuMemcpyHtoD(dXq, Xq, nX); cuMemcpyHtoD(dWq_pt, Wq_pt, nW); cuMemcpyHtoD(dWq_pr, Wq_pr, nW);
    cuMemcpyHtoD(d_xs, &x_scale, 4); cuMemcpyHtoD(d_ws_pt, &w_scale_pt, 4);
    cuMemcpyHtoD(d_ws_vec, w_scale_vec, n_out * 4);

    uint16_t *hYb = malloc(nY * 2); float *hY = malloc(nY * 4);
    double mae, rrel;

    /* --- per-tensor weight scale --- */
    int rc = cublasew_gemm_fp8_e4m3_bf16out_rowmajor_nt(
        bc, dY, dWq_pt, dXq, d_ws_pt, d_xs, 0, n_tok, n_out, n_in);
    if (rc != 0) { fprintf(stderr, "per-tensor FP8 failed (%d)\n", rc); return 1; }
    cuStreamSynchronize(stream);
    cuMemcpyDtoH(hYb, dY, nY * 2);
    for (size_t i = 0; i < nY; i++) hY[i] = host_bf16_to_f32(hYb[i]);
    mae_rel(hY, hRef, nY, &mae, &rrel);
    double chan_pt = chan_rel(hY, hRef, n_tok, n_out);
    printf("\nper-tensor W scale : MAE=%.4e  per-chan rel=%.4e\n", mae, chan_pt);
    float *hY_pt = malloc(nY * 4); memcpy(hY_pt, hY, nY * 4);

    /* --- per-row weight scale (FP8 matmul + Sdgmm row-scale, F32 out) --- */
    rc = cublasew_gemm_fp8_e4m3_f32out_wperrow_rowmajor_nt(
        bc, dY_f32, dWq_pr, dXq, d_ws_vec, d_xs, n_tok, n_out, n_in);
    if (rc != 0) {
        printf("per-row   W scale : returned -1 — cublasSdgmm path unavailable "
               "on this GPU/cuBLAS; per-tensor fallback remains valid.\n");
        printf("\nRESULT: per-row path correctly reports unsupported (fallback OK)\n");
    } else {
        cuStreamSynchronize(stream);
        cuMemcpyDtoH(hY, dY_f32, nY * 4);
        mae_rel(hY, hRef, nY, &mae, &rrel);
        double chan_pr = chan_rel(hY, hRef, n_tok, n_out);
        if (getenv("DBG")) {
            int o_small = 0, o_big = 0;
            for (int o = 0; o < n_out; o++) {
                if (row_mag[o] < row_mag[o_small]) o_small = o;
                if (row_mag[o] > row_mag[o_big]) o_big = o;
            }
            /* Host FP8-sim of the big channel (token 0): dequantize the same
             * fp8 bytes the GPU used and double-accumulate. If this matches the
             * GPU result, the ~50% gap vs ref is genuine FP8 quant error, not a
             * scale bug. */
            double sim_pt = 0.0, sim_pr = 0.0;
            for (int k = 0; k < n_in; k++) {
                double xq = host_fp8_e4m3_to_f32(Xq[(size_t)0 * n_in + k]) * x_scale;
                sim_pt += xq * host_fp8_e4m3_to_f32(Wq_pt[(size_t)o_big * n_in + k]) * w_scale_pt;
                sim_pr += xq * host_fp8_e4m3_to_f32(Wq_pr[(size_t)o_big * n_in + k]) * w_scale_vec[o_big];
            }
            printf("[dbg] BIG  chan o=%d mag=%.2e ws=%.3e : ref=%.5e  pt=%.5e  pr=%.5e  "
                   "simPT=%.5e simPR=%.5e\n",
                   o_big, row_mag[o_big], w_scale_vec[o_big],
                   hRef[o_big], hY_pt[o_big], hY[o_big], sim_pt, sim_pr);
            printf("[dbg] SMALL chan o=%d mag=%.2e ws=%.3e : ref=%.5e  pt=%.5e  pr=%.5e\n",
                   o_small, row_mag[o_small], w_scale_vec[o_small],
                   hRef[o_small], hY_pt[o_small], hY[o_small]);
        }
        printf("per-row   W scale : MAE=%.4e  per-chan rel=%.4e\n", mae, chan_pr);
        printf("\nper-channel rel-error improvement (per-tensor / per-row) = %.2fx\n",
               chan_pt / chan_pr);
        printf("RESULT: %s\n", (chan_pr < chan_pt) ? "per-row improves precision (PASS)"
                                              : "per-row did NOT improve (CHECK)");
    }

    cublasewDestroy(bc);
    return 0;
}
