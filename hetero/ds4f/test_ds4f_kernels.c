/* test_ds4f_kernels.c — correctness gate for the AVX2 DS4F decode matvec
 * kernels in common/ds4f_matvec_avx2.h.
 *
 * Each kernel is checked against an independent scalar reference written
 * directly from the documented weight layout (NOT by calling any shared
 * helper), so a layout misunderstanding shows up as a failure rather than
 * being reproduced identically on both sides.
 *
 * Tolerances:
 *   BF16 / BF16_PV / FP8 / MXFP4-f32  exact same arithmetic, only the f32
 *                                     summation order differs -> tight relative
 *                                     tolerance against the reference magnitude.
 *   Q8_PV                             integer dot, exact up to the f32 scale
 *                                     multiply -> tight.
 *   MXFP4 W4A8                        quantizes the ACTIVATION to int8 per 32
 *                                     values, so it is checked with a looser
 *                                     bound and its error is reported.
 *
 * Build: make test_ds4f_kernels && ./build/test_ds4f_kernels
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "ds4f.h"

#define K 4096

static uint32_t rng_state = 0x1234567u;
static uint32_t rnd(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rndf(void) { return ((float)(rnd() >> 8) / 8388608.0f) - 1.0f; }

static int g_fail = 0;

static void check(const char *name, const float *got, const float *want,
                  int n, float tol) {
    float norm = 0.f, maxrel = 0.f;
    for (int i = 0; i < n; i++) norm += fabsf(want[i]);
    norm = norm / n;
    if (norm < 1e-6f) norm = 1e-6f;
    for (int i = 0; i < n; i++) {
        float rel = fabsf(got[i] - want[i]) / norm;
        if (rel > maxrel) maxrel = rel;
    }
    int ok = maxrel <= tol;
    if (!ok) g_fail = 1;
    printf("  %-28s max rel %.3e (tol %.0e)  %s\n", name, maxrel, tol, ok ? "PASS" : "FAIL");
    if (!ok)
        for (int i = 0; i < n; i++)
            printf("      [%d] got %14.6f want %14.6f\n", i, got[i], want[i]);
}

/* ---------------- BF16 ---------------- */
static void test_bf16(void) {
    uint16_t *w = malloc(sizeof(uint16_t) * 8 * K);
    float *x = malloc(sizeof(float) * K);
    for (int i = 0; i < 8 * K; i++) w[i] = (uint16_t)(rnd() >> 16);
    for (int i = 0; i < K; i++) x[i] = rndf();

    float got[8], want[8];
    matvec_bf16_8row(got, w, w+K, w+2*K, w+3*K, w+4*K, w+5*K, w+6*K, w+7*K, x, K);
    for (int r = 0; r < 8; r++) {
        double a = 0.0;
        for (int i = 0; i < K; i++) {
            uint32_t b = (uint32_t)w[(size_t)r * K + i] << 16; float f;
            memcpy(&f, &b, sizeof(f));
            a += (double)f * x[i];
        }
        want[r] = (float)a;
    }
    check("matvec_bf16_8row", got, want, 8, 1e-5f);
    free(w); free(x);
}

/* ---------------- BF16_PV ----------------
 * group of 8 rows = 4 pair buffers; pair p holds rows 2p,2p+1 interleaved as
 * [a0,b0,a1,b1,...] over K elements (2K halfwords). */
static void test_bf16_pv(void) {
    uint16_t *g = malloc(sizeof(uint16_t) * 8 * K);
    float *x = malloc(sizeof(float) * K);
    for (int i = 0; i < 8 * K; i++) g[i] = (uint16_t)(rnd() >> 16);
    for (int i = 0; i < K; i++) x[i] = rndf();

    float got[8], want[8];
    matvec_bf16_8row_pv(got, g, g + 2*K, g + 4*K, g + 6*K, x, K);
    for (int p = 0; p < 4; p++) {
        const uint16_t *pp = g + (size_t)p * 2 * K;
        for (int half = 0; half < 2; half++) {
            double a = 0.0;
            for (int i = 0; i < K; i++) {
                uint32_t b = (uint32_t)pp[2 * i + half] << 16; float f;
                memcpy(&f, &b, sizeof(f));
                a += (double)f * x[i];
            }
            want[2 * p + half] = (float)a;
        }
    }
    check("matvec_bf16_8row_pv", got, want, 8, 1e-5f);
    free(g); free(x);
}

/* ---------------- Q8_PV ----------------
 * K/64 blocks of 528 B: 8 fp16 row scales (16 B) then 8 rows x 64 int8. */
static void test_q8_pv(void) {
    int nb = K / 64;
    uint8_t *grp = malloc((size_t)nb * 528);
    float *x = malloc(sizeof(float) * K);
    for (int b = 0; b < nb; b++) {
        uint8_t *blk = grp + (size_t)b * 528;
        uint16_t *scl = (uint16_t *)blk;
        for (int r = 0; r < 8; r++) scl[r] = ggml_fp32_to_fp16(0.01f + 0.001f * (float)(rnd() % 50));
        int8_t *qs = (int8_t *)(blk + 16);
        for (int i = 0; i < 8 * 64; i++) qs[i] = (int8_t)((int)(rnd() % 255) - 127);
    }
    for (int i = 0; i < K; i++) x[i] = rndf();

    int8_t *xq = malloc(K); float *xs = malloc(sizeof(float) * nb);
    ds4f_quant_x_sdot_into(x, K, xq, xs);

    float got[8], want[8];
    matvec_sdot_8row(got, grp, xq, xs, K);
    for (int r = 0; r < 8; r++) {
        double a = 0.0;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = grp + (size_t)b * 528;
            const uint16_t *scl = (const uint16_t *)blk;
            const int8_t *qs = (const int8_t *)(blk + 16) + (size_t)r * 64;
            int32_t d = 0;
            for (int c = 0; c < 64; c++) d += (int32_t)qs[c] * (int32_t)xq[(size_t)b * 64 + c];
            a += (double)d * ((double)ggml_fp16_to_fp32(scl[r]) * xs[b]);
        }
        want[r] = (float)a;
    }
    check("matvec_sdot_8row (Q8_PV)", got, want, 8, 1e-5f);
    free(grp); free(x); free(xq); free(xs);
}

/* ---------------- FP8 E4M3 ---------------- */
static void test_fp8(void) {
    uint8_t *w = malloc((size_t)8 * K);
    float *x = malloc(sizeof(float) * K);
    int sbc = (K + 127) / 128;
    uint8_t *es = malloc(sbc);
    uint32_t lut[256];
    ds4f_init_fp8_e4m3fn_lut(lut);
    for (size_t i = 0; i < (size_t)8 * K; i++) w[i] = (uint8_t)(rnd() >> 24);
    for (int i = 0; i < K; i++) x[i] = rndf();
    for (int i = 0; i < sbc; i++) es[i] = (uint8_t)(120 + (rnd() % 12));

    float got[8], want[8];
    matvec_fp8e4m3_8row(got, w, w+K, w+2*K, w+3*K, w+4*K, w+5*K, w+6*K, w+7*K,
                        es, lut, x, K);
    for (int r = 0; r < 8; r++) {
        double a = 0.0;
        for (int c = 0; c < K; c++) {
            float wf; uint32_t bits = lut[w[(size_t)r * K + c]];
            memcpy(&wf, &bits, sizeof(wf));
            a += (double)wf * ggml_e8m0_to_fp32(es[c >> 7]) * x[c];
        }
        want[r] = (float)a;
    }
    check("matvec_fp8e4m3_8row", got, want, 8, 1e-5f);
    free(w); free(x); free(es);
}

/* ---------------- MXFP4 ---------------- */
static void test_mxfp4(void) {
    static const float kv[16] = { 0.f,1.f,2.f,3.f,4.f,6.f,8.f,12.f,
                                  0.f,-1.f,-2.f,-3.f,-4.f,-6.f,-8.f,-12.f };
    size_t rw = K / 2, rs = K / 32;
    uint8_t *w = malloc(8 * rw), *s = malloc(8 * rs);
    float *x = malloc(sizeof(float) * K);
    for (size_t i = 0; i < 8 * rw; i++) w[i] = (uint8_t)(rnd() >> 24);
    for (size_t i = 0; i < 8 * rs; i++) s[i] = (uint8_t)(120 + (rnd() % 12));
    for (int i = 0; i < K; i++) x[i] = rndf();

    float want[8];
    for (int r = 0; r < 8; r++) {
        const uint8_t *wr = w + (size_t)r * rw, *sr = s + (size_t)r * rs;
        double a = 0.0;
        for (int b = 0; b < K / 32; b++) {
            double p = 0.0;
            for (int j = 0; j < 16; j++) {
                uint8_t byte = wr[(size_t)b * 16 + j];
                p += (double)kv[byte & 0xf] * x[b * 32 + j];
                p += (double)kv[byte >> 4]  * x[b * 32 + j + 16];
            }
            a += p * ggml_e8m0_to_fp32(sr[b]);
        }
        want[r] = (float)a;
    }

    float got[8];
    matvec_mxfp4_8row(got, w, w+rw, w+2*rw, w+3*rw, w+4*rw, w+5*rw, w+6*rw, w+7*rw,
                      s, s+rs, s+2*rs, s+3*rs, s+4*rs, s+5*rs, s+6*rs, s+7*rs, x, K);
    check("matvec_mxfp4_8row (f32 act)", got, want, 8, 1e-5f);

    /* W4A8: activation quantized to int8 per 32-block. Random nibbles make the
     * true dot a random walk, the worst case for activation quantization, so
     * this bound is deliberately loose -- the meaningful gate is the model-level
     * logit comparison, not this. */
    int8_t *xq = malloc(K);
    float *xs = malloc(sizeof(float) * (K / 32)), *xc = malloc(sizeof(float) * (K / 32));
    ds4f_mxfp4_quant_act(x, K, xq, xs, xc);
    for (int r = 0; r < 8; r++)
        matvec_mxfp4_1row_i8(got + r, w + (size_t)r * rw, s + (size_t)r * rs, xq, xs, xc, K);
    check("matvec_mxfp4_1row_i8 (W4A8)", got, want, 8, 5e-2f);

    free(xq); free(xs); free(xc);

    /* ---- RAW on-disk layout (zero-copy experts) ----
     * Build the on-disk bytes, apply ds4f_copy_worker's exact repack to get the
     * arena form, and require the raw kernels on the on-disk bytes to agree with
     * the repacked kernels on the repacked bytes. This is the gate that lets
     * expert tensors point straight into the mapped safetensors shards. */
    uint8_t *wraw = malloc(8 * rw), *sraw = malloc(8 * rs);
    for (size_t i = 0; i < 8 * rw; i++) wraw[i] = (uint8_t)(rnd() >> 24);
    for (size_t i = 0; i < 8 * rs; i++) sraw[i] = (uint8_t)(120 + (rnd() % 12));

    uint8_t *wrp = malloc(8 * rw), *srp = malloc(8 * rs);
    for (int r = 0; r < 8; r++) {
        const uint8_t *sw = wraw + (size_t)r * rw; uint8_t *dw = wrp + (size_t)r * rw;
        for (size_t b = 0; b < rw / 16; b++) {
            const uint8_t *sb = sw + b * 16; uint8_t *db = dw + b * 16;
            for (int j = 0; j < 16; j++) {
                uint8_t lo = (j & 1) ? (uint8_t)(sb[j >> 1] >> 4)       : (uint8_t)(sb[j >> 1] & 0xf);
                uint8_t hi = (j & 1) ? (uint8_t)(sb[(j >> 1) + 8] >> 4) : (uint8_t)(sb[(j >> 1) + 8] & 0xf);
                db[j] = (uint8_t)((hi << 4) | lo);
            }
        }
        const uint8_t *ss = sraw + (size_t)r * rs; uint8_t *ds = srp + (size_t)r * rs;
        for (size_t j = 0; j < rs; j++) { uint8_t e = ss[j]; ds[j] = e ? (uint8_t)(e - 1) : 0; }
    }

    float ref[8];
    for (int r = 0; r < 8; r++)
        matvec_mxfp4_1row(ref + r, wrp + (size_t)r * rw, srp + (size_t)r * rs, x, K);

    float *xp = malloc(sizeof(float) * K);
    ds4f_mxfp4_perm_act_f32(x, K, xp);
    for (int r = 0; r < 8; r++)
        matvec_mxfp4_1row_f32_raw(got + r, wraw + (size_t)r * rw, sraw + (size_t)r * rs, xp, K);
    check("mxfp4 raw f32 == repacked", got, ref, 8, 1e-6f);

    int8_t *rq = malloc(K);
    float *rsq = malloc(sizeof(float) * (K / 32)), *rcq = malloc(sizeof(float) * (K / 32));
    ds4f_mxfp4_quant_act_raw(x, K, rq, rsq, rcq);
    for (int r = 0; r < 8; r++)
        matvec_mxfp4_1row_i8_raw(got + r, wraw + (size_t)r * rw, sraw + (size_t)r * rs,
                                 rq, rsq, rcq, K);
    check("mxfp4 raw W4A8 vs repacked", got, ref, 8, 5e-2f);

    free(wraw); free(sraw); free(wrp); free(srp); free(xp); free(rq); free(rsq); free(rcq);
    free(w); free(s); free(x);
}

int main(void) {
    printf("DS4F AVX2 decode kernels vs scalar reference (K=%d)\n\n", K);
    test_bf16();
    test_bf16_pv();
    test_q8_pv();
    test_fp8();
    test_mxfp4();
    printf("\n%s\n", g_fail ? "FAIL" : "ALL PASS");
    return g_fail;
}
