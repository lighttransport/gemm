/* Correctness tests for the pair-interleaved q38d kernels.
 * 1. F4/F6 repack reproduces the original compact tile weights exactly.
 * 2. Q6_K/Q4_K repack reproduces the GGML reference dequantization.
 * 3. SVE prepare matches the scalar quantizer byte for byte.
 * 4. SVE group kernels match a double reference over the same quantized
 *    activations (tight relative tolerance), for A8 and A16. */
#include <stdlib.h>
#include <stdio.h>
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../../common/ggml_dequant.h"
#include "../qwen38_lowbit.h"
#include "q38d_kern.h"
#include <stdio.h>
#include <stdlib.h>

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}
static float frand(void) { return (float)(rnd() % 2000001) / 1e6f - 1.f; }
static int failures;
#define CHECK(c, ...) do { if (!(c)) { failures++; fprintf(stderr, __VA_ARGS__); } } while (0)

static void fill_act(float *x, int n, int outliers) {
    for (int i = 0; i < n; i++) x[i] = frand() * (1 + (rnd() % 7));
    for (int i = 0; i < outliers; i++) x[rnd() % n] = frand() * 900.f;
    x[3] = 0; /* exact zero */
    for (int i = 64; i < 80 && i < n; i++) x[i] = 0; /* all-zero 16-block */
}

static int test_matrix(int fmt, uint8_t *g, int cols, int groups) {
    float *x = malloc((size_t)cols * 4);
    fill_act(x, cols, 5);
    int bad = 0;
    for (int arith = 8; arith <= 16; arith += 8) {
        q38d_act a = {cols, arith, NULL, NULL, NULL, NULL}, b = a;
        a.q = malloc(q38d_act_qbytes(cols, arith)); b.q = malloc(q38d_act_qbytes(cols, arith));
        a.sc = malloc((size_t)cols / 16 * 4); b.sc = malloc((size_t)cols / 16 * 4);
        a.sum = malloc((size_t)cols / 16 * 4); b.sum = malloc((size_t)cols / 16 * 4);
        q38d_prepare_ref(&a, x);
#ifdef __ARM_FEATURE_SVE
        q38d_prepare_sve(&b, x, 1.f, NULL);
        CHECK(!memcmp(a.q, b.q, q38d_act_qbytes(cols, arith)), "prepare q mismatch a%d\n", arith);
        CHECK(!memcmp(a.sc, b.sc, (size_t)cols / 16 * 4), "prepare sc mismatch a%d\n", arith);
        CHECK(!memcmp(a.sum, b.sum, (size_t)cols / 16 * 4), "prepare sum mismatch a%d\n", arith);
        size_t gb = q38d_group_bytes(fmt, cols);
        float *out = malloc((size_t)groups * 8 * 4), ref[8];
        q38d_gemv(out, g, fmt, &a, 0, groups, groups * 8, 0);
        double worst = 0;
        for (int gi = 0; gi < groups; gi++) {
            q38d_group_ref(ref, g + gi * gb, fmt, &a);
            double norm = 0;
            for (int r = 0; r < 8; r++) norm += (double)ref[r] * ref[r];
            norm = sqrt(norm / 8) + 1e-30;
            for (int r = 0; r < 8; r++) {
                double e = fabs((double)out[gi * 8 + r] - ref[r]) / norm;
                if (e > worst) worst = e;
            }
        }
        if (fmt == Q38D_Q4K) {
            double wq = 0;
            q38d_asm_variant = 5;
            for (int gi = 0; gi < groups; gi++) {
                float oa[8];
                q38d_group_asm(oa, g + gi * gb, fmt, arith, &a, 0, 8);
                q38d_group_ref(ref, g + gi * gb, fmt, &a);
                double norm = 0;
                for (int r = 0; r < 8; r++) norm += (double)ref[r] * ref[r];
                norm = sqrt(norm / 8) + 1e-30;
                for (int r = 0; r < 8; r++) { double e = fabs((double)oa[r] - ref[r]) / norm; if (e > wq) wq = e; }
            }
            printf("   q4k asm max_rel_err=%.3e\n", wq);
            if (wq > 2e-6) { bad = 1; failures++; }
        }
        if (fmt == Q38D_Q8K) {
            double wq = 0;
            for (int gi = 0; gi < groups; gi++) {
                float oa[8];
                q38d_group_asm(oa, g + gi * gb, fmt, arith, &a, 0, 8);
                q38d_group_ref(ref, g + gi * gb, fmt, &a);
                double norm = 0;
                for (int r = 0; r < 8; r++) norm += (double)ref[r] * ref[r];
                norm = sqrt(norm / 8) + 1e-30;
                for (int r = 0; r < 8; r++) { double e = fabs((double)oa[r] - ref[r]) / norm; if (e > wq) wq = e; }
            }
            printf("   q8k asm max_rel_err=%.3e\n", wq);
            if (wq > 2e-6) { bad = 1; failures++; }
        }
        if ((fmt == Q38D_F4 || fmt == Q38D_F6) && groups >= 2) { /* dual kernel: groups 0 and 1 as A and B */
            float oa[16], ob[8], oc[8];
            q38d_gemv_dual_fmt(oa, oa + 8, g, g + gb, &a, 0, 1, fmt);
            q38d_asm_variant = 7;
            q38d_group_asm(ob, g, fmt, arith, &a, 0, 8);
            q38d_group_asm(oc, g + gb, fmt, arith, &a, 0, 8);
            double e8 = 0;
            for (int r = 0; r < 8; r++) {
                e8 = fmax(e8, fabs((double)oa[r] - ob[r]) / (fabs(ob[r]) + 1e-3));
                e8 = fmax(e8, fabs((double)oa[8 + r] - oc[r]) / (fabs(oc[r]) + 1e-3));
            }
            printf("   dual max_rel_diff=%.3e\n", e8);
            if (e8 > 1e-5) { bad = 1; failures++; }
        }
        if (fmt != Q38D_Q4K && fmt != Q38D_Q8K) { /* assembly path vs double reference */
            double wa = 0;
            for (int gi = 0; gi < groups; gi++) {
                float oa[8];
                q38d_group_asm(oa, g + gi * gb, fmt, arith, &a, 0, 8);
                q38d_group_ref(ref, g + gi * gb, fmt, &a);
                double norm = 0;
                for (int r = 0; r < 8; r++) norm += (double)ref[r] * ref[r];
                norm = sqrt(norm / 8) + 1e-30;
                for (int r = 0; r < 8; r++) { double e = fabs((double)oa[r] - ref[r]) / norm; if (e > wa) wa = e; }
            }
            for (int gi = 0; gi < groups; gi++) {
                float oa[8], ob[8];
                q38d_asm_variant = 1; q38d_group_asm(oa, g + gi * gb, fmt, arith, &a, 0, 8);
                q38d_asm_variant = 2; q38d_group_asm(ob, g + gi * gb, fmt, arith, &a, 0, 8);
                if (memcmp(oa, ob, sizeof oa)) { failures++; fprintf(stderr, "asm variants differ fmt %d\n", fmt); break; }
                q38d_asm_variant = 4; q38d_group_asm(ob, g + gi * gb, fmt, arith, &a, 0, 8);
                if (memcmp(oa, ob, sizeof oa)) { failures++; fprintf(stderr, "asm4 differs fmt %d\n", fmt); break; }
                q38d_f6_variant = 5;
                q38d_asm_variant = 5; q38d_group_asm(ob, g + gi * gb, fmt, arith, &a, 0, 8);
                { double e5 = 0; for (int r = 0; r < 8; r++) e5 = fmax(e5, fabs((double)oa[r] - ob[r]) / (fabs(oa[r]) + 1e-3));
                  if (e5 > 1e-5) { failures++; fprintf(stderr, "asm5 differs fmt %d (%g)\n", fmt, e5); break; } }
                q38d_f6_variant = 9; q38d_group_asm(ob, g + gi * gb, fmt, arith, &a, 0, 8);
                { double e9 = 0; for (int r = 0; r < 8; r++) e9 = fmax(e9, fabs((double)oa[r] - ob[r]) / (fabs(oa[r]) + 1e-3));
                  if (e9 > 1e-5) { failures++; fprintf(stderr, "asm9 differs fmt %d (%g)\n", fmt, e9); break; } }
                q38d_asm_variant = 7; q38d_group_asm(ob, g + gi * gb, fmt, arith, &a, 0, 8);
                { double e7 = 0; for (int r = 0; r < 8; r++) e7 = fmax(e7, fabs((double)oa[r] - ob[r]) / (fabs(oa[r]) + 1e-3));
                  if (e7 > 1e-5) { failures++; fprintf(stderr, "asm7 differs fmt %d (%g)\n", fmt, e7); break; } }
                q38d_asm_variant = 8; q38d_group_asm(ob, g + gi * gb, fmt, arith, &a, 0, 8);
                { double e9 = 0; for (int r = 0; r < 8; r++) e9 = fmax(e9, fabs((double)oa[r] - ob[r]) / (fabs(oa[r]) + 1e-3));
                  if (e9 > 1e-5) { failures++; fprintf(stderr, "asm7c differs fmt %d (%g)\n", fmt, e9); break; } }
                q38d_asm_variant = 6; q38d_group_asm(ob, g + gi * gb, fmt, arith, &a, 0, 8);
                { double e6 = 0; for (int r = 0; r < 8; r++) e6 = fmax(e6, fabs((double)oa[r] - ob[r]) / (fabs(oa[r]) + 1e-3));
                  if (e6 > 1e-5) { failures++; fprintf(stderr, "asm6 differs fmt %d (%g)\n", fmt, e6); break; } }
                q38d_asm_variant = 2;
            }
            printf("   asm max_rel_err=%.3e\n", wa);
            if (wa > 2e-6) { bad = 1; failures++; }
        }
        /* mode=1 accumulates */
        float *out2 = malloc((size_t)groups * 8 * 4);
        for (int i = 0; i < groups * 8; i++) out2[i] = 1.5f;
        q38d_gemv(out2, g, fmt, &a, 0, groups, groups * 8, 1);
        int addok = 1;
        for (int i = 0; i < groups * 8; i++) if (out2[i] != out[i] + 1.5f) addok = 0;
        CHECK(addok, "fmt %d a%d accumulate mismatch\n", fmt, arith);
        printf("fmt=%d cols=%d arith=%d max_rel_err=%.3e\n", fmt, cols, arith, worst);
        if (worst > 2e-6) { bad = 1; failures++; }
        /* quantization error vs F32 activations (informational) */
        q38d_act f = {cols, 0, NULL, NULL, NULL, x};
        double qerr = 0, qn = 0;
        for (int gi = 0; gi < groups && gi < 4; gi++) {
            q38d_group_ref(ref, g + gi * gb, fmt, &f);
            for (int r = 0; r < 8; r++) {
                double e = out[gi * 8 + r] - ref[r];
                qerr += e * e; qn += (double)ref[r] * ref[r];
            }
        }
        printf("   activation quantization rel L2 vs F32 = %.3e\n", sqrt(qerr / qn));
        /* exact-weight F32 kernel vs double reference */
        {
            double w2 = 0;
            for (int gi = 0; gi < groups; gi++) {
                float o32[8];
                q38d_group_f32_sve(o32, g + gi * gb, fmt, x, cols, 0, 8);
                q38d_group_ref(ref, g + gi * gb, fmt, &f);
                double norm = 0;
                for (int r = 0; r < 8; r++) norm += (double)ref[r] * ref[r];
                norm = sqrt(norm / 8) + 1e-30;
                for (int r = 0; r < 8; r++) { double e = fabs((double)o32[r] - ref[r]) / norm; if (e > w2) w2 = e; }
            }
            printf("   f32 kernel max_rel_err=%.3e\n", w2);
            if (w2 > 2e-6) { bad = 1; failures++; }
        }
        /* unit quantizer reproduces whole-vector quantizer */
        {
            q38d_act u = a;
            u.q = malloc(q38d_act_qbytes(cols, arith)); u.sc = malloc((size_t)cols / 16 * 4); u.sum = malloc((size_t)cols / 16 * 4);
            q38d_prepare_range(&u, x, 0, cols / 32);
            int same = !memcmp(u.q, a.q, q38d_act_qbytes(cols, arith)) && !memcmp(u.sc, a.sc, (size_t)cols / 16 * 4)
                       && !memcmp(u.sum, a.sum, (size_t)cols / 16 * 4);
            CHECK(same, "unit prepare mismatch a%d\n", arith);
            free(u.q); free(u.sc); free(u.sum);
        }
        free(out); free(out2);
#endif
        free(a.q); free(b.q); free(a.sc); free(b.sc); free(a.sum); free(b.sum);
    }
    free(x);
    return !bad;
}

int main(void) {
#ifdef __ARM_FEATURE_SVE
    {
        double worst = 0;
        for (int i = 0; i < 200000; i++) {
            float v = -87.f + 175.f * (float)i / 200000.f;
            double e = fabs((double)q38d_expf(v) - exp((double)v)) / exp((double)v);
            if (e > worst) worst = e;
        }
        printf("exp max rel err %.3e\n", worst);
        CHECK(worst < 3e-7, "exp accuracy\n");
    }
#endif
    const int cols = 1024, rows = 24, groups = rows / 8;
    /* ---- F4 ---- */
    {
        size_t rb = (size_t)cols / 64 * 36;
        uint8_t *raw = malloc(rows * rb);
        for (size_t i = 0; i < rows * rb; i++) raw[i] = (uint8_t)rnd();
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < cols / 64; b++)
                for (int s = 0; s < 4; s++) {
                    uint8_t *sc = raw + r * rb + b * 36 + s;
                    *sc = (uint8_t)(rnd() % 128); /* includes 0 and 0x7f sentinel */
                    if (rnd() % 5 == 0) *sc = (uint8_t)(rnd() % 8); /* subnormal */
                }
        size_t tb = q38_lowbit_bytes(Q38_LB_NVFP4, rows, cols);
        uint8_t *tiles = malloc(tb);
        q38_lowbit_pack_nvfp4(tiles, tb, raw, rb, rows, cols);
        size_t gb = q38d_group_bytes(Q38D_F4, cols);
        uint8_t *g = aligned_alloc(256, gb * groups);
        for (int gi = 0; gi < groups; gi++)
            q38d_repack_f4(g + gi * gb, tiles + gi * (tb / groups), cols);
        float *w0 = malloc(cols * 4);
        int exact = 1;
        for (int r = 0; r < rows; r++) {
            q38_lowbit_dequant_row(w0, tiles, Q38_LB_NVFP4, rows, cols, r);
            for (int k = 0; k < cols; k++)
                if (w0[k] != q38d_weight(g + (r / 8) * gb, Q38D_F4, cols, r % 8, k)) exact = 0;
        }
        CHECK(exact, "F4 repack mismatch\n");
        printf("F4 repack exact=%d\n", exact);
        test_matrix(Q38D_F4, g, cols, groups);
        free(raw); free(tiles); free(g); free(w0);
    }
    /* ---- F6 ---- */
    {
        float *src = malloc((size_t)rows * cols * 4);
        for (int i = 0; i < rows * cols; i++) src[i] = frand() * ldexpf(1.f, -(int)(rnd() % 9));
        size_t tb = q38_lowbit_bytes(Q38_LB_FP6_E2M3, rows, cols);
        uint8_t *tiles = malloc(tb);
        q38_lowbit_pack_fp6(tiles, tb, src, cols, rows, cols);
        size_t gb = q38d_group_bytes(Q38D_F6, cols);
        uint8_t *g = aligned_alloc(256, gb * groups);
        for (int gi = 0; gi < groups; gi++)
            q38d_repack_f6(g + gi * gb, tiles + gi * (tb / groups), cols);
        float *w0 = malloc(cols * 4);
        int exact = 1;
        for (int r = 0; r < rows; r++) {
            q38_lowbit_dequant_row(w0, tiles, Q38_LB_FP6_E2M3, rows, cols, r);
            for (int k = 0; k < cols; k++)
                if (w0[k] != q38d_weight(g + (r / 8) * gb, Q38D_F6, cols, r % 8, k)) exact = 0;
        }
        CHECK(exact, "F6 repack mismatch\n");
        printf("F6 repack exact=%d\n", exact);
        test_matrix(Q38D_F6, g, cols, groups);
        free(src); free(tiles); free(g); free(w0);
    }
    /* ---- Q6_K and Q4_K ---- */
    for (int type = 0; type < 3; type++) {
        int gt = type == 1 ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K, fmt = type == 1 ? Q38D_Q4K : type == 2 ? Q38D_Q8K : Q38D_Q6K;
        size_t rb = (size_t)cols / 256 * (type == 1 ? 144 : 210);
        uint8_t *raw = malloc(rows * rb);
        for (size_t i = 0; i < rows * rb; i++) raw[i] = (uint8_t)rnd();
        for (int r = 0; r < rows; r++)
            for (int sb = 0; sb < cols / 256; sb++) {
                uint8_t *blk = raw + r * rb + sb * (type == 1 ? 144 : 210);
                uint16_t h1 = (uint16_t)(0x1000 + rnd() % 0x1800), h2 = (uint16_t)(0x1000 + rnd() % 0x1800);
                if (type == 1) { memcpy(blk, &h1, 2); memcpy(blk + 2, &h2, 2); }
                else memcpy(blk + 208, &h1, 2);
            }
        size_t gb = q38d_group_bytes(fmt, cols);
        uint8_t *g = aligned_alloc(256, gb * groups);
        for (int gi = 0; gi < groups; gi++)
            (type == 1 ? q38d_repack_q4k : type == 2 ? q38d_repack_q8k : q38d_repack_q6k)(g + gi * gb, raw + gi * 8 * rb, rb, 8, cols);
        float *w0 = malloc(cols * 4);
        int exact = 1;
        double worst = 0;
        for (int r = 0; r < rows; r++) {
            dequant_row(gt, raw + r * rb, w0, cols);
            for (int k = 0; k < cols; k++) {
                float w = q38d_weight(g + (r / 8) * gb, fmt, cols, r % 8, k);
                if (w0[k] != w) {
                    exact = 0;
                    double e = fabs((double)w0[k] - w) / (fabs(w0[k]) + 1e-3);
                    if (e > worst) worst = e;
                }
            }
        }
        /* Q4_K: d*q - m may round differently from a fused multiply-add. */
        printf("%s repack exact=%d worst_rel=%.3e\n", type == 1 ? "Q4K" : type == 2 ? "Q8K" : "Q6K", exact, worst);
        CHECK(worst < 1e-6, "repack mismatch type %d\n", type);
        test_matrix(fmt, g, cols, groups);
        free(raw); free(g); free(w0);
    }
    printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures != 0;
}
