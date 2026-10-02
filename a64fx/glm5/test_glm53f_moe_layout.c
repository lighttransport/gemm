#define _POSIX_C_SOURCE 200809L
#include "glm53f_moe_grouped_native.h"
#include <stdlib.h>
#include <omp.h>
enum { LAYOUT_H = 4096, LAYOUT_INTER = 512 };
typedef struct { const uint8_t *gu, *down; int gu_type, down_q6, inter; } layout_expert;
#include <stdio.h>

static void *test_alloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 256, n ? n : 256)) exit(2);
    if (!p) exit(2);
    return p;
}
static void fill_weights(uint8_t *p, int rows, int K, int type) {
    int bytes = type == 2 ? GMN_Q6K_BYTES : (int)gmn_sblk_bytes(type);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < K / 256; ++b) {
            uint8_t *q = p + ((size_t)r * (K / 256) + b) * bytes;
            for (int i = 0; i < bytes; ++i) q[i] = (uint8_t)(r * 31 + b * 13 + i * 17 + i / 7);
            uint16_t d = (uint16_t)(0x1800 + (r * 17 + b * 31) % 768), dm = 0x1400;
            if (type == 2) memcpy(q + bytes - 2, &d, 2);
            else { memcpy(q, &d, 2); memcpy(q + 2, &dm, 2); }
        }
}
static void legacy(const layout_expert *ep, int count, const int *pairs,
        const int8_t *xq, const float *xs, const float *bt, float *out, int padding) {
    enum { LIMIT = 96 };
    int8_t *xp = test_alloc(LIMIT * LAYOUT_H), *xp2 = test_alloc(LIMIT * LAYOUT_INTER);
    int8_t *aq = test_alloc(LIMIT * LAYOUT_INTER);
    float *xsp = test_alloc(LIMIT * (LAYOUT_H / 32) * 4), *Bg = test_alloc(LIMIT * (LAYOUT_H / 32) * 4);
    float *as = test_alloc(LIMIT * (LAYOUT_INTER / 16) * 4), *asp = test_alloc(LIMIT * (LAYOUT_INTER / 16) * 4);
    float *yg = test_alloc(LIMIT * (2 * LAYOUT_INTER + 64) * 4), *yd = test_alloc((size_t)LIMIT * 4160 * 4);
    float *mina = test_alloc((LAYOUT_H / 32) * 64 * 4);
    uint8_t *cb = test_alloc((GMN_KC / 32) * GMN_BLK);
    static const int8_t zero[LAYOUT_H]; static const float zxs[LAYOUT_H / 32];
    int inter = ep->inter, ldg = 2 * inter + padding, sb = ep->down_q6 ? 16 : 32, nb = inter / sb;
    for (int off = 0; off < count; off += LIMIT) {
        int m = count - off < LIMIT ? count - off : LIMIT, mp = (m + 5) / 6 * 6;
        for (int g = 0; g < mp; g += 6) {
            const int8_t *rows[6]; const float *sc[6];
            for (int u = 0; u < 6; ++u) {
                int t = g + u < m ? pairs[2 * (off + g + u)] : -1;
                rows[u] = t >= 0 ? xq + (size_t)t * LAYOUT_H : zero;
                sc[u] = t >= 0 ? xs + (size_t)t * (LAYOUT_H / 32) : zxs;
            }
            gmn_pack6(xp + (size_t)g * LAYOUT_H, xsp + (size_t)g * (LAYOUT_H / 32), rows, sc, LAYOUT_H);
        }
        for (int p = 0; p < mp; ++p) {
            if (p < m) memcpy(Bg + (size_t)p * (LAYOUT_H / 32), bt + (size_t)pairs[2 * (off + p)] * (LAYOUT_H / 32), (LAYOUT_H / 32) * 4);
            else memset(Bg + (size_t)p * (LAYOUT_H / 32), 0, (LAYOUT_H / 32) * 4);
        }
        gmn_gemm(ep->gu_type, ep->gu, gmn_row_bytes(ep->gu_type, LAYOUT_H), LAYOUT_H,
                 2 * inter, mp, xp, xsp, Bg, yg, ldg, cb, mina, 1);
        if (ep->down_q6) gmn_swiglu_quant16(yg, ldg, inter, m, mp, aq, as);
        else gmn_swiglu_quant(yg, ldg, inter, m, mp, aq, as, Bg);
        for (int g = 0; g < mp; g += 6) {
            const int8_t *rows[6]; const float *sc[6];
            for (int u = 0; u < 6; ++u) { rows[u] = aq + (size_t)(g + u) * inter; sc[u] = as + (size_t)(g + u) * nb; }
            gmn_pack6_sb(xp2 + (size_t)g * inter, asp + (size_t)g * nb, rows, sc, inter, sb);
        }
        if (ep->down_q6) gmn_gemm_q6(ep->down, GMN_Q6K_BYTES, inter, LAYOUT_H, mp, xp2, asp, yd, 4160, cb);
        else gmn_gemm(GMN_TYPE_Q5K, ep->down, gmn_row_bytes(GMN_TYPE_Q5K, inter), inter,
                      LAYOUT_H, mp, xp2, asp, Bg, yd, 4160, cb, mina, 1);
        for (int p = 0; p < m; ++p) {
            int t = pairs[2 * (off + p)], k = pairs[2 * (off + p) + 1];
            memcpy(out + ((size_t)t * 4 + k) * LAYOUT_H, yd + (size_t)p * 4160, LAYOUT_H * 4);
        }
    }
    free(xp); free(xp2); free(aq); free(xsp); free(Bg); free(as); free(asp);
    free(yg); free(yd); free(mina); free(cb);
}
int main(void) {
    layout_expert expert[4]; uint8_t *gu[4], *down[4];
    for (int e = 0; e < 4; ++e) {
        int inter = e >= 2 ? 512 : 256, gt = e % 2 ? GMN_TYPE_Q5K : GMN_TYPE_Q4K, dq = e == 1;
        gu[e] = test_alloc((size_t)2 * inter * gmn_row_bytes(gt, LAYOUT_H));
        down[e] = test_alloc((size_t)LAYOUT_H * (dq ? GMN_Q6K_BYTES : gmn_row_bytes(GMN_TYPE_Q5K, inter)));
        fill_weights(gu[e], 2 * inter, LAYOUT_H, gt); fill_weights(down[e], LAYOUT_H, inter, dq ? 2 : GMN_TYPE_Q5K);
        expert[e] = (layout_expert){gu[e], down[e], gt, dq, inter};
    }
    const int sizes[] = {5, 94, 97, 149, 193, 385, 17};
    int failed = 0, cases = 0;
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        int n = sizes[si], count[4], base[5] = {0};
        float *xf = test_alloc((size_t)n * LAYOUT_H * 4), *xs = test_alloc((size_t)n * (LAYOUT_H / 32) * 4);
        float *bt = test_alloc((size_t)n * (LAYOUT_H / 32) * 4); int8_t *xq = test_alloc((size_t)n * LAYOUT_H);
        float *ref = test_alloc((size_t)n * 4 * LAYOUT_H * 4), *out = test_alloc((size_t)n * 4 * LAYOUT_H * 4);
        for (int t = 0; t < n; ++t) {
            for (int i = 0; i < LAYOUT_H; ++i) xf[(size_t)t * LAYOUT_H + i] = t == 0 ? 0.f : (float)((t * 13 + i * 19) % 1999 - 999) * .00071f;
            gmn_quant_row(xf + (size_t)t * LAYOUT_H, LAYOUT_H, xq + (size_t)t * LAYOUT_H,
                          xs + (size_t)t * (LAYOUT_H / 32), bt + (size_t)t * (LAYOUT_H / 32));
        }
        for (int e = 0; e < 4; ++e) { count[e] = n > e * 2 ? n - e * 2 : 0; base[e + 1] = base[e] + count[e]; }
        int *list = test_alloc((size_t)base[4] * 2 * sizeof(int));
        for (int e = 0; e < 4; ++e)
            for (int p = 0; p < count[e]; ++p) { list[2 * (base[e] + p)] = (p * 13 + e * 17) % n; list[2 * (base[e] + p) + 1] = e; }
        for (size_t i = 0; i < (size_t)n * 4 * LAYOUT_H; ++i) ref[i] = out[i] = 1234.5f;
        for (int e = 0; e < 4; ++e) legacy(expert + e, count[e], list + 2 * base[e], xq, xs, bt, ref, 0);
        for (int rep = 0; rep < 3; ++rep) {
            for (int e = 0; e < 4; ++e) legacy(expert + e, count[e], list + 2 * base[e], xq, xs, bt, out, 64);
            int rc = 0;
            int exact = !memcmp(ref, out, (size_t)n * 4 * LAYOUT_H * 4);
            failed |= rc != 0 || !exact; ++cases;
            printf("MOE_LAYOUT_UNIT threads=%d tokens=%d repetition=%d rc=%d routes_and_guards=%s %s\n",
                   omp_get_max_threads(), n, rep, rc, exact ? "BIT_EXACT" : "FAIL", rc || !exact ? "FAIL" : "PASS");
        }
        free(list); free(xf); free(xs); free(bt); free(xq); free(ref); free(out);
    }
    for (int e = 0; e < 4; ++e) { free(gu[e]); free(down[e]); }
    printf("GLM53F_MOE_LAYOUT threads=%d cases=%d Q4_Q5_gu_Q5_Q6_down=BIT_%s %s\n",
           omp_get_max_threads(), cases, failed ? "FAIL" : "EXACT", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
