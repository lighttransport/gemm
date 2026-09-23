#define _GNU_SOURCE
#include <stdlib.h>
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include <arm_sve.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float old_dot(const float *w, const float *x, int k) {
    svbool_t pg = svptrue_b32();
    svfloat32_t a = svdup_f32(0);
    int vl = (int)svcntw();
    for (int i = 0; i < k; i += vl)
        a = svmla_x(pg, a, svld1(pg, w + i), svld1(pg, x + i));
    return svaddv_f32(pg, a);
}

static void block_dot3(float y[3], uint32_t type, const uint8_t *src,
                       const float *x, int k) {
    float wb[256] __attribute__((aligned(64)));
    svbool_t pg = svptrue_b32();
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0;
    size_t bs = type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q6_K);
    int vl = (int)svcntw();
    for (int ib = 0; ib < k / 256; ib++) {
        if (type == GGML_TYPE_Q4_K)
            dequantize_row_q4_K(src + (size_t)ib * bs, wb, 256);
        else
            dequantize_row_q6_K(src + (size_t)ib * bs, wb, 256);
        int base = ib * 256;
        for (int j = 0; j < 256; j += vl) {
            svfloat32_t v = svld1_f32(pg, wb + j);
            a0 = svmla_f32_m(pg, a0, v, svld1_f32(pg, x + base + j));
            a1 = svmla_f32_m(pg, a1, v, svld1_f32(pg, x + k + base + j));
            a2 = svmla_f32_m(pg, a2, v, svld1_f32(pg, x + 2 * k + base + j));
        }
    }
    y[0] = svaddv_f32(pg, a0);
    y[1] = svaddv_f32(pg, a1);
    y[2] = svaddv_f32(pg, a2);
}

int main(void) {
    enum { K = 5120, ROWS = 48 };
    float *x = malloc(3 * K * sizeof(float));
    float *row = malloc(K * sizeof(float));
    if (!x || !row) return 2;
    for (int i = 0; i < 3 * K; i++) x[i] = ((i * 29) % 251 - 125) * (1.0f / 128);
    int bad = 0;
    for (int type = GGML_TYPE_Q4_K; type <= GGML_TYPE_Q6_K; type += 2) {
        size_t bs = type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q6_K);
        uint8_t *weights = malloc((size_t)ROWS * (K / 256) * bs);
        if (!weights) return 2;
        for (int r = 0; r < ROWS; r++)
            for (int ib = 0; ib < K / 256; ib++) {
                uint8_t *b = weights + ((size_t)r * (K / 256) + ib) * bs;
                for (size_t j = 0; j < bs; j++) b[j] = (uint8_t)(r * 37 + ib * 23 + j * 17);
                if (type == GGML_TYPE_Q4_K) {
                    ((block_q4_K *)b)->d = 0x3800;
                    ((block_q4_K *)b)->dmin = 0x3000;
                } else ((block_q6_K *)b)->d = 0x3800;
            }
        for (int r = 0; r < ROWS; r++) {
            const uint8_t *w = weights + (size_t)r * (K / 256) * bs;
            dequant_row(type, w, row, K);
            float ref[3], got[3];
            for (int t = 0; t < 3; t++) ref[t] = old_dot(row, x + (size_t)t * K, K);
            block_dot3(got, type, w, x, K);
            for (int t = 0; t < 3; t++)
                if (memcmp(&ref[t], &got[t], sizeof(float))) {
                    if (bad < 6) printf("mismatch type=%d row=%d tok=%d ref=%a got=%a\n",
                                        type, r, t, ref[t], got[t]);
                    bad++;
                }
        }
        printf("type=%d rows=%d outputs=%d bad=%d\n", type, ROWS, ROWS * 3, bad);
        free(weights);
    }
    free(x); free(row);
    return bad ? 1 : 0;
}
