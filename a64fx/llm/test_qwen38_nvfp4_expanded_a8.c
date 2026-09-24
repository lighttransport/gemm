/* Compare execution-layout repacks against independent FP4 source decoding. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float d[8]; uint8_t qs[64]; } source_subblock;
typedef struct { source_subblock s[4]; } source_block;
typedef struct { int8_t lo[64], hi[64]; } packed_act;
extern void q38_nvfp4_packed_a8_prepare(packed_act *, const int8_t *, int);
extern int q38_nvfp4_packed8_iscale_repack(void *, const void *, int);
extern int q38_nvfp4_packed8_iscale_a8_rows(float *, const void *,
                                         const packed_act *, float, int, int);
extern int q38_nvfp4_packed5_repack(void *, const void *, int);
extern int q38_nvfp4_packed6_repack(void *, const void *, int);
extern int q38_nvfp4_packed5_a8_rows(float *, const void *,
                                   const packed_act *, float, int, int);
extern int q38_nvfp4_packed6_a8_rows(float *, const void *,
                                   const packed_act *, float, int, int);
static uint32_t rng = 0x52ef41d9;
static uint32_t next_random(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}

static int check(int rows, int cols) {
    const int blocks = rows / 8 * (cols / 64), nb = cols / 64;
    source_block *src = malloc((size_t)blocks * sizeof(*src));
    void *w4 = malloc((size_t)blocks * 352);
    void *w5 = malloc((size_t)blocks * 416);
    void *w6 = malloc((size_t)blocks * 480);
    int8_t *digits = malloc((size_t)cols);
    packed_act *act = malloc((size_t)(cols / 16) * sizeof(*act));
    float *y4 = malloc((size_t)rows * sizeof(float));
    float *y5 = malloc((size_t)rows * sizeof(float));
    float *y6 = malloc((size_t)rows * sizeof(float));
    if (!src || !w4 || !w5 || !w6 || !digits || !act || !y4 || !y5 || !y6)
        return 0;
    for (int b = 0; b < blocks; b++) {
        for (int s = 0; s < 4; s++) {
            for (int r = 0; r < 8; r++)
                src[b].s[s].d[r] = b % 17 == 0 ? 0.0f :
                    (s == 0 ? 1.0f : (float)(next_random() % 8)) * 0x1p-10f;
            for (int j = 0; j < 64; j++) src[b].s[s].qs[j] = (uint8_t)next_random();
        }
    }
    for (int k = 0; k < cols; k++) digits[k] = (int8_t)(next_random() % 256 - 128);
    q38_nvfp4_packed_a8_prepare(act, digits, cols);
    int ok = q38_nvfp4_packed8_iscale_repack(w4, src, blocks) &&
             q38_nvfp4_packed5_repack(w5, src, blocks) &&
             q38_nvfp4_packed6_repack(w6, src, blocks) &&
             q38_nvfp4_packed8_iscale_a8_rows(y4, w4, act, 0x1p-7f, rows, cols) &&
             q38_nvfp4_packed5_a8_rows(y5, w5, act, 0x1p-7f, rows, cols) &&
             q38_nvfp4_packed6_a8_rows(y6, w6, act, 0x1p-7f, rows, cols);
    static const int8_t code[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    for (int row = 0; row < rows && ok; row++) {
        double expected = 0.0;
        for (int ib = 0; ib < nb; ib++)
            for (int s = 0; s < 4; s++) {
                const source_subblock *p = &src[(row / 8) * nb + ib].s[s];
                int dot = 0;
                for (int j = 0; j < 8; j++) {
                    uint8_t z = p->qs[(row % 8) * 8 + j];
                    dot += code[z & 15] * digits[ib * 64 + s * 16 + j];
                    dot += code[z >> 4] * digits[ib * 64 + s * 16 + j + 8];
                }
                expected += dot * (double)p->d[row % 8] * 0x1p-7;
            }
        if (!isfinite(y5[row]) || !isfinite(y6[row]) ||
            fabs(y5[row] - expected) > 1e-5 * (1.0 + fabs(expected)) ||
            fabs(y6[row] - expected) > 1e-5 * (1.0 + fabs(expected)) ||
            memcmp(y4 + row, y5 + row, sizeof(float)) ||
            memcmp(y4 + row, y6 + row, sizeof(float))) {
            fprintf(stderr, "row=%d expected=%g compact=%g five=%g six=%g\n",
                    row, expected, y4[row], y5[row], y6[row]);
            ok = 0;
        }
    }
    src[0].s[0].d[0] = -1.0f;
    if (q38_nvfp4_packed5_repack(w5, src, 1) ||
        q38_nvfp4_packed6_repack(w6, src, 1)) ok = 0;
    printf("expanded rows=%d cols=%d correct=%d\n", rows, cols, ok);
    free(src); free(w4); free(w5); free(w6); free(digits); free(act);
    free(y4); free(y5); free(y6);
    return ok;
}

int main(void) {
    return check(8, 64) && check(24, 192) && check(64, 5120) ? 0 : 1;
}
