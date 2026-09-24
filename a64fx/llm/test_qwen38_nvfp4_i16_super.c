/* Exercise full-K INT32 overflow and rare UE4M3 corrections. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int q38_nvfp4_i16_super_register(const void *, int, int);
extern int q38_nvfp4_i16_super_mt(float *, const void *, const float *,
                                  int, int, int);
extern void q38_nvfp4_i16_super_release(void);

enum { ROWS = 64, COLS = 17408, N = 3 };
typedef struct { uint8_t d[8], qs[64]; } subblock;
typedef struct { subblock s[4]; } block;

int main(void) {
    size_t blocks = (size_t)ROWS / 8 * (COLS / 64);
    block *w = calloc(blocks, sizeof(*w));
    float *x = malloc((size_t)N * COLS * sizeof(*x));
    float *y = malloc((size_t)N * ROWS * sizeof(*y));
    if (!w || !x || !y) return 1;
    for (size_t b = 0; b < blocks; b++)
        for (int s = 0; s < 4; s++) {
            subblock *p = &w[b].s[s];
            memset(p->qs, 0x77, sizeof(p->qs));
            memset(p->d, b % (COLS / 64) == 0 && s == 0 ? 11 : 10,
                   sizeof(p->d));
        }
    for (int k = 0; k < COLS; k++) {
        x[k] = 1.0f;
        x[COLS + k] = -1.0f;
        x[2 * COLS + k] = k & 1 ? -1.0f : 1.0f;
    }
    if (q38_nvfp4_i16_super_register(w, ROWS, COLS) ||
        !q38_nvfp4_i16_super_mt(y, w, x, ROWS, COLS, 1)) return 1;
    float expected = ((COLS - 16) * 120.0f + 16 * 132.0f) / 1024.0f;
    for (int row = 0; row < ROWS; row++) {
        if (fabsf(y[row] - expected) > 0.01f ||
            fabsf(y[ROWS + row] + expected) > 0.01f ||
            fabsf(y[2 * ROWS + row]) > 0.01f) {
            fprintf(stderr, "FAIL row=%d expected=%g got=%g,%g,%g\n",
                    row, expected, y[row], y[ROWS + row], y[2 * ROWS + row]);
            return 1;
        }
    }
    q38_nvfp4_i16_super_release();
    free(w); free(x); free(y);
    puts("PASS i16_super overflow rare_scale rows=64 cols=17408");
    return 0;
}
