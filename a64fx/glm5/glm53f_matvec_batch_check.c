#include <arm_sve.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "glm53f_expert_kern.h"

enum { ROWS = 4096, COLS = 4096, TOKENS = 4, REPEAT = 8 };

static void *aligned_buffer(size_t bytes) {
    void *p = NULL;
    return posix_memalign(&p, 256, bytes) ? NULL : p;
}

int main(void) {
    uint8_t *w = aligned_buffer((size_t)ROWS * COLS);
    float *scale = aligned_buffer((size_t)(ROWS / 128) * (COLS / 128) * sizeof(float));
    float *x = aligned_buffer((size_t)TOKENS * COLS * sizeof(float));
    float *reference = aligned_buffer((size_t)TOKENS * ROWS * sizeof(float));
    float *batch = aligned_buffer((size_t)TOKENS * ROWS * sizeof(float));
    if (!w || !scale || !x || !reference || !batch) return 2;
    for (size_t i = 0; i < (size_t)ROWS * COLS; ++i)
        w[i] = (uint8_t)((i * 13 + 17) % 120);
    for (size_t i = 0; i < (size_t)(ROWS / 128) * (COLS / 128); ++i)
        scale[i] = 0.0005f * (float)(1 + i % 7);
    for (int t = 0; t < TOKENS; ++t)
        for (int i = 0; i < COLS; ++i)
            x[(size_t)t * COLS + i] = (float)(((i * 17 + t * 29) % 251) - 125) / 125.0f;

    for (int t = 0; t < TOKENS; ++t)
        glm53f_mv_fp8_block128_bits(reference + (size_t)t * ROWS,
            w, scale, x + (size_t)t * COLS, ROWS, COLS);
    glm53f_mv_fp8_block128_bits_batch(batch, w, scale, x, TOKENS, ROWS, COLS);

    double t0 = omp_get_wtime();
    for (int z = 0; z < REPEAT; ++z)
        for (int t = 0; t < TOKENS; ++t)
            glm53f_mv_fp8_block128_bits(reference + (size_t)t * ROWS,
                w, scale, x + (size_t)t * COLS, ROWS, COLS);
    double seq = omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    for (int z = 0; z < REPEAT; ++z)
        glm53f_mv_fp8_block128_bits_batch(batch, w, scale, x, TOKENS, ROWS, COLS);
    double batched = omp_get_wtime() - t0;

    double diff2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < (size_t)TOKENS * ROWS; ++i) {
        double d = (double)reference[i] - batch[i];
        diff2 += d * d;
        ref2 += (double)reference[i] * reference[i];
    }
    double rel_l2 = sqrt(diff2 / (ref2 + 1e-30));
    int ok = rel_l2 < 2e-6;
    printf("GLM53F_MATVEC_BATCH rows=%d cols=%d tokens=%d rel_l2=%.9g "
           "seq_ms=%.3f batch_ms=%.3f speedup=%.3f %s\n",
           ROWS, COLS, TOKENS, rel_l2, seq * 1e3 / REPEAT,
           batched * 1e3 / REPEAT, seq / batched, ok ? "PASS" : "FAIL");
    free(batch); free(reference); free(x); free(scale); free(w);
    return ok ? 0 : 1;
}
