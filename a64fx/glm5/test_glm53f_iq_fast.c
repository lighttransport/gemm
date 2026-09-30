/* Compares the full-width Q4_K/Q5_K decode row kernels against the reference loops and times both.
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I. -Ikern test_glm53f_iq_fast.c -lm
 * (includes glm53f_iq_bridge.c so the static reference rows are visible) */
#include "glm53f_iq_bridge.c"
#include "glm53f_iq_fast.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    pthread_once(&glm5_iq_lut_once, glm5_iq_init_luts);
    const int H = 4096, rows = 2048;
    iqf_init();
    srand(7);
    for (int type = 0; type < 2; ++type) {
        const int gt = type ? GLM53F_GGML_Q5_K : GLM53F_GGML_Q4_K;
        const size_t rb = glm53f_iq_row_size(gt, H);
        uint8_t *w = aligned_alloc(256, (size_t)rows * rb);
        for (size_t i = 0; i < (size_t)rows * rb; ++i) w[i] = (uint8_t)rand();
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < H / 256; ++b) {
                uint8_t *blk = w + (size_t)r * rb + (size_t)b * (type ? 176 : 144);
                _Float16 d = (_Float16)(0.001f * (1 + rand() % 50)), dm = (_Float16)(0.001f * (1 + rand() % 50));
                memcpy(blk, &d, 2); memcpy(blk + 2, &dm, 2);
            }
        float *x = malloc(H * 4);
        for (int i = 0; i < H; ++i) x[i] = (float)(rand() % 2001 - 1000) / 500.0f;
        glm5_iq_q8_block xq[16];
        glm5_iq_quant_q8(xq, x, H);
        iqf_act act[16] __attribute__((aligned(64)));
        iqf_prepare(act, (const iqf_src_block *)xq, 16);
        float *ref = malloc(rows * 4), *got = malloc(rows * 4);
        for (int r = 0; r < rows; ++r) ref[r] = iq_row(gt, w + (size_t)r * rb, xq, H / 256);
        iqf_rows(got, w, rb, rows, act, H / 256, type);
        double se = 0, sr = 0, mx = 0;
        for (int r = 0; r < rows; ++r) { double d = got[r] - ref[r]; se += d * d; sr += (double)ref[r] * ref[r]; if (fabs(d) > mx) mx = fabs(d); }
        printf("%s: rel_l2=%.3e max_abs=%.3e (rms ref %.3f)\n", type ? "Q5_K" : "Q4_K", sqrt(se / sr), mx, sqrt(sr / rows));
        /* single-thread timing, rows resident in HBM/L2 mix (2048 rows = 4.7 MB) */
        for (int rep = 0; rep < 3; ++rep) {
            double t0 = now(); for (int it = 0; it < 20; ++it) for (int r = 0; r < rows; ++r) ref[r] = iq_row(gt, w + (size_t)r * rb, xq, H / 256);
            double t1 = now(); for (int it = 0; it < 20; ++it) iqf_rows(got, w, rb, rows, act, H / 256, type);
            double t2 = now();
            printf("  old %.1f cyc/row(2GHz)  new %.1f cyc/row  speedup %.2fx\n", (t1 - t0) / (20.0 * rows) * 2e9, (t2 - t1) / (20.0 * rows) * 2e9, (t1 - t0) / (t2 - t1));
        }
        free(w); free(x); free(ref); free(got);
    }
    return 0;
}
