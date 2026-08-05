#define _GNU_SOURCE
#include "k3_quant.h"

#include <stdio.h>
#include <stdlib.h>

static uint32_t rng_state = 0x31415926u;
static uint32_t rnd_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static int check_type(const char *name, int type, int cols) {
    const int rows = 17;
    const size_t rb = k3_quant_row_bytes(type, cols);
    uint8_t *w = (uint8_t *)calloc((size_t)rows, rb);
    float *x = (float *)malloc((size_t)cols * sizeof(*x));
    float *ref = (float *)malloc((size_t)rows * sizeof(*ref));
    float *got = (float *)malloc((size_t)rows * sizeof(*got));
    if (!w || !x || !ref || !got) return 2;
    for (int i = 0; i < cols; ++i)
        x[i] = ((float)(rnd_u32() & 0xffffu) / 32768.0f - 1.0f) * 0.2f;
    for (int r = 0; r < rows; ++r) {
        uint8_t *row = w + (size_t)r * rb;
        for (size_t i = 0; i < rb; ++i) row[i] = (uint8_t)rnd_u32();
        /* Random bytes are valid structurally, but keep every FP16 scale
         * finite and positive so the numerical check is meaningful. */
        if (type == K3_Q_Q8_0) {
            for (int b = 0; b < cols / 32; ++b) {
                uint16_t d = 0x2e66;
                memcpy(row + (size_t)b * 34, &d, sizeof d);
            }
        } else {
            for (int b = 0; b < cols / 256; ++b) {
                uint16_t d = 0x2e66;
                memcpy(row + (size_t)b * (rb / (size_t)(cols / 256)),
                       &d, sizeof d);
            }
        }
        ref[r] = k3_quant_dot_row_ref(row, type, x, cols);
    }
    k3_quant_matrix m = {w, type, rows, cols, rb};
    int rc = k3_quant_matvec(got, &m, x, 48);
    double se = 0.0, sr = 0.0, mx = 0.0;
    for (int r = 0; r < rows; ++r) {
        double d = (double)got[r] - ref[r];
        se += d * d; sr += (double)ref[r] * ref[r];
        if (fabs(d) > mx) mx = fabs(d);
    }
    double rel = sqrt(se / (sr + 1e-30));
    printf("[%s] rows=%d cols=%d rel_l2=%.3e max_abs=%.3e %s\n",
           name, rows, cols, rel, mx, rc == 0 && rel < 1e-6 ? "OK" : "FAIL");
    int bad = rc || rel >= 1e-6;
    if (type >= K3_Q_IQ1_S && type <= K3_Q_IQ3_XXS) {
        for (int mode = K3_QUANT_SVE_A16; mode <= K3_QUANT_SVE_Q8; ++mode) {
            int mrc = k3_quant_matvec_mode(got, &m, x, 48, mode);
            double mse = 0.0, mse_ref = 0.0;
            for (int r = 0; r < rows; ++r) {
                double d = (double)got[r] - ref[r];
                mse += d * d; mse_ref += (double)ref[r] * ref[r];
            }
            double mrel = sqrt(mse / (mse_ref + 1e-30));
            printf("[%s %s] rel_l2=%.3e %s\n", name,
                   mode == K3_QUANT_SVE_A16 ? "a16" : "q8", mrel,
                   mrc == 0 && mrel < 0.08 ? "OK" : "FAIL");
            bad |= mrc || mrel >= 0.08;
        }
    }
    free(w); free(x); free(ref); free(got);
    return bad;
}

int main(void) {
    int bad = 0;
    bad |= check_type("Q8_0", K3_Q_Q8_0, 2048);
    bad |= check_type("IQ1_S", K3_Q_IQ1_S, 2048);
    bad |= check_type("IQ2_XS", K3_Q_IQ2_XS, 2048);
    bad |= check_type("IQ2_XXS", K3_Q_IQ2_XXS, 2048);
    bad |= check_type("IQ3_XXS", K3_Q_IQ3_XXS, 2048);
    puts(bad ? "K3 quantized kernel tests: FAIL" :
         "K3 quantized kernel tests: PASS");
    return bad ? 1 : 0;
}
