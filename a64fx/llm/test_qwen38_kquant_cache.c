/* Synthetic correctness checks for the shared Q5R/IQ4R decode layouts. */
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kquant_decode_cache.h"

enum { TEST_ROWS = 16, TEST_COLS = 512 };

static uint32_t test_random(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_weights(block_q5_K *q5, block_iq4_xs *iq4) {
    uint32_t state = 0x53f38a64u;
    size_t blocks = (size_t)TEST_ROWS * (TEST_COLS / 256);
    for (size_t b = 0; b < blocks; b++) {
        q5[b].d = ggml_fp32_to_fp16(0.0008f + 0.0001f * (float)(b % 7));
        q5[b].dmin = ggml_fp32_to_fp16(0.0003f + 0.00005f * (float)(b % 5));
        for (size_t i = 0; i < sizeof(q5[b].scales); i++)
            q5[b].scales[i] = (uint8_t)test_random(&state);
        for (size_t i = 0; i < sizeof(q5[b].qh); i++)
            q5[b].qh[i] = (uint8_t)test_random(&state);
        for (size_t i = 0; i < sizeof(q5[b].qs); i++)
            q5[b].qs[i] = (uint8_t)test_random(&state);

        iq4[b].d = ggml_fp32_to_fp16(0.0007f + 0.0001f * (float)(b % 9));
        iq4[b].scales_h = (uint16_t)test_random(&state);
        for (size_t i = 0; i < sizeof(iq4[b].scales_l); i++)
            iq4[b].scales_l[i] = (uint8_t)test_random(&state);
        for (size_t i = 0; i < sizeof(iq4[b].qs); i++)
            iq4[b].qs[i] = (uint8_t)test_random(&state);
    }
}

static void fill_activation(float *x, int pattern) {
    uint32_t state = 0x1234abcdu;
    for (int i = 0; i < TEST_COLS; i++) {
        if (pattern == 0)
            x[i] = 0.75f * sinf((float)i * 0.031f) + 0.2f * cosf((float)i * 0.17f);
        else if (pattern == 1)
            x[i] = i % 31 == 0 ? sinf((float)i * 0.13f) : 0.0f;
        else if (pattern == 2)
            x[i] = (i & 1 ? -1.0f : 1.0f) * ldexpf(0.75f, i % 17 - 8);
        else
            x[i] = ((float)(test_random(&state) >> 8) *
                    (2.0f / 16777215.0f) - 1.0f);
    }
}

static int compare(const float *got, const float *want, float rel_limit,
                   int require_bit_exact, double *nrmse_out, double *max_out) {
    double err2 = 0.0, norm2 = 0.0, max_abs = 0.0;
    for (int r = 0; r < TEST_ROWS; r++) {
        double e = (double)got[r] - want[r];
        double a = fabs(e);
        err2 += e * e;
        norm2 += (double)want[r] * want[r];
        if (a > max_abs) max_abs = a;
    }
    double nrmse = sqrt(err2 / (norm2 + 1e-30));
    *nrmse_out = nrmse;
    *max_out = max_abs;
    if (require_bit_exact && memcmp(got, want, TEST_ROWS * sizeof(*got))) return -1;
    return nrmse <= rel_limit ? 0 : -1;
}

int main(void) {
    const int nb = TEST_COLS / 256;
    size_t compact_blocks = (size_t)TEST_ROWS * nb;
    block_q5_K *q5 = aligned_alloc(256, compact_blocks * sizeof(*q5));
    block_iq4_xs *iq4 = aligned_alloc(256, compact_blocks * sizeof(*iq4));
    size_t q5r_size = packed_q5r_bytes(TEST_ROWS, TEST_COLS);
    size_t iq4r_size = packed_iq4r_bytes(TEST_ROWS, TEST_COLS);
    uint8_t *q5r = aligned_alloc(256, (q5r_size + 255) & ~(size_t)255);
    uint8_t *iq4r = aligned_alloc(256, (iq4r_size + 255) & ~(size_t)255);
    float *x = aligned_alloc(256, TEST_COLS * sizeof(*x));
    float native_q5[TEST_ROWS], packed_q5[TEST_ROWS];
    float native_iq4[TEST_ROWS], packed_iq4[TEST_ROWS];
    if (!q5 || !iq4 || !q5r || !iq4r || !x) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    if (packed_q5r_bytes(7, TEST_COLS) || packed_iq4r_bytes(TEST_ROWS, 255) ||
        !q5r_size || !iq4r_size) {
        fprintf(stderr, "dimension validation failed\n");
        return 1;
    }
    fill_weights(q5, iq4);
    if (pack_q5r(q5r, q5, TEST_ROWS, TEST_COLS) ||
        pack_iq4r(iq4r, iq4, TEST_ROWS, TEST_COLS)) {
        fprintf(stderr, "pack failed\n");
        return 1;
    }
    for (int pattern = 0; pattern < 4; pattern++) {
        fill_activation(x, pattern);
        tf_kquant_a8_block qx[TEST_COLS / 256];
        tf_kquant_quant_a8(qx, x, TEST_COLS);
        for (int r = 0; r < TEST_ROWS; r++) {
            native_q5[r] = tf_q5_k_a8_dot_sve(q5 + (size_t)r * nb, qx, nb);
            native_iq4[r] = tf_iq4_xs_a8_dot_sve(iq4 + (size_t)r * nb, qx, nb);
        }
        if (run_packed_q5r(packed_q5, q5r, x, TEST_ROWS, TEST_COLS) ||
            run_packed_iq4r(packed_iq4, iq4r, x, TEST_ROWS, TEST_COLS)) {
            fprintf(stderr, "matvec failed\n");
            return 1;
        }
        double q5_nrmse, q5_max, iq4_nrmse, iq4_max;
        if (compare(packed_q5, native_q5, 2e-6f, 0, &q5_nrmse, &q5_max) ||
            compare(packed_iq4, native_iq4, 0.0f, 1, &iq4_nrmse, &iq4_max)) {
            fprintf(stderr, "pattern=%d mismatch q5_nrmse=%.3g q5_max=%.3g "
                    "iq4_nrmse=%.3g iq4_max=%.3g\n",
                    pattern, q5_nrmse, q5_max, iq4_nrmse, iq4_max);
            return 1;
        }
        printf("pattern=%d q5_nrmse=%.3g q5_max=%.3g "
               "iq4_bit_exact=1\n", pattern, q5_nrmse, q5_max);
    }
    printf("SENTINEL qwen38_kquant_cache=OK layout_version=%d q5r_bytes=%zu iq4r_bytes=%zu\n",
           TF_KQUANT_CACHE_LAYOUT_VERSION, q5r_size, iq4r_size);
    free(x);
    free(iq4r);
    free(q5r);
    free(iq4);
    free(q5);
    return 0;
}
