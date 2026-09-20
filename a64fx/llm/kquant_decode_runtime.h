/* Runtime bridge from qtensor to the validated Q5R/IQ4R sidecar layouts. */
#ifndef A64FX_KQUANT_DECODE_RUNTIME_H
#define A64FX_KQUANT_DECODE_RUNTIME_H

#include "qwen38_kquant_stage.h"
#include "kquant_decode_cache.h"

static inline int tf_kquant_cache_rows(float *dst, const qtensor *mat,
                                       const float *x,
                                       int row_start, int row_end) {
    if (!dst || !mat || !x || !mat->kquant_cache ||
        row_start < 0 || row_end < row_start || row_end > mat->n_rows ||
        mat->n_rows % TF_KQUANT_CACHE_ROWS ||
        mat->n_cols % TF_KQUANT_CACHE_COLS) return 0;
    if ((mat->type == GGML_TYPE_Q5_K &&
         mat->kquant_cache_format != Q38KC_FORMAT_Q5R) ||
        (mat->type == GGML_TYPE_IQ4_XS &&
         mat->kquant_cache_format != Q38KC_FORMAT_IQ4R) ||
        (mat->type != GGML_TYPE_Q5_K && mat->type != GGML_TYPE_IQ4_XS))
        return 0;

    int nb = mat->n_cols / TF_KQUANT_CACHE_COLS;
    size_t block_bytes = mat->kquant_cache_format == Q38KC_FORMAT_Q5R ?
        packed_q5r_block_bytes() : packed_iq4r_block_bytes();
    size_t group_bytes = (size_t)nb * block_bytes;
    kquant_cache_a8_block *qx =
        (kquant_cache_a8_block *)alloca((size_t)nb * sizeof(*qx));
    kquant_cache_quant_a8(qx, x, mat->n_cols);

    int group0 = row_start / TF_KQUANT_CACHE_ROWS;
    int group1 = (row_end + TF_KQUANT_CACHE_ROWS - 1) /
                 TF_KQUANT_CACHE_ROWS;
    for (int group = group0; group < group1; group++) {
        float out[TF_KQUANT_CACHE_ROWS];
        const uint8_t *weights = (const uint8_t *)mat->kquant_cache +
                                 (size_t)group * group_bytes;
        if (mat->kquant_cache_format == Q38KC_FORMAT_Q5R)
            packed_q5r_dot8(out, weights, qx, nb);
        else
            packed_iq4r_dot8(out, weights, qx, nb);
        int r0 = group * TF_KQUANT_CACHE_ROWS;
        int r1 = r0 + TF_KQUANT_CACHE_ROWS;
        if (r0 < row_start) r0 = row_start;
        if (r1 > row_end) r1 = row_end;
        for (int row = r0; row < r1; row++)
            dst[row] = out[row - group * TF_KQUANT_CACHE_ROWS];
    }
    return 1;
}

#endif
