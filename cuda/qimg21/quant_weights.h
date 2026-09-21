/* Optional row-scaled INT8 safetensors -> BF16 upload representation.
 * CPU dequantization keeps the original native BF16 compute path unchanged. */
#ifndef QIMG21_QUANT_WEIGHTS_H
#define QIMG21_QUANT_WEIGHTS_H
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint16_t *q21_read_int8_matrix(const char *path, size_t rows, size_t cols) {
    if (!rows || !cols || rows > SIZE_MAX / cols / 2 || rows > SIZE_MAX / 4) return NULL;
    st_context *st = safetensors_open(path);
    if (!st) return NULL;
    uint16_t *out = NULL;
    int wi = safetensors_find(st, "weight"), si = safetensors_find(st, "scale");
    if (wi < 0 || si < 0 || strcmp(safetensors_dtype(st, wi), "I8") ||
        strcmp(safetensors_dtype(st, si), "F32") || safetensors_ndims(st, wi) != 2 ||
        safetensors_ndims(st, si) != 1 || safetensors_shape(st, wi)[0] != rows ||
        safetensors_shape(st, wi)[1] != cols || safetensors_shape(st, si)[0] != rows ||
        safetensors_nbytes(st, wi) != rows * cols || safetensors_nbytes(st, si) != rows * 4)
        goto done;
    const int8_t *weight = safetensors_data(st, wi);
    const float *scale = safetensors_data(st, si);
    out = malloc(rows * cols * 2);
    if (!out) goto done;
    for (size_t r = 0; r < rows; r++) {
        if (!(scale[r] > 0) || !isfinite(scale[r])) goto invalid;
        for (size_t c = 0; c < cols; c++) {
            float value = (float)weight[r * cols + c] * scale[r];
            uint32_t bits;
            if (!isfinite(value) || weight[r * cols + c] == -128) goto invalid;
            memcpy(&bits, &value, sizeof(bits));
            out[r * cols + c] = (uint16_t)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
            if ((out[r * cols + c] & 0x7f80U) == 0x7f80U) goto invalid;
        }
    }
    goto done;
invalid:
    free(out); out = NULL;
done:
    safetensors_close(st);
    return out;
}
#endif
