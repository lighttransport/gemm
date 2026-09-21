/* Optional row-scaled INT8 safetensors -> BF16 upload representation.
 * CPU dequantization keeps the original native BF16 compute path unchanged. */
#ifndef QIMG21_QUANT_WEIGHTS_H
#define QIMG21_QUANT_WEIGHTS_H
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fenv.h>

static float q21_quant_source_value(const void *data, size_t index, int bf16) {
    float result;
    if (bf16) {
        uint16_t word;
        memcpy(&word, (const uint8_t *)data + index * 2, 2);
        uint32_t bits = (uint32_t)word << 16;
        memcpy(&result, &bits, 4);
    } else memcpy(&result, (const uint8_t *)data + index * 4, 4);
    return result;
}

/* Same quantization as quantize_weights.py, without a second model on disk.
 * Reconstruct one matrix at a time; vectors are not quantized. */
static uint16_t *q21_quantize_matrix_on_load(st_context *st, int index) {
    if (index < 0 || safetensors_ndims(st, index) != 2 || fegetround() != FE_TONEAREST) return NULL;
    const char *dtype = safetensors_dtype(st, index);
    int bf16 = !strcmp(dtype, "BF16");
    if (!bf16 && strcmp(dtype, "F32")) return NULL;
    size_t rows = safetensors_shape(st, index)[0], cols = safetensors_shape(st, index)[1];
    if (!rows || !cols || rows > SIZE_MAX / cols / 4 ||
        safetensors_nbytes(st, index) != rows * cols * (bf16 ? 2 : 4)) return NULL;
    const void *data = safetensors_data(st, index);
    uint16_t *out = malloc(rows * cols * 2);
    if (!out) return NULL;
    for (size_t r = 0; r < rows; r++) {
        float maximum = 0;
        for (size_t c = 0; c < cols; c++) {
            float value = q21_quant_source_value(data, r * cols + c, bf16);
            if (!isfinite(value)) goto invalid;
            maximum = fmaxf(maximum, fabsf(value));
        }
        float scale = maximum > 0 ? maximum / 127.0f : 1.0f;
        if (!(scale > 0) || !isfinite(scale)) goto invalid;
        for (size_t c = 0; c < cols; c++) {
            float value = q21_quant_source_value(data, r * cols + c, bf16);
            int quantized = (int)nearbyintf(value / scale);
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;
            float reconstructed = (float)quantized * scale;
            uint32_t bits;
            memcpy(&bits, &reconstructed, 4);
            out[r * cols + c] = (uint16_t)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
            if ((out[r * cols + c] & 0x7f80U) == 0x7f80U) goto invalid;
        }
    }
    return out;
invalid:
    free(out);
    return NULL;
}

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
