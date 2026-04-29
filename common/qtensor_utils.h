/*
 * qtensor_utils.h — shared qtensor struct + safetensors→qtensor loader.
 *
 * This consolidates the identical `qtensor` struct that used to be
 * re-declared (behind an ad-hoc guard ladder) by every single-header
 * model backbone, and the ~9 copies of `*_make_tensor()` that each of
 * those headers grew independently.
 *
 * Supports F32 / F16 / BF16 input tensors; BF16 is converted to F32
 * in a freshly malloc'd buffer (mmap'd data is read-only). The 4-slot
 * `dims[]` array holds the first up-to-4 dimensions; `n_rows` /
 * `n_cols` flatten the full shape for 2D GEMM-style access
 * (n_rows = shape[0], n_cols = prod(shape[1..nd])).
 *
 * Excluded:
 *   - common/transformer.h has an independent `qtensor` with an
 *     identical layout but an unrelated loading path (GGUF, not
 *     safetensors). It is intentionally left untouched.
 *   - common/qwen_image_dit.h has a divergent `qimg_st_make_tensor`
 *     with FP8 E4M3 support and a different signature; left untouched.
 */
#ifndef QTENSOR_UTILS_H
#define QTENSOR_UTILS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Coexist with common/transformer.h, which declares an identical
 * `qtensor` before this header is included in hybrid TUs. */
#ifndef TRANSFORMER_H
#ifndef QTENSOR_STRUCT_DEFINED
#define QTENSOR_STRUCT_DEFINED
typedef struct {
    void    *data;
    uint32_t type;       /* ggml_dtype */
    int      n_rows;
    int      n_cols;     /* product of all dims past the 0th */
    int      n_dims;     /* clamped to 4 for storage in dims[] */
    uint64_t dims[4];
} qtensor;
#endif
#endif

/* The loader needs the safetensors API. We only pull it in when the
 * caller has already included safetensors.h (matches the pre-existing
 * `#ifdef SAFETENSORS_H` gate in each consumer header). */
#ifdef SAFETENSORS_H

/* Build a qtensor from a safetensors tensor index.
 *
 * Returns a zeroed qtensor on negative idx or on unsupported dtype.
 * For BF16, allocates a freshly malloc'd F32 buffer; ownership of that
 * buffer is currently held by the consumer's dequant pipeline (which
 * keeps a parallel free-list). qt_make_tensor itself does not track it. */
static inline qtensor qt_make_tensor(st_context *st, int idx) {
    qtensor t = {0};
    if (idx < 0) return t;
    t.data = safetensors_data(st, idx);
    const char *dt = safetensors_dtype(st, idx);
    int is_bf16 = 0;
    if      (strcmp(dt, "F32")  == 0) t.type = GGML_TYPE_F32;
    else if (strcmp(dt, "F16")  == 0) t.type = GGML_TYPE_F16;
    else if (strcmp(dt, "BF16") == 0) { t.type = GGML_TYPE_F32; is_bf16 = 1; }
    else return (qtensor){0};

    int ndims_full = safetensors_ndims(st, idx);
    const uint64_t *shape = safetensors_shape(st, idx);
    /* qtensor.dims is a fixed 4-wide array. Clamp what we store there;
     * the full shape still drives n_rows/n_cols below. Conv weights
     * (e.g. out_C, 3, 3, 3, in_C) land as n_rows=out_C,
     * n_cols=27*in_C, which is what the GEMM path expects. */
    t.n_dims = ndims_full > 4 ? 4 : ndims_full;
    for (int d = 0; d < t.n_dims; d++) t.dims[d] = shape[d];
    if (ndims_full >= 2) {
        t.n_rows = (int)shape[0];
        t.n_cols = 1;
        for (int d = 1; d < ndims_full; d++) t.n_cols *= (int)shape[d];
    } else {
        t.n_cols = (int)shape[0];
        t.n_rows = 1;
    }

    if (is_bf16) {
        size_t numel = (size_t)t.n_cols * (size_t)t.n_rows;
        float *buf = (float *)malloc(numel * sizeof(float));
        const uint16_t *src = (const uint16_t *)t.data;
        for (size_t i = 0; i < numel; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&buf[i], &bits, 4);
        }
        t.data = buf;
    }
    return t;
}

/* Convenience wrappers for name→tensor lookup. qt_find returns -1 on
 * missing tensor; qt_find_opt treats absence as success and zeroes out. */
static inline int qt_find(st_context *st, const char *name, qtensor *out) {
    int idx = safetensors_find(st, name);
    if (idx < 0) { *out = (qtensor){0}; return -1; }
    *out = qt_make_tensor(st, idx);
    return 0;
}

static inline int qt_find_opt(st_context *st, const char *name, qtensor *out) {
    int idx = safetensors_find(st, name);
    if (idx < 0) { *out = (qtensor){0}; return 0; }
    *out = qt_make_tensor(st, idx);
    return 0;
}

#endif /* SAFETENSORS_H */

/* Dequant helpers — depend on ggml_dequant.h (for dequant_row +
 * ggml_fp16_to_fp32 + the GGML_TYPE_* enum). Consumer headers that
 * include ggml_dequant.h pick these up automatically. */
#ifdef GGML_DEQUANT_H

static inline int qt_numel(const qtensor *t) {
    if (!t->data) return 0;
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    return n;
}

static inline float *qt_dequant(const qtensor *t) {
    if (!t->data) return NULL;
    int n = qt_numel(t);
    if (n <= 0) return NULL;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else {
        int bs, ts;
        switch (t->type) {
            case GGML_TYPE_Q8_0: bs = 32;  ts = 34;  break;
            case GGML_TYPE_Q4_K: bs = 256; ts = 144; break;
            case GGML_TYPE_Q6_K: bs = 256; ts = 210; break;
            default: memset(buf, 0, (size_t)n * sizeof(float)); return buf;
        }
        size_t row_bytes = (size_t)((t->n_cols + bs - 1) / bs) * ts;
        for (int r = 0; r < t->n_rows; r++) {
            const void *row = (const uint8_t *)t->data + r * row_bytes;
            dequant_row(t->type, row, buf + r * t->n_cols, t->n_cols);
        }
    }
    return buf;
}

static inline void qt_dequant_row(const qtensor *t, int row, float *dst) {
    int n = t->n_cols;
    if (row < 0 || row >= t->n_rows) {
        memset(dst, 0, (size_t)n * sizeof(float));
        return;
    }
    if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data + (size_t)row * n;
        for (int i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(dst, (const float *)t->data + (size_t)row * n,
               (size_t)n * sizeof(float));
    } else {
        int bs, ts;
        switch (t->type) {
            case GGML_TYPE_Q8_0: bs = 32;  ts = 34;  break;
            case GGML_TYPE_Q4_K: bs = 256; ts = 144; break;
            case GGML_TYPE_Q6_K: bs = 256; ts = 210; break;
            default: memset(dst, 0, (size_t)n * sizeof(float)); return;
        }
        size_t row_bytes = (size_t)((n + bs - 1) / bs) * ts;
        dequant_row(t->type, (const uint8_t *)t->data + row * row_bytes, dst, n);
    }
}

#endif /* GGML_DEQUANT_H */

#endif /* QTENSOR_UTILS_H */
