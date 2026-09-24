#ifndef QWEN38_LOWBIT_MODEL_H
#define QWEN38_LOWBIT_MODEL_H
#include "qwen38_lowbit.h"
#include "gguf_loader.h"

typedef struct q38_lowbit_model q38_lowbit_model;
typedef struct {
    void *part[4];
    int first[5];
    int rows, cols, format, arithmetic;
    q38_lowbit_model *owner;
} q38_lowbit_matrix;

/* Bounded pread conversion into final anonymous allocations. With numa=1,
 * first touch is pinned to CPUs 12,24,36,48 and physical placement is checked.
 * The metadata GGUF must stay open until after model_free. No GGML codes change.
 * Only one dispatch at a time may use a model (as with transformer_model). */
q38_lowbit_model *q38_lowbit_model_load(gguf_context *g, int format,
    int arithmetic, int numa, size_t reserve_bytes);
/* Load an existing image; reject stale, truncated, or corrupt images. The
 * source GGUF supplies metadata and must have the same file sizes/mtimes and
 * tensor inventory used to create the image. No silent conversion fallback. */
q38_lowbit_model *q38_lowbit_model_load_image(gguf_context *g, int format,
    int arithmetic, int numa, size_t reserve_bytes, const char *path);
/* Atomic publication through a sibling temporary file. Bounded writeback
 * prevents a second model-sized dirty page cache from accumulating. */
int q38_lowbit_model_save_image(const q38_lowbit_model *model, const char *path);
void q38_lowbit_model_free(q38_lowbit_model *model);
const q38_lowbit_matrix *q38_lowbit_model_tensor(const gguf_context *g, int index);
int q38_lowbit_matrix_row(float *dst, const q38_lowbit_matrix *m, int row);
int q38_lowbit_matrix_rows(float *dst, const q38_lowbit_matrix *m,
    const float *x, int first, int last);
int q38_lowbit_matrix_begin(const q38_lowbit_matrix *m, const float *x);
void q38_lowbit_matrix_end(const q38_lowbit_matrix *m);
#endif
