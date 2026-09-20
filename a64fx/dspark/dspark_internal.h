#ifndef A64FX_DSPARK_INTERNAL_H
#define A64FX_DSPARK_INTERNAL_H

#include "dspark.h"

#include <stddef.h>
#include <stdint.h>

#define DS_HIDDEN 5120
#define DS_INTERMEDIATE 17408
#define DS_LAYERS 5
#define DS_HEADS 32
#define DS_KV_HEADS 8
#define DS_HEAD_DIM 128
#define DS_Q_DIM (DS_HEADS * DS_HEAD_DIM)
#define DS_KV_DIM (DS_KV_HEADS * DS_HEAD_DIM)
#define DS_VOCAB 248320
#define DS_MARKOV_RANK 256
#define DS_MAX_POSITION 262144
#define DS_MASK_TOKEN 248070
#define DS_RMS_EPS 1.0e-6f

typedef struct {
    const uint16_t *data;
    size_t rows;
    size_t cols;
} ds_bf16_matrix;

typedef struct {
    const uint16_t *input_norm;
    const uint16_t *post_norm;
    const uint16_t *q_norm;
    const uint16_t *k_norm;
    ds_bf16_matrix q_proj, k_proj, v_proj, o_proj;
    ds_bf16_matrix gate_proj, up_proj, down_proj;
} ds_layer;

typedef struct {
    uint8_t *codes;              /* [N/16, K/16, 8, 16] */
    uint8_t *scales;             /* [N/16, K/16, 16] */
    size_t n, k, groups;
    float global_scale;
    size_t code_bytes, scale_bytes;
} ds_nvfp4_matrix;

struct dspark_model {
    dspark_backend backend;
    int threads;
    int target_layer_ids[DSPARK_TARGET_TAPS];
    int mask_token_id;
    void *draft_arena;
    size_t draft_bytes;
    void *embedding_arena;
    size_t embedding_bytes;
    const uint16_t *embedding;
    ds_bf16_matrix fc;
    const uint16_t *hidden_norm;
    ds_layer layers[DS_LAYERS];
    const uint16_t *norm;
    ds_bf16_matrix markov_w1, markov_w2;
    const uint16_t *confidence_weight;
    float confidence_bias;
    ds_nvfp4_matrix lm_head;
    float rope_inv_freq[DS_HEAD_DIM / 2];
    float rope_attention_factor;
};

struct dspark_state {
    const dspark_model *model;
    size_t capacity;
    size_t cursor;
    uint16_t *key_cache[DS_LAYERS];
    uint16_t *value_cache[DS_LAYERS];
};

float ds_bf16_to_f32(uint16_t x);
uint16_t ds_f32_to_bf16(float x);
float ds_decode_e2m1(uint8_t x);
float ds_decode_e4m3(uint8_t x);

void ds_gemm_bf16(const dspark_model *model, const uint16_t *w,
                  size_t rows, size_t cols, const float *x, size_t m,
                  float *y);
void ds_gemm_bf16_pair(const dspark_model *model, const uint16_t *w0,
                       const uint16_t *w1, size_t rows, size_t cols,
                       const float *x, size_t m, float *y0, float *y1);
void ds_rmsnorm(const uint16_t *weight, const float *x, float *y,
                size_t rows, size_t cols, float eps, int threads);
void ds_head_rmsnorm(const uint16_t *weight, float *x, size_t rows,
                     size_t heads, size_t head_dim, float eps, int threads);
void ds_apply_rope(const dspark_model *model, float *x, size_t rows,
                   size_t heads, size_t position0, int threads);
float ds_dot_bf16(const float *x, const uint16_t *y, size_t n,
                  dspark_backend backend);
float ds_attention_scores_bf16(const float *q, const uint16_t *kc,
                               const uint16_t *kn, size_t context,
                               size_t total, size_t kv_head, float scale,
                               float *scores, dspark_backend backend);
void ds_attention_values_bf16(const float *scores, const uint16_t *vc,
                              const uint16_t *vn, size_t context,
                              size_t total, size_t kv_head, float norm,
                              float *out, dspark_backend backend);
void ds_nvfp4_gemm(const dspark_model *model, const ds_nvfp4_matrix *w,
                   const float *x, size_t m, float *y);

void *ds_anon_alloc(size_t bytes);
void ds_anon_free(void *ptr, size_t bytes);

#endif
