#ifndef GLM53F_IQ_BRIDGE_H
#define GLM53F_IQ_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

enum {
    GLM53F_GGML_Q8_0 = 8,
    GLM53F_GGML_Q2_K = 10,
    GLM53F_GGML_Q3_K = 11,
    GLM53F_GGML_Q4_K = 12,
    GLM53F_GGML_Q5_K = 13,
    GLM53F_GGML_Q6_K = 14,
    GLM53F_GGML_IQ2_XS = 17,
    GLM53F_GGML_IQ3_XXS = 18,
    GLM53F_GGML_IQ4_XS = 23,
    /* Runtime-only layout of GGUF Q8_0 rows (see glm53f_native_repack). */
    GLM53F_NATIVE_Q8_0R = 1008,
    GLM53F_NATIVE_Q8_0R16 = 1009
};

typedef struct {
    const uint8_t *gate_up;
    const uint8_t *down;
    int gate_type;
    int down_type;
    int inter;
} glm53f_iq_part;

int glm53f_iq_type_supported(int type);
size_t glm53f_iq_row_size(int type, int columns);
/* CMG-affine placement of one routed-expert part (see glm53f_iq_bridge.c); call before the pages are first touched. */
int glm53f_iq_affine_enabled(void);
void glm53f_iq_place_part(const uint8_t *gate_up, size_t gate_row_bytes, int gate_rows, const uint8_t *down,
                          size_t down_row_bytes, int down_rows);
int glm53f_iq_expert_weighted(
    float *output, const glm53f_iq_part *parts, const float *weights,
    int count, const float *input, float *gate_up, float *activation);
typedef struct glm53f_iq_batch_scratch glm53f_iq_batch_scratch;
glm53f_iq_batch_scratch *glm53f_iq_batch_scratch_create(void);
void glm53f_iq_batch_scratch_free(glm53f_iq_batch_scratch *scratch);
/* Up to four native verification positions, each with up to eight parts.
 * Parts/weights use part_stride entries per token. Grouping preserves each
 * token's scalar fast/reference selection and original route-sum order. */
int glm53f_iq_expert_weighted_batch(float *output,
    const glm53f_iq_part *parts, const float *weights, const int *counts,
    int part_stride, const float *input, int tokens,
    glm53f_iq_batch_scratch *scratch);

int glm53f_iq_matvec(
    float *output, const uint8_t *weight, int weight_type,
    int rows, int columns, const float *input);
/* Native GGUF matrices for the non-routed decode path: the routed-expert
 * types above plus Q8_0.  Activations follow llama.cpp's vec_dot contract:
 * Q8_K-style blocks for K/IQ weights and Q8_0 blocks for Q8_0 weights. */
typedef struct {
    float *output;
    const uint8_t *weight;
    int type, rows, columns;
} glm53f_native_matrix;

/* Shared-expert description for the fused routed+shared decode step. gu[0..1] / dn carry the output buffers
 * (gate, up, result); act is the SwiGLU buffer; act_x / act_h are native activation storages. */
typedef struct {
    glm53f_native_matrix gu[2], dn;
    void *act_x, *act_h;
    float *act;
    int rows, x_q8k, x_q80, h_q8k, h_q80;
} glm53f_iq_shared;
int glm53f_iq_expert_weighted_shared(float *output, const glm53f_iq_part *parts, const float *weights, int count,
                                     const float *input, const glm53f_iq_shared *sh);


int glm53f_native_type_supported(int type);
/* Bytes per row for any native type, including GLM53F_NATIVE_Q8_0R. */
size_t glm53f_native_row_size(int type, int columns);
/* Q8_0 with columns % 64 == 0: allocate a repacked copy (caller frees the
 * source) and set *output_type = GLM53F_NATIVE_Q8_0R.  Other types, or
 * GLM53F_NATIVE_NO_REPACK=1: *output = NULL and *output_type = type. */
int glm53f_native_repack(int type, const uint8_t *source, int rows,
                         int columns, uint8_t **output, int *output_type);
/* Keep head-local row indexing for matrices such as attn_v_b.weight. */
int glm53f_native_repack_rowwise(int type, const uint8_t *source, int rows,
                                 int columns, uint8_t **output, int *output_type);
size_t glm53f_native_act_bytes(int columns);
int glm53f_native_act_prepare(void *storage, const float *input, int columns,
                              int need_q8k, int need_q80);
/* Team version: call from every thread of an enclosing parallel region; quantizes in parallel and returns with the
 * activation ready (implicit barrier). Bit-identical to glm53f_native_act_prepare. */
int glm53f_native_act_prepare_team(void *storage, const float *input, int columns, int need_q8k, int need_q80);
/* Orphaned omp-for over all rows of up to eight same-input matrices; call
 * from every thread of an enclosing team after one thread prepared the
 * activation and the team synchronized. */
int glm53f_native_matvec_team(const glm53f_native_matrix *m, int count,
                              const void *activation);
int glm53f_native_matvec_prepared_n(const glm53f_native_matrix *m, int count, const void *activation);
int glm53f_native_matvec_n(const glm53f_native_matrix *m, int count,
                           const float *input);
/* Token-major inputs/outputs: input stride = columns, each output stride =
 * its matrix's rows. Each token has a separately prepared activation at
 * activation + token * activation_stride. The team entry point has an
 * implicit output barrier and must be called by every enclosing thread. */
int glm53f_native_matvec_batch_team(const glm53f_native_matrix *m, int count,
    const void *activation, size_t activation_stride, int tokens);
int glm53f_native_matvec_batch(const glm53f_native_matrix *m, int count,
    const float *input, int tokens);
int glm53f_iq_matvec_2(
    float *output0, const uint8_t *weight0, int weight0_type,
    float *output1, const uint8_t *weight1, int weight1_type,
    int rows, int columns, const float *input);

#endif
