#ifndef DS41F_ATTENTION_H
#define DS41F_ATTENTION_H
#include "ds41f_weights.h"
#include <stdint.h>
typedef struct {
    size_t capacity;
    int index_head_tiles;
    int rows_fresh_pages;
    int sparse_math;
    int sparse_tile; /* zero retains the original head-parallel control */
    uint8_t *compressed[4]; /* each row: KV288 + index68 */
    float *window;          /* reference execution buffer, 40*128*512 floats */
    float *rows;            /* reused 640x512 sparse-attention workspace */
    float pool_value[3][512],pool_score[3][512];
    int selected[512];size_t selected_count;
    uint8_t *candidate_blocks;
    uint8_t publication[356];
} ds41f_attention;
typedef struct {
    float qr[1280],kv[512];
    int selected[512];size_t selected_count;
    uint8_t publication[356];
} ds41f_attention_context;
int ds41f_attention_prepare(ds41f_attention *state,const ds41f_weights *weights,
                            int layer,size_t position,const float *x,ds41f_attention_context *context);
int ds41f_attention_apply(ds41f_attention *state,int layer,size_t position,
                          const ds41f_attention_context *context);
int ds41f_attention_project(ds41f_attention *state,const ds41f_weights *weights,
                            int layer,size_t position,const float *qr,
                            size_t first_head,size_t heads,float *projected);
int ds41f_attention_output(const ds41f_weights *weights,int layer,const float *projected,float *out);
int ds41f_index_scores(float *scores,const float *q,const float *weights,const uint8_t *rows,
                       size_t count,const uint8_t *candidates,int head_tiles);
int ds41f_attention_init(ds41f_attention *state,size_t max_tokens);
int ds41f_attention_place_workspace(ds41f_attention *state);
void ds41f_attention_free(ds41f_attention *state);
int ds41f_attention_step(ds41f_attention *state,const ds41f_weights *weights,
                         int layer,size_t position,const float *x,float *out);
/* Apply the source owner's published row on other ranks after each source
 * layer. Owner has already written it before evaluating sparse attention. */
int ds41f_attention_receive(ds41f_attention *state,int layer,size_t position,
                            const uint8_t publication[356]);
#endif
