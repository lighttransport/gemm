#ifndef DS41F_ATTENTION_H
#define DS41F_ATTENTION_H
#include "ds41f_weights.h"
#include <stdint.h>
typedef struct {
    size_t capacity;
    uint8_t *compressed[4]; /* each row: KV288 + index68 */
    float *window;          /* reference execution buffer, 40*128*512 floats */
    float pool_value[3][512],pool_score[3][512];
    int selected[512];size_t selected_count;
    uint8_t *candidate_blocks;
    uint8_t publication[356];
} ds41f_attention;
int ds41f_attention_init(ds41f_attention *state,size_t max_tokens);
void ds41f_attention_free(ds41f_attention *state);
int ds41f_attention_step(ds41f_attention *state,const ds41f_weights *weights,
                         int layer,size_t position,const float *x,float *out);
/* Apply the source owner's published row on other ranks after each source
 * layer. Owner has already written it before evaluating sparse attention. */
int ds41f_attention_receive(ds41f_attention *state,int layer,size_t position,
                            const uint8_t publication[356]);
#endif
