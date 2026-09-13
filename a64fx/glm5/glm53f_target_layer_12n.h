#ifndef GLM53F_TARGET_LAYER_12N_H
#define GLM53F_TARGET_LAYER_12N_H

#include <stdint.h>
#include "glm53f_mhc_sve.h"

typedef int (*glm53f_target_sublayer_12n)(
    void *context, float *output, const float *normalized_input);

typedef struct {
    glm53f_mhc_site attention_mhc;
    glm53f_mhc_site ffn_mhc;
    const uint16_t *input_norm;
    const uint16_t *post_attention_norm;
} glm53f_target_layer_weights_12n;

typedef struct {
    glm53f_mhc_scratch mhc;
    float sublayer_output[GLM53F_MHC_WIDTH];
} glm53f_target_layer_scratch_12n;

/* Exact checkpoint ordering for one decoder layer. Both callbacks receive the
 * collapsed, RMS-normalized 4096-vector and return a replicated 4096-vector;
 * distributed callbacks therefore complete their all-reduce before returning. */
static inline int glm53f_target_layer_forward_12n(
        float *streams, const glm53f_target_layer_weights_12n *weights,
        glm53f_target_sublayer_12n attention, void *attention_context,
        glm53f_target_sublayer_12n ffn, void *ffn_context,
        glm53f_target_layer_scratch_12n *scratch) {
    glm53f_mhc_pre_sve(&scratch->mhc, streams, &weights->attention_mhc,
                       weights->input_norm);
    if (attention(attention_context, scratch->sublayer_output,
                  scratch->mhc.normalized)) return -1;
    glm53f_mhc_post_sve(streams, scratch->sublayer_output, &scratch->mhc);

    glm53f_mhc_pre_sve(&scratch->mhc, streams, &weights->ffn_mhc,
                       weights->post_attention_norm);
    if (ffn(ffn_context, scratch->sublayer_output,
            scratch->mhc.normalized)) return -1;
    glm53f_mhc_post_sve(streams, scratch->sublayer_output, &scratch->mhc);
    return 0;
}

#endif
