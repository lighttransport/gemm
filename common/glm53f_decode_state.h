#ifndef GLM53F_DECODE_STATE_H
#define GLM53F_DECODE_STATE_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

enum {
    GLM53F_LINEAR_LAYERS = 34,
    GLM53F_SPARSE_LAYERS = 11,
    GLM53F_LOCAL_HEADS_12N = 6,
    GLM53F_HEAD_DIM = 128,
    GLM53F_CONV_HISTORY = 4
};

typedef struct {
    float *recurrent;
    float *conv;
    int sparse_length[GLM53F_SPARSE_LAYERS];
} glm53f_decode_state_12n;

static inline size_t glm53f_recurrent_bytes_12n(void) {
    return (size_t)GLM53F_LINEAR_LAYERS * GLM53F_LOCAL_HEADS_12N *
           GLM53F_HEAD_DIM * GLM53F_HEAD_DIM * sizeof(float);
}

static inline size_t glm53f_conv_bytes_12n(void) {
    return (size_t)GLM53F_LINEAR_LAYERS * 3 * GLM53F_LOCAL_HEADS_12N *
           GLM53F_HEAD_DIM * GLM53F_CONV_HISTORY * sizeof(float);
}

static inline int glm53f_decode_state_alloc_12n(glm53f_decode_state_12n *state) {
    memset(state, 0, sizeof(*state));
    if (posix_memalign((void **)&state->recurrent, 256,
                       glm53f_recurrent_bytes_12n()) ||
        posix_memalign((void **)&state->conv, 256,
                       glm53f_conv_bytes_12n())) {
        free(state->recurrent);
        free(state->conv);
        memset(state, 0, sizeof(*state));
        return -1;
    }
    return 0;
}

static inline void glm53f_decode_state_free_12n(glm53f_decode_state_12n *state) {
    free(state->recurrent);
    free(state->conv);
    memset(state, 0, sizeof(*state));
}

/* Sparse K/V storage is append-only. Restoring its logical lengths hides the
 * rejected suffix, while the recurrent and convolutional KDA state must be
 * restored byte-for-byte. */
static inline void glm53f_decode_state_save_12n(
        glm53f_decode_state_12n *dst, const glm53f_decode_state_12n *src) {
    memcpy(dst->recurrent, src->recurrent, glm53f_recurrent_bytes_12n());
    memcpy(dst->conv, src->conv, glm53f_conv_bytes_12n());
    memcpy(dst->sparse_length, src->sparse_length,
           sizeof(src->sparse_length));
}

#endif
