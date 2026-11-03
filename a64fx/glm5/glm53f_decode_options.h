#ifndef GLM53F_DECODE_OPTIONS_H
#define GLM53F_DECODE_OPTIONS_H

/* Model-owned decode policy. Prompt prefill always uses the exact defaults. */
typedef struct {
    int expert_top_k;
    int sparse_top_k;
    int index_refresh;
    int kda_int8;
} glm53f_decode_options;

static inline glm53f_decode_options glm53f_decode_options_default(void) {
    glm53f_decode_options o = {8, 2048, 1, 0};
    return o;
}
static inline int glm53f_decode_options_valid(const glm53f_decode_options *o) {
    return o && (o->expert_top_k == 8 || o->expert_top_k == 6 || o->expert_top_k == 4 || o->expert_top_k == 2) &&
        (o->sparse_top_k == 2048 || o->sparse_top_k == 1024 || o->sparse_top_k == 512 || o->sparse_top_k == 256) &&
        (o->index_refresh == 1 || o->index_refresh == 2 || o->index_refresh == 4 || o->index_refresh == 8) &&
        (o->kda_int8 == 0 || o->kda_int8 == 1);
}
#endif
