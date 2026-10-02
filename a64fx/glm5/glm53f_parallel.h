#ifndef GLM53F_PARALLEL_H
#define GLM53F_PARALLEL_H
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum { GLM53F_TP12 = 0, GLM53F_PP3_TP4 = 1 } glm53f_parallel_layout;
typedef struct {
    glm53f_parallel_layout layout;
    int microbatch, cuts[2];
} glm53f_parallel_config;
typedef struct {
    int world_rank, world_size, stage, stages, tp_rank, tp_size;
    int first_layer, end_layer;
} glm53f_parallel_map;

static inline glm53f_parallel_config glm53f_parallel_default(void) {
    glm53f_parallel_config c = {GLM53F_TP12, 1024, {15, 30}};
    return c;
}
static inline int glm53f_parallel_valid(const glm53f_parallel_config *c) {
    return c && (c->layout == GLM53F_TP12 || c->layout == GLM53F_PP3_TP4) &&
        (c->microbatch == 512 || c->microbatch == 1024 || c->microbatch == 2048) &&
        c->cuts[0] >= 1 && c->cuts[0] < c->cuts[1] && c->cuts[1] < 45;
}
static inline int glm53f_parallel_map_rank(const glm53f_parallel_config *c,
        int rank, int size, glm53f_parallel_map *m) {
    if (!glm53f_parallel_valid(c) || !m || size != 12 || rank < 0 || rank >= size) return -1;
    m->world_rank = rank; m->world_size = size;
    m->stages = c->layout == GLM53F_PP3_TP4 ? 3 : 1;
    m->tp_size = size / m->stages;
    m->stage = rank / m->tp_size; m->tp_rank = rank % m->tp_size;
    const int bounds[4] = {0, c->cuts[0], c->cuts[1], 45};
    m->first_layer = m->stages == 1 ? 0 : bounds[m->stage];
    m->end_layer = m->stages == 1 ? 45 : bounds[m->stage + 1];
    return 0;
}
/* Argument-only selection; never selects process-wide numerical switches. */
static inline int glm53f_parallel_option(glm53f_parallel_config *c,
        int argc, char **argv, int *index) {
    if (!c || !argv || !index || *index < 0 || *index >= argc) return -1;
    const char *key = argv[*index];
    if (strcmp(key, "--parallel-layout") && strcmp(key, "--pipeline-microbatch") &&
        strcmp(key, "--pipeline-cuts")) return 0;
    if (*index + 1 >= argc) return -1;
    const char *value = argv[++*index];
    glm53f_parallel_config next = *c;
    if (!strcmp(key, "--parallel-layout")) {
        if (!strcmp(value, "tp12")) next.layout = GLM53F_TP12;
        else if (!strcmp(value, "pp3-tp4")) next.layout = GLM53F_PP3_TP4;
        else return -1;
    } else {
        char *end; errno = 0;
        long a = strtol(value, &end, 10);
        if (errno || end == value || a < 1 || a > 2048) return -1;
        if (!strcmp(key, "--pipeline-microbatch")) {
            if (*end) return -1;
            next.microbatch = (int)a;
        } else {
            if (*end != ',') return -1;
            const char *tail = end + 1; errno = 0;
            long b = strtol(tail, &end, 10);
            if (errno || end == tail || *end || b < 1 || b >= 45) return -1;
            next.cuts[0] = (int)a; next.cuts[1] = (int)b;
        }
    }
    if (!glm53f_parallel_valid(&next)) return -1;
    *c = next;
    return 1;
}
/* Two FP32 four-stream transfer slots; checked before allocating. */
static inline int glm53f_pipeline_buffer_bytes(int microbatch, int flat, size_t *bytes) {
    if (!bytes || microbatch < 1 || microbatch > 2048 || flat < 4 || flat % 4 ||
        (size_t)microbatch > SIZE_MAX / (size_t)flat / sizeof(float) / 2) return -1;
    *bytes = (size_t)microbatch * flat * sizeof(float) * 2;
    return 0;
}
#endif
