#ifndef GLM53F_PIPELINE_H
#define GLM53F_PIPELINE_H
#include "glm53f_dist.h"

typedef enum { GLM53F_PIPELINE_SERIAL = 0, GLM53F_PIPELINE_OVERLAP = 1 } glm53f_pipeline_schedule;
typedef int (*glm53f_pipeline_callback)(void *context, const glm53f_dist *dist,
    float *streams, int offset, int tokens, int flat);
typedef struct {
    double compute_seconds, receive_seconds, send_wait_seconds;
    int microbatches, positions;
} glm53f_pipeline_profile;
/* All world ranks call. Producer runs on stage0; executor on each stage;
 * consumer on the last stage. Callbacks may use TP collectives, never world
 * collectives. Data is token-major four-stream FP32. Errors abort world once
 * communication starts so peers cannot remain waiting for an absent stage. */
int glm53f_pipeline_run(const glm53f_dist *d, int positions, int flat,
    glm53f_pipeline_schedule schedule, glm53f_pipeline_callback producer,
    glm53f_pipeline_callback executor, glm53f_pipeline_callback consumer,
    void *context, glm53f_pipeline_profile *profile);
#endif
