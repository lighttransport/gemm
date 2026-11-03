#ifndef GLM53F_KDA_QUEUE_H
#define GLM53F_KDA_QUEUE_H
#include <arm_sve.h>
#include "glm53f_expert_kern.h"

/* Independent projections only. Every team member supplies the same list;
 * the implicit barrier completes every output before a dependent queue runs.
 * Keep the reference four-row kernel and scalar remainder arithmetic intact. */
typedef struct {
    float *out;
    const uint16_t *weight;
    const float *input;
    int rows, cols;
} glm53f_kda_projection_job;

static inline void glm53f_kda_projection_queue_team(
        const glm53f_kda_projection_job *jobs, int count, int tokens) {
    int blocks = 0;
    for (int j = 0; j < count; ++j) blocks += (jobs[j].rows + 3) / 4;
#pragma omp for schedule(static)
    for (int block = 0; block < blocks; ++block) {
        int index = 0, row_block = block;
        while (row_block >= (jobs[index].rows + 3) / 4) {
            row_block -= (jobs[index].rows + 3) / 4;
            ++index;
        }
        const glm53f_kda_projection_job *job = jobs + index;
        int row = row_block * 4;
        if (row + 4 <= job->rows) {
            glm53f_matvec_bf16_4x4(job->out + row, job->rows,
                job->weight + (size_t)row * job->cols, job->input,
                tokens, job->cols);
        } else {
            for (int t = 0; t < tokens; ++t)
                for (int r = row; r < job->rows; ++r)
                    job->out[(size_t)t * job->rows + r] = glm53f_dot_bf16_sve(
                        job->weight + (size_t)r * job->cols,
                        job->input + (size_t)t * job->cols, job->cols);
        }
    }
}
#endif
