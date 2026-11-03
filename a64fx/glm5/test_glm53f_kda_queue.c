#include <stdio.h>
#include <omp.h>
#include "glm53f_kda_queue.h"

int main(void) {
    const int rows[] = {640, 640, 640, 128, 128, 5};
    enum { JOBS = 6, COLS = 256 };
    glm53f_kda_projection_job jobs[JOBS];
    float *reference[JOBS];
    float input[4 * COLS];
    for (int i = 0; i < 4 * COLS; ++i)
        input[i] = (float)((i * 31) % 253 - 126) / 127;
    for (int j = 0; j < JOBS; ++j) {
        uint16_t *w = malloc((size_t)rows[j] * COLS * sizeof(*w));
        float *out = malloc((size_t)4 * rows[j] * sizeof(*out));
        reference[j] = malloc((size_t)4 * rows[j] * sizeof(float));
        if (!w || !out || !reference[j]) return 2;
        for (int i = 0; i < rows[j] * COLS; ++i) {
            float value = (float)((i * 17 + j * 11) % 251 - 125) / 128;
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            w[i] = (uint16_t)(bits >> 16);
        }
        jobs[j] = (glm53f_kda_projection_job){out, w, input, rows[j], COLS};
    }
    for (int tokens = 1; tokens <= 4; ++tokens) {
        for (int j = 0; j < JOBS; ++j) {
            int r = 0;
            for (; r + 4 <= rows[j]; r += 4)
                glm53f_matvec_bf16_4x4(reference[j] + r, rows[j],
                    jobs[j].weight + (size_t)r * COLS, input, tokens, COLS);
            for (int t = 0; t < tokens; ++t)
                for (int k = r; k < rows[j]; ++k)
                    reference[j][t * rows[j] + k] = glm53f_dot_bf16_sve(
                        jobs[j].weight + (size_t)k * COLS, input + t * COLS, COLS);
        }
#pragma omp parallel
        {
            glm53f_kda_projection_queue_team(jobs, JOBS, tokens);
        }
        for (int j = 0; j < JOBS; ++j)
            if (memcmp(reference[j], jobs[j].out, (size_t)tokens * rows[j] * sizeof(float))) {
                fprintf(stderr, "KDA_QUEUE mismatch tokens=%d job=%d\n", tokens, j);
                return 1;
            }
        printf("KDA_QUEUE tokens=%d BIT_EXACT PASS\n", tokens);
    }
    for (int j = 0; j < JOBS; ++j) {
        free((void *)jobs[j].weight);
        free(jobs[j].out);
        free(reference[j]);
    }
    return 0;
}
