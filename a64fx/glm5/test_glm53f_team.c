/* Real work-sharing, delayed workers, reused stack contexts, repeated teams. */
#include "glm53f_team.h"
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N = 257, JOBS = 4000 };
struct job { uint64_t *output; uint64_t epoch; };
static void execute(void *context) {
    struct job *j = context;
    /* Skew the last worker before its completion publication. */
    if (omp_get_thread_num() + 1 == omp_get_num_threads() && j->epoch % 17 == 0)
        for (volatile int i = 0; i < 200; ++i) {}
#pragma omp for schedule(static)
    for (int i = 0; i < N; ++i) j->output[i] = j->epoch * N + i;
}
static void control(void *context) {
    int *failed = context;
    uint64_t output[N];
    for (uint64_t e = 1; e <= JOBS; ++e) {
        struct job j = {output, e};
        memset(output, 0, sizeof(output));
        glm53f_team_dispatch(execute, &j);
        for (int i = 0; i < N; ++i) *failed |= output[i] != e * N + i;
        /* The worker must no longer reference j or output here. */
        j.epoch = UINT64_MAX;
        memset(output, 0xff, sizeof(output));
    }
}
int main(void) {
    int failed = 0;
    double begin = 0;
    for (int r = 0; r < 3; ++r) {
        if (r == 1) begin = omp_get_wtime();
        glm53f_team_run(control, &failed);
    }
    double seconds = omp_get_wtime() - begin;
    printf("GLM53F_TEAM threads=%d jobs=%d repeats=3 warm_job_us=%.6f epochs=%s\n", omp_get_max_threads(), JOBS,
        seconds * 1e6 / (2 * JOBS), failed ? "FAIL" : "PASS");
    return failed;
}
