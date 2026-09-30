/* Synthetic single-node benchmark of the production Q8_0 decode matvec (glm53f_native_matvec_team on repacked Q8_0R rows)
 * at KDA / MLA / dense shapes, weights rotated through a pool larger than the caches.
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I. -Ikern -I../.. \
 *          bench_glm53f_native_q8.c glm53f_iq_bridge.c kern/glm53f_kern_*.c kern/glm53f_kern_gemm_asm.S -lm -lpthread
 * run:   OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active FLIB_BARRIER=HARD ./a.out [POOL=12] */
#define _GNU_SOURCE
#include "glm53f_iq_bridge.h"
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
typedef struct { const char *name; int count; int rows[6]; int cols; } shape;
int main(void) {
    const int pool = getenv("POOL") ? atoi(getenv("POOL")) : 12, iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 300;
    if (getenv("INTERLEAVE") && atoi(getenv("INTERLEAVE"))) { unsigned long m = 0xF0UL; syscall(SYS_set_mempolicy, 3L, &m, 8UL); }
    static const shape sh[] = {
        {"kda_front(3x768+2x128 x4096)", 5, {768, 768, 768, 128, 128, 0}, 4096},
        {"kda_oproj(4096 x768)", 1, {4096, 0, 0, 0, 0, 0}, 768},
        {"mla_q_a(1536 x4096)", 1, {1536, 0, 0, 0, 0, 0}, 4096},
        {"mla_kv_a(512 x4096)", 1, {512, 0, 0, 0, 0, 0}, 4096},
        {"dense_gate_up(2x1024 x4096)", 2, {1024, 1024, 0, 0, 0, 0}, 4096},
        {"shared_gate_up(2x192 x4096)", 2, {192, 192, 0, 0, 0, 0}, 4096},
        {"tiny(48 x4096) = fixed overhead", 1, {48, 0, 0, 0, 0, 0}, 4096},
    };
    const int team = getenv("TEAM") && atoi(getenv("TEAM"));
    float *x = aligned_alloc(256, 4096 * 4);
    for (int i = 0; i < 4096; ++i) x[i] = sinf(0.37f * i);
    for (unsigned si = 0; si < sizeof(sh) / sizeof(*sh); ++si) {
        const shape *s = &sh[si];
        glm53f_native_matrix (*mat)[6] = malloc(sizeof(*mat) * pool);
        size_t bytes = 0;
        for (int p = 0; p < pool; ++p)
            for (int m = 0; m < s->count; ++m) {
                const int rows = s->rows[m], cols = s->cols;
                const size_t sb = (size_t)(cols / 32) * 34;
                uint8_t *src = malloc((size_t)rows * sb);
                for (size_t i = 0; i < (size_t)rows * sb; ++i) src[i] = (uint8_t)(i * 2654435761u >> 15);
                for (int r = 0; r < rows; ++r) for (int b = 0; b < cols / 32; ++b) { _Float16 d = (_Float16)0.01f; memcpy(src + r * sb + b * 34, &d, 2); }
                uint8_t *out = NULL; int type = 0;
                glm53f_native_repack(8 /* GGML_TYPE_Q8_0 */, src, rows, cols, &out, &type);
                free(src);
                mat[p][m] = (glm53f_native_matrix){aligned_alloc(256, (size_t)rows * 4), out, type, rows, cols};
                if (!p) bytes += (size_t)rows * glm53f_native_row_size(type, cols);
            }
        void *act = aligned_alloc(256, glm53f_native_act_bytes(s->cols));
        double sum = 0, best = 1e9;
        for (int it = -20; it < iters; ++it) {
            const glm53f_native_matrix *mm = mat[(it + 20) % pool];
            double t0 = now();
#pragma omp parallel
            {
                if (team) glm53f_native_act_prepare_team(act, x, s->cols, 0, 1);
                else {
#pragma omp single
                    glm53f_native_act_prepare(act, x, s->cols, 0, 1);
                }
                glm53f_native_matvec_team(mm, s->count, act);
            }
            double dt = now() - t0;
            if (it >= 0) { sum += dt; if (dt < best) best = dt; }
            if (it == iters - 1 && si == 6) {
                double tp = now();
                for (int r = 0; r < 200; ++r) {
#pragma omp parallel
                    { }
                }
                double tq = now();
                for (int r = 0; r < 200; ++r) glm53f_native_act_prepare(act, x, s->cols, 0, 1);
                double tr = now();
                printf("  empty parallel region %.2f us, serial act_prepare(4096, q80) %.2f us\n", (tq - tp) / 200 * 1e6, (tr - tq) / 200 * 1e6);
            }
        }
        printf("%-34s type=%d weights %.2f MB: mean %.1f us best %.1f us -> %.0f GB/s\n", s->name, mat[0][0].type, bytes / 1e6, sum / iters * 1e6, best * 1e6, bytes / (sum / iters) / 1e9);
    }
    return 0;
}
