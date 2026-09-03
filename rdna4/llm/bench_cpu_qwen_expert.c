#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef void (*quant_fn)(const float *, void *, int64_t);
typedef void (*dot_fn)(int, float *, size_t, const void *, size_t,
                       const void *, size_t, int);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#ifndef BENCH_M
#define BENCH_M 11
#endif

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s LIBGGML_CPU SHARD Q4_OFFSET Q5_OFFSET\n", argv[0]);
        return 2;
    }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    quant_fn quant_q8k = (quant_fn)dlsym(lib, "quantize_row_q8_K");
    quant_fn quant_q81 = (quant_fn)dlsym(lib, "quantize_row_q8_1");
    dot_fn dot_q4k = (dot_fn)dlsym(lib, "ggml_vec_dot_q4_K_q8_K");
    dot_fn dot_q51 = (dot_fn)dlsym(lib, "ggml_vec_dot_q5_1_q8_1");
    if (!quant_q8k || !quant_q81 || !dot_q4k || !dot_q51) return 1;

    int fd = open(argv[2], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) return 1;
    const unsigned char *map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) return 1;
    const unsigned char *q4 = map + strtoull(argv[3], NULL, 0);
    const unsigned char *q5 = map + strtoull(argv[4], NULL, 0);

    enum { M = BENCH_M, EMBED = 2560, FF = 640, REPEAT = 100 };
    float *x = aligned_alloc(64, (size_t)M * EMBED * sizeof(float));
    float *gate = aligned_alloc(64, (size_t)M * FF * sizeof(float));
    float *up = aligned_alloc(64, (size_t)M * FF * sizeof(float));
    unsigned char *xq = aligned_alloc(64, (size_t)M * (EMBED / 256) * 292);
    unsigned char *gq = aligned_alloc(64, (size_t)M * (FF / 32) * 40);
    if (!x || !gate || !up || !xq || !gq) return 1;
    for (int i = 0; i < M * EMBED; i++) x[i] = sinf((float)i * 0.001f);
    for (int m = 0; m < M; m++)
        quant_q8k(x + (size_t)m * EMBED, xq + (size_t)m * 2920, EMBED);

    double start = now_sec();
    for (int rep = 0; rep < REPEAT; rep++) {
#pragma omp parallel for schedule(static)
        for (int row = 0; row < FF; row++) for (int m = 0; m < M; m++) {
            dot_q4k(EMBED, &gate[(size_t)m * FF + row], 0,
                    q4 + (size_t)row * 1440, 0, xq + (size_t)m * 2920, 0, 1);
            dot_q4k(EMBED, &up[(size_t)m * FF + row], 0,
                    q4 + (size_t)row * 1440, 0, xq + (size_t)m * 2920, 0, 1);
        }
        for (int m = 0; m < M; m++) {
            for (int i = 0; i < FF; i++) {
                float g = gate[(size_t)m * FF + i];
                gate[(size_t)m * FF + i] = g / (1.0f + expf(-g)) * up[(size_t)m * FF + i];
            }
            quant_q81(gate + (size_t)m * FF, gq + (size_t)m * 800, FF);
        }
#pragma omp parallel for schedule(static)
        for (int row = 0; row < EMBED; row++) for (int m = 0; m < M; m++) {
            float out;
            dot_q51(FF, &out, 0, q5 + (size_t)row * 480, 0,
                     gq + (size_t)m * 800, 0, 1);
            gate[(size_t)m * FF + (row % FF)] += out * 1e-30f;
        }
    }
    double elapsed = now_sec() - start;
    printf("threads=%d M=%d expert=%.3f ms effective_weight=%.2f GiB/s\n",
           omp_get_max_threads(), M, elapsed * 1e3 / REPEAT,
           (double)REPEAT * M * (2.0 * FF * 1440.0 + EMBED * 480.0) /
               elapsed / (double)(1ULL << 30));
    munmap((void *)map, st.st_size);
    close(fd);
    dlclose(lib);
    free(x); free(gate); free(up); free(xq); free(gq);
    return 0;
}
