#define _POSIX_C_SOURCE 200809L

#include "turboquant_cpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static float rand_f32(uint64_t *s) {
    uint32_t v = (uint32_t)(splitmix64(s) >> 40);
    return ((float)v / 16777216.0f) * 2.0f - 1.0f;
}

static void fill(float *x, int n, uint64_t seed) {
    uint64_t s = seed;
    for (int i = 0; i < n; i++) x[i] = rand_f32(&s);
}

int main(int argc, char **argv) {
    int n = 4096;
    int iters = 2000;
    int verify = 1;
    int bits = 3;
    uint64_t seed = 42;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bits") && i + 1 < argc) bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--bench-only")) verify = 0;
        else if (!strcmp(argv[i], "--verify")) verify = 1;
    }

    size_t bytes = bits == 4 ? tq4_row_bytes(n) : tq3_row_bytes(n);
    if (bytes == 0) {
        fprintf(stderr, "n must be a positive multiple of %d\n", TQ3_BLOCK_SIZE);
        return 2;
    }
    if (bits != 3 && bits != 4) {
        fprintf(stderr, "--bits must be 3 or 4\n");
        return 2;
    }

    float *x = (float *)aligned_alloc(64, (size_t)n * sizeof(float));
    float *y = (float *)aligned_alloc(64, (size_t)n * sizeof(float));
    float *dq = (float *)aligned_alloc(64, (size_t)n * sizeof(float));
    void *q = aligned_alloc(64, bytes);
    if (!x || !y || !dq || !q) return 2;

    fill(x, n, seed + 1);
    fill(y, n, seed + 2);

    int rc = bits == 4
        ? tq4_quantize_row_f32((tq4_block *)q, x, n, seed)
        : tq3_quantize_row_f32((tq3_block *)q, x, n, seed);
    rc |= bits == 4
        ? tq4_dequantize_row_f32(dq, (const tq4_block *)q, n, seed)
        : tq3_dequantize_row_f32(dq, (const tq3_block *)q, n, seed);
    if (rc != 0) {
        fprintf(stderr, "quant/dequant failed\n");
        return 2;
    }

    if (verify) {
        double norm_x = 0.0, norm_e = 0.0;
        double dot_ref = 0.0;
        for (int i = 0; i < n; i++) {
            double e = (double)x[i] - (double)dq[i];
            norm_x += (double)x[i] * (double)x[i];
            norm_e += e * e;
            dot_ref += (double)dq[i] * (double)y[i];
        }
        float dot_fast = bits == 4
            ? tq4_dot_row_f32((const tq4_block *)q, y, n, seed)
            : tq3_dot_row_f32((const tq3_block *)q, y, n, seed);
        double rel = sqrt(norm_e / fmax(norm_x, 1.0e-30));
        double dot_err = fabs((double)dot_fast - dot_ref);
        printf("backend=%s tq%d n=%d bytes=%zu compression_vs_f16=%.3fx rel_l2=%.6f dot_err=%.6g\n",
               tq3_cpu_backend(), bits, n, bytes, ((double)n * 2.0) / (double)bytes, rel, dot_err);
        if (rel > 0.35 || dot_err > 2.0e-3 * fmax(1.0, fabs(dot_ref))) {
            fprintf(stderr, "verification failed\n");
            return 1;
        }
    }

    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        if (bits == 4) tq4_quantize_row_f32((tq4_block *)q, x, n, seed);
        else tq3_quantize_row_f32((tq3_block *)q, x, n, seed);
    }
    double t1 = now_sec();
    for (int i = 0; i < iters; i++) {
        if (bits == 4) tq4_dequantize_row_f32(dq, (const tq4_block *)q, n, seed);
        else tq3_dequantize_row_f32(dq, (const tq3_block *)q, n, seed);
    }
    double t2 = now_sec();
    volatile float sink = 0.0f;
    for (int i = 0; i < iters; i++) {
        sink += bits == 4
            ? tq4_dot_row_f32((const tq4_block *)q, y, n, seed)
            : tq3_dot_row_f32((const tq3_block *)q, y, n, seed);
    }
    double t3 = now_sec();

    double gb = (double)n * sizeof(float) * (double)iters / 1.0e9;
    printf("quant %.3f GB/s dequant %.3f GB/s dot %.3f GB/s sink=%.6f\n",
           gb / (t1 - t0), gb / (t2 - t1), gb / (t3 - t2), sink);

    free(x);
    free(y);
    free(dq);
    free(q);
    return 0;
}
