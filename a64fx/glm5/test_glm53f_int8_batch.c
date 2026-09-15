#define _POSIX_C_SOURCE 200809L
#include "glm53f_int8.h"
#include <stdio.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void matrix(float *out, const int8_t *w, const float *scale,
        const int8_t *x, const float *xs, int rows, int cols, int tokens,
        int batch) {
    if (!batch) {
        for (int t = 0; t < tokens; ++t) {
#pragma omp parallel for schedule(static)
            for (int r = 0; r < rows; r += 64)
                glm53f_i8_dot64(out + (size_t)t * rows + r,
                    w + (size_t)r * cols, scale + r,
                    x + (size_t)t * cols, xs[t], cols);
        }
    } else {
#pragma omp parallel for collapse(2) schedule(static)
        for (int t = 0; t < tokens; t += 16)
            for (int r = 0; r < rows; r += 16) {
                int n = tokens - t;
                if (n > 16) n = 16;
                glm53f_i8_dot16_batch16(out + (size_t)t * rows + r, rows,
                    w + (size_t)(r / 64) * 64 * cols + (r % 64) * 4,
                    scale + r, x + (size_t)t * cols, cols, xs + t, n, cols);
            }
    }
}

static int check(int rows, int cols, int tokens, int reps) {
    int8_t *w = malloc((size_t)rows * cols);
    int8_t *x = malloc((size_t)tokens * cols);
    float *scale = malloc((size_t)rows * sizeof(float));
    float *xs = malloc((size_t)tokens * sizeof(float));
    float *ref = malloc((size_t)tokens * rows * sizeof(float));
    float *out = malloc((size_t)tokens * rows * sizeof(float));
    if (!w || !x || !scale || !xs || !ref || !out) return 2;
    for (int r = 0; r < rows; ++r) {
        scale[r] = r == 0 ? 0.0f : 0.0001f * (1 + r % 17);
        for (int c = 0; c < cols; ++c)
            w[(size_t)r * cols + c] = (int8_t)((r * 37 + c * 19) % 256 - 128);
    }
    for (int t = 0; t < tokens; ++t) {
        xs[t] = 0.001f * (1 + t % 13);
        for (int c = 0; c < cols; ++c)
            x[(size_t)t * cols + c] = t == 0 ? 0 : t == 1 ? 127 :
                t == 2 ? -128 : (int8_t)((t * 71 + c * 53) % 256 - 128);
    }
    matrix(ref, w, scale, x, xs, rows, cols, tokens, 0);
    matrix(out, w, scale, x, xs, rows, cols, tokens, 1);
    int failed = memcmp(ref, out, (size_t)tokens * rows * sizeof(float)) != 0;
    /* Independent integer oracle, not merely two calls into the SVE code. */
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r) {
            int64_t dot = 0;
            for (int c = 0; c < cols; ++c)
                dot += (int)w[(size_t)(r / 64) * 64 * cols +
                    (size_t)(c / 4) * 256 + (r % 64) * 4 + c % 4] *
                    (int)x[(size_t)t * cols + c];
            float expected = (float)dot * (scale[r] * xs[t]);
            if (memcmp(&expected, out + (size_t)t * rows + r, sizeof(float)))
                failed = 1;
        }
    printf("INT8_BATCH rows=%d cols=%d tokens=%d exact=%d\n", rows, cols, tokens, !failed);
    if (reps && !failed) {
        double time[2] = {0}, sum = 0;
        for (int rep = -2; rep < reps; ++rep)
            for (int mode = 0; mode < 2; ++mode) {
                double begin = seconds();
                matrix(out, w, scale, x, xs, rows, cols, tokens, mode);
                if (rep >= 0) time[mode] += seconds() - begin;
                sum += out[rows + rows / 2];
            }
        printf("INT8_BATCH_BENCH rows=%d cols=%d tokens=%d reps=%d "
               "scalar_ms=%.6f batch_ms=%.6f speedup=%.3f checksum=%.9g\n",
               rows, cols, tokens, reps, time[0]*1e3/reps, time[1]*1e3/reps,
               time[0]/time[1], sum);
    }
    free(out); free(ref); free(xs); free(scale); free(x); free(w);
    return failed;
}

int main(int argc, char **argv) {
    if (argc == 5) {
        int rows = atoi(argv[1]), cols = atoi(argv[2]), tokens = atoi(argv[3]);
        int reps = atoi(argv[4]);
        if (rows < 64 || rows % 64 || cols < 128 || cols > 4096 ||
            cols % 128 || tokens < 2 || tokens > 512 || reps < 1) return 2;
        return check(rows, cols, tokens, reps);
    }
    const int sizes[] = {128, 256, 512, 640, 768, 4096};
    const int batches[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33};
    int failed = 0;
    for (unsigned k = 0; k < sizeof(sizes)/sizeof(sizes[0]); ++k)
        for (unsigned t = 0; t < sizeof(batches)/sizeof(batches[0]); ++t)
            failed |= check(128, sizes[k], batches[t], 0);
    return failed;
}
