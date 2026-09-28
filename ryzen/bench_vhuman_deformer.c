/* Benchmark the virtual-human face deformer on a rig package.
 *   ./bench_vhuman_deformer rig_deformer.safetensors [frames]
 * Prints per-frame latency (single frames: sparse AXPY morphs) and batch
 * throughput (one sgemm_avx2 over all morphs), with and without the ML
 * correctives. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lightrig_mlp2.h"
#include "vhuman_deformer.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s rig_deformer.safetensors [frames]\n", argv[0]);
        return 2;
    }
    size_t frames = argc > 2 ? (size_t)atol(argv[2]) : 256;
    vh_deformer *d = vh_deformer_load(argv[1]);
    if (!d) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    size_t C = vh_deformer_controls(d), V = vh_deformer_vertices(d);
    float *x = malloc(sizeof(float) * C * frames), *out = malloc(sizeof(float) * V * 3 * frames);
    float *scratch = malloc(sizeof(float) * vh_deformer_batch_scratch(d, frames));
    if (!x || !out || !scratch) return 1;
    srand(1);
    for (size_t i = 0; i < C * frames; ++i) x[i] = (rand() % 10 == 0) ? (float)rand() / RAND_MAX : 0.f;
    printf("vertices %zu, controls %zu, morphs %zu, ml %s (mlp2 %s)\n", V, C, vh_deformer_morphs(d),
           vh_deformer_has_ml(d) ? "yes" : "no", lt_mlp2_f32_backend());
    for (int ml = 1; ml >= 0; --ml) {
        vh_deformer_eval(d, x, ml, out);                       /* warm up */
        double t = now();
        for (size_t f = 0; f < frames; ++f) vh_deformer_eval(d, x + f * C, ml, out);
        double single = (now() - t) / frames;
        t = now();
        vh_deformer_eval_batch(d, x, frames, ml, scratch, out);
        double batch = (now() - t) / frames;
        printf("ml=%d  single %.3f ms/frame  batch(%zu) %.3f ms/frame\n", ml, single * 1e3, frames, batch * 1e3);
    }
    free(x);
    free(out);
    free(scratch);
    vh_deformer_free(d);
    return 0;
}
