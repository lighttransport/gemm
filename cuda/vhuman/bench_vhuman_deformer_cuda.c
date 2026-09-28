/* GPU face deformer benchmark and CPU parity check.
 *   ./bench_vhuman_deformer_cuda rig_deformer.safetensors [frames] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "vhuman_deformer_cuda.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s rig_deformer.safetensors [frames]\n", argv[0]);
        return 2;
    }
    size_t frames = argc > 2 ? (size_t)atol(argv[2]) : 1024;
    vh_deformer *d = vh_deformer_load(argv[1]);
    if (!d) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    vh_gpu *g = vh_gpu_create(d, 0, 1);
    if (!g) { fprintf(stderr, "no CUDA device / driver\n"); vh_deformer_free(d); return 1; }
    size_t C = vh_deformer_controls(d), V = vh_deformer_vertices(d);
    float *x = malloc(sizeof(float) * C * frames), *out = malloc(sizeof(float) * V * 3 * frames), *ref = malloc(sizeof(float) * V * 3);
    srand(1);
    for (size_t i = 0; i < C * frames; ++i) x[i] = (rand() % 10 == 0) ? (float)rand() / RAND_MAX : 0.f;
    printf("%s: vertices %zu, morphs %zu, frames %zu\n", vh_gpu_name(g), V, vh_deformer_morphs(d), frames);
    double ms[4];
    vh_gpu_eval_batch(g, x, frames, 1, out, ms);        /* warm up */
    for (int mode = 0; mode < 3; ++mode) {
        int ml = mode < 2, ct = mode == 0;          /* ML + contacts, ML only, linear rig only */
        vh_deformer_set_contact_iterations(d, ct ? 4 : 0);
        if (vh_gpu_eval_batch(g, x, frames, ml, out, ms)) { fprintf(stderr, "launch failed\n"); return 1; }
        double err = 0;
        for (size_t f = 0; f < frames; f += frames / 8 + 1) {
            vh_deformer_eval(d, x + f * C, ml, ref);
            for (size_t i = 0; i < V * 3; ++i) err = fmax(err, fabs(ref[i] - out[f * V * 3 + i]));
        }
        printf("ml=%d contacts=%d  kernel %.3f ms (%.2f us/frame)  host rig %.3f ms  upload %.3f ms  download %.3f ms  max|gpu-cpu| %.2e m\n",
               ml, ct, ms[2], ms[2] * 1e3 / frames, ms[0], ms[1], ms[3], err);
    }
    free(x); free(out); free(ref);
    vh_gpu_free(g);
    vh_deformer_free(d);
    return 0;
}
