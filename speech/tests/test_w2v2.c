/* SPDX-License-Identifier: MIT
 * test_w2v2 <ja_align.safetensors> <input.npy (16 kHz f32)> <out_dir>: run the C encoder, dump stages. */
#define JA_BASE_IMPLEMENTATION
#define JA_W2V2_IMPLEMENTATION
#include "ja_base.h"
#include "w2v2.h"
#include "../../common/npy_io.h"
#include <sys/stat.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 4) return 1;
    int nd = 0, dims[8], f32 = 0;
    float *x = (float *)npy_load(argv[2], &nd, dims, &f32);
    if (!x || !f32) { fprintf(stderr, "bad input\n"); return 1; }
    mkdir(argv[3], 0755);
    struct timespec a, b, c;
    clock_gettime(CLOCK_MONOTONIC, &a);
    w2v2_model *m = w2v2_load(argv[1]);
    if (!m) return 1;
    clock_gettime(CLOCK_MONOTONIC, &b);
    w2v2_output o;
    w2v2_run(m, x, dims[0], &o, argv[3]);
    clock_gettime(CLOCK_MONOTONIC, &c);
    double tl = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
    double tr = (c.tv_sec - b.tv_sec) + (c.tv_nsec - b.tv_nsec) * 1e-9;
    printf("samples=%d frames=%d load=%.2fs run=%.3fs (RTF %.3f)\n", dims[0], o.T, tl, tr, tr / (dims[0] / 16000.0));
    w2v2_output_free(&o);
    w2v2_free(m);
    free(x);
    return 0;
}
