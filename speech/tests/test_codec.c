/* SPDX-License-Identifier: MIT
 * test_codec: decode codes.npy with the C codec and dump stages for ref/compare.py.
 *   ./test_codec <speech_tokenizer_dir> <codes.npy> <out_dir>
 */
#define SAFETENSORS_IMPLEMENTATION
#define QTTS_OPS_IMPLEMENTATION
#define QTTS_CODEC_IMPLEMENTATION
#include "safetensors.h"
#include "npy_io.h"
#include "qtts_ops.h"
#include "qtts_codec.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <tokenizer_dir> <codes.npy> <out_dir>\n", argv[0]); return 1; }
    int nd = 0, dims[8] = {0}, f32 = 0;
    int32_t *codes = (int32_t *)npy_load(argv[2], &nd, dims, &f32);
    if (!codes || nd != 2 || f32) { fprintf(stderr, "bad codes file\n"); return 1; }
    mkdir(argv[3], 0755);
    double t0 = now_s();
    qtts_codec *c = qtts_codec_load(argv[1]);
    if (!c) return 1;
    double t1 = now_s();
    int n = 0;
    float *wav = qtts_codec_decode(c, codes, dims[0], &n, argv[3]);
    double t2 = now_s();
    char p[1024];
    snprintf(p, sizeof(p), "%s/wav.npy", argv[3]);
    qt_npy_save_f32(p, wav, 1, &n);
    printf("frames=%d samples=%d load=%.2fs decode=%.3fs (RTF %.3f)\n", dims[0], n, t1 - t0, t2 - t1,
           (t2 - t1) / (n / 24000.0));
    free(wav); free(codes);
    qtts_codec_free(c);
    return 0;
}
