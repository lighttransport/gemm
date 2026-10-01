/* SPDX-License-Identifier: MIT */
#define SAFETENSORS_IMPLEMENTATION
#define QTTS_OPS_IMPLEMENTATION
#define QTTS_CODEC_IMPLEMENTATION
#include "safetensors.h"
#include "qtts_codec.h"
#include "qtts_codec_stream.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test_codec_stream <speech_tokenizer_dir>\n"); return 2; }
    qtts_codec *codec = qtts_codec_load(argv[1]);
    if (!codec) return 2;
    int T = codec->window + 5, nq = codec->nq;
    int32_t *codes = malloc((size_t)T * nq * sizeof(int32_t));
    for (int i = 0; i < T * nq; ++i) codes[i] = (i * 37 + 7) % codec->cb_size;
    int count;
    float *expected = qtts_codec_decode(codec, codes, T, &count, NULL);
    qtts_codec_stream *state = qtts_codec_stream_create(codec);
    double mse = 0; float maximum = 0;
    for (int frame = 0; frame < T; ++frame) {
        float *actual = qtts_codec_stream_push(state, codes + frame * nq);
        if (!actual) return 3;
        for (int i = 0; i < 1920; ++i) {
            float delta = fabsf(actual[i] - expected[frame * 1920 + i]);
            if (!isfinite(delta)) return 4;
            if (delta > maximum) maximum = delta;
            mse += (double)delta * delta;
        }
        free(actual);
    }
    printf("stateful codec: %d frames, KV eviction at %d, max error %.8g, RMSE %.8g\n", T, codec->window, maximum, sqrt(mse / count));
    qtts_codec_stream_free(state);
    state = qtts_codec_stream_create(codec);
    float *reset = qtts_codec_stream_push(state, codes);
    for (int i = 0; i < 1920; ++i) if (fabsf(reset[i] - expected[i]) > 2e-4f) return 5;
    free(reset); qtts_codec_stream_free(state); qtts_codec_free(codec); free(expected); free(codes);
    return maximum < 2e-4f ? 0 : 1;
}
