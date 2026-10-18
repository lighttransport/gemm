/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * en_align: English speech -> timed IPA phones / words, viseme curves, prosody (JSON).
 *
 *   en_align --model en_align.safetensors --wav in.wav [--text "the transcript"] [--phones "h ə l oʊ"]
 *            [--dict cmudict.dict] [--fps 30] [--out aux.json] [--posteriors post.npy] [--dump-dir dir]
 */
#define JA_ALIGN_IMPLEMENTATION
#include "../ja_align/ja_align.h"
#include "../ja_align/ja_wav.h"
#include "en_align.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static int cpu_encoder(void *ctx, const float *wav, int n, w2v2_output *out, const char *dump) {
    return w2v2_run((w2v2_model *)ctx, wav, n, out, dump);
}

int main(int argc, char **argv) {
    const char *model = NULL, *wav = NULL, *out = "aux.json", *post = NULL;
    en_align_opts o; en_align_opts_default(&o);
    o.dict_path = "/mnt/nvme01/models/speech/cmudict/cmudict.dict";
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--wav") && v) { wav = v; i++; }
        else if (!strcmp(a, "--text") && v) { o.text = v; i++; }
        else if (!strcmp(a, "--phones") && v) { o.phones = v; i++; }
        else if (!strcmp(a, "--dict") && v) { o.dict_path = v; i++; }
        else if (!strcmp(a, "--fps") && v) { o.fps = (float)atof(v); i++; }
        else if (!strcmp(a, "--out") && v) { out = v; i++; }
        else if (!strcmp(a, "--posteriors") && v) { post = v; i++; }
        else if (!strcmp(a, "--dump-dir") && v) { o.dump_dir = v; i++; }
        else { fprintf(stderr, "unknown or incomplete option %s\n", a); return 1; }
    }
    if (!model || !wav) { fprintf(stderr, "usage: %s --model en_align.safetensors --wav in.wav [--text transcript] [--out aux.json]\n", argv[0]); return 1; }
    if (o.dump_dir) mkdir(o.dump_dir, 0755);
    en_vocab vocab;
    if (en_vocab_load(&vocab, model)) return 1;
    int n = 0, sr = 0;
    float *x = wav_read(wav, &n, &sr);
    if (!x) { fprintf(stderr, "cannot read %s\n", wav); return 1; }
    double t0 = now_s();
    w2v2_model *m = w2v2_load(model);
    if (!m) return 1;
    double t1 = now_s();
    en_align_result r;
    if (en_align_run_ex(cpu_encoder, m, &vocab, x, n, sr, &o, &r)) { fprintf(stderr, "alignment failed\n"); return 1; }
    double t2 = now_s();
    en_align_write_json(&r, out);
    if (post) { int dims[2] = { r.r.T, r.r.n_phon_cls }; ja_npy_save_f32(post, r.r.phon_post, 2, dims); }
    fprintf(stderr, "%.2fs audio, %d frames, load %.2fs, align %.2fs (RTF %.3f), mode=%s, words %d (%d unknown)\n",
            r.r.duration, r.r.T, t1 - t0, t2 - t1, (t2 - t1) / r.r.duration, r.r.forced ? "forced" : "free", r.n_words, r.unknown_words);
    printf("%s\n", r.r.phoneme_text);
    en_align_result_free(&r);
    w2v2_free(m);
    free(x);
    return 0;
}
