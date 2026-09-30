/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * ja_align: Japanese speech -> timed phonemes / kana, viseme curves, prosody (JSON).
 *
 *   ja_align --model ja_align.safetensors --wav in.wav [--kana "きょうは…"] [--phonemes "ky o o w a"]
 *            [--fps 30] [--out aux.json] [--posteriors post.npy] [--dump-dir dir] [--cuda [--device N]]
 */
#define JA_ALIGN_IMPLEMENTATION
#include "ja_align.h"
#include "ja_wav.h"
#ifdef JA_WITH_CUDA
#define JA_CUDA_IMPLEMENTATION
#include "ja_cuda.h"
static int cuda_encoder(void *ctx, const float *wav, int n, w2v2_output *out, const char *dump) {
    return w2v2_cuda_run((w2v2_cuda *)ctx, wav, n, out, dump);
}
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model = NULL, *wav = NULL, *out = "aux.json", *post = NULL;
    int use_cuda = 0, device = 0;
    ja_align_opts o;
    ja_align_opts_default(&o);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--wav") && v) { wav = v; i++; }
        else if (!strcmp(a, "--kana") && v) { o.kana = v; i++; }
        else if (!strcmp(a, "--phonemes") && v) { o.phonemes = v; i++; }
        else if (!strcmp(a, "--fps") && v) { o.fps = (float)atof(v); i++; }
        else if (!strcmp(a, "--out") && v) { out = v; i++; }
        else if (!strcmp(a, "--posteriors") && v) { post = v; i++; }
        else if (!strcmp(a, "--dump-dir") && v) { o.dump_dir = v; i++; }
        else if (!strcmp(a, "--cuda") || !strcmp(a, "--rocm")) {
#ifdef JA_WITH_HIP
            if (strcmp(a,"--rocm")) { fprintf(stderr,"use --rocm for the HIP build\n"); return 1; }
#else
            if (strcmp(a,"--cuda")) { fprintf(stderr,"use the ROCm aligner build\n"); return 1; }
#endif
            use_cuda = 1;
        }
        else if (!strcmp(a, "--device") && v) { device = atoi(v); i++; }
        else { fprintf(stderr, "unknown or incomplete option %s\n", a); return 1; }
    }
    if (!model || !wav) {
        fprintf(stderr, "usage: %s --model ja_align.safetensors --wav in.wav [--kana reading] [--out aux.json]\n", argv[0]);
        return 1;
    }
    if (o.dump_dir) mkdir(o.dump_dir, 0755);
    int n = 0, sr = 0;
    float *x = wav_read(wav, &n, &sr);
    if (!x) { fprintf(stderr, "cannot read %s\n", wav); return 1; }
    double t0 = now_s();
    w2v2_model *m = NULL;
    ja_align_result r;
    int rc;
    double t1;
    if (use_cuda) {
#ifdef JA_WITH_CUDA
        w2v2_cuda *g = w2v2_cuda_create(model, device, 1);
        if (!g) return 1;
        t1 = now_s();
        rc = ja_align_run_ex(cuda_encoder, g, x, n, sr, &o, &r);
        w2v2_cuda_free(g);
#else
        (void)device;
        fprintf(stderr, "built without CUDA (make -C speech cuda)\n");
        return 1;
#endif
    } else {
        m = w2v2_load(model);
        if (!m) return 1;
        t1 = now_s();
        rc = ja_align_run(m, x, n, sr, &o, &r);
    }
    if (rc) { fprintf(stderr, "alignment failed\n"); return 1; }
    double t2 = now_s();
    ja_align_write_json(&r, out);
    if (post) { int dims[2] = { r.T, r.n_phon_cls }; ja_npy_save_f32(post, r.phon_post, 2, dims); }
    fprintf(stderr, "%.2fs audio, %d frames, load %.2fs, align %.2fs (RTF %.3f), mode=%s\n",
            r.duration, r.T, t1 - t0, t2 - t1, (t2 - t1) / r.duration, r.forced ? "forced" : "free");
    printf("%s\n%s\n", r.phoneme_text, r.kana_text);
    ja_align_result_free(&r);
    w2v2_free(m);
    free(x);
    return 0;
}
