/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * tts_ja: Japanese text -> Qwen3-TTS speech (wav) + ja_align animation cues (aux.json).
 *
 *   tts_ja --model <qwen3-tts dir> --aligner ja_align.safetensors --text "..." [--speaker Ono_Anna]
 *          [--instruct "..."] [--kana "pronunciation reading"] [--seed N] [--backend cpu|cuda]
 *          [--fps 30] [--out out.wav] [--aux aux.json]
 *
 * Without --kana the aligner runs alignment-free on the synthesized audio (the content is known
 * to be clean speech); with --kana it force-aligns the given reading.
 */
#define SAFETENSORS_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define QTTS_OPS_IMPLEMENTATION
#define QTTS_CODEC_IMPLEMENTATION
#define QTTS_TALKER_IMPLEMENTATION
#include "safetensors.h"
#include "gguf_loader.h"
#include "bpe_tokenizer.h"
#include "qtts_ops.h"
#include "qtts_codec.h"
#include "qtts_talker.h"
#include "qtts_tokenizer.h"
#include "wav_io.h"
#ifdef QTTS_WITH_CUDA
#define QTTS_CUDA_IMPLEMENTATION
#include "qtts_cuda.h"
#endif
#define JA_ALIGN_IMPLEMENTATION
#include "ja_align.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int32_t *tokenize(const bpe_vocab *v, const char *fmt, const char *text, int *n) {
    size_t len = strlen(fmt) + strlen(text) + 16;
    char *s = (char *)malloc(len);
    snprintf(s, len, fmt, text);
    int cnt = bpe_tokenize(v, s, -1, NULL, 0);
    int32_t *ids = (int32_t *)malloc(sizeof(int32_t) * (size_t)(cnt > 0 ? cnt : 1));
    *n = bpe_tokenize(v, s, -1, ids, cnt);
    free(s);
    return ids;
}

int main(int argc, char **argv) {
    const char *model = "/mnt/nvme01/models/speech/Qwen3-TTS-12Hz-1.7B-CustomVoice";
    const char *aligner = "/mnt/nvme01/models/speech/japanese-wav2vec2-large-hiragana-ctc/ja_align.safetensors";
    const char *text = NULL, *speaker = "Ono_Anna", *instruct = "", *out_wav = "out.wav", *aux = "aux.json";
    int use_cuda = 0;
    qtts_gen_params gp;
    qtts_gen_params_default(&gp);
    ja_align_opts ao;
    ja_align_opts_default(&ao);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--aligner") && v) { aligner = v; i++; }
        else if (!strcmp(a, "--text") && v) { text = v; i++; }
        else if (!strcmp(a, "--speaker") && v) { speaker = v; i++; }
        else if (!strcmp(a, "--instruct") && v) { instruct = v; i++; }
        else if (!strcmp(a, "--kana") && v) { ao.kana = v; i++; }
        else if (!strcmp(a, "--seed") && v) { gp.seed = strtoull(v, NULL, 10); i++; }
        else if (!strcmp(a, "--max-frames") && v) { gp.max_frames = atoi(v); i++; }
        else if (!strcmp(a, "--fps") && v) { ao.fps = (float)atof(v); i++; }
        else if (!strcmp(a, "--out") && v) { out_wav = v; i++; }
        else if (!strcmp(a, "--aux") && v) { aux = v; i++; }
        else if (!strcmp(a, "--backend") && v) { use_cuda = !strcmp(v, "cuda"); i++; }
        else { fprintf(stderr, "unknown or incomplete option %s\n", a); return 1; }
    }
    if (!text) { fprintf(stderr, "usage: %s --text \"日本語のテキスト\" [options]\n", argv[0]); return 1; }
    double t0 = now_s();
    bpe_vocab *vocab = qtts_tokenizer_load(model);
    if (!vocab) return 1;
    int n_ids = 0, n_inst = 0;
    int32_t *ids = tokenize(vocab, "<|im_start|>assistant\n%s<|im_end|>\n<|im_start|>assistant\n", text, &n_ids);
    int32_t *inst = *instruct ? tokenize(vocab, "<|im_start|>user\n%s<|im_end|>\n", instruct, &n_inst) : NULL;
    qtts_model *m = qtts_model_load(model, 2400);
    char tokdir[1024];
    snprintf(tokdir, sizeof(tokdir), "%s/speech_tokenizer", model);
    qtts_codec *codec = qtts_codec_load(tokdir);
    w2v2_model *w2v = w2v2_load(aligner);
    if (!m || !codec || !w2v) return 1;
    const qtts_backend *be = NULL;
#ifdef QTTS_WITH_CUDA
    qtts_cuda *gpu = NULL;
    qtts_backend gbe;
    if (use_cuda) {
        if (!(gpu = qtts_cuda_create(m, codec, 0, 1))) return 1;
        gbe = qtts_cuda_backend(gpu);
        be = &gbe;
    }
#else
    if (use_cuda) { fprintf(stderr, "built without CUDA\n"); return 1; }
#endif
    double t1 = now_s();
    qtts_gen_result res;
    if (qtts_generate(m, be, ids, n_ids, inst, n_inst, speaker, "Japanese", &gp, 0, &res, NULL, NULL)) return 1;
    int n = 0;
    float *wav;
#ifdef QTTS_WITH_CUDA
    if (use_cuda) wav = qtts_cuda_decode(gpu, res.codes, res.n_frames, &n, NULL);
    else
#endif
    wav = qtts_codec_decode(codec, res.codes, res.n_frames, &n, NULL);
    double t2 = now_s();
    wav_write_pcm16(out_wav, wav, n, 24000);
    ja_align_result r;
    if (ja_align_run(w2v, wav, n, 24000, &ao, &r)) { fprintf(stderr, "alignment failed\n"); return 1; }
    ja_align_write_json(&r, aux);
    double t3 = now_s();
    fprintf(stderr, "load %.1fs | tts %.2fs for %.2fs audio (RTF %.2f) | align %.2fs | %s + %s\n",
            t1 - t0, t2 - t1, n / 24000.0, (t2 - t1) / (n / 24000.0), t3 - t2, out_wav, aux);
    printf("%s\n%s\n", r.kana_text, r.phoneme_text);
    ja_align_result_free(&r);
    free(wav); free(ids); free(inst);
    qtts_gen_result_free(&res);
#ifdef QTTS_WITH_CUDA
    qtts_cuda_free(gpu);
#endif
    w2v2_free(w2v);
    qtts_codec_free(codec);
    qtts_model_free(m);
    bpe_vocab_free(vocab);
    return 0;
}
