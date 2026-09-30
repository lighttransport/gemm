/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * tts_ja: Japanese text -> Qwen3-TTS speech (wav) + ja_align animation cues (aux.json).
 *
 *   tts_ja --model <qwen3-tts dir> --aligner ja_align.safetensors --text "..." [--speaker Ono_Anna]
 *          [--instruct "..."] [--kana "pronunciation reading"] [--seed N] [--backend cpu|cuda]
 *          [--fps 30] [--out out.wav] [--aux aux.json]
 *          voice clone (--model <Base dir>): --ref-wav ref.wav [--ref-text "transcript"] [--xvec-only]
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
#define QTTS_SPK_IMPLEMENTATION
#define QTTS_CODEC_ENC_IMPLEMENTATION
#include "safetensors.h"
#include "gguf_loader.h"
#include "bpe_tokenizer.h"
#include "qtts_ops.h"
#include "qtts_codec.h"
#include "qtts_talker.h"
#include "qtts_tokenizer.h"
#include "qtts_spk.h"
#include "qtts_codec_enc.h"
#include "wav_io.h"
#include "qtts_clone.h"
#ifdef QTTS_WITH_CUDA
#define QTTS_CUDA_IMPLEMENTATION
#include "qtts_cuda.h"
#endif
#define JA_ALIGN_IMPLEMENTATION
#include "ja_align.h"
#ifdef QTTS_WITH_CUDA
#define JA_CUDA_IMPLEMENTATION
#include "ja_cuda.h"
static int cuda_encoder(void *ctx, const float *wav, int n, w2v2_output *out, const char *dump) {
    return w2v2_cuda_run((w2v2_cuda *)ctx, wav, n, out, dump);
}
#endif

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
    int use_cuda = 0, xvec_only = 0, device = 0;
    const char *ref_wav = NULL, *ref_text = NULL;
    qtts_gen_params gp;
    qtts_gen_params_default(&gp);
    ja_align_opts ao;
    ja_align_opts_default(&ao);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--device") && v) { device = atoi(v); i++; }
        else if (!strcmp(a, "--model") && v) { model = v; i++; }
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
        else if (!strcmp(a, "--backend") && v) { if (strcmp(v, "cpu") && strcmp(v, "cuda") && strcmp(v, "rocm")) { fprintf(stderr, "invalid backend %s\n", v); return 2; }
#ifdef QTTS_WITH_HIP
            if (!strcmp(v, "cuda")) { fprintf(stderr, "ROCm runner cannot select CUDA\n"); return 2; }
            use_cuda = !strcmp(v, "rocm");
#else
            if (!strcmp(v, "rocm")) { fprintf(stderr, "use the ROCm runner\n"); return 2; }
            use_cuda = !strcmp(v, "cuda");
#endif
            i++; }
        else if (!strcmp(a, "--ref-wav") && v) { ref_wav = v; i++; }
        else if (!strcmp(a, "--ref-text") && v) { ref_text = v; i++; }
        else if (!strcmp(a, "--xvec-only")) xvec_only = 1;
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
    /* CPU encoder weights are only needed when the aligner runs on the CPU */
    w2v2_model *w2v = use_cuda ? NULL : w2v2_load(aligner);
    if (!m || !codec || (!use_cuda && !w2v)) return 1;
    qtts_clone_state cs;
    memset(&cs, 0, sizeof(cs));
    if (ref_wav) {
        if (qtts_clone_prepare(&cs, model, vocab, ref_wav, ref_text, xvec_only, NULL)) return 1;
        gp.clone = &cs.vc;
        gp.streaming = 1;
        speaker = "";
    }
    const qtts_backend *be = NULL;
#ifdef QTTS_WITH_CUDA
    qtts_cuda *gpu = NULL;
    qtts_backend gbe;
    if (use_cuda) {
        if (!(gpu = qtts_cuda_create(m, codec, device, 1))) return 1;
        gbe = qtts_cuda_backend(gpu);
        be = &gbe;
    }
#else
    if (use_cuda) { fprintf(stderr, "built without CUDA\n"); return 1; }
#endif
    double t1 = now_s();
    qtts_gen_result res;
    if (qtts_generate(m, be, ids, n_ids, inst, n_inst, speaker, "Japanese", &gp, 0, &res, NULL, NULL)) return 1;
    int n = 0, dec_T = 0;
    float *wav;
    int32_t *dec_codes = qtts_clone_codes(&cs, res.codes, res.n_frames, &dec_T);
#ifdef QTTS_WITH_CUDA
    if (use_cuda) wav = qtts_cuda_decode(gpu, dec_codes, dec_T, &n, NULL);
    else
#endif
    wav = qtts_codec_decode(codec, dec_codes, dec_T, &n, NULL);
    n = qtts_clone_trim(&cs, wav, n, dec_T);
    if (dec_codes != res.codes) free(dec_codes);
    double t2 = now_s();
    wav_write_pcm16(out_wav, wav, n, 24000);
    ja_align_result r;
    int arc;
    double t_al0 = now_s();
#ifdef QTTS_WITH_CUDA
    if (use_cuda) {  /* created after TTS so each runtime keeps its own current context */
        w2v2_cuda *wg = w2v2_cuda_create(aligner, device, 0);
        if (!wg) return 1;
        t_al0 = now_s();
        arc = ja_align_run_ex(cuda_encoder, wg, wav, n, 24000, &ao, &r);
        w2v2_cuda_free(wg);
    } else
#endif
    arc = ja_align_run(w2v, wav, n, 24000, &ao, &r);
    if (arc) { fprintf(stderr, "alignment failed\n"); return 1; }
    ja_align_write_json(&r, aux);
    double t3 = now_s();
    fprintf(stderr, "load %.1fs | tts %.2fs for %.2fs audio (RTF %.2f) | align %.2fs | %s + %s\n",
            t1 - t0, t2 - t1, n / 24000.0, (t2 - t1) / (n / 24000.0), t3 - t_al0, out_wav, aux);
    printf("%s\n%s\n", r.kana_text, r.phoneme_text);
    ja_align_result_free(&r);
    free(wav); free(ids); free(inst);
    qtts_gen_result_free(&res);
    qtts_clone_free(&cs);
#ifdef QTTS_WITH_CUDA
    qtts_cuda_free(gpu);
#endif
    w2v2_free(w2v);
    qtts_codec_free(codec);
    qtts_model_free(m);
    bpe_vocab_free(vocab);
    return 0;
}
