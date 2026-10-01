/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * qwen3_tts: Qwen3-TTS (CustomVoice) text -> 24 kHz wav, CPU backend.
 *
 *   qwen3_tts --model <dir> --text "..." [--speaker Ono_Anna] [--language Japanese]
 *             [--instruct "..."] [--greedy] [--seed N] [--max-frames N] [--streaming]
 *             [--out out.wav] [--dump-dir dir] [--ids input_ids.npy] [--codes-out codes.npy]
 *             [--backend cpu|cuda] [--device N] [--codes-in codes.npy (decode only)]
 *   voice clone (Base model): --ref-wav ref.wav [--ref-text "transcript"] [--xvec-only]
 *             with --ref-text: in-context cloning (reference codes + transcript in the prompt);
 *             without it (or --xvec-only): speaker-embedding-only cloning.
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
#include "npy_io.h"
#include "qtts_ops.h"
#include "qtts_codec.h"
#include "qtts_codec_stream.h"
#include "qtts_talker.h"
#include "qtts_spk.h"
#include "qtts_codec_enc.h"
#include "qtts_tokenizer.h"
#include "wav_io.h"
#include "qtts_clone.h"
#ifdef QTTS_WITH_CUDA
#define QTTS_CUDA_IMPLEMENTATION
#include "qtts_cuda.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <signal.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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

static void save_npy_f32(const char *dir, const char *name, const float *d, int rows, int cols) {
    char p[1024];
    snprintf(p, sizeof(p), "%s/%s.npy", dir, name);
    int dims[2] = { rows, cols };
    qt_npy_save_f32(p, d, cols > 0 ? 2 : 1, dims);
}

static void on_frame(void *user, const int32_t *c, int frame) {
    (void)c;
    int *verbose = (int *)user;
    if (*verbose && frame % 25 == 0) { fprintf(stderr, "."); fflush(stderr); }
}

/* Native-endian corpus stream. Header: VHFEAT1\0, int32 hidden size;
 * records: int64 sample start, int32 codes[16], float32 hidden[H]. */
static volatile sig_atomic_t cancel_requested = 0;
static void request_cancel(int signal_number) { (void)signal_number; cancel_requested = 1; }
typedef struct {
    FILE *features, *pcm;
    qtts_codec_stream *decoder;
    double start, codec_seconds, codec_max_seconds;
    int emitted, headers, codec_frames;
} feature_sink;
static void feature_report(const feature_sink *sink) {
    if (sink->codec_frames) fprintf(stderr, "stream codec: frames=%d total_ms=%.3f mean_ms=%.3f max_ms=%.3f\n",
        sink->codec_frames, sink->codec_seconds*1000, sink->codec_seconds*1000/sink->codec_frames, sink->codec_max_seconds*1000);
}
static int feature_headers(feature_sink *sink, int H) {
    if (sink->headers) return 0;
    int32_t rate = 24000;
    if (sink->features && (fwrite("VHFEAT1\0", 1, 8, sink->features) != 8 || fwrite(&H, 4, 1, sink->features) != 1)) return 1;
    if (sink->pcm && (fwrite("VHPCM1\0\0", 1, 8, sink->pcm) != 8 || fwrite(&rate, 4, 1, sink->pcm) != 1)) return 1;
    sink->headers = 1;
    return 0;
}
static int on_feature(void *user, const int32_t *codes, const float *hidden, int H, int64_t start) {
    feature_sink *sink = user;
    if (feature_headers(sink, H)) return 1;
    FILE *f = sink->features;
    if (f) {
        if (fwrite(&start, 8, 1, f) != 1 || fwrite(codes, 4, 16, f) != 16 ||
            fwrite(hidden, 4, (size_t)H, f) != (size_t)H || fflush(f)) return 1;
    }
    if (sink->pcm) {
        double codec_start = now_s();
        float *pcm = qtts_codec_stream_push(sink->decoder, codes);
        double elapsed = now_s()-codec_start;
        sink->codec_seconds += elapsed;
        if (elapsed > sink->codec_max_seconds) sink->codec_max_seconds = elapsed;
        sink->codec_frames++;
        if (!pcm) return 1;
        int32_t count = 1920;
        int failed = fwrite(&start, 8, 1, sink->pcm) != 1 || fwrite(&count, 4, 1, sink->pcm) != 1 ||
            fwrite(pcm, 4, 1920, sink->pcm) != 1920 || fflush(sink->pcm);
        free(pcm);
        if (failed) return 1;
        if (!sink->emitted) fprintf(stderr, "stream first PCM: %.2f ms after generation start (excludes model load)\n", (now_s() - sink->start) * 1000);
    }
    sink->emitted = 1;
    return cancel_requested ? 1 : 0;
}

/* Persistent local stdin worker: one JSON {"text":"..."} request per line.
 * Files/pipes receive ordinary headers per utterance and explicit end markers.
 * SIGUSR1 cancels at a codec-frame boundary; state is reset for the next request. */
static int serve_requests(qtts_model *m, const qtts_backend *be, const bpe_vocab *vocab, qtts_codec *codec,
                          const char *speaker, const char *language, const qtts_gen_params *params,
                          const char *features_path) {
    FILE *features = fopen(features_path, "wb");
    if (!features) return 1;
    struct sigaction action;
    memset(&action, 0, sizeof(action)); action.sa_handler = request_cancel; action.sa_flags = SA_RESTART;
    sigaction(SIGUSR1, &action, NULL);
    if (fwrite("VHTTSRDY", 1, 8, stdout) != 8 || fflush(stdout)) { fclose(features); return 1; }
    char *line = NULL; size_t capacity = 0; int status = 0;
    while (getline(&line, &capacity, stdin) > 0) {
        if (strlen(line) > 16384) { status = 1; break; }
        json_val *request = json_parse(line, (int)strlen(line));
        const json_val *value = request ? json_obj_get(request, "text") : NULL;
        if (!value || value->type != JSON_STRING || value->str.len < 1 || value->str.len > 4096 ||
            memchr(value->str.ptr, '\0', (size_t)value->str.len)) {
            json_free(request); status = 1; break;
        }
        int count = 0;
        int32_t *ids = tokenize(vocab, "<|im_start|>assistant\n%s<|im_end|>\n<|im_start|>assistant\n", value->str.ptr, &count);
        json_free(request);
        if (count + params->max_frames + 32 > 2400) { free(ids); status = 1; break; }
        cancel_requested = 0;
        qtts_codec_stream *decoder = qtts_codec_stream_create(codec);
        feature_sink sink = { .features=features, .pcm=stdout, .decoder=decoder, .start=now_s() };
        if (!decoder || feature_headers(&sink, qtts_model_hidden(m))) { qtts_codec_stream_free(decoder); free(ids); status = 1; break; }
        qtts_gen_result result;
        int rc = qtts_generate_ex(m, be, ids, count, NULL, 0, speaker, language, params, 0, &result,
                                  NULL, NULL, on_feature, &sink);
        feature_report(&sink);
        free(ids); qtts_codec_stream_free(decoder); qtts_gen_result_free(&result);
        if ((rc && !cancel_requested) || ferror(features) || ferror(stdout)) { status = 1; break; }
        int64_t end = -1; int32_t zero = 0;
        int H = qtts_model_hidden(m);
        float *zeros = calloc((size_t)H + 16, sizeof(float));
        int failed = !zeros || fwrite(&end, 8, 1, features) != 1 || fwrite(zeros, 4, (size_t)H + 16, features) != (size_t)H + 16 ||
                     fflush(features) || fwrite(&end, 8, 1, stdout) != 1 || fwrite(&zero, 4, 1, stdout) != 1 || fflush(stdout);
        free(zeros);
        if (failed) { status = 1; break; }
    }
    free(line); fclose(features);
    return status;
}

int main(int argc, char **argv) {
    const char *model = "/mnt/nvme01/models/speech/Qwen3-TTS-12Hz-1.7B-CustomVoice";
    const char *text = "今日はいい天気ですね。", *speaker = "Ono_Anna", *language = "Japanese";
    const char *instruct = "", *out_wav = "out.wav", *dump = NULL, *ids_npy = NULL, *codes_out = NULL;
    qtts_gen_params gp;
    qtts_gen_params_default(&gp);
    int verbose = 1, use_cuda = 0, device = 0;
    int threads = 8;
    const char *codes_in = NULL, *ref_wav = NULL, *ref_text = NULL;
    int xvec_only = 0, streaming_set = 0;
    int serve_stdin = 0;
    const char *features_out = NULL, *pcm_out = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--text") && v) { text = v; i++; }
        else if (!strcmp(a, "--speaker") && v) { speaker = v; i++; }
        else if (!strcmp(a, "--language") && v) { language = v; i++; }
        else if (!strcmp(a, "--instruct") && v) { instruct = v; i++; }
        else if (!strcmp(a, "--out") && v) { out_wav = v; i++; }
        else if (!strcmp(a, "--dump-dir") && v) { dump = v; i++; }
        else if (!strcmp(a, "--ids") && v) { ids_npy = v; i++; }
        else if (!strcmp(a, "--codes-out") && v) { codes_out = v; i++; }
        else if (!strcmp(a, "--features-out") && v) { features_out = v; i++; }
        else if (!strcmp(a, "--threads") && v) { threads = atoi(v); i++; }
        else if (!strcmp(a, "--serve-stdin")) { serve_stdin = 1; }
        else if (!strcmp(a, "--pcm-out") && v) { pcm_out = v; i++; }
        else if (!strcmp(a, "--seed") && v) { gp.seed = strtoull(v, NULL, 10); i++; }
        else if (!strcmp(a, "--max-frames") && v) { gp.max_frames = atoi(v); i++; }
        else if (!strcmp(a, "--temperature") && v) { gp.temperature = gp.sub_temperature = (float)atof(v); i++; }
        else if (!strcmp(a, "--top-k") && v) { gp.top_k = gp.sub_top_k = atoi(v); i++; }
        else if (!strcmp(a, "--backend") && v) { use_cuda = !strcmp(v, "cuda"); i++; }
        else if (!strcmp(a, "--device") && v) { device = atoi(v); i++; }
        else if (!strcmp(a, "--codes-in") && v) { codes_in = v; i++; }
        else if (!strcmp(a, "--greedy")) gp.greedy = 1;
        else if (!strcmp(a, "--streaming")) { gp.streaming = 1; streaming_set = 1; }
        else if (!strcmp(a, "--non-streaming")) { gp.streaming = 0; streaming_set = 1; }
        else if (!strcmp(a, "--ref-wav") && v) { ref_wav = v; i++; }
        else if (!strcmp(a, "--ref-text") && v) { ref_text = v; i++; }
        else if (!strcmp(a, "--xvec-only")) xvec_only = 1;
        else if (!strcmp(a, "--quiet")) verbose = 0;
        else { fprintf(stderr, "unknown or incomplete option %s\n", a); return 1; }
    }
    if (dump) mkdir(dump, 0755);
    if (serve_stdin && (!pcm_out || strcmp(pcm_out, "-") || !features_out || codes_in || ref_wav || dump || ids_npy || *instruct)) {
        fprintf(stderr, "--serve-stdin requires --pcm-out - and --features-out pipe, no clone/fixtures/instruct\n"); return 1;
    }
    double t0 = now_s();
    bpe_vocab *vocab = qtts_tokenizer_load(model);
    if (!vocab) return 1;
    int n_ids = 0, n_inst = 0;
    int32_t *ids = NULL, *inst = NULL;
    if (ids_npy) {
        int nd = 0, dims[8], f32 = 0;
        ids = (int32_t *)npy_load(ids_npy, &nd, dims, &f32);
        if (!ids || f32) { fprintf(stderr, "bad --ids file\n"); return 1; }
        n_ids = dims[0];
    } else {
        ids = tokenize(vocab, "<|im_start|>assistant\n%s<|im_end|>\n<|im_start|>assistant\n", text, &n_ids);
    }
    if (*instruct) inst = tokenize(vocab, "<|im_start|>user\n%s<|im_end|>\n", instruct, &n_inst);

    if (threads < 1 || threads > 256) { fprintf(stderr, "--threads must be1..256\n"); return 1; }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    qtts_model *m = qtts_model_load(model, 2400);
    char tokdir[1024];
    snprintf(tokdir, sizeof(tokdir), "%s/speech_tokenizer", model);
    qtts_codec *codec = qtts_codec_load(tokdir);
    if (!m || !codec) return 1;

    /* voice clone: speaker embedding (+ reference codes and transcript for ICL) */
    qtts_clone_state cs;
    memset(&cs, 0, sizeof(cs));
    if (ref_wav) {
        if (qtts_clone_prepare(&cs, model, vocab, ref_wav, ref_text, xvec_only, dump)) return 1;
        gp.clone = &cs.vc;
        if (!streaming_set) gp.streaming = 1;   /* generate_voice_clone defaults to the streaming text feed */
        speaker = "";
    }
    const qtts_backend *be = NULL;
#ifdef QTTS_WITH_CUDA
    qtts_cuda *gpu = NULL;
    qtts_backend gbe;
    if (use_cuda) {
        gpu = qtts_cuda_create(m, pcm_out ? NULL : codec, device, verbose);
        if (!gpu) { fprintf(stderr, "CUDA backend unavailable\n"); return 1; }
        gbe = qtts_cuda_backend(gpu);
        be = &gbe;
    }
#else
    (void)device;
    if (use_cuda) { fprintf(stderr, "built without CUDA (make cuda)\n"); return 1; }
#endif
    double t1 = now_s();
    if (verbose) fprintf(stderr, "loaded in %.2fs; %d text ids; backend %s\n", t1 - t0, n_ids, use_cuda ? "cuda" : "cpu");
    if (serve_stdin) {
        int rc = serve_requests(m, be, vocab, codec, speaker, language, &gp, features_out);
        free(ids); free(inst);
#ifdef QTTS_WITH_CUDA
        qtts_cuda_free(gpu);
#endif
        qtts_codec_free(codec); qtts_model_free(m); bpe_vocab_free(vocab);
        return rc;
    }

    qtts_gen_result res;
    memset(&res, 0, sizeof(res));
    FILE *features = features_out ? fopen(features_out, "wb") : NULL;
    if ((features_out && !features) || ((features_out || pcm_out) && (codes_in || ref_wav))) {
        fprintf(stderr, "--features-out requires CustomVoice generation and a writable path\n");
        if (features) fclose(features);
        return 1;
    }
    FILE *pcm_file = pcm_out ? (!strcmp(pcm_out, "-") ? stdout : fopen(pcm_out, "wb")) : NULL;
    qtts_codec_stream *stream_decoder = pcm_out ? qtts_codec_stream_create(codec) : NULL;
    if (pcm_out && (!pcm_file || !stream_decoder)) {
        fprintf(stderr, "cannot initialize stateful PCM output\n");
        if (features) fclose(features);
        if (pcm_file && pcm_file != stdout) fclose(pcm_file);
        qtts_codec_stream_free(stream_decoder);
        return 1;
    }
    feature_sink sink = { .features=features, .pcm=pcm_file, .decoder=stream_decoder, .start=now_s() };
    if (codes_in) {  /* decode-only: codes from an .npy */
        int nd = 0, dims[8], f32 = 0;
        res.codes = (int32_t *)npy_load(codes_in, &nd, dims, &f32);
        if (!res.codes || nd != 2) { fprintf(stderr, "bad --codes-in\n"); return 1; }
        res.n_frames = dims[0];
    } else {
        int rc = qtts_generate_ex(m, be, ids, n_ids, inst, n_inst, speaker, language, &gp, dump != NULL, &res,
                                  on_frame, &verbose, (features || pcm_file) ? on_feature : NULL, &sink);
        feature_report(&sink);
        int io_error = features ? ferror(features) : 0;
        if (features && fclose(features)) io_error = 1;
        if (pcm_file && pcm_file != stdout && fclose(pcm_file)) io_error = 1;
        qtts_codec_stream_free(stream_decoder);
        if (rc || io_error) { qtts_gen_result_free(&res); return 1; }
    }
    double t2 = now_s();
    int n = 0;
    float *wav;
    /* ICL: decode [reference codes + generated codes], then cut the reference part */
    int dec_T = 0;
    int32_t *dec_codes = qtts_clone_codes(&cs, res.codes, res.n_frames, &dec_T);
#ifdef QTTS_WITH_CUDA
    if (use_cuda && !pcm_out) wav = qtts_cuda_decode(gpu, dec_codes, dec_T, &n, dump);
    else
#endif
    wav = qtts_codec_decode(codec, dec_codes, dec_T, &n, dump);
    n = qtts_clone_trim(&cs, wav, n, dec_T);
    if (dec_codes != res.codes) free(dec_codes);
    if (!wav) { fprintf(stderr, "decode failed\n"); return 1; }
    double t3 = now_s();
    wav_write_pcm16(out_wav, wav, n, qtts_codec_sample_rate(codec));
    double dur = n / (double)qtts_codec_sample_rate(codec);
    fprintf(stderr, "\nframes=%d audio=%.2fs talker=%.2fs (RTF %.2f) codec=%.2fs (RTF %.2f) -> %s\n",
            res.n_frames, dur, t2 - t1, (t2 - t1) / (dur > 0 ? dur : 1), t3 - t2, (t3 - t2) / (dur > 0 ? dur : 1), out_wav);
    if (dump) {
        mkdir(dump, 0755);
        char p[1024];
        snprintf(p, sizeof(p), "%s/input_ids.npy", dump);
        qt_npy_save_i32(p, ids, 1, &n_ids);
        int H = qtts_model_hidden(m), V = qtts_model_codec_vocab(m);
        if (res.prefill_embeds) {
            save_npy_f32(dump, "prefill_embeds", res.prefill_embeds, res.prefill_len, H);
            save_npy_f32(dump, "step_logits", res.step_logits, res.n_steps, V);
            save_npy_f32(dump, "talker_hidden", res.step_hidden, res.n_steps, H);
        }
        snprintf(p, sizeof(p), "%s/codes.npy", dump);
        int cd[2] = { res.n_frames, 16 };
        qt_npy_save_i32(p, res.codes, 2, cd);
        snprintf(p, sizeof(p), "%s/wav.npy", dump);
        qt_npy_save_f32(p, wav, 1, &n);
    }
    if (codes_out) { int cd[2] = { res.n_frames, 16 }; qt_npy_save_i32(codes_out, res.codes, 2, cd); }
    free(wav); free(ids); free(inst); qtts_clone_free(&cs);
    qtts_gen_result_free(&res);
#ifdef QTTS_WITH_CUDA
    qtts_cuda_free(gpu);
#endif
    qtts_codec_free(codec);
    qtts_model_free(m);
    bpe_vocab_free(vocab);
    return 0;
}
