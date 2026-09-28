/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * qtts_clone.h - voice-clone setup and output assembly shared by the CLIs (Base model).
 *
 *   qtts_clone_prepare  reference wav (any rate, resampled to 24 kHz) -> x-vector, and for ICL
 *                       (ref_text given, xvec_only == 0) the reference codes + transcript ids
 *   qtts_clone_codes    [reference codes + generated codes] for decoding (ICL)
 *   qtts_clone_trim     drop the reference part of the decoded waveform, like the reference
 *                       wrapper: cut = n_ref / (n_ref + n_gen) * n_samples
 *
 * Header-only static functions; include after the qtts_* implementations and bpe_tokenizer.h.
 */
#ifndef QTTS_CLONE_H
#define QTTS_CLONE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    qtts_voice_clone vc;
    float *spk_emb;
    int32_t *ref_codes, *ref_ids;
} qtts_clone_state;

static int qtts_clone_prepare(qtts_clone_state *cs, const char *model_dir, const bpe_vocab *vocab,
                              const char *ref_wav, const char *ref_text, int xvec_only, const char *dump) {
    memset(cs, 0, sizeof(*cs));
    int rn = 0, rsr = 0;
    float *rw = wav_read(ref_wav, &rn, &rsr);
    if (!rw) { fprintf(stderr, "cannot read %s\n", ref_wav); return -1; }
    if (rsr != 24000) {
        int n24 = 0;
        float *r24 = wav_resample(rw, rn, rsr, 24000, &n24);
        free(rw); rw = r24; rn = n24;
    }
    qtts_spk *spk = qtts_spk_load(model_dir);
    if (!spk) { free(rw); return -1; }
    int D = qtts_spk_dim(spk);
    cs->spk_emb = (float *)malloc(sizeof(float) * (size_t)D);
    float *mel = NULL;
    int mel_T = 0;
    if (qtts_spk_embed(spk, rw, rn, cs->spk_emb, dump ? &mel : NULL, &mel_T)) {
        fprintf(stderr, "speaker embedding failed\n"); qtts_spk_free(spk); free(rw); return -1;
    }
    qtts_spk_free(spk);
    if (dump) {
        char p[1024];
        int md[2] = { mel_T, 128 };
        snprintf(p, sizeof(p), "%s/spk_mel.npy", dump); qt_npy_save_f32(p, mel, 2, md);
        snprintf(p, sizeof(p), "%s/spk_emb.npy", dump); qt_npy_save_f32(p, cs->spk_emb, 1, &D);
        free(mel);
    }
    cs->vc.spk_emb = cs->spk_emb;
    if (ref_text && *ref_text && !xvec_only) {
        char tokdir[1024];
        snprintf(tokdir, sizeof(tokdir), "%s/speech_tokenizer", model_dir);
        qtts_cenc *enc = qtts_cenc_load(tokdir);
        if (!enc) { fprintf(stderr, "cannot load the speech tokenizer encoder\n"); free(rw); return -1; }
        cs->ref_codes = qtts_cenc_encode(enc, rw, rn, &cs->vc.n_ref, dump);
        qtts_cenc_free(enc);
        const char *fmt = "<|im_start|>assistant\n%s<|im_end|>\n";
        size_t len = strlen(fmt) + strlen(ref_text) + 8;
        char *s = (char *)malloc(len);
        snprintf(s, len, fmt, ref_text);
        int cnt = bpe_tokenize(vocab, s, -1, NULL, 0);
        cs->ref_ids = (int32_t *)malloc(sizeof(int32_t) * (size_t)(cnt > 0 ? cnt : 1));
        cs->vc.n_ref_ids = bpe_tokenize(vocab, s, -1, cs->ref_ids, cnt);
        free(s);
        cs->vc.ref_codes = cs->ref_codes;
        cs->vc.ref_ids = cs->ref_ids;
        if (dump) {
            char p[1024];
            int cd[2] = { cs->vc.n_ref, 16 };
            snprintf(p, sizeof(p), "%s/ref_codes.npy", dump); qt_npy_save_i32(p, cs->ref_codes, 2, cd);
            snprintf(p, sizeof(p), "%s/ref_ids.npy", dump); qt_npy_save_i32(p, cs->ref_ids, 1, &cs->vc.n_ref_ids);
        }
    }
    free(rw);
    return 0;
}

/* returns codes to decode (malloc'd when ICL prepends the reference) and their frame count */
static int32_t *qtts_clone_codes(const qtts_clone_state *cs, int32_t *gen, int n_gen, int *T) {
    int nr = cs ? cs->vc.n_ref : 0;
    *T = nr + n_gen;
    if (!nr) return gen;
    int32_t *c = (int32_t *)malloc(sizeof(int32_t) * (size_t)(*T) * 16);
    memcpy(c, cs->ref_codes, sizeof(int32_t) * (size_t)nr * 16);
    memcpy(c + (size_t)nr * 16, gen, sizeof(int32_t) * (size_t)n_gen * 16);
    return c;
}

static int qtts_clone_trim(const qtts_clone_state *cs, float *wav, int n, int T) {
    int nr = cs ? cs->vc.n_ref : 0;
    if (!nr || !wav) return n;
    int cut = (int)((double)nr / (T > 0 ? T : 1) * n);
    memmove(wav, wav + cut, sizeof(float) * (size_t)(n - cut));
    return n - cut;
}

static void qtts_clone_free(qtts_clone_state *cs) {
    free(cs->spk_emb); free(cs->ref_codes); free(cs->ref_ids);
    memset(cs, 0, sizeof(*cs));
}

#endif /* QTTS_CLONE_H */
