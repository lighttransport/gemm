/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * en_align.h - English speech -> timed IPA phones, word spans, viseme curves and prosody.
 *
 * The same pipeline as ja_align.h with the English CTC model (w2v2.h, single final-layer
 * head over IPA tokens): resample -> posteriors -> free (greedy) or forced alignment to the
 * text's pronunciation (CMUdict + letter-to-sound fallback, en_phoneme.h) -> intervals ->
 * 15-class viseme curves -> prosody. Output: "en_align.v1" (the ja_align.v1 layout with
 * `words` instead of `kana`). Include after ja_align.h with JA_ALIGN_IMPLEMENTATION.
 */
#ifndef EN_ALIGN_H
#define EN_ALIGN_H

#include "../ja_align/ja_align.h"
#include "en_phoneme.h"

typedef struct {
    const char *text;      /* optional transcript -> forced alignment through G2P */
    const char *phones;    /* optional space-separated IPA tokens (overrides text) */
    const char *dict_path; /* CMU dictionary file; NULL or missing: letter-to-sound only */
    float fps, ramp, sil_gap, sil_db, pad;
    const char *dump_dir;
} en_align_opts;

typedef struct {
    ja_align_result r;      /* phones, intervals, visemes, prosody (kana unused) */
    ja_unit *words; int n_words;
    char text[4096];
    int unknown_words;
} en_align_result;

static void en_align_opts_default(en_align_opts *o) {
    ja_align_opts j; ja_align_opts_default(&j);
    memset(o, 0, sizeof(*o));
    o->fps = j.fps; o->ramp = j.ramp; o->sil_gap = j.sil_gap; o->sil_db = j.sil_db; o->pad = j.pad;
}

static int en_align_run_ex(ja_encoder_fn enc, void *enc_ctx, const en_vocab *vocab, const float *wav, int n, int sr,
                           const en_align_opts *o, en_align_result *er) {
    memset(er, 0, sizeof(*er));
    ja_align_result *r = &er->r;
    ja_align_opts jo; ja_align_opts_default(&jo);
    jo.fps = o->fps; jo.ramp = o->ramp; jo.sil_gap = o->sil_gap; jo.sil_db = o->sil_db; jo.pad = o->pad; jo.dump_dir = o->dump_dir;
    int n16 = 0;
    float *x = ja_resample(wav, n, sr, 16000, &n16);
    r->duration = n16 / 16000.0;
    r->hop = 0.01f;
    r->n_prosody = 1 + (n16 - 1) / 160;
    r->rms_db = (float *)malloc(sizeof(float) * (size_t)r->n_prosody);
    r->f0 = (float *)malloc(sizeof(float) * (size_t)r->n_prosody);
    r->aper = (float *)malloc(sizeof(float) * (size_t)r->n_prosody);
    ja_rms_db(x, n16, 400, 160, r->rms_db, r->n_prosody);
    ja_yin(x, n16, 16000, 640, 160, 60.0f, 600.0f, 0.15f, r->f0, r->aper, r->n_prosody);
    w2v2_output out;
    int pf = (int)lroundf(o->pad / (float)JA_FRAME), ps = pf * 320;
    float *xp = (float *)calloc((size_t)n16 + 2 * (size_t)ps, sizeof(float));
    memcpy(xp + ps, x, sizeof(float) * (size_t)n16);
    int rc_run = enc(enc_ctx, xp, n16 + 2 * ps, &out, o->dump_dir);
    free(xp); free(x);
    if (rc_run) return -1;
    if (pf > 0) {
        int keep = (int)ceil(r->duration / JA_FRAME);
        if (keep > out.T - pf) keep = out.T - pf;
        if (keep < 1) keep = 1;
        memmove(out.phon_logp, out.phon_logp + (size_t)pf * out.n_phon, sizeof(float) * (size_t)keep * out.n_phon);
        out.T = keep;
    }
    if (out.n_phon != vocab->n) { fprintf(stderr, "en_align: vocabulary (%d) does not match the model head (%d)\n", vocab->n, out.n_phon); w2v2_output_free(&out); return -1; }
    r->T = out.T; r->n_phon_cls = out.n_phon;
    r->phon_post = (float *)malloc(sizeof(float) * (size_t)out.T * out.n_phon);
    for (size_t i = 0; i < (size_t)out.T * out.n_phon; i++) r->phon_post[i] = expf(out.phon_logp[i]);
    int cap = out.T + 8;
    ctc_seg *segs = (ctc_seg *)calloc((size_t)cap, sizeof(ctc_seg));
    int *tgt = (int *)malloc(sizeof(int) * (size_t)cap), *word_of = (int *)malloc(sizeof(int) * (size_t)cap);
    int *path = (int *)malloc(sizeof(int) * (size_t)out.T);
    int L = -1, nwords = 0;
    en_dict dict; memset(&dict, 0, sizeof(dict));
    if (o->phones && *o->phones) {
        char buf[8192]; snprintf(buf, sizeof(buf), "%s", o->phones);
        L = 0;
        for (char *t = strtok(buf, " \t\n"); t; t = strtok(NULL, " \t\n")) {
            int id = en_vocab_index(vocab, t);
            if (id < 0) { fprintf(stderr, "en_align: unknown token '%s'\n", t); L = -1; break; }
            if (L < cap) { word_of[L] = L; tgt[L++] = id; }
        }
        nwords = L > 0 ? L : 0;
    } else if (o->text && *o->text) {
        if (o->dict_path && en_dict_load(&dict, o->dict_path)) fprintf(stderr, "en_align: no dictionary at %s; letter-to-sound only\n", o->dict_path);
        L = en_text_to_ids(vocab, &dict, o->text, tgt, word_of, cap, &nwords, &er->unknown_words);
        snprintf(er->text, sizeof(er->text), "%s", o->text);
    }
    int ns;
    if (L > 0) {
        float tot;
        if (ctc_forced_align(out.phon_logp, out.T, out.n_phon, tgt, L, path, segs, &tot)) {
            fprintf(stderr, "en_align: forced alignment failed (text too long for the audio)\n");
            ns = ctc_greedy_segments(out.phon_logp, out.T, out.n_phon, segs, cap);
            L = -1;
        } else {
            ns = L; r->forced = 1;
            ctc_segmentation_params sp = { (float)JA_FRAME, 30 };
            float *tim = (float *)malloc(sizeof(float) * (size_t)L), *cp = (float *)malloc(sizeof(float) * (size_t)out.T);
            ctc_segmentation(out.phon_logp, out.T, out.n_phon, tgt, L, &sp, tim, cp, &r->utt_start, &r->utt_end, &r->utt_conf);
            free(tim); free(cp);
        }
    } else {
        if (L == 0) fprintf(stderr, "en_align: no pronounceable words, using free alignment\n");
        ns = ctc_greedy_segments(out.phon_logp, out.T, out.n_phon, segs, cap);
    }
    r->phones = (ja_unit *)calloc((size_t)(ns > 0 ? ns : 1), sizeof(ja_unit));
    r->n_phones = ns;
    int tl = 0;
    for (int i = 0; i < ns; i++) {
        ja_unit *u = &r->phones[i];
        snprintf(u->sym, sizeof(u->sym), "%s", vocab->tok[segs[i].label]);
        u->id = segs[i].label; u->start = segs[i].start * JA_FRAME; u->end = segs[i].end * JA_FRAME;
        u->conf = segs[i].conf; u->viseme = en_viseme(u->sym);
        tl += snprintf(r->phoneme_text + tl, sizeof(r->phoneme_text) - (size_t)tl, "%s%s", i ? " " : "", u->sym);
        if (tl >= (int)sizeof(r->phoneme_text)) tl = (int)sizeof(r->phoneme_text) - 1;
    }
    /* word spans from the forced phone spans */
    if (r->forced && nwords > 0) {
        er->words = (ja_unit *)calloc((size_t)nwords, sizeof(ja_unit));
        er->n_words = nwords;
        for (int w = 0; w < nwords; w++) { er->words[w].start = -1; er->words[w].id = w; }
        for (int i = 0; i < ns; i++) {
            ja_unit *u = &er->words[word_of[i]];
            if (u->start < 0) u->start = r->phones[i].start;
            u->end = r->phones[i].end;
            u->conf += r->phones[i].conf;
            u->id++;                                      /* phone count + word index, normalised below */
        }
        for (int w = 0; w < nwords; w++) { int cnt = er->words[w].id - w; er->words[w].conf = cnt ? er->words[w].conf / cnt : 0; er->words[w].id = w; }
        /* word texts */
        int w = 0; const char *p = er->text;
        while (*p && w < nwords) {
            while (*p && !(isalpha((unsigned char)*p) || (unsigned char)*p >= 0x80 || *p == '\'')) p++;
            const char *s = p;
            while (*p && (isalpha((unsigned char)*p) || (unsigned char)*p >= 0x80 || *p == '\'')) p++;
            if (p > s) { snprintf(er->words[w].sym, sizeof(er->words[w].sym), "%.*s", (int)(p - s), s); w++; }
        }
    }
    r->kana = (ja_unit *)calloc(1, sizeof(ja_unit)); r->n_kana = 0;
    en_dict_free(&dict);
    free(segs); free(tgt); free(word_of); free(path);
    w2v2_output_free(&out);
    ja__build_intervals(r, &jo);
    ja__build_visemes(r, &jo);
    return 0;
}

static int en_align_write_json(const en_align_result *er, const char *path) {
    const ja_align_result *r = &er->r;
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "{\n  \"format\": \"en_align.v1\",\n  \"duration\": %.4f,\n", r->duration);
    fprintf(f, "  \"mode\": \"%s\",\n", r->forced ? "forced" : "free");
    if (r->forced)
        fprintf(f, "  \"utterance\": {\"start\": %.3f, \"end\": %.3f, \"confidence\": %.5f},\n", r->utt_start, r->utt_end, r->utt_conf);
    fprintf(f, "  \"phoneme_text\": \"%s\",\n  \"unknown_words\": %d,\n", r->phoneme_text, er->unknown_words);
    ja__json_units(f, "phones", r->phones, r->n_phones, 1);
    ja__json_units(f, "intervals", r->intervals, r->n_intervals, 1);
    ja__json_units(f, "words", er->words, er->n_words, 0);
    fprintf(f, "  \"visemes\": {\n    \"fps\": %.3f,\n    \"names\": [", r->fps);
    for (int v = 0; v < JA_N_VIS; v++) fprintf(f, "%s\"%s\"", v ? ", " : "", ja_viseme_names[v]);
    fprintf(f, "],\n    \"frames\": [");
    for (int i = 0; i < r->n_vis; i++) {
        fprintf(f, "%s\n      [", i ? "," : "");
        for (int v = 0; v < JA_N_VIS; v++) fprintf(f, "%s%.3f", v ? "," : "", r->visemes[(size_t)i * JA_N_VIS + v]);
        fprintf(f, "]");
    }
    fprintf(f, "\n    ]\n  },\n");
    fprintf(f, "  \"prosody\": {\n    \"hop\": %.3f,\n    \"rms_db\": ", r->hop);
    ja__json_floats(f, r->rms_db, r->n_prosody, 1, 0, "%.1f");
    fprintf(f, ",\n    \"f0_hz\": ");
    ja__json_floats(f, r->f0, r->n_prosody, 1, 0, "%.1f");
    fprintf(f, ",\n    \"aperiodicity\": ");
    ja__json_floats(f, r->aper, r->n_prosody, 1, 0, "%.3f");
    fprintf(f, "\n  },\n  \"posteriors\": {\"frame\": %.3f, \"frames\": %d, \"classes\": %d}\n}\n", JA_FRAME, r->T, r->n_phon_cls);
    fclose(f);
    return 0;
}

static void en_align_result_free(en_align_result *er) {
    free(er->words);
    ja_align_result_free(&er->r);
    memset(er, 0, sizeof(*er));
}

#endif /* EN_ALIGN_H */
