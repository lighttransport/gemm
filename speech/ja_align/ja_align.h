/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * ja_align.h - Japanese speech -> timed phonemes/kana, viseme curves and prosody.
 *
 * Pipeline: resample to 16 kHz -> wav2vec2 dual-CTC posteriors (w2v2.h)
 *   -> phoneme timing: alignment-free (greedy CTC spans) or forced alignment to a
 *      caller-supplied kana reading / phoneme string (ctc.h)
 *   -> contiguous phone intervals (onset-to-onset; silence inserted in low-energy gaps)
 *   -> 15-class viseme curves at a requested fps with linear co-articulation ramps
 *   -> 100 Hz RMS energy (dBFS), YIN F0 and aperiodicity (ja_dsp.h)
 * ja_align_write_json documents the output format (see speech/README.md).
 *
 * Define JA_ALIGN_IMPLEMENTATION once; it pulls in the implementations of all ja_* headers.
 */
#ifndef JA_ALIGN_H
#define JA_ALIGN_H

#ifdef JA_ALIGN_IMPLEMENTATION
#define JA_BASE_IMPLEMENTATION
#define JA_W2V2_IMPLEMENTATION
#define JA_CTC_IMPLEMENTATION
#define JA_DSP_IMPLEMENTATION
#endif
#include "ja_base.h"
#include "w2v2.h"
#include "ctc.h"
#include "ja_dsp.h"
#include "ja_phoneme.h"

typedef struct {
    const char *kana;      /* optional kana reading -> forced alignment of phonemes and kana */
    const char *phonemes;  /* optional space-separated phonemes (overrides kana for the phone track) */
    float fps;             /* viseme curve rate (default 30) */
    float ramp;            /* co-articulation ramp half-width in seconds (default 0.035) */
    float sil_gap;         /* minimum low-energy gap turned into silence (default 0.12 s) */
    float sil_db;          /* silence threshold relative to the loudest frame (default -35 dB) */
    float pad;             /* seconds of silence added on both sides before the encoder (default 0.5);
                            * wav2vec2 CTC misses speech that starts right at the buffer edge */
    const char *dump_dir;  /* optional .npy dumps of encoder stages / posteriors */
} ja_align_opts;

typedef struct {
    char sym[16];
    int id;                /* vocabulary index */
    double start, end;     /* seconds */
    float conf;            /* mean posterior over the CTC span */
    int viseme;            /* JA_VIS_* (phones only) */
} ja_unit;

typedef struct {
    double duration;
    int forced;                         /* 1 if a reading was aligned */
    double utt_conf, utt_start, utt_end; /* CTC-segmentation confidence (forced only) */
    ja_unit *phones; int n_phones;      /* raw CTC spans */
    ja_unit *intervals; int n_intervals;/* contiguous phone intervals incl. "sil" */
    ja_unit *kana; int n_kana;
    char phoneme_text[4096], kana_text[4096];
    int T, n_phon_cls;                  /* posterior frames (50 Hz) */
    float *phon_post;                   /* [T][n_phon_cls] probabilities */
    float hop;                          /* prosody hop (0.01 s) */
    int n_prosody;
    float *rms_db, *f0, *aper;
    float fps;
    int n_vis;
    float *visemes;                     /* [n_vis][JA_N_VIS] weights summing to 1 */
} ja_align_result;

void ja_align_opts_default(ja_align_opts *o);
int  ja_align_run(w2v2_model *m, const float *wav, int n, int sr, const ja_align_opts *o, ja_align_result *r);
int  ja_align_write_json(const ja_align_result *r, const char *path);
void ja_align_result_free(ja_align_result *r);

#endif /* JA_ALIGN_H */

#ifdef JA_ALIGN_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JA_FRAME 0.02

void ja_align_opts_default(ja_align_opts *o) {
    memset(o, 0, sizeof(*o));
    o->fps = 30.0f;
    o->ramp = 0.035f;
    o->sil_gap = 0.12f;
    o->sil_db = -35.0f;
    o->pad = 0.5f;
}

static int ja__parse_phonemes(const char *s, int *ids, int cap) {
    char buf[8192];
    snprintf(buf, sizeof(buf), "%s", s);
    int n = 0;
    for (char *t = strtok(buf, " \t\n"); t; t = strtok(NULL, " \t\n")) {
        int id = ja_phoneme_index(t);
        if (id < 0) { fprintf(stderr, "ja_align: unknown phoneme '%s'\n", t); return -1; }
        if (n < cap) ids[n++] = id;
    }
    return n;
}

static double ja__energy_at(const ja_align_result *r, double t) {
    int i = (int)lround(t / r->hop);
    if (i < 0) i = 0;
    if (i >= r->n_prosody) i = r->n_prosody - 1;
    return r->rms_db[i];
}

static void ja__add_interval(ja_align_result *r, int cap, const char *sym, int id, double s, double e,
                             float conf, int vis) {
    if (r->n_intervals >= cap || e <= s) return;
    ja_unit *u = &r->intervals[r->n_intervals++];
    snprintf(u->sym, sizeof(u->sym), "%s", sym);
    u->id = id; u->start = s; u->end = e; u->conf = conf; u->viseme = vis;
}

static void ja__build_intervals(ja_align_result *r, const ja_align_opts *o) {
    float mx = -200.0f;
    for (int i = 0; i < r->n_prosody; i++) if (r->rms_db[i] > mx) mx = r->rms_db[i];
    float thr = mx + o->sil_db;
    int cap = 2 * r->n_phones + 4;
    r->intervals = (ja_unit *)calloc((size_t)cap, sizeof(ja_unit));
    for (int i = 0; i < r->n_phones; i++) {
        const ja_unit *p = &r->phones[i];
        double on = p->start;
        if (i == 0) {
            ja__add_interval(r, cap, "sil", 0, 0.0, on, 1.0f, JA_VIS_SIL);
        } else if (r->n_intervals > 0) {
            /* the previous phone lasts while the signal stays above the silence threshold;
             * a remaining quiet gap of at least sil_gap becomes silence */
            ja_unit *prev = &r->intervals[r->n_intervals - 1];
            double cut = prev->end;
            while (cut < on && ja__energy_at(r, cut) >= thr) cut += r->hop;
            if (on - cut >= o->sil_gap) {
                prev->end = cut;
                ja__add_interval(r, cap, "sil", 0, cut, on, 1.0f, JA_VIS_SIL);
            } else {
                prev->end = on;
            }
        }
        ja__add_interval(r, cap, p->sym, p->id, on, p->end > on ? p->end : on + JA_FRAME, p->conf, p->viseme);
    }
    /* trailing: extend the last phone while voiced, then silence */
    if (r->n_intervals > 0) {
        ja_unit *last = &r->intervals[r->n_intervals - 1];
        double e = last->end;
        while (e < r->duration && e - last->end < 0.25 && ja__energy_at(r, e) >= thr) e += r->hop;
        last->end = e < r->duration ? e : r->duration;
        ja__add_interval(r, cap, "sil", 0, last->end, r->duration, 1.0f, JA_VIS_SIL);
    } else {
        ja__add_interval(r, cap, "sil", 0, 0.0, r->duration, 1.0f, JA_VIS_SIL);
    }
    /* geminate hold `cl`: take the mouth shape of the next consonant */
    for (int i = 0; i < r->n_intervals; i++)
        if (r->intervals[i].viseme < 0)
            r->intervals[i].viseme = i + 1 < r->n_intervals ? r->intervals[i + 1].viseme : JA_VIS_SIL;
}

static void ja__build_visemes(ja_align_result *r, const ja_align_opts *o) {
    r->fps = o->fps;
    r->n_vis = (int)ceil(r->duration * o->fps) + 1;
    r->visemes = (float *)calloc((size_t)r->n_vis * JA_N_VIS, sizeof(float));
    double ramp = o->ramp;
    for (int f = 0; f < r->n_vis; f++) {
        double t = f / (double)o->fps;
        float *w = r->visemes + (size_t)f * JA_N_VIS;
        for (int i = 0; i < r->n_intervals; i++) {
            const ja_unit *u = &r->intervals[i];
            if (t < u->start - ramp || t > u->end + ramp) continue;
            double a = (t - (u->start - ramp)) / (2 * ramp), b = ((u->end + ramp) - t) / (2 * ramp);
            double k = (a < 1 ? a : 1) * (b < 1 ? b : 1);
            if (k <= 0) continue;
            /* devoiced vowels (uppercase) are articulated weakly */
            if (u->sym[0] >= 'A' && u->sym[0] <= 'Z' && u->sym[0] != 'N') k *= 0.5;
            w[u->viseme] += (float)k;
        }
        double s = 0.0;
        for (int v = 0; v < JA_N_VIS; v++) s += w[v];
        if (s <= 1e-6) { w[JA_VIS_SIL] = 1.0f; continue; }
        for (int v = 0; v < JA_N_VIS; v++) w[v] = (float)(w[v] / s);
    }
}

static void ja__unit_from_seg(ja_unit *u, const ctc_seg *s, const char *const *names) {
    snprintf(u->sym, sizeof(u->sym), "%s", names[s->label]);
    u->id = s->label;
    u->start = s->start * JA_FRAME;
    u->end = s->end * JA_FRAME;
    u->conf = s->conf;
    u->viseme = names == ja_phoneme_names ? ja_phoneme_viseme(u->sym) : 0;
}

int ja_align_run(w2v2_model *m, const float *wav, int n, int sr, const ja_align_opts *o, ja_align_result *r) {
    memset(r, 0, sizeof(*r));
    int n16 = 0;
    float *x = ja_resample(wav, n, sr, 16000, &n16);
    r->duration = n16 / 16000.0;
    /* prosody at 100 Hz */
    r->hop = 0.01f;
    r->n_prosody = 1 + (n16 - 1) / 160;
    r->rms_db = (float *)malloc(sizeof(float) * (size_t)r->n_prosody);
    r->f0 = (float *)malloc(sizeof(float) * (size_t)r->n_prosody);
    r->aper = (float *)malloc(sizeof(float) * (size_t)r->n_prosody);
    ja_rms_db(x, n16, 400, 160, r->rms_db, r->n_prosody);
    ja_yin(x, n16, 16000, 640, 160, 60.0f, 600.0f, 0.15f, r->f0, r->aper, r->n_prosody);

    w2v2_output out;
    /* pad whole 20 ms frames of silence on both sides, run, then drop the padded frames */
    int pf = (int)lroundf(o->pad / (float)JA_FRAME), ps = pf * 320;
    float *xp = (float *)calloc((size_t)n16 + 2 * (size_t)ps, sizeof(float));
    memcpy(xp + ps, x, sizeof(float) * (size_t)n16);
    int rc_run = w2v2_run(m, xp, n16 + 2 * ps, &out, o->dump_dir);
    free(xp);
    free(x);
    if (rc_run) return -1;
    if (pf > 0) {
        int keep = (int)ceil(r->duration / JA_FRAME);
        if (keep > out.T - pf) keep = out.T - pf;
        if (keep < 1) keep = 1;
        memmove(out.phon_logp, out.phon_logp + (size_t)pf * out.n_phon, sizeof(float) * (size_t)keep * out.n_phon);
        memmove(out.kana_logp, out.kana_logp + (size_t)pf * out.n_kana, sizeof(float) * (size_t)keep * out.n_kana);
        out.T = keep;
    }
    r->T = out.T;
    r->n_phon_cls = out.n_phon;
    r->phon_post = (float *)malloc(sizeof(float) * (size_t)out.T * out.n_phon);
    for (size_t i = 0; i < (size_t)out.T * out.n_phon; i++) r->phon_post[i] = expf(out.phon_logp[i]);

    int cap = out.T + 8;
    ctc_seg *segs = (ctc_seg *)calloc((size_t)cap, sizeof(ctc_seg));
    int *tgt = (int *)malloc(sizeof(int) * (size_t)cap);
    int *path = (int *)malloc(sizeof(int) * (size_t)out.T);
    int L = -1;
    if (o->phonemes && *o->phonemes) L = ja__parse_phonemes(o->phonemes, tgt, cap);
    else if (o->kana && *o->kana) L = ja_kana_to_phonemes(o->kana, tgt, cap, NULL, 0);
    int ns;
    if (L > 0) {
        float tot;
        if (ctc_forced_align(out.phon_logp, out.T, out.n_phon, tgt, L, path, segs, &tot)) {
            fprintf(stderr, "ja_align: forced alignment failed (reading too long for the audio)\n");
            ns = ctc_greedy_segments(out.phon_logp, out.T, out.n_phon, segs, cap);
        } else {
            ns = L;
            r->forced = 1;
            ctc_segmentation_params sp = { (float)JA_FRAME, 30 };
            float *tim = (float *)malloc(sizeof(float) * (size_t)L);
            float *cp = (float *)malloc(sizeof(float) * (size_t)out.T);
            if (ctc_segmentation(out.phon_logp, out.T, out.n_phon, tgt, L, &sp, tim, cp,
                                 &r->utt_start, &r->utt_end, &r->utt_conf) == 0) {
            }
            free(tim); free(cp);
        }
    } else {
        if (L == 0 || ((o->kana && *o->kana) && L < 0)) fprintf(stderr, "ja_align: unusable reading, using free alignment\n");
        ns = ctc_greedy_segments(out.phon_logp, out.T, out.n_phon, segs, cap);
    }
    r->phones = (ja_unit *)calloc((size_t)(ns > 0 ? ns : 1), sizeof(ja_unit));
    r->n_phones = ns;
    int tl = 0;
    for (int i = 0; i < ns; i++) {
        ja__unit_from_seg(&r->phones[i], &segs[i], ja_phoneme_names);
        tl += snprintf(r->phoneme_text + tl, sizeof(r->phoneme_text) - (size_t)tl, "%s%s", i ? " " : "",
                       r->phones[i].sym);
        if (tl >= (int)sizeof(r->phoneme_text)) tl = (int)sizeof(r->phoneme_text) - 1;
    }
    /* kana track: forced to the reading when given, else greedy */
    int nk = -1;
    if (o->kana && *o->kana) {
        int Lk = ja_kana_to_ids(o->kana, tgt, cap);
        float tot;
        if (Lk > 0 && !ctc_forced_align(out.kana_logp, out.T, out.n_kana, tgt, Lk, path, segs, &tot)) nk = Lk;
    }
    if (nk < 0) nk = ctc_greedy_segments(out.kana_logp, out.T, out.n_kana, segs, cap);
    r->kana = (ja_unit *)calloc((size_t)(nk > 0 ? nk : 1), sizeof(ja_unit));
    r->n_kana = nk;
    tl = 0;
    for (int i = 0; i < nk; i++) {
        ja__unit_from_seg(&r->kana[i], &segs[i], ja_kana_names);
        tl += snprintf(r->kana_text + tl, sizeof(r->kana_text) - (size_t)tl, "%s", r->kana[i].sym);
        if (tl >= (int)sizeof(r->kana_text)) tl = (int)sizeof(r->kana_text) - 1;
    }
    free(segs); free(tgt); free(path);
    w2v2_output_free(&out);
    ja__build_intervals(r, o);
    ja__build_visemes(r, o);
    return 0;
}

static void ja__json_units(FILE *f, const char *key, const ja_unit *u, int n, int with_vis) {
    fprintf(f, "  \"%s\": [", key);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s\n    {\"s\": \"%s\", \"start\": %.3f, \"end\": %.3f, \"conf\": %.4f", i ? "," : "",
                u[i].sym, u[i].start, u[i].end, u[i].conf);
        if (with_vis) fprintf(f, ", \"viseme\": \"%s\"", ja_viseme_names[u[i].viseme]);
        fprintf(f, "}");
    }
    fprintf(f, "\n  ],\n");
}

static void ja__json_floats(FILE *f, const float *x, int n, int stride, int k, const char *fmt) {
    fprintf(f, "[");
    for (int i = 0; i < n; i++) {
        if (i) fprintf(f, ",");
        fprintf(f, fmt, x[(size_t)i * stride + k]);
    }
    fprintf(f, "]");
}

int ja_align_write_json(const ja_align_result *r, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "{\n  \"format\": \"ja_align.v1\",\n  \"duration\": %.4f,\n", r->duration);
    fprintf(f, "  \"mode\": \"%s\",\n", r->forced ? "forced" : "free");
    if (r->forced)
        fprintf(f, "  \"utterance\": {\"start\": %.3f, \"end\": %.3f, \"confidence\": %.5f},\n",
                r->utt_start, r->utt_end, r->utt_conf);
    fprintf(f, "  \"phoneme_text\": \"%s\",\n  \"kana_text\": \"%s\",\n", r->phoneme_text, r->kana_text);
    ja__json_units(f, "phones", r->phones, r->n_phones, 1);
    ja__json_units(f, "intervals", r->intervals, r->n_intervals, 1);
    ja__json_units(f, "kana", r->kana, r->n_kana, 0);
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
    fprintf(f, "\n  },\n  \"posteriors\": {\"frame\": %.3f, \"frames\": %d, \"classes\": %d}\n}\n",
            JA_FRAME, r->T, r->n_phon_cls);
    fclose(f);
    return 0;
}

void ja_align_result_free(ja_align_result *r) {
    free(r->phones); free(r->intervals); free(r->kana); free(r->phon_post);
    free(r->rms_db); free(r->f0); free(r->aper); free(r->visemes);
    memset(r, 0, sizeof(*r));
}

#endif /* JA_ALIGN_IMPLEMENTATION */
