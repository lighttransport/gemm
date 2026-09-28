/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * ctc.h - CTC decoding / alignment over per-frame log posteriors [T][V] (blank = 0).
 *
 *   ctc_greedy_segments  best-path decode with per-token frame spans and confidence.
 *   ctc_forced_align     Viterbi forced alignment of a known label sequence, written
 *                        from the CTC trellis definition (Graves et al., 2006):
 *                        states = blank-interleaved targets, transitions stay / +1 /
 *                        +2 (skip a blank between distinct labels).
 *   ctc_segmentation     CTC-segmentation (Kürzinger et al., 2020) following the
 *                        Apache-2.0 reference `ctc-segmentation` package (single
 *                        window, T <= 8000 frames): per-token timings, per-frame
 *                        char_probs and the min-mean-over-L utterance confidence.
 */
#ifndef JA_CTC_H
#define JA_CTC_H

typedef struct {
    int label;      /* vocabulary index (1..V-1) */
    int start, end; /* frame span [start, end) */
    float conf;     /* mean posterior probability of `label` over the span */
    float score;    /* sum of log posteriors of the label over the span */
} ctc_seg;

/* Returns the number of segments (<= cap). */
int ctc_greedy_segments(const float *logp, int T, int V, ctc_seg *out, int cap);

/* Viterbi alignment. path[T] receives the label per frame (0 = blank); segs[L] one span
 * per target. Returns 0, or -1 if the labels cannot fit in T frames. */
int ctc_forced_align(const float *logp, int T, int V, const int *tgt, int L,
                     int *path, ctc_seg *segs, float *total_logp);

typedef struct {
    float index_duration;  /* seconds per frame (0.02) */
    int score_min_mean_over_L; /* 30 */
} ctc_segmentation_params;

/* timings[L]: start time (s) of each target token; char_probs[T]; *conf: utterance
 * confidence (min mean log prob over windows of L frames); start_s, end_s: utterance span.
 * Returns 0, or -1 on failure (too long / too many tokens). */
int ctc_segmentation(const float *logp, int T, int V, const int *tgt, int L,
                     const ctc_segmentation_params *p, float *timings, float *char_probs,
                     double *start_s, double *end_s, double *conf);

#endif /* JA_CTC_H */

#ifdef JA_CTC_IMPLEMENTATION

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void ctc__finish_seg(const float *logp, int V, ctc_seg *s) {
    double sp = 0.0, sl = 0.0;
    for (int t = s->start; t < s->end; t++) {
        float l = logp[(size_t)t * V + s->label];
        sl += l;
        sp += exp(l);
    }
    int n = s->end - s->start;
    s->score = (float)sl;
    s->conf = n > 0 ? (float)(sp / n) : 0.0f;
}

int ctc_greedy_segments(const float *logp, int T, int V, ctc_seg *out, int cap) {
    int n = 0, prev = 0, start = 0;
    for (int t = 0; t <= T; t++) {
        int best = 0;
        if (t < T) {
            const float *r = logp + (size_t)t * V;
            for (int v = 1; v < V; v++) if (r[v] > r[best]) best = v;
        }
        if (t == T || best != prev) {
            if (prev != 0 && n < cap) {
                out[n].label = prev; out[n].start = start; out[n].end = t;
                ctc__finish_seg(logp, V, &out[n]);
                n++;
            }
            start = t;
            prev = best;
        }
    }
    return n;
}

int ctc_forced_align(const float *logp, int T, int V, const int *tgt, int L,
                     int *path, ctc_seg *segs, float *total_logp) {
    int S = 2 * L + 1;
    int need = L;
    for (int i = 1; i < L; i++) need += tgt[i] == tgt[i - 1];
    if (L == 0 || need > T) return -1;
    float *a = (float *)malloc(sizeof(float) * (size_t)S);
    float *b = (float *)malloc(sizeof(float) * (size_t)S);
    unsigned char *bp = (unsigned char *)malloc((size_t)T * S); /* 0 stay, 1 from s-1, 2 from s-2 */
    const float NEG = -INFINITY;
#define EXT(s) (((s) & 1) ? tgt[(s) >> 1] : 0)
    for (int s = 0; s < S; s++) a[s] = NEG;
    a[0] = logp[0];
    a[1] = logp[tgt[0]];
    memset(bp, 0, (size_t)S);
    for (int t = 1; t < T; t++) {
        const float *r = logp + (size_t)t * V;
        unsigned char *bt = bp + (size_t)t * S;
        for (int s = 0; s < S; s++) {
            float best = a[s];
            unsigned char arg = 0;
            if (s >= 1 && a[s - 1] > best) { best = a[s - 1]; arg = 1; }
            if (s >= 2 && (s & 1) && EXT(s) != EXT(s - 2) && a[s - 2] > best) { best = a[s - 2]; arg = 2; }
            b[s] = best == NEG ? NEG : best + r[EXT(s)];
            bt[s] = arg;
        }
        float *tmp = a; a = b; b = tmp;
    }
    int s = a[S - 1] >= a[S - 2] ? S - 1 : S - 2;
    if (total_logp) *total_logp = a[s];
    for (int i = 0; i < L; i++) { segs[i].label = tgt[i]; segs[i].start = -1; segs[i].end = -1; }
    for (int t = T - 1; t >= 0; t--) {
        path[t] = EXT(s);
        if (s & 1) {
            ctc_seg *g = &segs[s >> 1];
            if (g->end < 0) g->end = t + 1;
            g->start = t;
        }
        s -= bp[(size_t)t * S + s];
    }
#undef EXT
    for (int i = 0; i < L; i++) ctc__finish_seg(logp, V, &segs[i]);
    free(a); free(b); free(bp);
    return 0;
}

int ctc_segmentation(const float *logp, int T, int V, const int *tgt, int L,
                     const ctc_segmentation_params *p, float *timings, float *char_probs,
                     double *start_s, double *end_s, double *conf) {
    (void)V;
    const float MAXP = -10000000000.0f;   /* config.max_prob (table init) */
    const float PMAX = -1000000000.0f;    /* prob_max inside the fill */
    if (T > 8000 || T < 1) return -1;
    /* ground truth: [-1, blank, tokens..., blank] (prepare_token_list for one utterance) */
    int C = L + 3;
    int *gt = (int *)malloc(sizeof(int) * (size_t)C);
    gt[0] = -1; gt[1] = 0;
    for (int i = 0; i < L; i++) gt[2 + i] = tgt[i];
    gt[C - 1] = 0;
    if (L > 0 && tgt[L - 1] == 0) C--; /* no double blank at the end */
    int utt_begin0 = 1, utt_begin1 = C - 1;
    float *tab = (float *)malloc(sizeof(float) * (size_t)T * C);
    for (size_t i = 0; i < (size_t)T * C; i++) tab[i] = MAXP;
#define TB(t, c) tab[(size_t)(t) * C + (c)]
#define LP(t, v) logp[(size_t)(t) * V + (v)]
    TB(0, 0) = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int t = c == 0 ? 1 : 0; t < T; t++) {
            float sw = PMAX, mx = PMAX;
            if (gt[c] != -1) {
                float pp = (t - 1 < 0 || c == 0) ? PMAX : TB(t - 1, c - 1) + LP(t, gt[c]);
                if (pp > sw) sw = pp;
                if (LP(t, gt[c]) > mx) mx = LP(t, gt[c]);
            }
            float st;
            if (t - 1 < 0) st = PMAX;
            else if (c == 0) st = 0.0f;  /* preamble_transition_cost_zero */
            else st = TB(t - 1, c) + (LP(t, 0) > mx ? LP(t, 0) : mx);
            TB(t, c) = sw > st ? sw : st;
        }
    }
    int c = C - 1, t = 0;
    for (int i = 1; i < T; i++) if (TB(i, c) > TB(t, c)) t = i;
    double *tim = (double *)calloc((size_t)C, sizeof(double));
    for (int i = 0; i < T; i++) char_probs[i] = 0.0f;
    int ok = 0;
    while (t != 0 || c != 0) {
        if (t < 0 || c < 0) { ok = -1; break; }
        float delta = INFINITY, mxl = MAXP, swp = MAXP;
        int have = 0;
        if (gt[c] != -1) {
            swp = c > 0 ? LP(t, gt[c]) : MAXP;
            /* python negative index wrap for t-1 < 0 */
            int tp = t - 1 < 0 ? T - 1 : t - 1;
            float est = TB(t, c) - TB(tp, c - 1 < 0 ? C - 1 : c - 1);
            delta = fabsf(swp - est);
            have = 1;
            if (swp > mxl) mxl = swp;
        }
        float stp = t > 0 ? (LP(t, 0) > mxl ? LP(t, 0) : mxl) : MAXP;
        int tp = t - 1 < 0 ? T - 1 : t - 1;
        float est_st = TB(t, c) - TB(tp, c);
        if (have && fabsf(stp - est_st) > delta) {
            if (c > 0) {
                tim[c] = t * (double)p->index_duration;
                char_probs[t] = mxl;
            }
            c -= 1;
            t -= 1;
        } else {
            char_probs[t] = stp;
            t -= 1;
        }
    }
#undef TB
#undef LP
    if (ok == 0) {
        for (int i = 0; i < L; i++) timings[i] = (float)tim[2 + i];
        /* determine_utterance_segments for the single utterance */
        double mid0 = (tim[utt_begin0] + tim[utt_begin0 - 1]) / 2;
        double s0 = tim[utt_begin0 + 1] - 0.5 > mid0 ? tim[utt_begin0 + 1] - 0.5 : mid0;
        double mid1 = (tim[utt_begin1] + tim[utt_begin1 - 1]) / 2;
        double e0 = tim[utt_begin1 - 1] + 0.5 < mid1 ? tim[utt_begin1 - 1] + 0.5 : mid1;
        int st_t = (int)lround(s0 / p->index_duration), en_t = (int)lround(e0 / p->index_duration);
        int n = p->score_min_mean_over_L;
        double mn;
        if (en_t <= st_t) mn = -10000000000.0;
        else if (en_t - st_t <= n) {
            double s = 0.0;
            for (int i = st_t; i < en_t; i++) s += char_probs[i];
            mn = s / (en_t - st_t);
        } else {
            mn = 0.0;
            for (int i = st_t; i < en_t - n; i++) {
                double s = 0.0;
                for (int j = i; j < i + n; j++) s += char_probs[j];
                if (s / n < mn) mn = s / n;
            }
        }
        *start_s = s0; *end_s = e0; *conf = mn;
    }
    free(gt); free(tab); free(tim);
    return ok;
}

#endif /* JA_CTC_IMPLEMENTATION */
