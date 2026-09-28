/* SPDX-License-Identifier: MIT
 * test_ja_ctc            : synthetic checks (Viterbi optimality vs brute force, kana->phoneme table)
 * test_ja_ctc <ref_dir>  : compare against oracle dumps from ref/align_reference.py
 *                          (torchaudio forced_align path, ctc-segmentation timings/char_probs/confidence)
 */
#define JA_BASE_IMPLEMENTATION
#define JA_CTC_IMPLEMENTATION
#include "ja_base.h"
#include "ctc.h"
#include "ja_phoneme.h"
#include "../../common/npy_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

/* brute force: enumerate every frame labeling that collapses to tgt, return the best score */
static float brute_best(const float *lp, int T, int V, const int *tgt, int L) {
    int total = 1;
    for (int t = 0; t < T; t++) total *= V;
    float best = -INFINITY;
    int *lab = (int *)malloc(sizeof(int) * (size_t)T);
    for (int code = 0; code < total; code++) {
        int c = code;
        for (int t = 0; t < T; t++) { lab[t] = c % V; c /= V; }
        int out[64], n = 0, prev = 0;
        for (int t = 0; t < T; t++) {
            if (lab[t] != 0 && lab[t] != prev) out[n++] = lab[t];
            prev = lab[t];
        }
        if (n != L) continue;
        int ok = 1;
        for (int i = 0; i < L; i++) ok &= out[i] == tgt[i];
        if (!ok) continue;
        float s = 0;
        for (int t = 0; t < T; t++) s += lp[t * V + lab[t]];
        if (s > best) best = s;
    }
    free(lab);
    return best;
}

static void synthetic(void) {
    srand(7);
    for (int trial = 0; trial < 200; trial++) {
        int V = 3 + rand() % 2, T = 3 + rand() % 5, L = 1 + rand() % 3;
        float lp[16 * 8];
        for (int t = 0; t < T; t++) {
            float s = 0;
            for (int v = 0; v < V; v++) { lp[t * V + v] = (float)rand() / RAND_MAX + 0.05f; s += lp[t * V + v]; }
            for (int v = 0; v < V; v++) lp[t * V + v] = logf(lp[t * V + v] / s);
        }
        int tgt[4];
        for (int i = 0; i < L; i++) tgt[i] = 1 + rand() % (V - 1);
        int path[16];
        ctc_seg segs[4];
        float tot = 0;
        int rc = ctc_forced_align(lp, T, V, tgt, L, path, segs, &tot);
        float bb = brute_best(lp, T, V, tgt, L);
        if (bb == -INFINITY) { CHECK(rc != 0, "trial %d: infeasible but aligned", trial); continue; }
        CHECK(rc == 0 && fabsf(tot - bb) < 1e-4f, "trial %d: viterbi %.5f brute %.5f", trial, tot, bb);
        /* the returned path must collapse to tgt and score tot */
        float s = 0; int out[16], n = 0, prev = 0;
        for (int t = 0; t < T; t++) { s += lp[t * V + path[t]]; if (path[t] && path[t] != prev) out[n++] = path[t]; prev = path[t]; }
        int ok = n == L;
        for (int i = 0; ok && i < L; i++) ok &= out[i] == tgt[i];
        CHECK(ok && fabsf(s - tot) < 1e-4f, "trial %d: path does not realize the score", trial);
    }
    /* kana -> phonemes */
    struct { const char *k, *p; } cases[] = {
        {"こんにちわ", "k o N n i ch i w a"}, {"キョーワ", "ky o o w a"}, {"がっこう", "g a cl k o u"},
        {"ティーシャツ", "t i i sh a ts u"}, {"ふぁいる", "f a i r u"}, {"じゅうでん", "j u u d e N"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        int ids[64]; char txt[256];
        int n = ja_kana_to_phonemes(cases[i].k, ids, 64, txt, sizeof(txt));
        CHECK(n > 0 && !strcmp(txt, cases[i].p), "kana '%s' -> '%s' (want '%s')", cases[i].k, txt, cases[i].p);
    }
    printf("synthetic: %s\n", fails ? "FAILED" : "ok (200 viterbi trials vs brute force, kana table)");
}

static void *load(const char *dir, const char *name, int *nd, int *dims) {
    char p[1024]; int f32;
    snprintf(p, sizeof(p), "%s/%s", dir, name);
    return npy_load(p, nd, dims, &f32);
}

static void compare_ref(const char *dir, const char *kind) {
    int nd, d[8], nd2, d2[8], nd3, d3[8];
    char nm[128];
    snprintf(nm, sizeof(nm), "%s_logp.npy", kind);
    float *lp = (float *)load(dir, nm, &nd, d);
    snprintf(nm, sizeof(nm), "fa_%s_targets.npy", kind);
    int *tgt = (int *)load(dir, nm, &nd2, d2);
    snprintf(nm, sizeof(nm), "fa_%s_path.npy", kind);
    int *ref_path = (int *)load(dir, nm, &nd3, d3);
    if (!lp || !tgt || !ref_path) { printf("%s: no fixtures, skipped\n", kind); return; }
    int T = d[0], V = d[1], L = d2[0];
    int *path = (int *)malloc(sizeof(int) * (size_t)T);
    ctc_seg *segs = (ctc_seg *)malloc(sizeof(ctc_seg) * (size_t)L);
    float tot;
    int rc = ctc_forced_align(lp, T, V, tgt, L, path, segs, &tot);
    int diff = 0;
    for (int t = 0; t < T; t++) diff += path[t] != ref_path[t];
    double ref_tot = 0;
    for (int t = 0; t < T; t++) ref_tot += lp[(size_t)t * V + ref_path[t]];
    CHECK(rc == 0 && diff == 0, "%s forced_align: %d/%d frames differ (score %.4f vs oracle %.4f)", kind, diff, T, tot, ref_tot);
    printf("%s forced_align vs torchaudio: %d/%d frames differ, logp %.4f vs %.4f\n", kind, diff, T, tot, ref_tot);

    snprintf(nm, sizeof(nm), "ctcseg_%s_timings.npy", kind);
    float *rt = (float *)load(dir, nm, &nd, d2);
    snprintf(nm, sizeof(nm), "ctcseg_%s_charprobs.npy", kind);
    float *rc_p = (float *)load(dir, nm, &nd, d3);
    if (rt && rc_p) {
        ctc_segmentation_params sp = { 0.02f, 30 };
        float *tim = (float *)malloc(sizeof(float) * (size_t)L);
        float *cp = (float *)malloc(sizeof(float) * (size_t)T);
        double s0, e0, conf;
        int r2 = ctc_segmentation(lp, T, V, tgt, L, &sp, tim, cp, &s0, &e0, &conf);
        /* reference timings array is over ground_truth [-1, blank, tokens.., blank]: tokens start at 2 */
        double mt = 0, mc = 0;
        for (int i = 0; i < L; i++) { double e = fabs(tim[i] - rt[2 + i]); if (e > mt) mt = e; }
        for (int t = 0; t < T; t++) { double e = fabs(cp[t] - rc_p[t]); if (e > mc) mc = e; }
        char jp[1024];
        snprintf(jp, sizeof(jp), "%s/ctcseg_%s.json", dir, kind);
        FILE *f = fopen(jp, "r");
        double rs = 0, re = 0, rconf = 0;
        if (f) {
            char buf[4096]; size_t n = fread(buf, 1, sizeof(buf) - 1, f); buf[n] = 0; fclose(f);
            char *q = strstr(buf, "[[");
            if (q) sscanf(q + 2, "%lf, %lf, %lf", &rs, &re, &rconf);
        }
        CHECK(r2 == 0 && mt < 1e-6 && mc < 1e-4 && fabs(conf - rconf) < 1e-4,
              "%s ctc_segmentation: timing err %.3g, char_prob err %.3g, conf %.6f vs %.6f", kind, mt, mc, conf, rconf);
        printf("%s ctc_segmentation vs package: max timing err %.3g s, max char_prob err %.3g, "
               "span %.2f-%.2f vs %.2f-%.2f, conf %.6f vs %.6f\n", kind, mt, mc, s0, e0, rs, re, conf, rconf);
        free(tim); free(cp);
    }
    free(lp); free(tgt); free(ref_path); free(path); free(segs); free(rt); free(rc_p);
}

int main(int argc, char **argv) {
    synthetic();
    if (argc > 1) {
        compare_ref(argv[1], "phoneme");
        compare_ref(argv[1], "kana");
    }
    printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
