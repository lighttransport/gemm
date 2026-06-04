/* DS4F Tier-B2 prefill-math validation harness. Calls the Tier-B2 kernels in
 * common/ds4f.h and writes tierb2_c.txt with the SAME line order as the pure-Python
 * reference tools/ds4f_tierb2_ref.py (which writes tierb2_py.txt). Compare with:
 *   paste tierb2_py.txt tierb2_c.txt |
 *     awk '{d=$2-$4;a=d<0?-d:d;if(a>m){m=a;w=$1}}END{print "max-abs",m,"@",w}'
 *
 * Build (native A64FX):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -D_GNU_SOURCE \
 *       -I../../common -o build/ds4f_tierb2_test ds4f_tierb2_test.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "ds4f.h"

static FILE *OUT;
static void emit(const char *name, double v) { fprintf(OUT, "%-16s %.7e\n", name, v); }

/* deterministic f32 generator — identical to gv() in ds4f_tierb2_ref.py */
static float gv(long seed, long i) {
    long t = (i * 1315423911L + seed * 2654435761L) % 1000003L;
    return (float)((double)(t - 500001) / 250000.0);
}

/* small RoPE table (double freq/ang -> f32), identical to ds4f_tierb2_ref.rope_table */
static void rope_table(int seqlen, int rd, float base, float *cosb, float *sinb) {
    int half = rd / 2;
    for (int pos = 0; pos < seqlen; pos++)
        for (int k = 0; k < half; k++) {
            double freq = 1.0 / pow((double)base, (2.0 * k) / rd);
            double ang = (double)pos * freq;
            cosb[pos * half + k] = (float)cos(ang);
            sinb[pos * half + k] = (float)sin(ang);
        }
}

/* drive ds4f_compress_prefill on synthetic gv() weights/inputs and emit out[nwin*d]. */
static void run_compress(const char *tag, long base, int seqlen, int dim, int d,
                         int rd, int ratio, int rotate) {
    int coff = (ratio == 4) ? 2 : 1, W = coff * d, half = rd / 2;
    float *wkv = malloc((size_t)W * dim * 4), *wgate = malloc((size_t)W * dim * 4);
    float *ape = malloc((size_t)ratio * W * 4);
    uint16_t *nw = malloc((size_t)d * 2);
    float *x = malloc((size_t)seqlen * dim * 4);
    float *cosb = malloc((size_t)seqlen * half * 4), *sinb = malloc((size_t)seqlen * half * 4);
    for (int o = 0; o < W; o++)
        for (int i = 0; i < dim; i++) {
            wkv[o * dim + i] = gv(base + o, i);
            wgate[o * dim + i] = gv(base + 1000 + o, i);
        }
    for (int r = 0; r < ratio; r++)
        for (int c = 0; c < W; c++) ape[r * W + c] = gv(base + 2000 + r, c);
    for (int e = 0; e < d; e++) nw[e] = ds4f_f32_bf16(gv(base + 3000, e));
    for (int pos = 0; pos < seqlen; pos++)
        for (int i = 0; i < dim; i++) x[pos * dim + i] = gv(base + 4000 + pos, i);
    rope_table(seqlen, rd, 10000.0f, cosb, sinb);
    int nwin = (seqlen - seqlen % ratio) / ratio;
    float *out = malloc((size_t)nwin * d * 4);
    ds4f_compress_prefill(x, seqlen, dim, d, rd, ratio, wkv, wgate, ape, nw,
                          cosb, sinb, 1e-6f, rotate, out);
    char nm[32];
    for (int w = 0; w < nwin; w++)
        for (int e = 0; e < d; e++) {
            snprintf(nm, sizeof nm, "%s_%d_%d", tag, w, e);
            emit(nm, (double)out[w * d + e]);
        }
    free(wkv); free(wgate); free(ape); free(nw); free(x); free(cosb); free(sinb); free(out);
}

/* drive the indexer: per query s, q=wq_b(qr[s]) reshaped [H][hd], RoPE last rd dims
 * @pos s + rotate + fp4 per head; weights=wproj(x[s])*wscale; compressor fills kvc
 * (rotate=1); index_score over T compressed cols; masked top-k (thr=(s+1)/ratio).
 * Emits finite scores (t<min(thr,T)) and the sorted selected idx set padded to k. */
static void run_indexer(const char *tag, long base, int seqlen, int dim, int qlora,
                        int H, int hd, int rd, int ratio, int k, int offset) {
    int half = rd / 2, Wc = 2 * hd; float eps = 1e-6f;        /* compressor overlap coff=2 */
    float *wqb = malloc((size_t)H * hd * qlora * 4), *wproj = malloc((size_t)H * dim * 4);
    float *cwkv = malloc((size_t)Wc * dim * 4), *cwgate = malloc((size_t)Wc * dim * 4);
    float *cape = malloc((size_t)ratio * Wc * 4);
    uint16_t *cnorm = malloc((size_t)hd * 2);
    float *x = malloc((size_t)seqlen * dim * 4), *qr = malloc((size_t)seqlen * qlora * 4);
    float *cosb = malloc((size_t)seqlen * half * 4), *sinb = malloc((size_t)seqlen * half * 4);
    for (int o = 0; o < H * hd; o++)
        for (int i = 0; i < qlora; i++) wqb[o * qlora + i] = gv(base + o, i);
    for (int h = 0; h < H; h++)
        for (int i = 0; i < dim; i++) wproj[h * dim + i] = gv(base + 100 + h, i);
    for (int o = 0; o < Wc; o++)
        for (int i = 0; i < dim; i++) {
            cwkv[o * dim + i] = gv(base + 200 + o, i);
            cwgate[o * dim + i] = gv(base + 300 + o, i);
        }
    for (int r = 0; r < ratio; r++)
        for (int c = 0; c < Wc; c++) cape[r * Wc + c] = gv(base + 400 + r, c);
    for (int e = 0; e < hd; e++) cnorm[e] = ds4f_f32_bf16(gv(base + 500, e));
    for (int pos = 0; pos < seqlen; pos++)
        for (int i = 0; i < dim; i++) x[pos * dim + i] = gv(base + 600 + pos, i);
    for (int pos = 0; pos < seqlen; pos++)
        for (int i = 0; i < qlora; i++) qr[pos * qlora + i] = gv(base + 700 + pos, i);
    rope_table(seqlen, rd, 10000.0f, cosb, sinb);

    int T = (seqlen - seqlen % ratio) / ratio;
    float *kvc = malloc((size_t)T * hd * 4);
    ds4f_compress_prefill(x, seqlen, dim, hd, rd, ratio, cwkv, cwgate, cape, cnorm,
                          cosb, sinb, eps, 1, kvc);

    float sm_scale = (float)(1.0 / sqrt((double)hd));
    float inv_sqrt_H = (float)(1.0 / sqrt((double)H));
    float wscale = sm_scale * inv_sqrt_H;

    float *qheads = malloc((size_t)H * hd * 4), *weights = malloc((size_t)H * 4);
    float *score = malloc((size_t)T * 4); int *sel = malloc((size_t)k * sizeof(int));
    char nm[40];
    for (int s = 0; s < seqlen; s++) {
        for (int o = 0; o < H * hd; o++) {                   /* q = wq_b(qr[s]) */
            const float *w = wqb + (size_t)o * qlora, *xq = qr + (size_t)s * qlora;
            float a = 0.f; for (int i = 0; i < qlora; i++) a += w[i] * xq[i];
            qheads[o] = a;
        }
        for (int h = 0; h < H; h++) {                        /* RoPE+rotate+fp4 per head */
            float *qh = qheads + (size_t)h * hd;
            ds4f_rope_apply(qh + (hd - rd), cosb, sinb, s, half, 0);
            ds4f_rotate_activation(qh, hd);
            ds4f_fp4_act_quant_inplace(qh, hd, 32);
        }
        for (int h = 0; h < H; h++) {                        /* weights = wproj(x[s])*wscale */
            const float *w = wproj + (size_t)h * dim, *xs = x + (size_t)s * dim;
            float a = 0.f; for (int i = 0; i < dim; i++) a += w[i] * xs[i];
            weights[h] = a * wscale;
        }
        ds4f_index_score(qheads, kvc, weights, H, hd, T, score);
        int thr = (s + 1) / ratio, tlim = thr < T ? thr : T;
        for (int t = 0; t < tlim; t++) {
            snprintf(nm, sizeof nm, "%s_is_%d_%d", tag, s, t); emit(nm, (double)score[t]);
        }
        ds4f_index_topk(score, T, thr, k, offset, sel);
        for (int n = 0; n < k; n++) {
            snprintf(nm, sizeof nm, "%s_it_%d_%d", tag, s, n); emit(nm, (double)sel[n]);
        }
    }
    free(wqb); free(wproj); free(cwkv); free(cwgate); free(cape); free(cnorm);
    free(x); free(qr); free(cosb); free(sinb); free(kvc);
    free(qheads); free(weights); free(score); free(sel);
}

int main(void) {
    OUT = fopen("tierb2_c.txt", "w");
    if (!OUT) { perror("tierb2_c.txt"); return 1; }
    char nm[32];

    /* ---- 1. window index helper (prefill) ---- */
    enum { WIN = 4, WSEQ = 10 };
    int wrow[WSEQ];
    for (int s = 0; s < WSEQ; s++) {
        int wq = ds4f_window_idx_prefill(WIN, WSEQ, s, wrow);
        for (int c = 0; c < (WSEQ < WIN ? WSEQ : WIN); c++) {
            (void)wq; snprintf(nm, sizeof nm, "wi_%d_%d", s, c); emit(nm, (double)wrow[c]);
        }
    }

    /* ---- 2. compress index helper (prefill) ---- */
    enum { CRATIO = 2, CSEQ = 8, COFF = 100 };
    int crow[CSEQ];
    for (int s = 0; s < CSEQ; s++) {
        ds4f_compress_idx_prefill(CRATIO, CSEQ, COFF, s, crow);
        for (int t = 0; t < CSEQ / CRATIO; t++) {
            snprintf(nm, sizeof nm, "ci_%d_%d", s, t); emit(nm, (double)crow[t]);
        }
    }

    /* ---- 3. sparse_attn ---- */
    enum { SA_M = 6, SA_H = 3, SA_D = 8, SA_N = 12, SA_TOPK = 5 };
    static float q[SA_M * SA_H * SA_D], kv[SA_N * SA_D], sink[SA_H], o[SA_M * SA_H * SA_D];
    static int topk[SA_M * SA_TOPK];
    for (int s = 0; s < SA_M; s++)
        for (int hd = 0; hd < SA_H; hd++)
            for (int dd = 0; dd < SA_D; dd++)
                q[(s * SA_H + hd) * SA_D + dd] = gv(1 + s * 100 + hd * 10, dd);
    for (int n = 0; n < SA_N; n++)
        for (int dd = 0; dd < SA_D; dd++)
            kv[n * SA_D + dd] = gv(2 + n * 10, dd);
    for (int hd = 0; hd < SA_H; hd++) sink[hd] = gv(3, hd);
    for (int s = 0; s < SA_M; s++) {
        for (int t = 0; t < SA_TOPK - 1; t++) topk[s * SA_TOPK + t] = (s + t * 2) % SA_N;
        topk[s * SA_TOPK + (SA_TOPK - 1)] = -1;
    }
    float scale = (float)(1.0 / sqrt((double)SA_D));
    ds4f_sparse_attn(q, kv, sink, topk, SA_M, SA_H, SA_D, SA_TOPK, scale, o);
    for (int s = 0; s < SA_M; s++)
        for (int hd = 0; hd < SA_H; hd++)
            for (int dd = 0; dd < SA_D; dd++) {
                snprintf(nm, sizeof nm, "sa_%d_%d_%d", s, hd, dd);
                emit(nm, (double)o[(s * SA_H + hd) * SA_D + dd]);
            }

    /* ---- 4. hash routing (tid2eid lookup) ---- */
    enum { HNACT = 4, HNEXP = 16 };
    int toks[4] = {3, 17, 0, 9};
    for (int ti = 0; ti < 4; ti++)
        for (int k = 0; k < HNACT; k++) {
            int eid = (toks[ti] * 7 + k * 3) % HNEXP;
            snprintf(nm, sizeof nm, "hash_%d_%d", ti, k); emit(nm, (double)eid);
        }

    /* ---- 5. Compressor (prefill gated pooling) ---- */
    run_compress("c1", 5000, 10, 6, 8, 4, 4, 1);   /* overlap + rotate (indexer compressor) */
    run_compress("c2", 5000, 10, 6, 8, 4, 4, 0);   /* overlap, no rotate (layer compressor) */
    run_compress("c3", 6000, 8, 6, 8, 4, 2, 0);    /* non-overlap (ratio!=4), no rotate */

    /* ---- 6. Indexer (prefill scoring + masked top-k) ---- */
    run_indexer("ix", 8000, 12, 6, 6, 4, 8, 4, 4, 2, 100);

    fclose(OUT);
    printf("wrote tierb2_c.txt\n");
    return 0;
}
