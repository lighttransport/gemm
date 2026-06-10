/*
 * ds4f_runner.c - DeepSeek-V4-Flash synthetic inference harness (Stage 1).
 *
 * Single-node perf/plumbing run. Weights are SYNTHETIC (filled in HBM, no disk
 * load), logits MEANINGLESS. Validates: allocator fits ~21 GB at ep_size=11/
 * rank=0, end-to-end MLA+MoE forward runs, NUMA first-touch, per-token
 * throughput and effective on-demand-dequant GB/s.
 *
 * Build (native A64FX):
 *   make -C a64fx/llm ds4f_runner CC=fcc OPENMP=1
 * Run:
 *   LLM_THREADS=48 DS4F_EP_SIZE=11 DS4F_EP_RANK=0 \
 *   DS4F_PREFILL=8 DS4F_MAXGEN=16 build/ds4f_runner
 *
 * Env:
 *   LLM_THREADS    compute threads (default 48)
 *   DS4F_CMGS      CMGs to pin across (default 4)
 *   DS4F_EP_SIZE   expert-parallel world size (default 11)
 *   DS4F_EP_RANK   this rank's expert shard (default 0)
 *   DS4F_PREFILL   synthetic prefill tokens (default 8)
 *   DS4F_MAXGEN    synthetic decode tokens (default 16)
 *   DS4F_MAXPOS    KV cache capacity / max position (default 4096)
 *   DS4F_LAYERS    override n_layers (default 43; small for quick smoke)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

#include "ds4f.h"

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* resident set size in bytes from /proc/self/statm (pages * page_size) */
static size_t rss_bytes(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long total = 0, res = 0;
    if (fscanf(f, "%ld %ld", &total, &res) != 2) res = 0;
    fclose(f);
    return (size_t)res * (size_t)sysconf(_SC_PAGESIZE);
}

static int envi(const char *k, int dflt) {
    const char *v = getenv(k); return v ? atoi(v) : dflt;
}

/* splitmix64 for synthetic activations */
static uint64_t sm_state;
static double sm_next(void) {
    sm_state += 0x9E3779B97F4A7C15ull;
    uint64_t z = sm_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= (z >> 31);
    return (double)(z >> 11) / (double)(1ull << 53);   /* [0,1) */
}

int main(void) {
    int n_threads = envi("LLM_THREADS", 48);
    int n_cmgs    = envi("DS4F_CMGS", 4);
    int ep_size   = envi("DS4F_EP_SIZE", 11);
    int ep_rank   = envi("DS4F_EP_RANK", 0);
    int prefill   = envi("DS4F_PREFILL", 8);
    int maxgen    = envi("DS4F_MAXGEN", 16);
    int maxpos    = envi("DS4F_MAXPOS", 4096);
    int layers    = envi("DS4F_LAYERS", 0);
    int ctx_warm  = envi("DS4F_CTX_WARM", 0);   /* prefill synthetic KV to this ctx, decode from there */

    ds4f_config cfg = ds4f_config_from_env();
    cfg.max_pos = maxpos;
    if (layers > 0) cfg.n_layers = layers;
    /* DS4F_FORCE_RATIO=R: override EVERY layer's compress_ratio to R (bypass the
     * 0/1/last-dense rule) so a pure all-sparse asymptote can be measured. */
    int force_ratio = envi("DS4F_FORCE_RATIO", 0);
    if (force_ratio > 0)
        for (int L = 0; L < cfg.n_layers; L++) cfg.compress_ratios[L] = force_ratio;
    if (prefill + maxgen > cfg.max_pos) {
        fprintf(stderr, "prefill+maxgen (%d) exceeds max_pos (%d)\n",
                prefill + maxgen, cfg.max_pos);
        return 1;
    }
    if (ctx_warm + maxgen > cfg.max_pos) {
        fprintf(stderr, "ctx_warm+maxgen (%d) exceeds max_pos (%d)\n",
                ctx_warm + maxgen, cfg.max_pos);
        return 1;
    }

    int no = ds4f_n_owned(cfg.n_experts, ep_rank, ep_size);
    printf("=== DS4F synthetic harness (Stage 1) ===\n");
    printf("layers=%d hidden=%d vocab=%d  heads=%d q_head_dim=%d kv_lora=%d\n",
           cfg.n_layers, cfg.hidden, cfg.vocab, cfg.n_heads, cfg.q_head_dim, cfg.kv_lora);
    printf("experts=%d active=%d moe_inter=%d  ep_size=%d ep_rank=%d -> owned=%d/layer\n",
           cfg.n_experts, cfg.n_active, cfg.moe_inter, ep_size, ep_rank, no);
    printf("threads=%d cmgs=%d  prefill=%d maxgen=%d max_pos=%d\n",
           n_threads, n_cmgs, prefill, maxgen, maxpos);

    int dense_bf16 = envi("DS4F_FP8_BF16", 0);
    int dense_mxfp4 = envi("DS4F_DENSE_MXFP4", 0);      /* overrides FP8/BF16: 0.53 B/elem */
    const char *pv_e = getenv("DS4F_BF16_PV");          /* auto-on with predequant unless explicitly set */
    int bf16_pv = (pv_e && *pv_e) ? (atoi(pv_e) != 0) : dense_bf16;
    size_t arena_est = ds4f_arena_size(&cfg, ep_rank, ep_size, dense_bf16,
                                       envi("DS4F_TIERB2", 0) && !envi("DS4F_INT8_KV", 0));
    const char *dlabel = dense_mxfp4 ? "MXFP4(split,0.53B)" :
                         dense_bf16 ? (bf16_pv ? "BF16(predequant,pv)" : "BF16(predequant)") : "FP8(on-demand)";
    printf("arena reservation: %.2f GB   dense=%s\n", arena_est / (1024.0*1024.0*1024.0), dlabel);
    fflush(stdout);

    /* DS4F_REAL=1: load REAL weights from each node's staged blob (see
     * run_ds4f_stage_11n.sh) instead of the synthetic fill. blob_dir defaults to
     * DS4F_STAGE_DIR else /local/ds4f inside ds4f_load_real. The loader forces
     * dense=FP8 on-demand (ignores the DS4F_FP8_BF16/DENSE_MXFP4 dense knobs;
     * they only describe the synthetic fill). */
    int real_weights = envi("DS4F_REAL", 0);
    const char *blob_dir = getenv("DS4F_STAGE_DIR");
    double t_alloc0 = now_sec();
    ds4f_model *m = real_weights
        ? ds4f_load_real(cfg, ep_rank, ep_size, blob_dir, n_threads, n_cmgs)
        : ds4f_alloc_synth(cfg, ep_rank, ep_size, n_threads, n_cmgs);
    if (!m) { fprintf(stderr, "model alloc/load failed\n"); return 1; }
    double t_alloc = now_sec() - t_alloc0;
    size_t rss = rss_bytes();
    printf("alloc+first-touch: %.2f s   arena_used=%.2f GB   RSS=%.2f GB\n",
           t_alloc, m->arena_used / (1024.0*1024.0*1024.0),
           rss / (1024.0*1024.0*1024.0));
    if (m->sparse) {
        int nsp = 0; for (int L = 0; L < cfg.n_layers; L++) if (cfg.compress_ratios[L]) nsp++;
        printf("sparse indexer: ON  topk=%d index_dim=%d  (%d/%d layers sparse; dense when nP<=topk)\n",
               cfg.index_topk, cfg.index_head_dim, nsp, cfg.n_layers);
    }
    if (m->tierb2) {
        int ncsa = 0, nhca = 0;
        for (int L = 0; L < cfg.n_layers; L++) {
            if (cfg.compress_ratios[L] == 4) ncsa++;
            else if (cfg.compress_ratios[L]) nhca++;
        }
        printf("Tier-B2: ON (exact forced)  index_topk=%d index_dim=%d index_heads=%d  "
               "(%d CSA + %d HCA layers; compressed-KV folded into window softmax)\n",
               cfg.index_topk, cfg.index_head_dim, cfg.index_n_heads, ncsa, nhca);
    }
    fflush(stdout);

    int C = cfg.hidden;
    float *x = (float *)aligned_alloc(256, (size_t)C * 4);

    /* batched-prefill knobs:
     *   DS4F_PREFILL_BATCH = M_TILE > 0  -> process prefill in M-token GEMM tiles
     *                                       (needs DS4F_EXACT=1; FP8 dense batches
     *                                        via tile-dequant, FP8_BF16=1 optional).
     *   DS4F_PREFILL_CHECK = 1           -> also run token-at-a-time and diff the
     *                                       per-token argmax + final hidden state. */
    int prefill_batch = envi("DS4F_PREFILL_BATCH", 0);
    int prefill_check = envi("DS4F_PREFILL_CHECK", 0);
    if (prefill_batch > DS4F_MAX_MTILE) prefill_batch = DS4F_MAX_MTILE;
    if (prefill_batch > 0 && !m->exact) {
        fprintf(stderr, "DS4F_PREFILL_BATCH requires DS4F_EXACT=1 (batched path is exact-only)\n");
        return 1;
    }

    sm_state = 0xD5F00D ^ ((uint64_t)ep_rank << 32);
    double t_pf = 0; size_t pf_bytes = 0;
    int nan_count = 0; double xnorm = 0.0;

    if (prefill_check && prefill_batch > 0 && prefill > 0) {
        /* ---- correctness mode: batched vs token-at-a-time on identical inputs ---- */
        int M = prefill < prefill_batch ? prefill : prefill_batch;
        printf("\n[PREFILL_CHECK] batched(M=%d) vs token-at-a-time, %d tokens\n", M, M);
        float *X  = (float *)aligned_alloc(256, (size_t)M * C * 4);
        for (int mm = 0; mm < M; mm++)
            for (int i = 0; i < C; i++) X[mm*C+i] = (float)(sm_next() * 2.0 - 1.0);
        ds4f_alloc_prefill_batch(m, M);
        int *bt = (int *)malloc((size_t)M * 4);
        ds4f_forward_prefill(m, X, M, 0, bt);
        float *bh = (float *)aligned_alloc(256, (size_t)M * C * 4);
        memcpy(bh, m->p_x, (size_t)M * C * 4);
        int mism = 0; double maxabs = 0.0, maxmag = 0.0; int nnan = 0;
        for (int mm = 0; mm < M; mm++) {
            memcpy(x, X + (size_t)mm*C, (size_t)C*4);
            int tk = ds4f_forward_token(m, x, mm);
            if (tk != bt[mm]) { mism++;
                if (mism <= 5) printf("  arg mismatch tok %d: batch=%d token=%d\n", mm, bt[mm], tk); }
            for (int i = 0; i < C; i++) {
                double a = x[i], b = bh[(size_t)mm*C+i], d = fabs(a - b);
                if (d > maxabs) maxabs = d;
                if (fabs(a) > maxmag) maxmag = fabs(a);
                if (!(b == b)) nnan++;
            }
        }
        /* hidden magnitudes are huge with synthetic weights (||x|| ~1e6+), so gate
         * on the RELATIVE diff (GEMM K-tile reassociation ~1e-4) and exact argmax. */
        double maxrel = maxabs / (maxmag + 1e-6);
        int pass = (mism == 0) && (maxrel < 1e-3) && (nnan == 0);
        printf("  argmax mismatches: %d/%d   max-abs=%.3e (rel %.3e, |x|max=%.3e)   NaNs: %d\n",
               mism, M, maxabs, maxrel, maxmag, nnan);
        printf("  => %s\n", pass ? "PASS" : "FAIL");
        free(X); free(bt); free(bh);
        free(x); ds4f_free(m);
        return pass ? 0 : 1;
    }

    if (prefill_batch > 0 && prefill > 0) {
        /* ---- batched prefill (M-token GEMM tiles) ---- */
        ds4f_alloc_prefill_batch(m, prefill_batch);
        float *X = (float *)aligned_alloc(256, (size_t)prefill_batch * C * 4);
        int *bt = (int *)malloc((size_t)prefill_batch * 4);
        double t_pf0 = now_sec();
        for (int base = 0; base < prefill; base += prefill_batch) {
            int M = prefill - base < prefill_batch ? prefill - base : prefill_batch;
            for (int mm = 0; mm < M; mm++)
                for (int i = 0; i < C; i++) X[mm*C+i] = (float)(sm_next() * 2.0 - 1.0);
            m->bytes_read = 0;
            ds4f_forward_prefill(m, X, M, base, bt);
            pf_bytes += m->bytes_read;
        }
        t_pf = now_sec() - t_pf0;
        /* finiteness of the last tile's last token (kept in p_x) */
        int Mlast = prefill % prefill_batch; if (Mlast == 0) Mlast = prefill_batch;
        const float *xl = m->p_x + (size_t)(Mlast-1)*C;
        for (int i = 0; i < C; i++) { if (!(xl[i] == xl[i])) nan_count++; xnorm += (double)xl[i]*xl[i]; }
        free(X); free(bt);
    } else {
        /* ---- prefill (sequential token-at-a-time) ---- */
        double t_pf0 = now_sec();
        for (int p = 0; p < prefill; p++) {
            for (int i = 0; i < C; i++) x[i] = (float)(sm_next() * 2.0 - 1.0);
            m->bytes_read = 0;
            ds4f_forward_token(m, x, p);
            pf_bytes += m->bytes_read;
        }
        t_pf = now_sec() - t_pf0;
        for (int i = 0; i < C; i++) { if (!(x[i] == x[i])) nan_count++; xnorm += (double)x[i]*x[i]; }
    }

    if (prefill > 0) {
        double per = t_pf / prefill;
        printf("\nprefill: %d tok  %.1f ms/tok  %.2f tok/s   %.1f GB/s/tok-weights%s\n",
               prefill, per*1e3, prefill / t_pf,
               (pf_bytes / (double)prefill) / per / 1e9,
               prefill_batch > 0 ? "  [batched]" : "");
        printf("         last-hidden ||x||=%.3e  NaNs=%d\n", sqrt(xnorm), nan_count);
    }
    /* per-phase prefill profile (DS4F_PROF=1; printed before decode wipes m->prof) */
    { double psum = 0; for (int i = 0; i <= DS4F_P_TB2PREP; i++) psum += m->prof[i]; /* TB2* sub-timers excluded (overlap TB2PREP) */
      if (psum > 0 && prefill > 0) {
        printf("\nper-phase prefill profile (ms/tok, %% of accounted):\n");
        for (int i = 0; i < DS4F_NPHASE; i++) {
            double ms = m->prof[i] / prefill * 1e3;
            if (ms <= 0) continue;
            printf("  %-9s %7.3f ms  %5.1f%%\n", ds4f_prof_names[i], ms, 100.0*m->prof[i]/psum);
        }
        printf("  %-9s %7.3f ms  (sum of accounted phases)\n", "TOTAL", psum/prefill*1e3);
      } }

    /* ---- optional synthetic ctx-warm: prefill KV to a large context cheaply,
     * so decode-attn cost at that ctx can be measured without npos real tokens.
     * Decode then starts at pos=ctx_warm (overrides prefill base). ---- */
    int dec_base = prefill;
    if (ctx_warm > 0) {
        double tw0 = now_sec();
        ds4f_warm_kv(m, ctx_warm);
        ds4f_warm_tb2(m, ctx_warm);   /* fill compressed caches too (no-op unless tierb2) */
        printf("\nctx-warm: filled synthetic KV%s [0,%d)  %.2f s  (decode from pos=%d)\n",
               m->tierb2 ? "+compressed" : "", ctx_warm, now_sec()-tw0, ctx_warm);
        dec_base = ctx_warm;
    }

    /* ---- decode (M=1, token-at-a-time) ---- */
    memset(m->prof, 0, sizeof(m->prof));
    double t_dec0 = now_sec();
    size_t dec_bytes = 0;
    int last_tok = 0;
    for (int g = 0; g < maxgen; g++) {
        int pos = dec_base + g;
        for (int i = 0; i < C; i++) x[i] = (float)(sm_next() * 2.0 - 1.0);
        m->bytes_read = 0;
        last_tok = ds4f_forward_token(m, x, pos);
        dec_bytes += m->bytes_read;
    }
    double t_dec = now_sec() - t_dec0;

    if (maxgen > 0) {
        double per = t_dec / maxgen;
        printf("\ndecode:  %d tok  %.1f ms/tok  %.2f tok/s   %.1f GB/s/tok-weights\n",
               maxgen, per*1e3, maxgen / t_dec,
               (dec_bytes / (double)maxgen) / per / 1e9);
        printf("         bytes/tok=%.2f GB (FP8 dense + %d owned MXFP4 experts/layer)\n",
               (dec_bytes / (double)maxgen) / 1e9, no);
        printf("         last argmax token=%d (meaningless)\n", last_tok);
    }

    /* per-phase profile (only populated when DS4F_PROF=1) */
    double psum = 0; for (int i = 0; i <= DS4F_P_TB2PREP; i++) psum += m->prof[i]; /* TB2* sub-timers excluded (overlap TB2PREP) */
    if (psum > 0 && maxgen > 0) {
        printf("\nper-phase decode profile (ms/tok, %% of accounted):\n");
        for (int i = 0; i < DS4F_NPHASE; i++) {
            double ms = m->prof[i] / maxgen * 1e3;
            if (ms <= 0) continue;
            printf("  %-9s %7.3f ms  %5.1f%%\n", ds4f_prof_names[i], ms, 100.0*m->prof[i]/psum);
        }
        printf("  %-9s %7.3f ms  (sum of accounted phases)\n", "TOTAL", psum/maxgen*1e3);
    }

    printf("\nRSS final=%.2f GB  (budget 32 GB/node)\n", rss_bytes()/(1024.0*1024.0*1024.0));
    free(x);
    ds4f_free(m);
    return 0;
}
