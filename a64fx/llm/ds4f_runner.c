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

    ds4f_config cfg = ds4f_default_config();
    cfg.max_pos = maxpos;
    if (layers > 0) cfg.n_layers = layers;
    if (prefill + maxgen > cfg.max_pos) {
        fprintf(stderr, "prefill+maxgen (%d) exceeds max_pos (%d)\n",
                prefill + maxgen, cfg.max_pos);
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
    const char *pv_e = getenv("DS4F_BF16_PV");          /* auto-on with predequant unless explicitly set */
    int bf16_pv = (pv_e && *pv_e) ? (atoi(pv_e) != 0) : dense_bf16;
    size_t arena_est = ds4f_arena_size(&cfg, ep_rank, ep_size, dense_bf16);
    printf("arena reservation: %.2f GB   dense=%s\n", arena_est / (1024.0*1024.0*1024.0),
           dense_bf16 ? (bf16_pv ? "BF16(predequant,pv)" : "BF16(predequant)") : "FP8(on-demand)");
    fflush(stdout);

    double t_alloc0 = now_sec();
    ds4f_model *m = ds4f_alloc_synth(cfg, ep_rank, ep_size, n_threads, n_cmgs);
    double t_alloc = now_sec() - t_alloc0;
    size_t rss = rss_bytes();
    printf("alloc+first-touch: %.2f s   arena_used=%.2f GB   RSS=%.2f GB\n",
           t_alloc, m->arena_used / (1024.0*1024.0*1024.0),
           rss / (1024.0*1024.0*1024.0));
    fflush(stdout);

    int C = cfg.hidden;
    float *x = (float *)aligned_alloc(256, (size_t)C * 4);

    /* ---- prefill (sequential token-at-a-time; batched deferred) ---- */
    sm_state = 0xD5F00D ^ ((uint64_t)ep_rank << 32);
    double t_pf0 = now_sec();
    size_t pf_bytes = 0;
    for (int p = 0; p < prefill; p++) {
        for (int i = 0; i < C; i++) x[i] = (float)(sm_next() * 2.0 - 1.0);
        m->bytes_read = 0;
        ds4f_forward_token(m, x, p);
        pf_bytes += m->bytes_read;
    }
    double t_pf = now_sec() - t_pf0;

    /* check finiteness of the last prefill hidden state */
    int nan_count = 0; double xnorm = 0.0;
    for (int i = 0; i < C; i++) { if (!(x[i] == x[i])) nan_count++; xnorm += (double)x[i]*x[i]; }

    if (prefill > 0) {
        double per = t_pf / prefill;
        printf("\nprefill: %d tok  %.1f ms/tok  %.2f tok/s   %.1f GB/s/tok-weights\n",
               prefill, per*1e3, prefill / t_pf,
               (pf_bytes / (double)prefill) / per / 1e9);
        printf("         last-hidden ||x||=%.3e  NaNs=%d\n", sqrt(xnorm), nan_count);
    }

    /* ---- decode (M=1, token-at-a-time) ---- */
    memset(m->prof, 0, sizeof(m->prof));
    double t_dec0 = now_sec();
    size_t dec_bytes = 0;
    int last_tok = 0;
    for (int g = 0; g < maxgen; g++) {
        int pos = prefill + g;
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
    double psum = 0; for (int i = 0; i < 8; i++) psum += m->prof[i];
    if (psum > 0 && maxgen > 0) {
        printf("\nper-phase decode profile (ms/tok, %% of accounted):\n");
        for (int i = 0; i < 8; i++) {
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
