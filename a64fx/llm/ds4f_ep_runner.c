/*
 * ds4f_ep_runner.c - DeepSeek-V4-Flash synthetic EP harness (Stage 2).
 *
 * Multi-node expert-parallel decode/prefill over pure uTofu (no MPI in the
 * binary; mpiexec only places one rank per node). Each rank owns a disjoint
 * ~1/N slice of the 256 routed experts (e % N == rank); dense weights
 * (MLA + shared expert + router + head) are REPLICATED and computed
 * redundantly. The per-layer MoE combine is a single tp_allreduce_sum over the
 * routed-expert partial [hidden] (Stage 2). Shared expert stays local.
 *
 * Weights are SYNTHETIC (filled in HBM, no disk). Logits are MEANINGLESS; the
 * validation targets are: per-node memory fit (~20-26 GB), cross-rank lockstep
 * (identical step count + identical synthetic argmax, since activations are
 * seeded identically on every rank and the all-reduce makes s_route identical),
 * and decode/prefill throughput split into compute vs all-reduce comm.
 *
 * Build (native A64FX):
 *   make -C a64fx/llm ds4f_ep_runner CC=fcc OPENMP=1
 * Run (after tofu_topo_helper writes tofu_topo.txt, 1 proc/node):
 *   mpiexec -n 11 [-vcoordfile vcoord] build/ds4f_ep_runner
 *
 * Env (in addition to ds4f.h's DS4F_*):
 *   LLM_THREADS    compute threads (default 48)
 *   DS4F_PREFILL   synthetic prefill tokens (default 8)
 *   DS4F_MAXGEN    synthetic decode tokens (default 16)
 *   DS4F_MAXPOS    KV cache capacity / max position (default 4096)
 *   DS4F_LAYERS    override n_layers (default 43)
 *   DS4F_FP8_BF16  predequant dense FP8->BF16 (default 0 = on-demand FP8)
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <fcntl.h>
#include <errno.h>
#include <utofu.h>

#include "ds4f.h"
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

#define MAX_NODES 128  /* DS4P 96-node 1M run (96=8x12=6x16); 108 for co-serving. Was 96; 32 for 11-node DS4F */
#define RUN_STAG  DEMO_STAG
#define WAIT_TIMEOUT_SEC 300.0   /* tolerate cold first-touch skew across ranks */

static FILE *g_log = NULL;
static void logmsg(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    if (g_log) { va_list ap2; va_copy(ap2, ap); vfprintf(g_log, fmt, ap2); va_end(ap2); fflush(g_log); }
    vfprintf(stderr, fmt, ap); va_end(ap);
}
static void die(const char *what, int rc) { logmsg("FATAL: %s (rc=%d)\n", what, rc); exit(1); }
static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int envi(const char *k, int d) { const char *v = getenv(k); return (v && *v) ? atoi(v) : d; }

/* splitmix64 -> deterministic synthetic activations (IDENTICAL on every rank) */
static uint64_t sm_state;
static double sm_next(void) {
    sm_state += 0x9E3779B97F4A7C15ull;
    uint64_t z = sm_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= (z >> 31);
    return (double)(z >> 11) / (double)(1ull << 53);
}
static size_t rss_bytes(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long total = 0, res = 0;
    if (fscanf(f, "%ld %ld", &total, &res) != 2) res = 0;
    fclose(f);
    return (size_t)res * (size_t)sysconf(_SC_PAGESIZE);
}

/* ---- real greedy generation (coding-task quality test) ---- */
#define DS4F_EOS_ID 1
/* token id -> input embedding (BF16 row [vocab,hidden] -> f32). The forward's
 * first op is the input RMSNorm (no embedding scaling), so the raw widened row
 * is exactly the activation ds4f_forward_token expects. */
static void embed_lookup(const ds4f_model *m, int tok, float *x) {
    int C = m->cfg.hidden;
    if (tok < 0 || tok >= m->cfg.vocab) tok = 0;
    if (m->emb_rows < m->cfg.vocab) {   /* DS4F_TP_EMBED: vocab-sharded -> owner fills its row, others zero */
        for (int i = 0; i < C; i++) x[i] = 0.f;
        if (tok >= m->emb_r0 && tok < m->emb_r0 + m->emb_rows) {
            const uint16_t *row = m->embed + (size_t)(tok - m->emb_r0) * C;
            for (int i = 0; i < C; i++) { union { uint32_t u; float f; } z; z.u = (uint32_t)row[i] << 16; x[i] = z.f; }
        }
        if (m->ar_cb) m->ar_cb(x, C, m->ar_ctx);   /* sum owner's row + zeros -> full embedding (BIT-EXACT) */
        return;
    }
    const uint16_t *row = m->embed + (size_t)tok * C;
    for (int i = 0; i < C; i++) {
        union { uint32_t u; float f; } z; z.u = (uint32_t)row[i] << 16; x[i] = z.f;
    }
}

/* ================= HTTP serve mode (DS4F_SERVE): load once, loop on requests ================= */
/* ---- sampling: temperature / top-k / top-p / presence & repeat penalty ----
 * The head is replicated in the serve config, so every rank holds the SAME full logits. A
 * deterministic PRNG (SplitMix64) seeded identically per request and advanced in lockstep makes
 * all 11 ranks draw the SAME token -> lockstep preserved (== the greedy-argmax guarantee). */
static uint64_t ds4f_rng_state;
static inline double ds4f_rng_u01_st(uint64_t *st) {       /* uniform [0,1) from an explicit SplitMix64 state */
    uint64_t z = (*st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}
static inline double ds4f_rng_u01(void) { return ds4f_rng_u01_st(&ds4f_rng_state); }
typedef struct { float temp, top_p, pres_pen, rep_pen; int top_k, rep_last_n; } ds4f_sampler;

static const float *ds4f_srt_key;                          /* qsort key (single-threaded serve path) */
static int ds4f_srt_desc(const void *a, const void *b) {
    float fa = ds4f_srt_key[*(const int *)a], fb = ds4f_srt_key[*(const int *)b];
    return (fa < fb) - (fa > fb);
}
/* Sample the next token from an explicit logits buffer `lg[V]` and per-caller PRNG `*rng`, given recent
 * token history (for penalties). temp<=0 -> greedy argmax. Mutates lg (temperature/penalties applied in
 * place; the caller's buffer is regenerated next step). Deterministic in (lg, *rng): identical logits +
 * identical rng state on every rank -> identical token -> lockstep preserved. The batched path calls this
 * once per active sequence with that sequence's own full-logit row + own rng state. */
static int ds4f_sample_logits(float *lg, int V, const ds4f_sampler *sp, uint64_t *rng,
                              const int *hist, int nhist) {
    if (sp->temp <= 0.f) {                                  /* greedy */
        int best = 0; float bv = lg[0];
        for (int v = 1; v < V; v++) if (lg[v] > bv) { bv = lg[v]; best = v; }
        return best;
    }
    static int *idx = NULL; static float *prob = NULL;      /* scratch, sized once to vocab */
    if (!idx) { idx = (int *)ds4f_xmalloc((size_t)V * sizeof(int), "idx"); prob = (float *)ds4f_xmalloc((size_t)V * sizeof(float), "prob"); }
    /* repeat + presence penalty over the last rep_last_n tokens (once per unique token) */
    if (sp->rep_pen != 1.f || sp->pres_pen != 0.f) {
        int ln = sp->rep_last_n > 0 ? sp->rep_last_n : 64;
        int h0 = nhist > ln ? nhist - ln : 0;
        for (int i = h0; i < nhist; i++) {
            int t = hist[i]; if (t < 0 || t >= V) continue;
            int first = 1; for (int j = h0; j < i; j++) if (hist[j] == t) { first = 0; break; }
            if (!first) continue;
            if (sp->rep_pen != 1.f)  lg[t] = lg[t] > 0.f ? lg[t] / sp->rep_pen : lg[t] * sp->rep_pen;
            if (sp->pres_pen != 0.f) lg[t] -= sp->pres_pen;
        }
    }
    float inv = 1.f / sp->temp;                             /* temperature + sort desc */
    for (int v = 0; v < V; v++) { lg[v] *= inv; idx[v] = v; }
    ds4f_srt_key = lg; qsort(idx, (size_t)V, sizeof(int), ds4f_srt_desc);
    int kcut = (sp->top_k > 0 && sp->top_k < V) ? sp->top_k : V;
    float mx = lg[idx[0]], sum = 0.f;                       /* stable softmax over top-k */
    for (int i = 0; i < kcut; i++) { float e = expf(lg[idx[i]] - mx); prob[i] = e; sum += e; }
    int npc = kcut;                                         /* top-p nucleus */
    if (sp->top_p < 1.f && sp->top_p > 0.f) {
        float cum = 0.f, thr = sp->top_p * sum;
        for (int i = 0; i < kcut; i++) { cum += prob[i]; if (cum >= thr) { npc = i + 1; break; } }
    }
    float nsum = 0.f; for (int i = 0; i < npc; i++) nsum += prob[i];   /* renormalize + draw */
    double r = ds4f_rng_u01_st(rng) * nsum, acc = 0.0;
    for (int i = 0; i < npc; i++) { acc += prob[i]; if (acc >= r) return idx[i]; }
    return idx[npc - 1];
}
/* single-stream serve wrapper: sample from m->s_logits with the global PRNG (bit-identical to before). */
static int ds4f_sample(ds4f_model *m, const ds4f_sampler *sp, const int *hist, int nhist) {
    return ds4f_sample_logits(m->s_logits, m->cfg.vocab, sp, &ds4f_rng_state, hist, nhist);
}
/* reset per-request state: the compressor/indexer RING state carries between requests (position-
 * indexed KV/cmp/idx caches self-overwrite at pos 0). int8 KV/cmp calibration re-runs per request. */
static void ds4f_serve_reset(ds4f_model *m) {
    ds4f_config *c = &m->cfg;
    for (int L = 0; L < c->n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        int ratio = c->compress_ratios[L];
        if (m->tierb2 && ratio) {
            ds4f_compress_state_reset(ly->cmp_kv_state, ly->cmp_score_state, ratio, c->kv_lora);
            if (ratio == 4) ds4f_compress_state_reset(ly->idx_cmp_kv_state, ly->idx_cmp_score_state, ratio, c->index_head_dim);
        }
        ly->kv_caln = 0; ly->kv_frozen = 0; ly->cmp_caln = 0; ly->cmp_frozen = 0;   /* int8 recalibrate */
        ly->sel_cache_pos = -1;   /* DS4F_IDX_REUSE: invalidate the cached selection at each request boundary */
    }
}
/* one generation: prefill the prompt positions [pf_from, np) -- pf_from>0 reuses a cached prefix
 * already in the KV/compressor caches (see the prefix-search in the serve loop) -- then decode up to
 * max_new (stop on eos). sp->temp>0 -> sample (temp/top-k/top-p/penalties) from the per-position
 * logits, else greedy argmax. Returns n_gen written to out_ids. All ranks lockstep. */
static int ds4f_serve_gen(ds4f_model *m, const int *pids, int np, int max_new, float *x, int *out_ids,
                          const ds4f_sampler *sp, int pf_from) {
    int C = m->cfg.hidden, V = m->cfg.vocab, pf_last = 0, have_logits = 0;
    int pf_gemm = envi("DS4F_PREFILL_GEMM", 0), K = envi("DS4F_PREFILL_K", 32);
    int sampling = sp && sp->temp > 0.f;
    m->want_full_logits = sampling;   /* sampling reads every logit (temp/top_p/top_k); greedy only needs
                                        * argmax -> lets TP_HEAD decode use the cheap argmax-merge path */
    if (pf_from < 0 || pf_from >= np) pf_from = 0;       /* must prefill >=1 position (for pf_last logits) */
    if (K < 1) K = 1; if (K > 128) K = 128;   /* verify path caps at 128 */
    if (pf_gemm && !m->has_mtp) {                       /* batched-verify prefill */
        ds4f_alloc_prefill_batch(m, K);
        size_t hcC = (size_t)m->cfg.hc_mult * C;
        float *Xin = (float *)ds4f_xalloc(64, (size_t)K*C*4, "Xin"), *vhc = (float *)ds4f_xalloc(64, (size_t)K*hcC*4, "vhc");
        int lastM = 0;
        for (int base = pf_from; base < np; base += K) {
            int M = np - base < K ? np - base : K; lastM = M;
            for (int mm = 0; mm < M; mm++) embed_lookup(m, pids[base+mm], Xin + (size_t)mm*C);
            int ot[128]; ds4f_forward_verify(m, Xin, M, base, ot, vhc); pf_last = ot[M-1];
        }
        free(Xin); free(vhc);
        /* first-token sampling needs the last prompt position's logits; verify leaves them in
         * m->p_logits[lastM-1] (full vocab only when the head is replicated, which the serve config is). */
        if (m->head.rows == V && m->p_logits) {
            memcpy(m->s_logits, m->p_logits + (size_t)(lastM-1)*V, (size_t)V*4); have_logits = 1;
        }
    } else {                                            /* token-by-token prefill (m->s_logits left set) */
        for (int p = pf_from; p < np; p++) { embed_lookup(m, pids[p], x); pf_last = ds4f_forward_token(m, x, p); }
        have_logits = 1;
    }
    int *hist = (int *)ds4f_xmalloc((size_t)(np + max_new) * sizeof(int), "hist");   /* penalty history: prompt + gen */
    memcpy(hist, pids, (size_t)np * sizeof(int)); int nh = np;
    int n_gen = 0, cur = (sampling && have_logits) ? ds4f_sample(m, sp, hist, nh) : pf_last;
    for (int g = 0; g < max_new; g++) {
        out_ids[n_gen++] = cur; hist[nh++] = cur;
        if (cur == DS4F_EOS_ID) break;
        embed_lookup(m, cur, x);
        int am = ds4f_forward_token(m, x, np + g);       /* fills m->s_logits, returns argmax */
        cur = sampling ? ds4f_sample(m, sp, hist, nh) : am;
    }
    free(hist);
    return n_gen;
}
static long ds4f_read_seq(const char *path);   /* fwd (defined below) */
static void barrier(void);                      /* fwd (defined below) */
static int MyRank;                              /* fwd tentative def (defined with N below) */
/* ================= batched serve: decode up to B sequences concurrently (DS4F_SERVE_BATCH) =================
 * The throughput path. B persistent per-sequence cache bundles (ds4f_alloc_decode_batch, alloc once); each
 * request is prefilled into its own bundle, then all still-active sequences are decode-stepped TOGETHER via
 * one ds4f_forward_verify (dense GEMM + the per-layer EP reduce amortize across the batch -- dbbench M=16
 * measured ~32 tok/s aggregate vs ~17 single-stream). Greedy (per-seq argmax); each sequence independent
 * (its own bundle + position), so a seq's output does not depend on its batch-mates. All ranks read the
 * same batch request (shared FS) -> lockstep. Static batching: a batch is prefilled + drained to completion
 * before the next is admitted (dynamic mid-flight admission is a follow-on).
 * Request  file: "BATCH <N>\n" then per seq: "<max_new>\n<id id ...>\n".
 * Response file: "<N>\n" then per seq: "<gen id id ...>\n". */
static void ds4f_serve_batch_loop(ds4f_model *m, int B, int maxpos,
                                  const char *reqf, const char *respf,
                                  const char *reqseqf, const char *respseqf) {
    ds4f_config *c = &m->cfg; int Cc = c->hidden, L = c->n_layers; size_t hcC = (size_t)c->hc_mult * Cc;
    int K = envi("DS4F_PREFILL_K", 64); if (K < 1) K = 1; if (K > 128) K = 128;
    int pf_gemm = envi("DS4F_PREFILL_GEMM", 1) && !m->has_mtp;
    int mtile = B > K ? B : K; if (mtile > 128) mtile = 128;
    m->want_full_logits = 0;                 /* greedy: cheap argmax-merge head */
    ds4f_alloc_prefill_batch(m, mtile);
    ds4f_alloc_decode_batch(m, B);           /* B persistent bundles; bundle i = &bundles[i*L] */
    ds4f_lseq *bundles = m->dec_batch_seq;
    ds4f_lseq *view = (ds4f_lseq *)ds4f_xmalloc((size_t)B * L * sizeof(ds4f_lseq), "view");
    int   *vpos = (int *)ds4f_xmalloc((size_t)B * sizeof(int), "vpos");
    /* Xb/hcb feed forward_verify with up to max(B, prefill-chunk K) rows -> size to mtile, not B. */
    float *Xb   = (float *)ds4f_xalloc(64, (size_t)mtile * Cc * 4, "Xb");
    float *hcb  = (float *)ds4f_xalloc(64, (size_t)mtile * hcC * 4, "hcb");
    int   *otb  = (int *)ds4f_xmalloc((size_t)mtile * sizeof(int), "otb");
    int   *Xin  = (int *)ds4f_xmalloc((size_t)maxpos * sizeof(int), "Xin");
    typedef struct { int *ids; int nids, np, pos, mnew, active;
                     ds4f_sampler samp; uint64_t rng; int samples; } bseq;
    bseq *S = (bseq *)ds4f_xcalloc((size_t)B, sizeof(bseq), "S");
    long req_ctr = 0;   /* monotonic per-request counter (identical across ranks) -> default seed source */
    for (int i = 0; i < B; i++) S[i].ids = (int *)ds4f_xmalloc((size_t)(maxpos + 1) * sizeof(int), "ids");
    float *xscr = (float *)ds4f_xalloc(64, (size_t)Cc * 4, "xscr");   /* per-token embed scratch */
    long last_seq = ds4f_read_seq(reqseqf);
    if (MyRank == 0) logmsg("SERVE-BATCH ready: B=%d maxpos=%d prefill_gemm=%d K=%d\n", B, maxpos, pf_gemm, K);
    for (;;) {
        long seq = last_seq;
        if (MyRank == 0) while ((seq = ds4f_read_seq(reqseqf)) <= last_seq) usleep(2000);
        barrier();
        int N = 0;
        FILE *rf = fopen(reqf, "r");
        if (rf) {
            char line[256]; int want = 0;
            if (fgets(line, sizeof line, rf)) sscanf(line, "BATCH %d", &want);
            if (want > B) want = B;
            for (int i = 0; i < want; i++) {
                int mnew = 256;
                /* per-seq line: "max_new [temp top_p top_k seed rep_pen pres_pen]" (sampling optional). */
                float temp = 0.f, top_p = 1.f, rep_pen = 1.f, pres_pen = 0.f; int top_k = 0; long seed = 0;
                if (!fgets(line, sizeof line, rf)) break;
                sscanf(line, "%d %f %f %d %ld %f %f", &mnew, &temp, &top_p, &top_k, &seed, &rep_pen, &pres_pen);
                int np = 0, v; char *tok, *sp;
                char *ln = NULL; size_t lc = 0;
                if (getline(&ln, &lc, rf) < 0) { free(ln); break; }
                for (tok = strtok_r(ln, " \t\n", &sp); tok && np < maxpos; tok = strtok_r(NULL, " \t\n", &sp))
                    { if (sscanf(tok, "%d", &v) == 1) Xin[np++] = v; }
                free(ln);
                if (mnew < 0) mnew = 0; if (np + mnew > maxpos) mnew = maxpos > np ? maxpos - np : 0;
                bseq *s = &S[N]; s->np = np; s->mnew = mnew; s->nids = 0; s->pos = 0;
                memcpy(s->ids, Xin, (size_t)np * sizeof(int)); s->nids = np;
                /* per-seq sampler + PRNG (identical on every rank -> identical draws -> lockstep). seed 0
                 * => derive from a monotonic per-request counter (unique across batches, rank-deterministic). */
                s->samp.temp = temp; s->samp.top_p = top_p; s->samp.top_k = top_k;
                s->samp.rep_pen = rep_pen; s->samp.pres_pen = pres_pen; s->samp.rep_last_n = 64;
                s->samples = (temp > 0.f);
                { uint64_t bs = seed ? (uint64_t)seed : (uint64_t)(req_ctr + 1);
                  s->rng = bs * 0x9E3779B97F4A7C15ULL; }
                req_ctr++;
                N++;
            }
            fclose(rf);
        }
        double t0 = now_sec(); long pf_tokens = 0;
        /* ---- prefill each request into its own bundle (single-seq path: dec_batch_seq=NULL) ---- */
        for (int i = 0; i < N; i++) {
            bseq *s = &S[i];
            for (int l = 0; l < L; l++) ds4f_lseq_apply(&m->layers[l], &bundles[(size_t)i*L + l]);
            m->dec_batch_seq = NULL; m->dec_batch_pos = NULL;
            ds4f_serve_reset(m);
            int first = DS4F_EOS_ID;
            if (s->np > 0) {
                if (pf_gemm) {
                    for (int base = 0; base < s->np; base += K) {
                        int M = s->np - base < K ? s->np - base : K;
                        for (int mm = 0; mm < M; mm++) embed_lookup(m, s->ids[base+mm], Xb + (size_t)mm*Cc);
                        m->want_full_logits = (s->samples && base + K >= s->np);   /* last chunk: first-token draw */
                        int ot[128]; ds4f_forward_verify(m, Xb, M, base, ot, hcb); first = ot[M-1];
                        if (m->want_full_logits)
                            first = ds4f_sample_logits(m->p_logits_full + (size_t)(M-1)*c->vocab, c->vocab,
                                                       &s->samp, &s->rng, s->ids, s->np);
                    }
                } else {
                    m->want_full_logits = s->samples;
                    for (int p = 0; p < s->np; p++) { embed_lookup(m, s->ids[p], xscr); first = ds4f_forward_token(m, xscr, p); }
                    if (s->samples) first = ds4f_sample(m, &s->samp, s->ids, s->np);
                }
                pf_tokens += s->np;
            }
            m->want_full_logits = 0;
            /* persist this request's per-sequence calibration scalars into its bundle (prefill wrote the
             * bundle's buffers via the applied pointers; the frozen/caln ints must be captured for decode). */
            for (int l = 0; l < L; l++) ds4f_lseq_capture(&bundles[(size_t)i*L + l], &m->layers[l]);
            s->pos = s->np;   /* the first gen token (= last-prefill argmax) occupies position np; the
                               * first decode step forwards it AT pos=np (then pos advances). Do NOT
                               * pre-increment pos here or position np's KV is never written (a gap). */
            s->active = (s->np > 0 && s->mnew > 0);
            if (s->active) { s->ids[s->nids++] = first;             /* record it; decode forwards it at pos=np */
                             if (first == DS4F_EOS_ID || s->nids - s->np >= s->mnew) s->active = 0; }
        }
        double t_pf = now_sec() - t0;
        /* ---- decode the active set together, one forward_verify per step ---- */
        long dec_steps = 0, dec_toks = 0;
        for (;;) {
            int na = 0, map[128];
            for (int i = 0; i < N && na < B; i++) if (S[i].active) {
                map[na] = i;
                embed_lookup(m, S[i].ids[S[i].nids-1], Xb + (size_t)na*Cc);
                vpos[na] = S[i].pos;
                for (int l = 0; l < L; l++) view[(size_t)na*L + l] = bundles[(size_t)i*L + l];
                na++;
            }
            if (na == 0) break;
            int any_sample = 0;                              /* full [na,vocab] logits iff a seq samples */
            for (int a = 0; a < na; a++) if (S[map[a]].samples) { any_sample = 1; break; }
            m->want_full_logits = any_sample;
            m->dec_batch_seq = view; m->dec_batch_pos = vpos;
            ds4f_forward_verify(m, Xb, na, 0, otb, hcb);
            /* view is a COPY of the active bundles -> sync the per-seq state captured into it back home. */
            for (int a = 0; a < na; a++) for (int l = 0; l < L; l++)
                ds4f_lseq_sync(&bundles[(size_t)map[a]*L + l], &view[(size_t)a*L + l]);
            dec_steps++; dec_toks += na;
            for (int a = 0; a < na; a++) {
                bseq *s = &S[map[a]];
                int tok = s->samples ? ds4f_sample_logits(m->p_logits_full + (size_t)a*c->vocab, c->vocab,
                                                          &s->samp, &s->rng, s->ids, s->nids)
                                     : otb[a];
                s->ids[s->nids++] = tok; s->pos++;
                if (tok == DS4F_EOS_ID || s->nids - s->np >= s->mnew) s->active = 0;
            }
        }
        double dt = now_sec() - t0;
        if (MyRank == 0) {
            FILE *of = fopen(respf, "w");
            if (of) {
                fprintf(of, "%d\n", N);
                for (int i = 0; i < N; i++) {
                    int ng = S[i].nids - S[i].np;   /* generated tokens (after the prompt) */
                    for (int g = 0; g < ng; g++) fprintf(of, "%d%s", S[i].ids[S[i].np + g], g+1 < ng ? " " : "");
                    fprintf(of, "\n");
                }
                fclose(of);
            }
            long rq = ds4f_read_seq(reqseqf);
            FILE *sf = fopen(respseqf, "w"); if (sf) { fprintf(sf, "%ld\n", rq); fclose(sf); }
            long tot_gen = 0; for (int i = 0; i < N; i++) tot_gen += S[i].nids - S[i].np;
            logmsg("SERVE-BATCH req#%ld: N=%d prefill=%ld tok in %.2fs, decode %ld steps %ld tok in %.2fs -> %.1f tok/s agg\n",
                   rq, N, pf_tokens, t_pf, dec_steps, dec_toks, dt - t_pf, (dt-t_pf) > 0 ? tot_gen/(dt-t_pf) : 0.0);
        }
        barrier();
        last_seq = ds4f_read_seq(reqseqf);
    }
}
static long ds4f_read_seq(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return 0;
    long v = 0; if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); return v;
}

/* ================= DYNAMIC continuous batching (DS4F_SERVE_DYNAMIC) =================
 * True continuous batching: requests are admitted MID-FLIGHT (as slots free), so a fast request never
 * waits for slow batch-mates. Queue protocol on the shared FS (base = reqf minus ".req"):
 *   <base>.qhead      monotonic count of enqueued requests (frontend bumps AFTER writing the q file)
 *   <base>.q.<id>     request id: "<max_new>\n<id id ...>\n"
 *   <base>.r.<id>     response for id (rank 0 writes on retirement): "<gen id ...>\n"
 * LOCKSTEP across the 11 EP ranks is the constraint: every rank must run the SAME sequence of collectives
 * (each prefill + each decode step is a barrier-synchronized forward_verify). Two decisions must match on
 * every rank: (a) RETIREMENT -- deterministic, greedy argmax is identical across ranks (replicated head);
 * (b) ADMISSION count -- rank 0 reads qhead and BROADCASTS it via ar_cb (rank-0-authoritative sum), so all
 * ranks admit the same requests into the same (lowest-free) slots each round. Idle rounds (no active, none
 * pending) still call the broadcast + usleep on every rank, staying aligned. */
/* ================= socket transport (DS4F_SERVE_SOCK) =================
 * The file protocol's admission/response latency is dominated by cross-node FEFS/LLIO cache visibility
 * (~tens of seconds: client writes q.<id> on the login node, the compute-node runner sees it only after
 * the attr/dentry cache refreshes). TCP over the Tofu IP interface between the login node and the runner's
 * rank-0 compute node is ~0.7 ms round-trip (measured), so rank 0 hosts a TCP listener and the frontend
 * connects per request. ONLY rank 0 touches sockets; the received request is broadcast to the EP group via
 * ar_cb exactly like the file path, so lockstep is unchanged. Frame = [u32 BE length][payload]; request
 * payload is the same 2-line text as a q.<id> file ("max_new temp top_p top_k seed rep_pen pres_pen\n
 * ids...\n"); response payload is "gen_id gen_id ...\n". One request per connection (frontend closes after
 * reading the response); connect is cheap. */
#define DS4F_SOCK_MAXCONN 256
typedef struct { int fd; unsigned char *buf; int len, cap; int have; } ds4f_sconn;  /* have=full frame buffered */
typedef struct { int listen_fd; ds4f_sconn conns[DS4F_SOCK_MAXCONN]; } ds4f_sock;

static void ds4f_sock_myip(char *out, size_t n) {   /* prefer a Tofu 10.x IPv4 addr for this node */
    struct ifaddrs *ifa, *p; out[0] = 0;
    if (getifaddrs(&ifa) != 0) return;
    for (p = ifa; p; p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &((struct sockaddr_in *)p->ifa_addr)->sin_addr, ip, sizeof ip);
        if (!strncmp(ip, "10.", 3)) { snprintf(out, n, "%s", ip); break; }
    }
    freeifaddrs(ifa);
}
/* rank 0: create the non-blocking listener, publish "<ip> <port>" to <base>.sock (atomic temp+rename). */
static int ds4f_sock_init(ds4f_sock *S, const char *base) {
    memset(S, 0, sizeof *S);
    S->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (S->listen_fd < 0) return -1;
    int one = 1; setsockopt(S->listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = 0;   /* ephemeral port */
    if (bind(S->listen_fd, (struct sockaddr *)&a, sizeof a) != 0) return -1;
    socklen_t al = sizeof a; getsockname(S->listen_fd, (struct sockaddr *)&a, &al);
    if (listen(S->listen_fd, 128) != 0) return -1;
    fcntl(S->listen_fd, F_SETFL, O_NONBLOCK);
    char ip[64]; ds4f_sock_myip(ip, sizeof ip);
    char path[1152], tmp[1160]; snprintf(path, sizeof path, "%s.sock", base); snprintf(tmp, sizeof tmp, "%s.sock.t", base);
    FILE *f = fopen(tmp, "w"); if (f) { fprintf(f, "%s %d\n", ip, (int)ntohs(a.sin_port)); fclose(f); rename(tmp, path); }
    return 0;
}
static void ds4f_sock_close(ds4f_sock *S, int i) {
    if (S->conns[i].fd < 0) return;
    close(S->conns[i].fd); free(S->conns[i].buf);
    S->conns[i].fd = -1; S->conns[i].buf = NULL; S->conns[i].len = S->conns[i].cap = S->conns[i].have = 0;
}
/* accept pending connections + drain readable bytes into each conn's buffer, marking those with a complete
 * [u32 len][payload] frame as have=1. Non-blocking; call once per admit round on rank 0. */
static void ds4f_sock_poll(ds4f_sock *S) {
    for (;;) {                                        /* accept all pending */
        int c = accept(S->listen_fd, NULL, NULL);
        if (c < 0) break;
        fcntl(c, F_SETFL, O_NONBLOCK);
        int one = 1; setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        int slot = -1; for (int i = 0; i < DS4F_SOCK_MAXCONN; i++) if (S->conns[i].fd == 0 || S->conns[i].fd == -1) { slot = i; break; }
        if (slot < 0) { close(c); continue; }         /* table full -> drop */
        S->conns[slot].fd = c; S->conns[slot].buf = NULL; S->conns[slot].len = S->conns[slot].cap = S->conns[slot].have = 0;
    }
    for (int i = 0; i < DS4F_SOCK_MAXCONN; i++) {
        ds4f_sconn *cn = &S->conns[i];
        if (cn->fd <= 0 || cn->have) continue;
        for (;;) {
            if (cn->len + 4096 > cn->cap) { cn->cap = cn->cap ? cn->cap * 2 : 8192; cn->buf = (unsigned char *)realloc(cn->buf, cn->cap); }
            ssize_t r = recv(cn->fd, cn->buf + cn->len, cn->cap - cn->len, 0);
            if (r > 0) { cn->len += (int)r; continue; }
            if (r == 0) { ds4f_sock_close(S, i); break; }               /* peer closed before a full frame */
            break;                                                      /* EAGAIN */
        }
        if (cn->fd > 0 && cn->len >= 4) {                                /* have the length prefix? */
            uint32_t need; memcpy(&need, cn->buf, 4); need = ntohl(need);
            if (cn->len >= 4 + (int)need) cn->have = 1;                  /* full frame buffered */
        }
    }
}
/* pop the first conn with a complete request: copy its payload (NUL-terminated) into out, reset that
 * conn's read buffer (so the frame isn't re-detected while it awaits its response), return the conn index
 * (-1 if none ready). The conn stays open until ds4f_sock_respond. */
static int ds4f_sock_take(ds4f_sock *S, char *out, int outcap) {
    for (int i = 0; i < DS4F_SOCK_MAXCONN; i++) if (S->conns[i].fd > 0 && S->conns[i].have) {
        uint32_t need; memcpy(&need, S->conns[i].buf, 4); need = ntohl(need);
        int n = (int)need < outcap - 1 ? (int)need : outcap - 1;
        memcpy(out, S->conns[i].buf + 4, n); out[n] = 0;
        S->conns[i].len = 0; S->conns[i].have = 0;      /* consumed; conn awaits its response */
        return i;
    }
    return -1;
}
/* send a framed response on conn i, then close it (one request per connection). */
static void ds4f_sock_respond(ds4f_sock *S, int i, const int *g, int ng) {
    if (i < 0 || S->conns[i].fd <= 0) return;
    char *body = (char *)ds4f_xmalloc((size_t)ng * 12 + 2, "body"); int bl = 0;
    for (int k = 0; k < ng; k++) bl += sprintf(body + bl, "%d%s", g[k], k+1<ng?" ":"");
    body[bl++] = '\n';
    uint32_t nl = htonl((uint32_t)bl);
    unsigned char *frame = (unsigned char *)ds4f_xmalloc(4 + bl, "frame");
    memcpy(frame, &nl, 4); memcpy(frame + 4, body, bl);
    int off = 0, tot = 4 + bl, fd = S->conns[i].fd;
    while (off < tot) { ssize_t w = send(fd, frame + off, tot - off, 0);
        if (w > 0) off += (int)w; else if (w < 0 && errno == EAGAIN) continue; else break; }
    free(body); free(frame);
    ds4f_sock_close(S, i);
}

/* atomic response write (temp + rename) so the frontend, which polls for <base>.r.<id> existence,
 * never reads a partially-written file. */
static void ds4f_dyn_resp(const char *base, long id, const int *g, int ng) {
    char fin[1152], tmp[1160];
    snprintf(fin, sizeof fin, "%s.r.%ld", base, id);
    snprintf(tmp, sizeof tmp, "%s.r.%ld.t", base, id);
    FILE *of = fopen(tmp, "w");
    if (of) { for (int i = 0; i < ng; i++) fprintf(of, "%d%s", g[i], i+1<ng?" ":""); fprintf(of, "\n"); fclose(of); rename(tmp, fin); }
}
static void ds4f_serve_dynbatch_loop(ds4f_model *m, int B, int maxpos, const char *reqf) {
    ds4f_config *c = &m->cfg; int Cc = c->hidden, L = c->n_layers; size_t hcC = (size_t)c->hc_mult * Cc;
    int K = envi("DS4F_PREFILL_K", 64); if (K < 1) K = 1; if (K > 128) K = 128;
    int pf_gemm = envi("DS4F_PREFILL_GEMM", 1) && !m->has_mtp;
    int mtile = B > K ? B : K; if (mtile > 128) mtile = 128;
    m->want_full_logits = 0;
    ds4f_alloc_prefill_batch(m, mtile);
    ds4f_alloc_decode_batch(m, B);
    ds4f_lseq *bundles = m->dec_batch_seq;
    ds4f_lseq *view = (ds4f_lseq *)ds4f_xmalloc((size_t)B * L * sizeof(ds4f_lseq), "view");
    int   *vpos = (int *)ds4f_xmalloc((size_t)B * sizeof(int), "vpos");
    float *Xb   = (float *)ds4f_xalloc(64, (size_t)mtile * Cc * 4, "Xb");
    float *hcb  = (float *)ds4f_xalloc(64, (size_t)mtile * hcC * 4, "hcb");
    int   *otb  = (int *)ds4f_xmalloc((size_t)mtile * sizeof(int), "otb");
    /* admission broadcast buffer: [ok,mnew,np, temp,top_p,top_k,seed,rep_pen,pres_pen, ids...] -- rank 0
     * reads the fresh q.<id> file and BROADCASTS it so every rank admits an identical request (payload AND
     * sampling params). Cross-node FS-cache skew must never make ranks disagree on the collective's M
     * (that deadlocks the prefill/decode all-reduce) NOR on the per-seq sampler/seed (that would make ranks
     * draw different tokens -> lockstep divergence). */
    const int HDR = 9;
    float *admit_bc = (float *)ds4f_xalloc(64, ((size_t)maxpos + HDR + 1) * sizeof(float), "admit_bc");
    typedef struct { int *ids; int nids, np, pos, mnew, active; long id;
                     ds4f_sampler samp; uint64_t rng; int samples; int cfd; } dseq;
    dseq *S = (dseq *)ds4f_xcalloc((size_t)B, sizeof(dseq), "S");
    for (int i = 0; i < B; i++) { S[i].ids = (int *)ds4f_xmalloc((size_t)(maxpos + 1) * sizeof(int), "ids"); S[i].cfd = -1; }
    char base[1024]; int rl = (int)strlen(reqf); if (rl > 4) rl -= 4;   /* strip ".req" */
    snprintf(base, sizeof base, "%.*s", rl, reqf);
    /* socket transport (rank 0 only): TCP listener over the Tofu IP, published to <base>.sock. Requests
     * arrive on sockets instead of q.<id> files (no cross-node FS-cache admission latency). */
    int sock_mode = envi("DS4F_SERVE_SOCK", 0);
    ds4f_sock sock; char *sockpb = (char *)ds4f_xmalloc((size_t)maxpos * 8 + 64, "sockpb");
    if (sock_mode && MyRank == 0) { if (ds4f_sock_init(&sock, base) != 0) { logmsg("DS4F_SERVE_SOCK: listener init failed -> file mode\n"); sock_mode = 0; } }
    char qheadf[1088]; snprintf(qheadf, sizeof qheadf, "%s.qhead", base);
    if (MyRank == 0) {   /* ensure qhead exists BEFORE the first read -- else the node-local FS caches a
                          * "not found" and rank 0 never sees the frontend's later writes (the reason the
                          * launcher pre-creates reqseq). Create-if-absent, preserving any existing value. */
        FILE *qf = fopen(qheadf, "a"); if (qf) fclose(qf);
    }
    barrier();
    long qnext = ds4f_read_seq(qheadf);   /* skip requests enqueued before we came up (best-effort) */
    long done_count = 0; double t_last = now_sec(); long tok_since = 0;
    if (MyRank == 0) logmsg("SERVE-DYNBATCH ready: B=%d maxpos=%d prefill_gemm=%d K=%d transport=%s (continuous batching)\n",
                            B, maxpos, pf_gemm, K, sock_mode ? "socket" : "file");
    for (;;) {
        /* ---- admit into free slots by PROBING q.<qnext> directly. We do NOT read a qhead counter: its
         *      value change ("1\n"->"4\n", same 2-byte size) is not reliably visible on the compute-node
         *      FEFS/LLIO client (it caches the small file by size+coarse-mtime and never refetches -> rank 0
         *      sees a stale qhead and never admits). New-FILE existence IS coherent (create takes a lock,
         *      negative-dentry TTL is short), so rank 0 probes+reads q.<qnext> and broadcasts [ok,mnew,np,
         *      ids]; ab[0]<0.5 => no request waiting. Broadcasting keeps every rank's admit lockstep -- a
         *      per-rank read would race the FS cache into divergent np -> prefill all-reduce deadlock. int
         *      ids fit float32 exactly (vocab << 2^24). ---- */
        for (int i = 0; i < B; i++) {
            if (S[i].active) continue;                       /* slot busy (deterministic across ranks) */
            dseq *s = &S[i];
            int cap = maxpos + HDR, taken_cfd = -1;
            float *ab = admit_bc; memset(ab, 0, (size_t)cap * sizeof(float));
            if (MyRank == 0) {
                /* rank 0 fills ab[] from ONE waiting request (socket conn or q.<qnext> file); the parsed
                 * payload is identical text in both modes. ab[0]=1 => a request was taken. */
                char line1[256] = {0}, *ids_line = NULL, *pbuf = NULL; int got = 0;
                if (sock_mode) {
                    ds4f_sock_poll(&sock);
                    int ci = ds4f_sock_take(&sock, sockpb, (int)((size_t)maxpos*8+64));
                    if (ci >= 0) { taken_cfd = ci; pbuf = sockpb; got = 1;
                        char *nl = strchr(pbuf, '\n');
                        if (nl) { *nl = 0; snprintf(line1, sizeof line1, "%s", pbuf); ids_line = nl + 1; }
                        else { snprintf(line1, sizeof line1, "%s", pbuf); ids_line = pbuf + strlen(pbuf); } }
                } else {
                    char qf[1152]; snprintf(qf, sizeof qf, "%s.q.%ld", base, qnext);
                    FILE *f = fopen(qf, "r");
                    if (f) { char idl[65536] = {0};
                        if (fgets(line1, sizeof line1, f)) got = 1;
                        if (fgets(idl, sizeof idl, f)) { pbuf = strdup(idl); ids_line = pbuf; }
                        fclose(f); }
                }
                if (got) { int mnew = 256, np = 0, v; char *tk, *sp;
                    float temp = 0.f, top_p = 1.f, rep_pen = 1.f, pres_pen = 0.f; int top_k = 0; long seed = 0;
                    /* line 1: "mnew [temp top_p top_k seed rep_pen pres_pen]" (sampling optional, greedy default) */
                    sscanf(line1, "%d %f %f %d %ld %f %f", &mnew, &temp, &top_p, &top_k, &seed, &rep_pen, &pres_pen);
                    if (ids_line) for (tk = strtok_r(ids_line, " \t\n", &sp); tk && np < maxpos; tk = strtok_r(NULL, " \t\n", &sp))
                        if (sscanf(tk, "%d", &v) == 1) ab[HDR + np++] = (float)v;
                    ab[0] = 1.f; ab[1] = (float)mnew; ab[2] = (float)np;
                    ab[3] = temp; ab[4] = top_p; ab[5] = (float)top_k; ab[6] = (float)seed;
                    ab[7] = rep_pen; ab[8] = pres_pen;
                }
                if (!sock_mode && pbuf) free(pbuf);           /* strdup from the file path */
            }
            if (m->ar_cb && m->ep_size > 1) m->ar_cb(ab, cap, m->ar_ctx);
            if (ab[0] < 0.5f) break;                         /* nothing waiting -> retry next round */
            int mnew = (int)(ab[1] + 0.5f), np = (int)(ab[2] + 0.5f);
            for (int k = 0; k < np; k++) s->ids[k] = (int)(ab[HDR + k] + 0.5f);
            if (mnew < 0) mnew = 0; if (np + mnew > maxpos) mnew = maxpos > np ? maxpos - np : 0;
            s->np = np; s->mnew = mnew; s->nids = np; s->id = qnext; qnext++; s->cfd = taken_cfd;
            /* per-sequence sampler + PRNG (all ranks derive identical values -> identical draws -> lockstep).
             * temp<=0 => greedy (uses the batched argmax). seed 0 => derive from the request id (reproducible
             * per id; distinct across concurrent requests). */
            s->samp.temp = ab[3]; s->samp.top_p = ab[4]; s->samp.top_k = (int)(ab[5] + 0.5f);
            s->samp.rep_pen = ab[7]; s->samp.pres_pen = ab[8]; s->samp.rep_last_n = 64;
            s->samples = (s->samp.temp > 0.f);
            { long seed = (long)(ab[6] + 0.5f); uint64_t bs = seed ? (uint64_t)seed : (uint64_t)(s->id + 1);
              s->rng = bs * 0x9E3779B97F4A7C15ULL; }
            /* prefill into slot i's bundle (single-seq path) */
            for (int l = 0; l < L; l++) ds4f_lseq_apply(&m->layers[l], &bundles[(size_t)i*L + l]);
            m->dec_batch_seq = NULL; m->dec_batch_pos = NULL;
            ds4f_serve_reset(m);
            int first = DS4F_EOS_ID;
            if (np > 0) {
                if (pf_gemm) { for (int bs = 0; bs < np; bs += K) { int M = np - bs < K ? np - bs : K;
                        for (int mm = 0; mm < M; mm++) embed_lookup(m, s->ids[bs+mm], Xb + (size_t)mm*Cc);
                        /* sampling needs the LAST prompt position's full logits (to draw the first gen token);
                         * only the last chunk pays the full-[M,vocab] reduce -- earlier chunks stay greedy. */
                        m->want_full_logits = (s->samples && bs + K >= np);
                        int ot[128]; ds4f_forward_verify(m, Xb, M, bs, ot, hcb); first = ot[M-1];
                        if (m->want_full_logits)
                            first = ds4f_sample_logits(m->p_logits_full + (size_t)(M-1)*c->vocab, c->vocab,
                                                       &s->samp, &s->rng, s->ids, s->np); } }
                else { m->want_full_logits = s->samples;
                    for (int p = 0; p < np; p++) { embed_lookup(m, s->ids[p], Xb); first = ds4f_forward_token(m, Xb, p); }
                    if (s->samples) first = ds4f_sample(m, &s->samp, s->ids, s->np); }
            }
            m->want_full_logits = 0;
            /* persist this request's per-sequence calibration scalars into its bundle (see the static loop). */
            for (int l = 0; l < L; l++) ds4f_lseq_capture(&bundles[(size_t)i*L + l], &m->layers[l]);
            s->pos = np; s->active = (np > 0 && mnew > 0);
            if (s->active) { s->ids[s->nids++] = first;
                             if (first == DS4F_EOS_ID || s->nids - s->np >= s->mnew) s->active = 0; }
            if (!s->active && MyRank == 0) {                 /* prefill-only or immediate stop -> respond now */
                if (s->cfd >= 0) ds4f_sock_respond(&sock, s->cfd, s->ids + s->np, s->nids - s->np);
                else ds4f_dyn_resp(base, s->id, s->ids + s->np, s->nids - s->np);
                s->cfd = -1; done_count++;
            }
        }
        /* ---- gather the active set ---- */
        int na = 0, map[128];
        for (int i = 0; i < B; i++) if (S[i].active) {
            map[na] = i;
            embed_lookup(m, S[i].ids[S[i].nids-1], Xb + (size_t)na*Cc);
            vpos[na] = S[i].pos;
            for (int l = 0; l < L; l++) view[(size_t)na*L + l] = bundles[(size_t)i*L + l];
            na++;
        }
        if (na == 0) { usleep(2000); continue; }             /* idle: all ranks sleep, re-poll qhead */
        /* ---- one decode step over the active set ---- */
        int any_sample = 0;                                  /* if ANY active seq samples, reconstruct full
                                                              * [na,vocab] logits (greedy seqs just argmax it). */
        for (int a = 0; a < na; a++) if (S[map[a]].samples) { any_sample = 1; break; }
        m->want_full_logits = any_sample;
        m->dec_batch_seq = view; m->dec_batch_pos = vpos;
        ds4f_forward_verify(m, Xb, na, 0, otb, hcb);
        /* view is a COPY of the active bundles -> sync the per-seq state captured into it back home. */
        for (int a = 0; a < na; a++) for (int l = 0; l < L; l++)
            ds4f_lseq_sync(&bundles[(size_t)map[a]*L + l], &view[(size_t)a*L + l]);
        tok_since += na;
        for (int a = 0; a < na; a++) {
            dseq *s = &S[map[a]];
            int tok = s->samples ? ds4f_sample_logits(m->p_logits_full + (size_t)a*c->vocab, c->vocab,
                                                      &s->samp, &s->rng, s->ids, s->nids)
                                 : otb[a];
            s->ids[s->nids++] = tok; s->pos++;
            if (tok == DS4F_EOS_ID || s->nids - s->np >= s->mnew) {   /* retire -> respond, free slot */
                s->active = 0;
                if (MyRank == 0) {
                    if (s->cfd >= 0) ds4f_sock_respond(&sock, s->cfd, s->ids + s->np, s->nids - s->np);
                    else ds4f_dyn_resp(base, s->id, s->ids + s->np, s->nids - s->np);
                    s->cfd = -1; done_count++;
                }
            }
        }
        if (MyRank == 0) { double now = now_sec();
            if (now - t_last > 5.0) { logmsg("SERVE-DYNBATCH: %ld done, active=%d, %.1f tok/s agg (window)\n",
                                             done_count, na, tok_since/(now-t_last));
                                      t_last = now; tok_since = 0; } }
    }
}
/* ---- persist a context to disk (system-prompt cache / KV save-load) ----
 * File = [magic][npos][cfg_hash][ids(npos)][full cache snapshot for npos positions]. cfg_hash pins the
 * cache-mode layout (int8/int4/maxpos) so a mismatched runner refuses the blob instead of corrupting.
 * Caches are replicated (no-CP) -> any rank's blob restores identically; rank 0 writes, all ranks read. */
#define DS4F_CTX_MAGIC 0x44533443u   /* 'DS4C' */
/* context-parallel: the compressed caches are slot-sharded per node, so each rank holds a DIFFERENT
 * shard and must save/load its own per-rank file. Replicated (no-CP) keeps a single shared file. */
static int ds4f_ctx_sharded(void) {
    return envi("DS4F_CP", 0) || envi("DS4F_CP_IDX", 0) || envi("DS4F_CP_SHARD", 0);
}
static void ds4f_ctx_rankpath(ds4f_model *m, const char *path, char *out, size_t n) {
    if (ds4f_ctx_sharded()) snprintf(out, n, "%s.rank%02d", path, m->ep_rank);
    else                    snprintf(out, n, "%s", path);
}
static uint32_t ds4f_ctx_cfg_hash(ds4f_model *m) {
    int sh = ds4f_ctx_sharded();   /* under sharding the layout depends on the topology + this rank */
    uint32_t h = 2166136261u;
    int v[9] = { m->cfg.max_pos, m->int8_kv, envi("DS4F_INT8_CMP",0), envi("DS4F_INT4_CMP",0),
                 envi("DS4F_IDX_INT4",0), m->cfg.n_layers, sh,
                 sh ? m->ep_size : 0, sh ? m->ep_rank : 0 };
    for (int i = 0; i < 9; i++) { h ^= (uint32_t)v[i]; h *= 16777619u; }
    return h;
}
static int ds4f_ctx_write_file(ds4f_model *m, const char *path, const int *ids, int npos) {
    size_t sz; ds4f_ctx_snap(m, NULL, npos, 0, &sz);
    char *buf = (char *)ds4f_xmalloc(sz, "buf"); if (!buf) return -1;
    ds4f_ctx_snap(m, buf, npos, 0, NULL);
    char tmp[1100]; snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb"); if (!f) { free(buf); return -1; }
    uint32_t hdr[3] = { DS4F_CTX_MAGIC, (uint32_t)npos, ds4f_ctx_cfg_hash(m) };
    int ok = fwrite(hdr, 4, 3, f) == 3 && fwrite(ids, sizeof(int), npos, f) == (size_t)npos
             && fwrite(buf, 1, sz, f) == sz;
    fclose(f); free(buf);
    if (ok) { rename(tmp, path); return 0; }   /* atomic publish */
    remove(tmp); return -1;
}
static int ds4f_ctx_read_file(ds4f_model *m, const char *path, int *ids, int *out_npos, int maxpos) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    uint32_t hdr[3];
    if (fread(hdr, 4, 3, f) != 3 || hdr[0] != DS4F_CTX_MAGIC) { fclose(f); return -2; }
    int npos = (int)hdr[1];
    if (npos > maxpos || hdr[2] != ds4f_ctx_cfg_hash(m)) { fclose(f); return -3; }  /* wrong ctx-cache config */
    if (fread(ids, sizeof(int), npos, f) != (size_t)npos) { fclose(f); return -1; }
    size_t sz; ds4f_ctx_snap(m, NULL, npos, 0, &sz);
    char *buf = (char *)ds4f_xmalloc(sz, "buf");   /* ds4f_xmalloc aborts on OOM; no NULL check needed */
    if (fread(buf, 1, sz, f) != sz) { free(buf); fclose(f); return -1; }
    ds4f_ctx_snap(m, buf, npos, 1, NULL);           /* restore into the (replicated) caches */
    free(buf); fclose(f); *out_npos = npos; return 0;
}

/* ---- topology (tofu_topo.txt; written once by tofu_topo_helper) ----
 * TOPO_PATH (default tofu_topo.txt) may be overridden per process via TOFU_TOPO_PATH
 * so co-resident EP groups (e.g. ds4p on 96 nodes + ds4f on 12) can keep SEPARATE
 * topo files in one allocation. tofu_topo_helper honors the same env. */
static const char *topo_path(void) {
    const char *t = getenv("TOFU_TOPO_PATH");
    return (t && *t) ? t : TOPO_PATH;
}
static int read_topo(uint8_t coords[][TOFU_NCOORDS]) {
    const char *tp = topo_path();
    FILE *f = fopen(tp, "r");
    if (!f) { fprintf(stderr, "cannot open %s (run tofu_topo_helper first)\n", tp); exit(1); }
    int n = 0; char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n >= MAX_NODES) { fprintf(stderr, "too many nodes\n"); exit(1); }
        unsigned r, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u", &r, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) != 7)
            { fprintf(stderr, "malformed line: %s", line); exit(1); }
        if ((int)r != n) { fprintf(stderr, "%s ranks out of order\n", tp); exit(1); }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    if (n < 1) { fprintf(stderr, "%s lists %d node(s)\n", tp, n); exit(1); }
    return n;
}

/* ---- uTofu state (barrier region; the all-reduce keeps its own region) ---- */
static int             N, MyRank;
static char           *Region;
static size_t          SEND_OFF, BAR_BASE, SlotSend, SlotB;
static utofu_vcq_hdl_t Vcq;
static utofu_stadd_t   Base;
static utofu_vcq_id_t  PeerVcq[MAX_NODES];
static utofu_stadd_t   PeerBase[MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static uint64_t        Bt = 1;

static inline size_t bar_recv_off(int s) { return BAR_BASE + (size_t)s * SlotB; }
static inline size_t bar_go_off(void)    { return BAR_BASE + (size_t)N * SlotB; }

static void put_issue(utofu_vcq_id_t pv, utofu_stadd_t s, utofu_stadd_t d, size_t len, int drain) {
    int rc; void *cb;
    for (;;) { rc = utofu_put(Vcq, pv, s, d, len, 0, FLAGS, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(Vcq, 0, &cb); }
    if (rc != UTOFU_SUCCESS) die("utofu_put", rc);
    if (drain) { do { rc = utofu_poll_tcq(Vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
                 if (rc != UTOFU_SUCCESS) die("utofu_poll_tcq", rc); }
}
static void wait_ge(volatile uint64_t *q, uint64_t v, const char *what) {
    double ts = now_sec();
    while (*q < v) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die(what, -1);
}
/* fan-in to rank 0, fan-out release. robust=1 retries (startup skew). */
static void barrier_robust(int robust) {
    uint64_t t = ++Bt;
    char *sb = Region + SEND_OFF;
    if (MyRank == 0) {
        for (int s = 1; s < N; s++)
            wait_ge((volatile uint64_t *)(Region + bar_recv_off(s)), t, "barrier fan-in");
        for (int s = 1; s < N; s++) {
            *(volatile uint64_t *)sb = t;
            put_issue(PeerVcq[s], Base + SEND_OFF, PeerBase[s] + bar_go_off(), 8, 1);
        }
    } else {
        volatile uint64_t *go = (volatile uint64_t *)(Region + bar_go_off());
        double ts = now_sec();
        do {
            *(volatile uint64_t *)sb = t;
            put_issue(PeerVcq[0], Base + SEND_OFF, PeerBase[0] + bar_recv_off(MyRank), 8, 1);
            if (!robust) { wait_ge(go, t, "barrier release"); break; }
            for (int a = 0; a < 50 && *go < t; a++) usleep(2000);
            if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("bootstrap barrier timeout", -1);
        } while (*go < t);
    }
}
static void barrier(void) { barrier_robust(0); }   /* tp_comm_init's barrier_fn */

/* ---- EP combine all-reduce callback (model -> tp_allreduce.h) ---- */
static double g_ar_secs = 0.0; static long g_ar_calls = 0;
static void ep_ar_callback(float *buf, int count, void *ctx) {
    tp_comm *c = (tp_comm *)ctx;
    int mc = c->max_count > 0 ? c->max_count : count;
    double t0 = now_sec();
    for (int off = 0; off < count; ) {
        int n = count - off; if (n > mc) n = mc;
        tp_allreduce_sum(c, buf + off, n);
        off += n;
    }
    g_ar_secs += now_sec() - t0; g_ar_calls++;
}
/* (val, global-idx) argmax all-reduce callback (TP_HEAD batched-prefill head merge). */
static void ep_argmax_callback(float *val, int32_t *idx, void *ctx) {
    tp_allreduce_argmax((tp_comm *)ctx, val, idx);
}
/* MAX all-reduce callback (Phase-2 CP online-softmax combine: global per-head max). */
static void ep_armax_callback(float *buf, int count, void *ctx) {
    tp_comm *c = (tp_comm *)ctx;
    int mc = c->max_count > 0 ? c->max_count : count;
    double t0 = now_sec();
    for (int off = 0; off < count; ) {
        int n = count - off; if (n > mc) n = mc;
        tp_allreduce_max(c, buf + off, n);
        off += n;
    }
    g_ar_secs += now_sec() - t0; g_ar_calls++;
}

/* ---- CLI front-end: map --flags to the existing DS4F_* env config, so the runner is driven by
 * command-line args instead of `export` soup. Named flags cover the common knobs; `--set K=V` reaches
 * any DS4F_* var. Parsed FIRST in main (before any envi()/getenv() read) via setenv, so the existing
 * config sites are untouched and no-args behavior is byte-identical (backward compatible). Ported from
 * glm5_ep_runner.c's glm5_cli/glm5_apply_numa (model-agnostic). */
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif
/* NUMA-local weight placement (the measured 1.40x e2e, bit-identical lever): interleave THIS process's
 * future allocations across all CMGs -- in-process equivalent of `numactl --interleave=all`. ds4f decode
 * is fp8-stream memory-bound (same regime as glm5), so it benefits identically. OMP thread affinity is
 * read by the runtime at init, so the launch script still exports OMP_PROC_BIND=close / OMP_PLACES=cores
 * (best-effort setenv here as a fallback). Default ON via DS4F_NUMA. */
static void ds4f_apply_numa(int on){
    if(!on) return;
    unsigned long nodemask=~0UL;                 /* all NUMA nodes */
    syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, (unsigned long)(8*sizeof nodemask));
    setenv("OMP_PROC_BIND","close",0);           /* 0 = don't override if the script already set it */
    setenv("OMP_PLACES","cores",0);
    fprintf(stderr,"NUMA interleave on (MPOL_INTERLEAVE all-CMG; OMP_PROC_BIND=close/OMP_PLACES=cores)\n");
}
static void ds4f_cli_usage(void){
    fprintf(stderr,
      "ds4f_ep_runner [--flags]  (DeepSeek-V4-Flash; all map to DS4F_* env; env still works as fallback)\n"
      "  --numa[=0|1]        NUMA-interleave weights (default ON; the 1.40x bit-identical decode lever)\n"
      "  --preset decode     bundle: FP8_BF16+Q8_DENSE+HC_PAR+HC_RMSPAR+TIERB2+MHC+OPROJ_FUSE+ATTN_SVE+TP_HEAD+TP_EMBED\n"
      "  --model DIR         DS4F_MODEL_DIR       --real N         DS4F_REAL\n"
      "  --stage-dir D       DS4F_STAGE_DIR       --ep-size N      DS4F_EP_SIZE\n"
      "  --nshards N         DS4F_NSHARDS         --layers N       DS4F_LAYERS (0=full 43)\n"
      "  --prefill N         DS4F_PREFILL         --prefill-batch N DS4F_PREFILL_BATCH\n"
      "  --max-gen N         DS4F_MAXGEN          --maxpos N       DS4F_MAXPOS\n"
      "  --max-new N         DS4F_MAX_NEW         --cp N           DS4F_CP\n"
      "  --cp-idx N          DS4F_CP_IDX          --int8-kv N      DS4F_INT8_KV\n"
      "  --int8-cmp N        DS4F_INT8_CMP        --mtp N          DS4F_MTP\n"
      "  --tierb2 N          DS4F_TIERB2          --mhc N          DS4F_MHC\n"
      "  --sparse N          DS4F_SPARSE          --prompt-ids F   DS4F_PROMPT_IDS\n"
      "  --gen-out FILE      DS4F_GEN_OUT         --set KEY=VAL    set any DS4F_* var\n");
}
static void ds4f_cli(int argc,char**argv){
    int numa=1;
    for(int i=1;i<argc;i++){
        char*a=argv[i]; if(strncmp(a,"--",2)) continue; a+=2;
        char*eq=strchr(a,'='); char*val=NULL;
        if(eq){ *eq=0; val=eq+1; }
        else if(i+1<argc && argv[i+1][0]!='-'){ val=argv[++i]; }
        if(!strcmp(a,"help")){ ds4f_cli_usage(); continue; }
        if(!strcmp(a,"numa")){ numa=val?atoi(val):1; continue; }
        if(!strcmp(a,"preset")&&val&&!strcmp(val,"decode")){
            setenv("DS4F_FP8_BF16","1",1); setenv("DS4F_Q8_DENSE","1",1);
            setenv("DS4F_HC_PAR","1",1);   setenv("DS4F_HC_RMSPAR","1",1);
            setenv("DS4F_TIERB2","1",1);   setenv("DS4F_MHC","1",1);
            setenv("DS4F_OPROJ_FUSE","1",1); setenv("DS4F_ATTN_SVE","1",1);
            setenv("DS4F_FLAGBAR","1",1);   /* per-worker flag barrier: +8% M=1 decode, bit-identical */
            setenv("DS4F_ATTN_GEMM","1",1); /* 8-head KV-reuse attention: -50% attn phase, bit-identical (default on anyway) */
            /* TP_HEAD: vocab-shard the lm_head (bf16, Q8_DENSE-independent) across the EP group. BOTH a
             * memory lever (RSS -0.96 GB, more ctx-ceiling headroom) AND a decode-speed lever: 11n A/B
             * (2026-07-08) measured 13.07->13.37 tok/s (+2.3%, 76.5->74.8 ms/tok). First cut of this
             * feature (683cfaf) measured SPEED-NEUTRAL because ds4f_forward_token's TP_HEAD branch did a
             * full [vocab] (517 KB) all-reduce-SUM instead of the cheap (val,idx) argmax-merge that
             * ds4f_forward_verify already used -- exactly cancelling the head-compute saved. Fixed by
             * routing forward_token's greedy-decode TP_HEAD path through ar_argmax_cb (see
             * m->want_full_logits in ds4f.h/ds4f_impl.h; sampling still needs the full logits vector and
             * keeps the old path). BIT-EXACT (gen_ids 64/64 identical both before and after the fix;
             * global argmax == max over disjoint per-shard local argmaxes). No-op at ep_size<=1. Disable
             * with `--set DS4F_TP_HEAD=0` after --preset. */
            setenv("DS4F_TP_HEAD","1",1);
            /* TP_EMBED: vocab-shard the input embedding table (bf16, ~1.06 GB full -> ~96 MB/node at N=11).
             * Unlike the old TP_HEAD bug, embed_lookup's TP path was cheap from the start: decode M=1 only
             * needs ONE row (the token's embedding) filled by its owning shard, zeroed elsewhere, then a
             * [hidden]=4096-float (16 KB) all-reduce-SUM -- not a full-vocab reduce. 11n A/B (2026-07-08):
             * decode 13.37->13.37 tok/s (unchanged, comm noise-level +0.3%), RSS 20.98->20.01 GB (-0.97 GB).
             * gen_ids 64/64 IDENTICAL (bit-exact: disjoint row + zero-fill + SUM reconstructs the exact row).
             * A pure memory win at zero speed cost -- disable with `--set DS4F_TP_EMBED=0` after --preset. */
            setenv("DS4F_TP_EMBED","1",1);
            continue;
        }
        if(!strcmp(a,"set")&&val){ char*e=strchr(val,'='); if(e){*e=0; setenv(val,e+1,1);} continue; }
        #define MAP(flag,var) if(!strcmp(a,flag)){ if(val) setenv(var,val,1); continue; }
        MAP("model","DS4F_MODEL_DIR")   MAP("real","DS4F_REAL")        MAP("stage-dir","DS4F_STAGE_DIR")
        MAP("ep-size","DS4F_EP_SIZE")   MAP("nshards","DS4F_NSHARDS")  MAP("layers","DS4F_LAYERS")
        MAP("prefill","DS4F_PREFILL")   MAP("prefill-batch","DS4F_PREFILL_BATCH")
        MAP("max-gen","DS4F_MAXGEN")    MAP("maxpos","DS4F_MAXPOS")    MAP("max-new","DS4F_MAX_NEW")
        MAP("cp","DS4F_CP")             MAP("cp-idx","DS4F_CP_IDX")    MAP("int8-kv","DS4F_INT8_KV")
        MAP("int8-cmp","DS4F_INT8_CMP") MAP("mtp","DS4F_MTP")          MAP("tierb2","DS4F_TIERB2")
        MAP("mhc","DS4F_MHC")           MAP("sparse","DS4F_SPARSE")
        MAP("prompt-ids","DS4F_PROMPT_IDS") MAP("gen-out","DS4F_GEN_OUT")
        #undef MAP
        fprintf(stderr,"ds4f_ep_runner: unknown flag --%s (try --help)\n",a);
    }
    ds4f_apply_numa(numa);
}

/* DS4F_BACKTRACE=1: dump a symbolized backtrace on SIGSEGV/SIGBUS/SIGFPE. The MPI launcher only ever
 * reports "PLE 0610 ... (rank=6)(sig=11)", which tells you nothing about WHERE -- and a multi-node
 * crash is otherwise very expensive to localize (no core, no debugger). Opt-in; costs nothing off. */
static void ds4f_crash_handler(int sig) {
    void *bt[64];
    int n = backtrace(bt, 64);
    /* Write to a per-rank file on the SHARED FS, not stderr: the MPI launcher DROPS rank stderr, so
     * a handler that only writes to fd 2 produces nothing and you are back to "sig=11, good luck".
     * (/local is node-private and the crashing rank is usually on another node -- $HOME is the one
     * place the message is guaranteed to survive.) */
    char path[256];
    const char *home = getenv("HOME");
    snprintf(path, sizeof path, "%s/ds4f_crash_rank%02d.txt", home ? home : ".", MyRank);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        char hdr[128];
        int hn = snprintf(hdr, sizeof hdr, "rank %d: FATAL signal %d -- backtrace (%d frames):\n",
                          MyRank, sig, n);
        ssize_t wr = write(fd, hdr, (size_t)hn); (void)wr;
        backtrace_symbols_fd(bt, n, fd);
        fsync(fd); close(fd);
    }
    char hdr2[128];
    int h2 = snprintf(hdr2, sizeof hdr2, "\n*** rank %d: FATAL signal %d (backtrace -> %s)\n",
                      MyRank, sig, path);
    ssize_t w2 = write(2, hdr2, (size_t)h2); (void)w2;
    backtrace_symbols_fd(bt, n, 2);
    _exit(128 + sig);
}

int main(int argc,char**argv){
    ds4f_cli(argc,argv);
    /* ON BY DEFAULT (DS4F_BACKTRACE=0 to disable). Costs nothing until something crashes, and turns
     * the launcher's useless "PLE 0610 ... (rank=6)(sig=11)" into a named frame. Finding that
     * ds4f_pf_qnr_worker -> ds4f_rope_apply was the fault took a whole session WITHOUT this. */
    if (envi("DS4F_BACKTRACE", 1)) {
        /* SIGALTSTACK is REQUIRED here, not optional. The first attempt at this handler produced
         * eleven 0-byte crash files: the handler was entered but died before its first write(). That
         * is the signature of a STACK OVERFLOW -- the handler runs on the very stack that just
         * overflowed, so it cannot do anything. Give it its own stack and it can finally speak. */
        static char altstk[SIGSTKSZ * 4];
        stack_t ss = { .ss_sp = altstk, .ss_size = sizeof altstk, .ss_flags = 0 };
        sigaltstack(&ss, NULL);
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = ds4f_crash_handler;
        sa.sa_flags = SA_ONSTACK;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGSEGV, &sa, NULL);
        sigaction(SIGBUS,  &sa, NULL);
        sigaction(SIGFPE,  &sa, NULL);
    }
    int rc;
    int n_threads = envi("LLM_THREADS", 48);
    int n_cmgs    = envi("DS4F_CMGS", 4);
    int prefill   = envi("DS4F_PREFILL", 8);
    int maxgen    = envi("DS4F_MAXGEN", 16);
    int maxpos    = envi("DS4F_MAXPOS", 4096);
    int layers    = envi("DS4F_LAYERS", 0);
    int ctx_warm  = envi("DS4F_CTX_WARM", 0);   /* fill synthetic KV+compressed to this ctx, decode from there */
    int prefill_batch = envi("DS4F_PREFILL_BATCH", 0);   /* >0: batched M-token GEMM prefill (needs exact + dense bf16) */
    if (prefill_batch > DS4F_MAX_MTILE) prefill_batch = DS4F_MAX_MTILE;

    /* ---- real greedy generation mode (coding-task quality test) ----
     * DS4F_PROMPT_IDS=<file of whitespace-separated token ids> turns on real
     * generation: prefill the prompt via embed-lookup, then greedy argmax decode
     * with embed feedback (stop on eos). All ranks read the SAME prompt file and
     * compute the SAME argmax (dense+head replicated) -> identical token feed ->
     * cross-rank lockstep preserved with no extra broadcast. */
    const char *prompt_ids_file = getenv("DS4F_PROMPT_IDS");
    const char *gen_out_file    = getenv("DS4F_GEN_OUT");
    int gen_mode = (prompt_ids_file && *prompt_ids_file);
    int max_new  = envi("DS4F_MAX_NEW", 256);
    int *prompt_ids = NULL, n_prompt = 0;
    if (gen_mode) {
        FILE *pfh = fopen(prompt_ids_file, "r");
        if (!pfh) die("cannot open DS4F_PROMPT_IDS file", -1);
        int cap = 1024; prompt_ids = (int *)ds4f_xmalloc((size_t)cap*sizeof(int), "cap");
        int v;
        while (fscanf(pfh, "%d", &v) == 1) {
            if (n_prompt >= cap) { cap *= 2; prompt_ids = (int *)realloc(prompt_ids, (size_t)cap*sizeof(int)); }
            prompt_ids[n_prompt++] = v;
        }
        fclose(pfh);
        if (n_prompt < 1) die("DS4F_PROMPT_IDS file has no ids", -1);
        prefill = n_prompt;     /* prefill the whole prompt token-at-a-time */
        maxgen  = max_new;      /* decode up to max_new new tokens */
        prefill_batch = 0;      /* greedy feedback needs the per-token embedding */
        ctx_warm = 0;           /* gen mode does not warm synthetic KV */
    }

    /* ---- uTofu bootstrap (single TNI) ---- */
    utofu_tni_id_t *tni_ids = NULL; size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS) die("utofu_get_onesided_tnis", rc);
    if (num_tnis < 1) die("no onesided TNIs", -1);

    uint8_t my_coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(my_coords);
    if (rc != UTOFU_SUCCESS) die("utofu_query_my_coords", rc);

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N = read_topo(topo);
    MyRank = -1;
    for (int r = 0; r < N; r++) if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) MyRank = r;
    if (MyRank == -1) { fprintf(stderr, "my coords not in %s\n", topo_path()); exit(1); }

    ds4f_set_rank(MyRank);   /* so library ds4f_fatal()/OOM messages can name the rank */

    /* ---------------- DURABLE PER-RUN LOGS ----------------
     * mpiexec does not forward rank stdout/stderr, so these files ARE the diagnostics. They used to
     * be written to the CWD with fopen(...,"w") -- which means THE NEXT RUN TRUNCATES THEM. That is
     * not a theoretical hazard: it destroyed a VERIFY_GATE verdict mid-benchmark (step [2] wiped
     * step [1]'s result) and made a PASSING gate look like a silent crash. And only rank 0 got a
     * log at all, so every other rank's logmsg() went nowhere.
     *
     * Now: $DS4F_LOG_DIR/run-<jobid>-<pid>/rank<NN>.{log,err}, one per rank, plus a `latest`
     * symlink for the wrappers. Nothing is ever overwritten. */
    {
        const char *ldir = getenv("DS4F_LOG_DIR"); if (!ldir || !*ldir) ldir = "logs";
        const char *jid  = getenv("PJM_JOBID");    if (!jid  || !*jid)  jid  = "nojob";
        static char rundir[512];
        snprintf(rundir, sizeof rundir, "%s/run-%s-%d", ldir, jid, (int)getppid());
        mkdir(ldir, 0755);                       /* ignore EEXIST */
        mkdir(rundir, 0755);
        char en[640], lg[640];
        snprintf(en, sizeof en, "%s/rank%02d.err", rundir, MyRank);
        snprintf(lg, sizeof lg, "%s/rank%02d.log", rundir, MyRank);
        if (!freopen(en, "w", stderr)) { /* keep going: better to run than to die over a log */ }
        setvbuf(stderr, NULL, _IOLBF, 0);
        g_log = fopen(lg, "w");                  /* EVERY rank gets a log, not just rank 0 */
        if (MyRank == 0) {                       /* logs/latest -> this run, for the wrappers */
            char link[600]; snprintf(link, sizeof link, "%s/latest", ldir);
            unlink(link);
            char target[520]; snprintf(target, sizeof target, "run-%s-%d", jid, (int)getppid());
            if (symlink(target, link) != 0) { /* non-fatal */ }
        }
    }

    int ep_rank = MyRank, ep_size = N;
    ds4f_config cfg = ds4f_config_from_env();
    cfg.max_pos = maxpos;
    if (layers > 0) cfg.n_layers = layers;
    if (prefill + maxgen > cfg.max_pos) die("prefill+maxgen exceeds max_pos", -1);
    if (ctx_warm + maxgen > cfg.max_pos) die("ctx_warm+maxgen exceeds max_pos", -1);

    int dense_bf16 = envi("DS4F_FP8_BF16", 0);
    const char *pv_e = getenv("DS4F_BF16_PV");          /* auto-on with predequant unless explicitly set */
    int bf16_pv = (pv_e && *pv_e) ? (atoi(pv_e) != 0) : dense_bf16;
    int no = ds4f_n_owned(cfg.n_experts, ep_rank, ep_size);
    /* mirror ds4f_load_real's dense_qt so the printed estimate matches the real arena
     * (DS4F_DENSE selects the offline-baked rep and overrides FP8_BF16). */
    ds4f_qtype est_qt = dense_bf16 ? (bf16_pv ? DS4F_BF16_PV : DS4F_BF16) : DS4F_FP8;
    {   const char *de = getenv("DS4F_DENSE");
        if (de && *de) {
            if      (!strcmp(de, "q8pv"))   est_qt = DS4F_Q8_PV;
            else if (!strcmp(de, "bf16pv")) est_qt = DS4F_BF16_PV;
        } }
    size_t arena_est = ds4f_arena_size(&cfg, ep_rank, ep_size, est_qt,
                                       envi("DS4F_TIERB2", 0) && !envi("DS4F_INT8_KV", 0));
    if (MyRank == 0)
        logmsg("=== DS4F EP synthetic harness (Stage 2): %d ranks ===\n"
               "layers=%d hidden=%d experts=%d active=%d  owned~%d/layer  dense=%s\n"
               "threads=%d prefill=%d maxgen=%d max_pos=%d ctx_warm=%d  arena~%.2f GB/node\n",
               N, cfg.n_layers, cfg.hidden, cfg.n_experts, cfg.n_active, no,
               dense_bf16 ? (bf16_pv ? "BF16(predequant,pv)" : "BF16(predequant)") : "FP8(on-demand)",
               n_threads, prefill, maxgen, maxpos, ctx_warm, arena_est/(1024.0*1024.0*1024.0));
    /* one-time per-node memory ceiling diagnostic: the OOM-killer (sig9) fires on the
     * cgroup/physical limit, which may be << HBM total. Print it so the ctx ceiling is
     * derived from the REAL usable budget, not the assumed 31.8 GB. */
    if (MyRank == 0) {
        const char *paths[] = { "/sys/fs/cgroup/memory.max",
                                "/sys/fs/cgroup/memory/memory.limit_in_bytes", NULL };
        for (int i = 0; paths[i]; i++) {
            FILE *cf = fopen(paths[i], "r");
            if (cf) { char buf[64] = {0}; if (fgets(buf, sizeof buf, cf)) {
                for (char *p = buf; *p; p++) if (*p=='\n') *p=0;
                double gb = atof(buf)/1e9;
                logmsg("NODE_CGROUP_LIMIT %s = %s (%.2f GB)\n", paths[i], buf, gb>0?gb:-1); }
                fclose(cf); break; }
        }
        FILE *mf = fopen("/proc/meminfo", "r");
        if (mf) { char line[128];
            while (fgets(line, sizeof line, mf))
                if (!strncmp(line,"MemTotal",8) || !strncmp(line,"MemAvailable",12) ||
                    !strncmp(line,"MemFree",7)  || !strncmp(line,"HugePages_Total",15) ||
                    !strncmp(line,"Hugetlb",7)) {
                    for (char *p=line; *p; p++) if (*p=='\n') *p=0;
                    logmsg("NODE_MEMINFO %s\n", line); }
            fclose(mf); }
    }

    /* ---- allocate this rank's shard (owned experts + replicated dense) ----
     * DS4F_REAL=1: load REAL staged weights for THIS rank (rank<MyRank>.blob in
     * DS4F_STAGE_DIR, else /local/ds4f) — the stager's PMIX_RANK pins the same
     * physical node the topo assigns EP rank MyRank, so the blob rank matches.
     * Else synthetic fill. Loader forces dense=FP8 (ignores the synth dense knobs). */
    int real_weights = envi("DS4F_REAL", 0);
    const char *blob_dir = getenv("DS4F_STAGE_DIR");

    /* ---------------- PRE-FLIGHT: fail in SECONDS, not after a 4-minute load ----------------
     * Every check here fired for real during development and cost far more than it should have:
     *
     *  - Missing stage. /local is node-local and is WIPED on every job restart, so the single most
     *    common failure is "the blob is gone". The loader already reports it, but only after the
     *    run has burned minutes -- and the message lands in a per-rank file the wrappers were not
     *    reading, which is how a wiped stage looked like a mysterious 3-second death.
     *  - Arena will not fit. Reported AFTER the weight load, i.e. after the expensive part.
     *  - EXACT=0 with a batched path. The batched forward always applies RoPE, but the RoPE tables
     *    are only built when exact is on -> NULL deref in a pool worker, reported as a bare sig=11.
     *    (ds4f_forward_verify now guards this too; catching it here names it before anything runs.) */
    {
        if (real_weights) {
            const char *bd = blob_dir && *blob_dir ? blob_dir : "/local/ds4f";
            char mp[1100];
            snprintf(mp, sizeof mp, "%s/rank%02d.manifest", bd, ep_rank);
            FILE *pf = fopen(mp, "r");
            if (!pf) ds4f_fatal("no staged weights: cannot open %s\n"
                                "       /local is node-local and is WIPED on every job restart.\n"
                                "       Re-run the stager (run_ds4f*_stage_*.sh) before this job.", mp);
            fclose(pf);
        }
        /* Any path that drives ds4f_forward_verify needs the RoPE tables, which need EXACT. */
        int batched = envi("DS4F_PREFILL_GEMM", 0) || envi("DS4F_DB_BENCH", 0)
                   || envi("DS4F_DECODE_BATCH", 0) || envi("DS4F_VERIFY_GATE", 0)
                   || (envi("DS4F_SERVE", 0) && envi("DS4F_SERVE_BATCH", 1) > 1);
        /* mirror ds4f_load's logic: TIERB2 and INT8_KV both force exact on (ds4f_impl.h:2415,2419) */
        int will_be_exact = envi("DS4F_EXACT", 0) || envi("DS4F_TIERB2", 0) || envi("DS4F_INT8_KV", 0);
        if (batched && !will_be_exact)
            ds4f_fatal("a batched path is enabled (PREFILL_GEMM/DB_BENCH/DECODE_BATCH/VERIFY_GATE/"
                       "SERVE_BATCH) but DS4F_EXACT=0.\n"
                       "       The batched forward always applies RoPE, and the RoPE tables are only\n"
                       "       built when exact is on -> it would segfault in a pool worker.\n"
                       "       Set DS4F_EXACT=1 (and DS4F_TIERB2=1 DS4F_MHC=1 for the real model).");
        double need_gb = ds4f_arena_size(&cfg, ep_rank, ep_size, DS4F_FP8, 0) / 1073741824.0;
        double avail   = ds4f_mem_avail_gb();
        if (avail > 0 && need_gb > avail)
            ds4f_fatal("the arena will not fit: needs ~%.2f GB, node has %.2f GB available.\n"
                       "       Lower DS4F_MAXPOS, enable the dense TP stack, or use more nodes.",
                       need_gb, avail);
        if (MyRank == 0)
            logmsg("preflight OK: stage present, arena ~%.2f GB vs %.2f GB available, exact=%d\n",
                   need_gb, avail, will_be_exact);
    }

    double ta0 = now_sec();
    ds4f_model *m = real_weights
        ? ds4f_load_real(cfg, ep_rank, ep_size, blob_dir, n_threads, n_cmgs)
        : ds4f_alloc_synth(cfg, ep_rank, ep_size, n_threads, n_cmgs);
    if (!m) ds4f_fatal("model alloc/load failed (see the lines above for the reason)");
    if (MyRank == 0 && m->tierb2) {
        int ncsa = 0, nhca = 0;
        for (int L = 0; L < cfg.n_layers; L++) {
            if (cfg.compress_ratios[L] == 4) ncsa++;
            else if (cfg.compress_ratios[L]) nhca++;
        }
        logmsg("Tier-B2: ON (exact forced)  index_topk=%d index_dim=%d index_heads=%d  "
               "(%d CSA + %d HCA layers; compressed-KV folded into window softmax)\n",
               cfg.index_topk, cfg.index_head_dim, cfg.index_n_heads, ncsa, nhca);
    }
    if (MyRank == 0 && m->tierb2) {   /* per-node ctx-cache accounting (CP MemFree A/B): the O(ctx) KV/cmp/idx
                                       * buffers this node allocated, and the CP-shardable subset (cmp_q4/idx int4). */
        size_t tot = 0, shard = 0; int KV = cfg.kv_lora, ihd = cfg.index_head_dim;
        for (int L = 0; L < cfg.n_layers; L++) { ds4f_layer *ly = &m->layers[L];
            int ratio = cfg.compress_ratios[L], nslot = ratio ? cfg.max_pos/ratio : 0;
            if (ly->kv_cache)   tot += (size_t)ly->kv_slots*KV*2;
            if (ly->cmp_kv)     tot += (size_t)nslot*KV*4;
            if (ly->cmp_q4)   { size_t b=(size_t)ly->cp_nslot*(KV/2);           tot+=b; shard+=b; }
            if (ly->cmp_q)      tot += (size_t)nslot*KV;
            if (ly->idx_kv8_4){ size_t b=(size_t)ly->idx_cp_nslot*(ihd/2);      tot+=b; shard+=b; }
            if (ly->idx_kv8)    tot += (size_t)nslot*ihd;
            if (ly->idx_pscale){size_t b=(size_t)(ly->idx_kv8_4?ly->idx_cp_nslot:nslot)*4; tot+=b; shard+=b; }
        }
        logmsg("CTX_CACHE: total=%.1f MB/node shardable=%.1f MB/node (max_pos=%d cp=%d cp_shard=%d)\n",
               tot/1048576.0, shard/1048576.0, cfg.max_pos, m->cp, envi("DS4F_CP_SHARD",0));
        /* DS4F_CP_IDX break-even guard. The idx-merge gathers ep_size*index_topk candidates per CSA layer --
         * a FIXED comm cost (measured ~56 ms/tok at N=11,k=512, and NOT reducible by packing the reduces:
         * that was tried, zero benefit) -- while the scan-sharding saving only GROWS with ctx (~27 ms even
         * at 128k). Break-even is ~276k ctx; below it CP_IDX is a NET LOSS (measured 16k: 7.81 -> 4.93
         * tok/s). Prefer DS4F_IDX_REUSE, which cuts the same O(T) scan with ZERO comm at any ctx. */
        if (m->cp && envi("DS4F_CP_IDX", 0) && cfg.max_pos < 262144)
            logmsg("WARN: DS4F_CP_IDX at max_pos=%d is a NET LOSS (idx-merge comm ~56 ms/tok is fixed; "
                   "break-even ~276k ctx). Measured 16k: 7.81 -> 4.93 tok/s. Use DS4F_IDX_REUSE instead.\n",
                   cfg.max_pos);
    }
    double ta1 = now_sec();
    {   char tn[64]; snprintf(tn, sizeof tn, "ds4f_ep_load_rank%02d.txt", MyRank);
        FILE *tf = fopen(tn, "w");
        if (tf) { fprintf(tf, "rank %d: alloc+first-touch=%.2fs arena_used=%.2f GB RSS=%.2f GB owned=%d/layer\n",
                          MyRank, ta1-ta0, m->arena_used/1e9, rss_bytes()/1e9, no); fclose(tf); }
    }

    /* ---- barrier region (own cache line per remote-written slot) ---- */
    SlotSend = DEMO_CACHE_LINE; SlotB = DEMO_CACHE_LINE; SEND_OFF = 0; BAR_BASE = SlotSend;
    size_t region_sz = BAR_BASE + (size_t)(N + 1) * SlotB;
    if (posix_memalign((void **)&Region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(Region, 0, region_sz);

    /* ---- VCQ + region registration; reconstruct peers by convention ---- */
    utofu_tni_id_t tni = tni_ids[0];
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &Vcq);
    if (rc != UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id", rc);
    utofu_vcq_id_t my_real;
    rc = utofu_query_vcq_id(Vcq, &my_real);
    if (rc != UTOFU_SUCCESS) die("utofu_query_vcq_id", rc);
    rc = utofu_reg_mem_with_stag(Vcq, Region, region_sz, RUN_STAG, 0, &Base);
    if (rc != UTOFU_SUCCESS) die("utofu_reg_mem_with_stag", rc);
    for (int r = 0; r < N; r++) {
        if (r == MyRank) { PeerVcq[r] = my_real; PeerBase[r] = Base; continue; }
        rc = utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &PeerVcq[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)", rc);
        utofu_set_vcq_id_path(&PeerVcq[r], NULL);
        rc = utofu_query_stadd(PeerVcq[r], RUN_STAG, &PeerBase[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(peer)", rc);
    }
    free(tni_ids);

    /* Robust startup barrier BEFORE tp_comm_init: every rank must have registered
     * its RUN_STAG region and be spinning before any NON-robust barrier (including
     * the one tp_comm_init issues internally) runs — otherwise the first barrier's
     * Puts race the peer's reg_mem and get silently dropped -> deadlock. Same
     * ordering as tp_runner.c (robust barrier at :969 ahead of tp_comm_init :988). */
    barrier_robust(1);

    /* ---- batched-prefill eligibility + size the all-reduce comm region so a whole
     * [M,C] partial fits in ONE tp_allreduce_sum. ep_ar_callback chunks by max_count,
     * so if max_count stayed = hidden the [M*hidden] partial would split into M sums
     * and the per-layer latency amortization (the big cluster lever) would be lost.
     * The argmax/decode sends only Put `count`-sized payloads, so the enlarged slot
     * costs no extra decode bandwidth (see tp_ar_send_argmax). ---- */
    if (prefill_batch > 0 && !m->exact) {
        if (MyRank == 0) logmsg("DS4F_PREFILL_BATCH needs DS4F_EXACT=1; using token-at-a-time prefill\n");
        prefill_batch = 0;
    }
    if (prefill_batch > 0 && (m->mhc || m->tierb2)) {
        if (MyRank == 0) logmsg("DS4F_PREFILL_BATCH unsupported with mHC/Tier-B2; using token-at-a-time prefill\n");
        prefill_batch = 0;
    }
    /* TP_ATTN shards wq_b by head -> batched prefill would need a per-layer [M,H] gather-reduce of
     * p_attn before the block-diagonal o-proj (heads not group-aligned), which measured a net LOSS
     * (comm ~57%, 37 < 47.85 tok/s). TP_ATTN is a decode-only memory lever; fall back to token-at-a-
     * time prefill (which shards wq_b correctly via the decode TP path). TP_SHARED/TP_OPROJ compose. */
    if (prefill_batch > 0 && getenv("DS4F_TP_ATTN") && atoi(getenv("DS4F_TP_ATTN"))) {
        if (MyRank == 0) logmsg("DS4F_TP_ATTN (wq_b head-shard) is decode-only; using token-at-a-time prefill\n");
        prefill_batch = 0;
    }
    if (prefill_batch > 0 && !dense_bf16 && MyRank == 0)
        logmsg("WARN: DS4F_PREFILL_BATCH without DS4F_FP8_BF16=1 -> dense falls back to per-token matvec (no speedup)\n");
    /* MXFP4 GEMM tile-dequant default-ON for batched prefill (Step 2o). The svtbl expert
     * GEMM is dequant-bound (flat ~84 Gmac/s); tile-dequanting nibbles->bf16 once + reusing
     * across M wins at M>=16 (real-weight 11n A/B: +7.9% prefill, token-EXACT argmax+||x||,
     * threshold-insensitive 1-16 => batched per-expert M>=16). LOSSLESS (relL2 2e-7), svtbl
     * kept for M<16. Inert outside batched prefill (decode uses mv_worker, not gemm_worker).
     * Explicit DS4F_MXFP4_GEMM_TILE overrides (incl. =0 to disable). */
    if (prefill_batch > 0 && getenv("DS4F_MXFP4_GEMM_TILE") == NULL) {
        m->mxfp4_gemm_tile = 16;
        if (MyRank == 0) logmsg("batched prefill: MXFP4 GEMM tile-dequant default ON (DS4F_MXFP4_GEMM_TILE=16)\n");
    }
    int ar_mtile = (prefill_batch > 0 && prefill > 0)
                 ? (prefill_batch < prefill ? prefill_batch : prefill) : 1;

    /* ---- all-reduce comm region (hidden*ar_mtile floats) + wire model hook ---- */
    static tp_comm comm;
    if (tp_comm_init(&comm, Vcq, PeerVcq, MyRank, N, cfg.hidden * ar_mtile, barrier) != 0)
        die("tp_comm_init", -1);
    m->ar_cb = ep_ar_callback; m->ar_ctx = &comm;
    m->ar_max_cb = ep_armax_callback; m->ar_max_ctx = &comm;
    m->ar_argmax_cb = ep_argmax_callback; m->ar_argmax_ctx = &comm;

    barrier_robust(1);   /* everyone past load + comm init (robust: tolerate skew) */
    if (MyRank == 0) logmsg("all %d ranks past bootstrap barrier; starting prefill%s\n",
                            N, prefill_batch > 0 ? " [batched M-token GEMM]" : "");

    /* ---- DS4F_DB_BENCH: batched concurrent decode throughput sweep. Decode M cold-start sequences
     * for ND steps at each M, report aggregate tok/s vs the roofline projection (step(M)~=fixed+perM). ---- */
    if (envi("DS4F_DB_BENCH", 0) > 0) {
        int C = m->cfg.hidden, hc = m->cfg.hc_mult, ND = envi("DS4F_DB_NTOK", 32), maxM = envi("DS4F_DB_MAXM", 16);
        int Ms[6] = {1, 2, 4, 8, 16, 32}, nM = 0; while (nM < 6 && Ms[nM] <= maxM) nM++;
        float *Xb  = (float *)ds4f_xalloc(64, (size_t)maxM*C*4, "Xb");
        float *hcb = (float *)ds4f_xalloc(64, (size_t)maxM*(size_t)hc*C*4, "hcb");
        int *cur = (int *)ds4f_xmalloc((size_t)maxM*sizeof(int), "cur"), *pos = (int *)ds4f_xmalloc((size_t)maxM*sizeof(int), "pos");
        int *ot  = (int *)ds4f_xmalloc((size_t)maxM*sizeof(int), "ot");
        if (MyRank == 0) logmsg("DECODE_BATCH throughput sweep (ND=%d steps/M):\n", ND);
        /* DS4F_TRACE=1: append a checkpoint per step to $HOME. When a multi-node run dies with a bare
         * "sig=11" and the crash handler itself cannot run (stack overflow -> 0-byte crash files),
         * the LAST line of this file is the last thing that completed. Crude, and decisive. */
        int trc = envi("DS4F_TRACE", 0);
        char trp[256]; FILE *trf = NULL;
        if (trc) { snprintf(trp, sizeof trp, "%s/ds4f_trace_rank%02d.txt", getenv("HOME") ? getenv("HOME") : ".", MyRank);
                   trf = fopen(trp, "w"); }
        #define TRACE(...) do { if (trf) { fprintf(trf, __VA_ARGS__); fflush(trf); } } while (0)
        for (int mi = 0; mi < nM; mi++) {
            int M = Ms[mi];
            TRACE("M=%d: serve_reset+free\n", M);
            ds4f_serve_reset(m); ds4f_free_decode_batch(m);   /* free prior M's cache sets (no leak/thrash) */
            for (int k = 0; k < M; k++) cur[k] = 100 + k*1000;
            for (int k = 0; k < M; k++) { embed_lookup(m, cur[k], Xb + (size_t)k*C); pos[k] = 0; }
            TRACE("M=%d: -> forward_decode_batch (warm)\n", M);
            ds4f_forward_decode_batch(m, Xb, pos, M, ot, hcb);        /* warm (alloc caches/buffers) */
            TRACE("M=%d: warm OK\n", M);
            for (int k = 0; k < M; k++) cur[k] = ot[k];
            int do_prof = envi("DS4F_PROF", 0);
            if (do_prof) for (int i = 0; i < DS4F_NPHASE; i++) m->prof[i] = 0;   /* reset per-M breakdown */
            barrier(); double t0 = now_sec();
            for (int t = 1; t <= ND; t++) {
                for (int k = 0; k < M; k++) { embed_lookup(m, cur[k], Xb + (size_t)k*C); pos[k] = t; }
                ds4f_forward_decode_batch(m, Xb, pos, M, ot, hcb);
                for (int k = 0; k < M; k++) cur[k] = ot[k];
            }
            double dt = now_sec() - t0;
            if (MyRank == 0) logmsg("  M=%2d: %.1f ms/step  aggregate %.1f tok/s  (%.2f tok/s/seq)\n",
                                    M, dt/ND*1e3, (double)M*ND/dt, (double)ND/dt);
            if (do_prof && MyRank == 0) {   /* per-step ms per phase -> which components scale with M (batching targets) */
                char pb[1024]; int pn = 0; pn += snprintf(pb+pn, sizeof pb-pn, "    prof M=%2d (ms/step):", M);
                for (int i = 0; i < DS4F_P_QKV_A; i++) { double ms = m->prof[i]/ND*1e3;
                    if (ms > 0.05) pn += snprintf(pb+pn, sizeof pb-pn, " %s=%.2f", ds4f_prof_names[i], ms); }
                logmsg("%s\n", pb);
            }
        }
        barrier(); exit(0);
    }

    /* ---- DS4F_DECODE_BATCH P1 isolation test: batched concurrent decode must isolate sequences.
     * Decode seqA solo (nseq=1) vs seqA+seqB batched (nseq=2); seqA's token stream must be IDENTICAL
     * (each sequence reads its own swapped cache set -> batching a neighbour changes nothing). ---- */
    if (envi("DS4F_DECODE_BATCH", 0) > 0) {
        int C = m->cfg.hidden, hc = m->cfg.hc_mult, ND = envi("DS4F_DB_NTOK", 16); if (ND > 64) ND = 64;
        int seedA = envi("DS4F_DB_SEEDA", 100), seedB = envi("DS4F_DB_SEEDB", 5000);
        float *Xb  = (float *)ds4f_xalloc(64, (size_t)2*C*4, "Xb");
        float *hcb = (float *)ds4f_xalloc(64, (size_t)2*(size_t)hc*C*4, "hcb");
        int aSolo[64], aBatch[64];
        ds4f_serve_reset(m); m->dec_batch_seq = NULL; m->dec_batch_pos = NULL; m->dec_nseq = 0;   /* seqA solo */
        { int cur = seedA;
          for (int t = 0; t < ND; t++) { embed_lookup(m, cur, Xb); int p[1] = { t }; int ot[1];
              ds4f_forward_decode_batch(m, Xb, p, 1, ot, hcb); aSolo[t] = cur = ot[0]; } }
        ds4f_serve_reset(m); m->dec_batch_seq = NULL; m->dec_batch_pos = NULL; m->dec_nseq = 0;   /* seqA + seqB */
        { int cA = seedA, cB = seedB;
          for (int t = 0; t < ND; t++) { embed_lookup(m, cA, Xb); embed_lookup(m, cB, Xb + C);
              int p[2] = { t, t }; int ot[2];
              ds4f_forward_decode_batch(m, Xb, p, 2, ot, hcb); aBatch[t] = cA = ot[0]; cB = ot[1]; } }
        if (MyRank == 0) {
            int match = 0; for (int t = 0; t < ND; t++) if (aSolo[t] == aBatch[t]) match++;
            logmsg("DECODE_BATCH isolation (seqA solo vs batched-with-seqB): %d/%d match -> %s\n",
                   match, ND, match == ND ? "PASS" : "FAIL");
            char b[512]; int n = 0; n += snprintf(b+n, sizeof b-n, "  solo : ");
            for (int t = 0; t < ND && t < 12; t++) n += snprintf(b+n, sizeof b-n, "%d ", aSolo[t]);
            n += snprintf(b+n, sizeof b-n, "\n  batch: ");
            for (int t = 0; t < ND && t < 12; t++) n += snprintf(b+n, sizeof b-n, "%d ", aBatch[t]);
            logmsg("%s\n", b);
        }
        barrier(); exit(0);
    }

    /* ---- DS4F_VERIFY_GATE: is ds4f_forward_verify (the batched / serve forward) numerically
     * EQUIVALENT to ds4f_forward_token (the gated single-stream matvec forward)?
     *
     * WHY THIS EXISTS: it caught a real one. Every batched path (PREFILL_GEMM, decode-batch,
     * batched serve) silently produced GARBAGE because ds4f_forward_verify fed p_o1 column 0 to a
     * wo_b that DS4F_TP_WOB had COLUMN-sharded (fixed 2026-07-12, ds4f_impl.h; must be p_o1+oi0).
     * Single-stream decode was fine throughout, because it is a different function.
     * DS4F_DB_BENCH only TIMES steps -- it never inspects what it generated -- which is how this
     * stayed hidden while we quoted 35.4 tok/s of "throughput". KEEP THIS GATE GREEN (16/16):
     * anything touching forward_verify, the TP shards, or the serve loops must re-run it.
     *
     * The test isolates the forward function and nothing else: BOTH arms prefill the same prompt
     * through the KNOWN-GOOD ds4f_forward_token path (so caches/Tier-B2 state are built identically),
     * then decode ND tokens -- arm A with ds4f_forward_token, arm B with ds4f_forward_verify(K=1).
     * Same caches, same positions, same prompt => ANY divergence is the verify forward itself.
     * (Expect a small reassoc-class tail: the GEMM reassociates. A prompt-ignoring/garbage stream
     *  is a BUG.) ---- */
    if (envi("DS4F_VERIFY_GATE", 0) > 0 && gen_mode && n_prompt > 0) {
        int Cc = m->cfg.hidden, hcm = m->cfg.hc_mult; size_t hcC = (size_t)hcm * Cc;
        int ND = envi("DS4F_VG_NTOK", 16); if (ND > 64) ND = 64;
        ds4f_alloc_prefill_batch(m, 8);      /* sizes m->v_* / m_tile: forward_verify SEGVs without it */
        float *xb  = (float *)ds4f_xalloc(64, (size_t)Cc * 4, "xb");
        float *hcb = (float *)ds4f_xalloc(64, hcC * 4, "hcb");
        int A[64], B[64];
        for (int arm = 0; arm < 2; arm++) {
            ds4f_serve_reset(m); m->dec_batch_seq = NULL; m->dec_batch_pos = NULL; m->dec_nseq = 0;
            int last = DS4F_EOS_ID;
            for (int p = 0; p < n_prompt; p++) {                  /* identical known-good prefill */
                embed_lookup(m, prompt_ids[p], xb);
                last = ds4f_forward_token(m, xb, p);
            }
            int cur = last;
            for (int t = 0; t < ND; t++) {
                (arm ? B : A)[t] = cur;
                embed_lookup(m, cur, xb);
                if (!arm) cur = ds4f_forward_token(m, xb, n_prompt + t);
                else { int ot[1]; ds4f_forward_verify(m, xb, 1, n_prompt + t, ot, hcb); cur = ot[0]; }
            }
        }
        int match = 0, pfx = 0;
        for (int t = 0; t < ND; t++) if (A[t] == B[t]) match++;
        while (pfx < ND && A[pfx] == B[pfx]) pfx++;
        const char *verdict = match == ND ? "PASS"
                            : (pfx >= ND/2 ? "reassoc-tail?" : "FAIL (verify forward is WRONG)");
        if (MyRank == 0) {
            logmsg("VERIFY_GATE (forward_token vs forward_verify K=1, IDENTICAL prefill): "
                   "%d/%d match, common prefix %d -> %s\n", match, ND, pfx, verdict);
            char b[512]; int n = 0; n += snprintf(b+n, sizeof b-n, "  token : ");
            for (int t = 0; t < ND && t < 12; t++) n += snprintf(b+n, sizeof b-n, "%d ", A[t]);
            n += snprintf(b+n, sizeof b-n, "\n  verify: ");
            for (int t = 0; t < ND && t < 12; t++) n += snprintf(b+n, sizeof b-n, "%d ", B[t]);
            logmsg("%s\n", b);
            /* DURABLE VERDICT. logmsg() writes only to ds4f_ep_rank00.txt, which the NEXT run
             * truncates -- so a benchmark that runs the gate and then anything else destroys the
             * gate's own result (that happened, and the gate's PASS looked like a silent crash).
             * Write the verdict somewhere nothing else clobbers, and print it to stdout too. */
            const char *vf = getenv("DS4F_VERIFY_GATE_OUT");
            char vpath[1024];
            snprintf(vpath, sizeof vpath, "%s", (vf && *vf) ? vf : "ds4f_verify_gate.txt");
            FILE *f = fopen(vpath, "w");
            if (f) {
                fprintf(f, "%d/%d match, common prefix %d -> %s\n", match, ND, pfx, verdict);
                fprintf(f, "token : "); for (int t = 0; t < ND; t++) fprintf(f, "%d ", A[t]);
                fprintf(f, "\nverify: "); for (int t = 0; t < ND; t++) fprintf(f, "%d ", B[t]);
                fprintf(f, "\n"); fclose(f);
            }
            printf("VERIFY_GATE: %d/%d match, common prefix %d -> %s\n", match, ND, pfx, verdict);
            fflush(stdout);
        }
        /* Exit code carries the verdict. The gate deliberately produces no gen ids, so the gen
         * wrapper reports rc=1 + "no gen_ids produced" -- which is indistinguishable from a crash.
         * Every rank computed A/B in lockstep, so each can decide for itself. 0 = PASS. */
        barrier(); exit(match == ND ? 0 : 3);
    }

    /* Stage-A self-test (post-barrier so every rank's comm is live): verify
     * tp_allreduce_max == serial max + lockstep (every rank computes the same expected
     * global max). Gated DS4F_CP_SELFTEST. Runs in lockstep on all ranks (seq stays aligned). */
    if (getenv("DS4F_CP_SELFTEST") && atoi(getenv("DS4F_CP_SELFTEST"))) {
        enum { NT = 19 };                         /* odd count: exercises the chunk tail */
        float buf[NT], exq[NT];
        for (int i = 0; i < NT; i++) {
            buf[i] = (float)((MyRank + 1) * (i + 1)) - 137.5f;    /* per-rank, includes negatives */
            exq[i] = (float)(N * (i + 1)) - 137.5f;               /* max over ranks is rank N-1 */
        }
        m->ar_max_cb(buf, NT, m->ar_max_ctx);
        int bad = 0; float worst = 0.f;
        for (int i = 0; i < NT; i++) { float d = buf[i] - exq[i]; d = d < 0 ? -d : d; if (d > worst) worst = d; if (d > 1e-3f) bad++; }
        fprintf(stderr, "[CP_SELFTEST rank %d] tp_allreduce_max %s (bad=%d worst=%.3e)\n",
                MyRank, bad ? "FAIL" : "PASS", bad, worst);
    }

    int C = cfg.hidden;
    float *x = (float *)ds4f_xalloc(256, (size_t)C * 4, "x");

    /* ---- HTTP serve loop: load once, then loop on requests from the (python) frontend via shared-FS
     * files. Rank 0 polls a request-seq counter; a barrier releases all ranks together; all read the
     * same prompt (shared FS) -> lockstep, no broadcast. Loops until killed. ---- */
    if (envi("DS4F_SERVE", 0)) {
        const char *reqf = getenv("DS4F_SERVE_REQ"), *respf = getenv("DS4F_SERVE_RESP");
        const char *reqseqf = getenv("DS4F_SERVE_REQSEQ"), *respseqf = getenv("DS4F_SERVE_RESPSEQ");
        if (!reqf || !respf || !reqseqf || !respseqf) die("DS4F_SERVE needs DS4F_SERVE_{REQ,RESP,REQSEQ,RESPSEQ}", -1);
        int maxpos = envi("DS4F_MAXPOS", 4096);
        int serve_batch = envi("DS4F_SERVE_BATCH", 1);   /* >1: concurrent batched decode (throughput path) */
        if (serve_batch > 1) {
            if (serve_batch > 64) serve_batch = 64;
            if (envi("DS4F_SERVE_DYNAMIC", 0))            /* continuous batching: mid-flight admission */
                ds4f_serve_dynbatch_loop(m, serve_batch, maxpos, reqf);
            else
                ds4f_serve_batch_loop(m, serve_batch, maxpos, reqf, respf, reqseqf, respseqf);
            /* never returns */
        }
        int *pids = (int *)ds4f_xmalloc((size_t)maxpos * sizeof(int), "pids");
        int *oids = (int *)ds4f_xmalloc((size_t)(maxpos + 1) * sizeof(int), "oids");
        int *fids = (int *)ds4f_xmalloc((size_t)(maxpos + 1) * sizeof(int), "fids");   /* scratch for disk load */
        /* prefix cache (context management): each slot's ids[0,len) is the token sequence held in the
         * KV/compressor caches. A request whose prompt EXTENDS its slot's sequence skips re-prefilling
         * the shared prefix and continues from len (the multi-turn TTFT win). Divergence -> full reset. */
        int prefix_cache = envi("DS4F_SERVE_PREFIX_CACHE", 1);
        /* multi-context slots: DS4F_SERVE_SLOTS independent conversations, one live in the caches at a
         * time. A request's slot id context-switches by snapshotting the live caches to the old slot and
         * restoring the new slot's snapshot (ds4f_ctx_snap). Each slot keeps its own token sequence. */
        int nslots = envi("DS4F_SERVE_SLOTS", 1); if (nslots < 1) nslots = 1; if (nslots > 64) nslots = 64;
        typedef struct { char *snap; int *ids; int len; int used; } serve_slot;
        serve_slot *slots = (serve_slot *)ds4f_xcalloc((size_t)nslots, sizeof(serve_slot), "slots");
        for (int i = 0; i < nslots; i++) slots[i].ids = (int *)ds4f_xmalloc((size_t)(maxpos + 1) * sizeof(int), "ids");
        int live = 0;
        /* system-prompt cache: preload a persisted context into slot 0 so every conversation starts with
         * it already prefilled (instant TTFT, survives restarts). Build it once with a ctl=save request. */
        const char *syscache = getenv("DS4F_SERVE_SYSCACHE");
        if (syscache && *syscache) {
            char rp[1100]; ds4f_ctx_rankpath(m, syscache, rp, sizeof rp);   /* per-rank shard under CP */
            int sn = 0, rc = ds4f_ctx_read_file(m, rp, slots[0].ids, &sn, maxpos);
            if (rc == 0) { slots[0].len = sn; slots[0].used = 1;
                if (MyRank == 0) logmsg("SERVE syscache: preloaded %d-token system prompt from %s%s\n",
                                        sn, syscache, ds4f_ctx_sharded() ? " [per-rank shards]" : ""); }
            else if (MyRank == 0) logmsg("SERVE syscache: %s not loaded (rc=%d)\n", syscache, rc);
        }
        long last_seq = ds4f_read_seq(reqseqf);   /* ignore any stale request present at startup */
        if (MyRank == 0) logmsg("SERVE ready: poll %s (seq=%ld), req %s -> resp %s  prefix_cache=%d slots=%d\n",
                                reqseqf, last_seq, reqf, respf, prefix_cache, nslots);
        for (;;) {
            long seq = last_seq;
            if (MyRank == 0) while ((seq = ds4f_read_seq(reqseqf)) <= last_seq) usleep(2000);
            barrier();                                          /* release all ranks (req file is written before its seq bumps) */
            int mnew = 256, np = 0, slot = 0, ctl = 0;
            char path[1024] = "";
            /* header: "max_new temp top_p top_k pres_pen rep_pen seed [slot] [ctl]" (missing -> greedy /
             * slot 0 / no ctl, so old requests still work). ctl bit0=load-before, bit1=save-after; when
             * ctl!=0 the NEXT line is the disk path. Then the prompt ids. */
            ds4f_sampler sp = { .temp = 0.f, .top_p = 1.f, .pres_pen = 0.f, .rep_pen = 1.f, .top_k = 0, .rep_last_n = 64 };
            long seed = 0;
            FILE *rf = fopen(reqf, "r");
            if (rf) { char line[512];
                      if (fgets(line, sizeof line, rf))
                          sscanf(line, "%d %f %f %d %f %f %ld %d %d", &mnew, &sp.temp, &sp.top_p, &sp.top_k,
                                 &sp.pres_pen, &sp.rep_pen, &seed, &slot, &ctl);
                      if (ctl) { char pl[1024]; if (fgets(pl, sizeof pl, rf)) {
                          pl[strcspn(pl, "\r\n")] = 0; snprintf(path, sizeof path, "%s", pl); } }
                      int v; while (np < maxpos && fscanf(rf, "%d", &v) == 1) pids[np++] = v; fclose(rf); }
            if (slot < 0 || slot >= nslots) slot = 0;
            if (np > maxpos) np = maxpos;                       /* truncate over-long prompt to the KV ceiling */
            if (mnew < 0) mnew = 0;                             /* mnew==0 = prefill only (cache priming) */
            if (np + mnew > maxpos) mnew = (maxpos > np) ? maxpos - np : 0;   /* keep prompt+gen <= MAXPOS */
            ds4f_rng_state = (uint64_t)seed;                    /* identical seed on every rank -> lockstep sampling */
            /* ---- context switch: make `slot` the live context (snapshot the outgoing, restore incoming) ---- */
            if (slot != live) {
                double ts0 = now_sec(); size_t save_sz = 0, rest_sz = 0;
                if (slots[live].len > 0) {
                    ds4f_ctx_snap(m, NULL, slots[live].len, 0, &save_sz);
                    slots[live].snap = (char *)realloc(slots[live].snap, save_sz);
                    ds4f_ctx_snap(m, slots[live].snap, slots[live].len, 0, NULL); slots[live].used = 1;
                }
                double ts1 = now_sec();
                if (slots[slot].used) { ds4f_ctx_snap(m, slots[slot].snap, slots[slot].len, 1, NULL);
                                        ds4f_ctx_snap(m, NULL, slots[slot].len, 0, &rest_sz); }
                else { ds4f_serve_reset(m); slots[slot].len = 0; }
                double ts2 = now_sec(); int from = live; live = slot;
                if (MyRank == 0) logmsg("SERVE switch %d->%d: save %.2fms (%d tok, %.1f MB) restore %.2fms (%d tok, %.1f MB)\n",
                    from, slot, (ts1-ts0)*1e3, slots[from].len, save_sz/1048576.0,
                    (ts2-ts1)*1e3, slots[slot].len, rest_sz/1048576.0);
            }
            /* ---- ctl load: restore a persisted context from disk into the live slot (all ranks read) ---- */
            int loaded = 0;
            if ((ctl & 1) && path[0]) {
                char rp[1100]; ds4f_ctx_rankpath(m, path, rp, sizeof rp);   /* each rank reads its own shard */
                int sn = 0, rc = ds4f_ctx_read_file(m, rp, fids, &sn, maxpos);
                if (rc == 0) { memcpy(slots[live].ids, fids, (size_t)sn * sizeof(int)); slots[live].len = sn; loaded = 1; }
                else if (MyRank == 0) logmsg("SERVE load %s failed rc=%d\n", path, rc);
            }
            int *ctx_ids = slots[live].ids; int ctx_len = slots[live].len;
            /* ---- prefix search: longest common prefix of the new prompt and the live slot's sequence ---- */
            int matched = 0, pf_from = 0;
            if (prefix_cache) { int lim = np < ctx_len ? np : ctx_len;
                while (matched < lim && pids[matched] == ctx_ids[matched]) matched++; }
            /* reuse ONLY a pure append (caches already at position ctx_len); else full reset+reprefill. */
            if (matched == ctx_len && matched < np && matched > 0) pf_from = matched;
            else { ds4f_serve_reset(m); pf_from = 0; matched = 0; }
            double t0 = now_sec();
            int ng = np > 0 ? ds4f_serve_gen(m, pids, np, mnew, x, oids, &sp, pf_from) : 0;
            double dt = now_sec() - t0;
            /* update the live slot's sequence: prompt[0,np) + fed-back tokens (trailing EOS is emitted but
             * never embedded, so it is NOT in the caches -> exclude it). */
            int appended = ng; if (ng > 0 && oids[ng-1] == DS4F_EOS_ID) appended = ng - 1;
            ctx_len = 0;
            for (int i = 0; i < np && ctx_len < maxpos; i++)       ctx_ids[ctx_len++] = pids[i];
            for (int i = 0; i < appended && ctx_len < maxpos; i++) ctx_ids[ctx_len++] = oids[i];
            slots[live].len = ctx_len; slots[live].used = 1;
            /* ---- ctl save: persist the live context to disk. Replicated -> rank 0 writes one shared
             * file; sharded (CP) -> EVERY rank writes its own shard (path.rankNN). ---- */
            if ((ctl & 2) && path[0]) {
                char rp[1100]; ds4f_ctx_rankpath(m, path, rp, sizeof rp);
                if (ds4f_ctx_sharded() || MyRank == 0) {
                    int rc = ds4f_ctx_write_file(m, rp, ctx_ids, ctx_len);
                    if (MyRank == 0) logmsg("SERVE save %s%s (%d tokens) rc=%d\n", path,
                                            ds4f_ctx_sharded() ? " [per-rank shards]" : "", ctx_len, rc);
                }
            }
            if (MyRank == 0) {
                FILE *of = fopen(respf, "w");
                if (of) { for (int i = 0; i < ng; i++) fprintf(of, "%d%s", oids[i], i+1 < ng ? " " : "\n"); fclose(of); }
                long rq = ds4f_read_seq(reqseqf);
                FILE *sf = fopen(respseqf, "w"); if (sf) { fprintf(sf, "%ld\n", rq); fclose(sf); }
                logmsg("SERVE req#%ld: slot=%d prompt=%d (cached %d%s, prefill %d) gen=%d in %.2fs (%.1f tok/s)\n",
                       rq, slot, np, matched, loaded ? "+loaded" : "", np - pf_from, ng, dt, ng > 0 ? ng / dt : 0.0);
            }
            barrier();                                          /* lockstep + ensure response visible before next */
            last_seq = ds4f_read_seq(reqseqf);
        }
    }

    /* ---- prefill (synthetic, identical activations on every rank) ----
     * Batched path (prefill_batch>0): ds4f_forward_prefill processes M tokens per call
     * so each weight is read from HBM once per M-tile (compute-bound) and the per-layer
     * EP all-reduce fires ONCE per tile instead of once per token (latency amortized
     * M-fold). The sm_next() draw order is identical to the sequential path, so X is
     * bit-identical on every rank -> replicated dense + lockstep argmax preserved. ---- */
    double t_pf0 = now_sec(); size_t pf_bytes = 0; g_ar_secs = 0; g_ar_calls = 0;
    int nan_count = 0; double xnorm = 0.0; int pf_last_tok = -1;
    int mtp_on = gen_mode && m->has_mtp;   /* DS4F_MTP self-spec: maintain MTP KV (prefill+decode) + measure accept rate */
    int spec_on = mtp_on && envi("DS4F_SPEC", 0);   /* DS4F_SPEC: gamma=1 speculative decode loop */
    float *xe = mtp_on ? (float *)ds4f_xalloc(64, (size_t)C * 4, "xe") : NULL;
    int mtp_prev_draft = -1, mtp_hits = 0, mtp_total = 0;
    sm_state = 0xD5F00D;   /* SAME seed on every rank -> replicated dense + valid all-reduce */
    if (prefill_batch > 0 && prefill > 0) {
        ds4f_alloc_prefill_batch(m, ar_mtile);
        float *X  = (float *)ds4f_xalloc(256, (size_t)ar_mtile * C * 4, "X");
        int   *bt = (int *)ds4f_xmalloc((size_t)ar_mtile * sizeof(int), "bt");
        for (int base = 0; base < prefill; base += ar_mtile) {
            int M = prefill - base < ar_mtile ? prefill - base : ar_mtile;
            for (int mm = 0; mm < M; mm++)
                for (int i = 0; i < C; i++) X[mm*C+i] = (float)(sm_next() * 2.0 - 1.0);
            m->bytes_read = 0;
            ds4f_forward_prefill(m, X, M, base, bt);
            pf_bytes += m->bytes_read;
            pf_last_tok = bt[M-1];
        }
        int Mlast = prefill % ar_mtile; if (Mlast == 0) Mlast = ar_mtile;
        const float *xl = m->p_x + (size_t)(Mlast-1)*C;
        for (int i = 0; i < C; i++) { if (!(xl[i] == xl[i])) nan_count++; xnorm += (double)xl[i]*xl[i]; }
        free(X); free(bt);
    } else {
        int tf_check = gen_mode && envi("DS4F_TF_CHECK", 0);
        int tf_correct = 0, tf_total = 0;
        /* DS4F_PREFILL_GEMM: batch the gen prefill through the MHC+Tier-B2-capable verify path in
         * chunks of K -> the dense projections become an M=K GEMM instead of K matvecs, and the
         * per-layer EP all-reduce fires once per K tokens instead of once per token. Attn/tb2/mHC
         * stay per-position (looped, causal). COHERENT not bit-identical to token-by-token (GEMM
         * reassoc); seeds the same decode. Not with MTP (per-token MTP KV maintenance).
         *
         * MEASURED (base 12n, K sweep, 2026-07-13): 17.88 -> 28.31 tok/s (+58%) from K=1 to K=128,
         * saturating past K~16. The win is mostly the GEMM, NOT the comm elision: ar_calls falls 39x
         * (6090 -> 156) but comm only 20.3% -> 14.9%, i.e. compute -14.5 ms/tok vs comm -6.0 ms/tok.
         * The floor is attn/tb2/mHC, which stay per-position. Default is now ON (K=32) in
         * run_ds4fbase_12n.sh -- it was off only because forward_verify was broken (see f9daca59). */
        int pf_gemm = gen_mode && !mtp_on && envi("DS4F_PREFILL_GEMM", 0);
        if (pf_gemm) {
            int K = envi("DS4F_PREFILL_K", 32); if (K < 1) K = 1; if (K > 128) K = 128;   /* verify path caps at 128 */
            ds4f_alloc_prefill_batch(m, K);
            size_t hcC = (size_t)m->cfg.hc_mult * C;
            float *Xin = (float *)ds4f_xalloc(64, (size_t)K * C * 4, "Xin");
            float *vhc = (float *)ds4f_xalloc(64, (size_t)K * hcC * 4, "vhc");
            int Mlast = 1;
            for (int base = 0; base < prefill; base += K) {
                int M = prefill - base < K ? prefill - base : K; Mlast = M;
                for (int mm = 0; mm < M; mm++) embed_lookup(m, prompt_ids[base + mm], Xin + (size_t)mm * C);
                m->bytes_read = 0; int ot[128];   /* verify writes M<=K argmaxes (was ot[8]: overrun at K>8) */
                ds4f_forward_verify(m, Xin, M, base, ot, vhc);
                pf_bytes += m->bytes_read; pf_last_tok = ot[M - 1];
            }
            const float *xl = vhc + (size_t)(Mlast - 1) * hcC;   /* last position's hc state (nan/norm probe) */
            for (int i = 0; i < C; i++) { if (!(xl[i] == xl[i])) nan_count++; xnorm += (double)xl[i]*xl[i]; }
            if (MyRank == 0) {
                logmsg("prefill via batched verify (DS4F_PREFILL_GEMM, K=%d)\n", K);
                double n = prefill > 0 ? prefill : 1;
                logmsg("  prefill-verify tb2 (ms/tok over %d pos): lcmp=%.3f qproj=%.3f scan=%.3f topk=%.3f icmp=%.3f wproj=%.3f rope=%.3f attn=%.3f\n",
                       prefill, m->prof[DS4F_P_TB2LCMP]/n*1e3, m->prof[DS4F_P_TB2QPROJ]/n*1e3,
                       m->prof[DS4F_P_TB2SCAN]/n*1e3, m->prof[DS4F_P_TB2TOPK]/n*1e3, m->prof[DS4F_P_TB2ICMP]/n*1e3,
                       m->prof[DS4F_P_TB2WPROJ]/n*1e3, m->prof[DS4F_P_TB2ROPE]/n*1e3, m->prof[DS4F_P_ATTN]/n*1e3);
            }
            free(Xin); free(vhc);
        } else {
        for (int p = 0; p < prefill; p++) {
            if (gen_mode) embed_lookup(m, prompt_ids[p], x);
            else for (int i = 0; i < C; i++) x[i] = (float)(sm_next() * 2.0 - 1.0);
            m->bytes_read = 0;
            pf_last_tok = ds4f_forward_token(m, x, p);
            pf_bytes += m->bytes_read;
            if (mtp_on) {   /* maintain MTP KV over the prompt: process token@(p+1) at position p+1 */
                int nt = (p + 1 < prefill) ? prompt_ids[p+1] : pf_last_tok;
                embed_lookup(m, nt, xe);
                mtp_prev_draft = ds4f_mtp_predict(m, m->s_x4, xe, p + 1, NULL);
            }
            /* teacher-forcing sanity: does argmax(pos p) predict prompt_ids[p+1]?
             * A correct LM hits ~50-80% on its own code text; ~0% == broken forward. */
            if (tf_check && p+1 < prefill) {
                int hit = (pf_last_tok == prompt_ids[p+1]);
                tf_correct += hit; tf_total++;
                if (MyRank == 0) logmsg("  TF p=%-2d in=%-6d pred=%-6d tgt=%-6d %s\n",
                                        p, prompt_ids[p], pf_last_tok, prompt_ids[p+1], hit?"HIT":".");
            }
        }
        if (tf_check && MyRank == 0)
            logmsg("TF_ACCURACY %d/%d = %.1f%% (prompt next-token; real LM ~50-80%%, broken ~0%%)\n",
                   tf_correct, tf_total, tf_total ? 100.0*tf_correct/tf_total : 0.0);
        for (int i = 0; i < C; i++) { if (!(x[i] == x[i])) nan_count++; xnorm += (double)x[i]*x[i]; }
        }
    }
    double t_pf = now_sec() - t_pf0;
    double pf_ar = g_ar_secs; long pf_calls = g_ar_calls;

    barrier();   /* lockstep check between phases */

    /* ---- optional long-ctx warm: fill synthetic KV + compressed caches to ctx_warm,
     * then decode from there. Deterministic (fixed per-layer seeds) + rank-independent
     * (local caches only, no all-reduce) -> lockstep preserved. Lets us measure decode
     * cost at long ctx without paying O(ctx^2) real prefill. ---- */
    int dec_base = prefill;
    if (ctx_warm > 0) {
        ds4f_warm_kv(m, ctx_warm);
        if (MyRank == 0) logmsg("RSS_AFTER_WARMKV=%.2f GB\n", rss_bytes()/1e9);
        ds4f_warm_tb2(m, ctx_warm);
        if (MyRank == 0) logmsg("RSS_AFTER_WARMTB2=%.2f GB (delta=cmp_kv/idx_kv)\n", rss_bytes()/1e9);
        dec_base = ctx_warm;
        if (MyRank == 0) logmsg("warmed synthetic KV+compressed caches to ctx=%d; decoding from there\n", ctx_warm);
    }

    /* ---- decode (M=1, token-at-a-time) ---- */
    if (envi("DS4F_PROF", 0) && MyRank == 0 && prefill > 0) {   /* prefill verify-section profile (before the decode reset) */
        logmsg("per-phase PREFILL (ms/tok over %d tok):\n", prefill);
        for (int i = 0; i < DS4F_NPHASE; i++)
            if (m->prof[i] > 0)
                logmsg("  %-10s %7.3f ms  %5.1f%%\n", ds4f_prof_names[i],
                       m->prof[i] / prefill * 1e3, 100.0 * m->prof[i] / (t_pf > 0 ? t_pf : 1));
    }
    memset(m->prof, 0, sizeof(m->prof));
    g_cmp_mv_secs = 0;   /* tb2lcmp/icmp matvec-dispatch attribution: reset over the decode window */
    double t_dec0 = now_sec(); size_t dec_bytes = 0; g_ar_secs = 0; g_ar_calls = 0;
    int last_tok = 0;
    int *gen_ids = NULL, n_gen = 0;
    if (spec_on) {
        /* gamma=1 speculative decode (M2 step 1: LOOPED verify -> validates the loop LOGIC byte-identical
         * to plain decode; the batched dense that makes it FASTER is M2b). Each step: draft token@(pos+2)
         * with the MTP, verify pos+1 (commit) + pos+2 (speculative, input the draft). On accept emit 2
         * tokens; on reject restore the compressor state + the saved hidden, redo pos+2 next step. */
        size_t hcf = (size_t)m->cfg.hc_mult * C, hcb = hcf * 4;
        int spec_batch = envi("DS4F_SPEC_BATCH", 0);   /* 1: BATCHED verify (M2b, COHERENT, the speedup) */
        gen_ids = (int *)ds4f_xmalloc((size_t)(max_new + 2) * sizeof(int), "s");
        char *snap = (char *)ds4f_xmalloc(ds4f_tb2_snap_bytes(m), "snap");
        float *hc_save = (float *)ds4f_xalloc(64, hcb, "hc_save"), *Xin = NULL, *vhc = NULL;
        if (spec_batch) { ds4f_alloc_prefill_batch(m, 2); Xin = (float *)ds4f_xalloc(64,(size_t)2*C*4, "Xin"); vhc = (float *)ds4f_xalloc(64, 2*hcb, "vhc"); }
        int pos = dec_base - 1, t_next = pf_last_tok, dec_steps = 0, accepts = 0, rejects = 0;
        while (n_gen < max_new) {
            gen_ids[n_gen++] = t_next;
            if (t_next == DS4F_EOS_ID) break;
            embed_lookup(m, t_next, xe);                              /* draft token@(pos+2) */
            int d = ds4f_mtp_predict(m, m->s_x4, xe, pos + 1, NULL);
            if (spec_batch) {                                        /* M2b: ONE batched verify of pos+1,pos+2 */
                embed_lookup(m, t_next, Xin); embed_lookup(m, d, Xin + C);
                ds4f_tb2_snap(m, snap, 0);
                int ot[2]; ds4f_forward_verify(m, Xin, 2, pos + 1, ot, vhc); dec_steps++;
                int m1 = ot[0], m2 = ot[1];
                if (n_gen >= max_new) { t_next = m1; break; }
                if (d == m1) {                                       /* ACCEPT: both committed */
                    gen_ids[n_gen++] = m1; t_next = m2; memcpy(m->s_x4, vhc + hcf, hcb); pos += 2; accepts++;
                    if (m1 == DS4F_EOS_ID) break;
                } else {                                             /* REJECT: undo pos+2, redo pos+1 (GEMM-consistent) */
                    ds4f_tb2_snap(m, snap, 1); embed_lookup(m, t_next, Xin);
                    ds4f_forward_verify(m, Xin, 1, pos + 1, ot, vhc); dec_steps++;
                    t_next = ot[0]; memcpy(m->s_x4, vhc, hcb); pos += 1; rejects++;
                }
            } else {                                                /* M2 step 1: LOOPED verify (byte-identical) */
                embed_lookup(m, t_next, x); m->bytes_read = 0;
                int m1 = ds4f_forward_token(m, x, pos + 1);
                dec_bytes += m->bytes_read; dec_steps++;
                if (n_gen >= max_new) { t_next = m1; break; }
                memcpy(hc_save, m->s_x4, hcb); ds4f_tb2_snap(m, snap, 0);
                embed_lookup(m, d, x); m->bytes_read = 0;
                int m2 = ds4f_forward_token(m, x, pos + 2);
                dec_bytes += m->bytes_read; dec_steps++;
                if (d == m1) { gen_ids[n_gen++] = m1; pos += 2; t_next = m2; accepts++; if (m1 == DS4F_EOS_ID) break; }
                else { ds4f_tb2_snap(m, snap, 1); memcpy(m->s_x4, hc_save, hcb); pos += 1; t_next = m1; rejects++; }
            }
        }
        last_tok = t_next; maxgen = n_gen; free(snap); free(hc_save); free(Xin); free(vhc);
        if (MyRank == 0) logmsg("SPEC%s accepts=%d rejects=%d emitted=%d forwards=%d -> %.3f tok/fwd (decode tok/s = real gen rate)\n",
                                spec_batch?"_BATCH":"", accepts, rejects, n_gen, dec_steps, dec_steps ? (double)n_gen/dec_steps : 0.0);
    } else if (gen_mode) {
        /* greedy: pf_last_tok is the prompt's first prediction (token at pos
         * dec_base). Feed it back, argmax->embed->next, until eos or max_new.
         * forward(x,pos) places `cur` at `pos` and predicts pos+1. */
        gen_ids = (int *)ds4f_xmalloc((size_t)(max_new + 1) * sizeof(int), "s");
        int gemm_decode = envi("DS4F_GEMM_DECODE", 0);   /* run each position through verify(K=1) -- the GEMM-path
                                                          * decode that the batched spec verify is token-identical to */
        if (gemm_decode) ds4f_alloc_prefill_batch(m, 2);
        int cur = pf_last_tok, dec_steps = 0;
        for (int g = 0; g < max_new; g++) {
            gen_ids[n_gen++] = cur;
            if (cur == DS4F_EOS_ID) break;
            int pos = dec_base + g;
            embed_lookup(m, cur, x);
            m->bytes_read = 0;
            if (gemm_decode) { int ot; ds4f_forward_verify(m, x, 1, pos, &ot, m->s_x4); cur = ot; }
            else cur = ds4f_forward_token(m, x, pos);
            dec_bytes += m->bytes_read;
            dec_steps++;
            if (mtp_on) {   /* check last step's draft vs the real next token (alpha), then draft for the next */
                if (mtp_prev_draft >= 0) { mtp_total++; if (mtp_prev_draft == cur) mtp_hits++; }
                if (cur != DS4F_EOS_ID) { embed_lookup(m, cur, xe);
                    mtp_prev_draft = ds4f_mtp_predict(m, m->s_x4, xe, pos + 1, NULL); }
            }
        }
        last_tok = cur;
        maxgen = dec_steps;     /* report tok/s over actual decode forward calls */
        if (mtp_on && MyRank == 0)
            logmsg("MTP_ALPHA %d/%d = %.1f%% (MTP draft == main next-tok; alpha -- the spec-decode gain predictor)\n",
                   mtp_hits, mtp_total, mtp_total ? 100.0*mtp_hits/mtp_total : 0.0);
    } else {
        for (int g = 0; g < maxgen; g++) {
            int pos = dec_base + g;
            for (int i = 0; i < C; i++) x[i] = (float)(sm_next() * 2.0 - 1.0);
            m->bytes_read = 0;
            last_tok = ds4f_forward_token(m, x, pos);
            dec_bytes += m->bytes_read;
        }
    }
    double t_dec = now_sec() - t_dec0;
    double dec_ar = g_ar_secs;

    barrier();   /* final lockstep barrier */

    /* ---- per-rank report ---- */
    {   char rn[64]; snprintf(rn, sizeof rn, "ds4f_ep_perf_rank%02d.txt", MyRank);
        FILE *rf = fopen(rn, "w");
        if (rf) {
            fprintf(rf, "rank %d/%d  owned=%d/layer  RSS=%.2f GB\n", MyRank, N, no, rss_bytes()/1e9);
            if (prefill > 0)
                fprintf(rf, "prefill: %d tok  %.1f ms/tok  %.2f tok/s  comm %.1f%%  ar_calls=%ld  argmax=%d%s\n",
                        prefill, t_pf/prefill*1e3, prefill/t_pf, 100.0*pf_ar/t_pf, pf_calls,
                        pf_last_tok, prefill_batch > 0 ? "  [batched]" : "");
            if (maxgen > 0)
                fprintf(rf, "decode:  %d tok  %.1f ms/tok  %.2f tok/s  comm %.1f%%  %.1f GB/s-weights\n",
                        maxgen, t_dec/maxgen*1e3, maxgen/t_dec, 100.0*dec_ar/t_dec,
                        (dec_bytes/(double)maxgen)/(t_dec/maxgen)/1e9);
            fprintf(rf, "last argmax=%d (synthetic; identical across ranks == lockstep ok)\n", last_tok);
            fprintf(rf, "prefill ||x||=%.3e NaNs=%d\n", sqrt(xnorm), nan_count);
            fclose(rf);
        }
    }
    if (MyRank == 0) {
        logmsg("\n=== rank0 summary (%d nodes, EP all-reduce combine) ===\n", N);
        if (prefill > 0)
            logmsg("prefill: %d tok  %.1f ms/tok  %.2f tok/s   comm %.1f%% (ar_calls=%ld argmax=%d)%s\n",
                   prefill, t_pf/prefill*1e3, prefill/t_pf, 100.0*pf_ar/t_pf, pf_calls, pf_last_tok,
                   prefill_batch > 0 ? "  [batched]" : "");
        if (maxgen > 0) {
            logmsg("decode:  %d tok  %.1f ms/tok  %.2f tok/s   comm %.1f%% (%.0f us/tok)\n",
                   maxgen, t_dec/maxgen*1e3, maxgen/t_dec, 100.0*dec_ar/t_dec, dec_ar/maxgen*1e6);
            logmsg("         %.2f GB/tok-weights  argmax=%d  NaNs=%d  RSS=%.2f GB\n",
                   (dec_bytes/(double)maxgen)/1e9, last_tok, nan_count, rss_bytes()/1e9);
        }
        /* psum = top-level phases only (0..TB2PREP); TB2SCAN.. are SUB-timers of TB2PREP and
         * would double-count. Their printed % is then a clean "% of decode" reading. */
        double psum = 0; for (int i = 0; i <= DS4F_P_TB2PREP; i++) psum += m->prof[i];
        if (psum > 0 && maxgen > 0) {
            logmsg("per-phase decode (ms/tok):\n");
            for (int i = 0; i < DS4F_NPHASE; i++) {
                double ms = m->prof[i]/maxgen*1e3; if (ms <= 0) continue;
                logmsg("  %-9s %7.3f ms  %5.1f%%\n", ds4f_prof_names[i], ms, 100.0*m->prof[i]/psum);
            }
            /* tb2lcmp+tb2icmp split: compressor MATVEC dispatch vs the serial softmax/state tail */
            logmsg("  (cmp_matvec %.3f ms  -> tail = tb2lcmp+tb2icmp - this)\n", g_cmp_mv_secs/maxgen*1e3);
        }
        /* gen mode: rank 0 writes the generated token-id stream for detokenize */
        if (gen_mode && gen_out_file && *gen_out_file) {
            FILE *gf = fopen(gen_out_file, "w");
            if (gf) {
                for (int i = 0; i < n_gen; i++) fprintf(gf, "%d%s", gen_ids[i], i+1 < n_gen ? " " : "\n");
                fclose(gf);
                logmsg("gen: wrote %d token ids to %s%s\n", n_gen, gen_out_file,
                       (n_gen > 0 && gen_ids[n_gen-1] == DS4F_EOS_ID) ? " (eos)" : " (max_new)");
            } else {
                logmsg("gen: WARNING could not open DS4F_GEN_OUT=%s for write\n", gen_out_file);
            }
        }
    }

    free(prompt_ids);
    free(gen_ids);
    free(x);
    ds4f_free(m);
    return 0;
}
