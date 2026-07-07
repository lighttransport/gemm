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
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
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
static inline double ds4f_rng_u01(void) {                  /* uniform [0,1) */
    uint64_t z = (ds4f_rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}
typedef struct { float temp, top_p, pres_pen, rep_pen; int top_k, rep_last_n; } ds4f_sampler;

static const float *ds4f_srt_key;                          /* qsort key (single-threaded serve path) */
static int ds4f_srt_desc(const void *a, const void *b) {
    float fa = ds4f_srt_key[*(const int *)a], fb = ds4f_srt_key[*(const int *)b];
    return (fa < fb) - (fa > fb);
}
/* Sample the next token from m->s_logits given recent token history (for penalties). temp<=0 ->
 * greedy argmax (bit-identical to ds4f_forward_token). Mutates m->s_logits (regenerated next step). */
static int ds4f_sample(ds4f_model *m, const ds4f_sampler *sp, const int *hist, int nhist) {
    int V = m->cfg.vocab;
    float *lg = m->s_logits;
    if (sp->temp <= 0.f) {                                  /* greedy */
        int best = 0; float bv = lg[0];
        for (int v = 1; v < V; v++) if (lg[v] > bv) { bv = lg[v]; best = v; }
        return best;
    }
    static int *idx = NULL; static float *prob = NULL;      /* scratch, sized once to vocab */
    if (!idx) { idx = (int *)malloc((size_t)V * sizeof(int)); prob = (float *)malloc((size_t)V * sizeof(float)); }
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
    double r = ds4f_rng_u01() * nsum, acc = 0.0;
    for (int i = 0; i < npc; i++) { acc += prob[i]; if (acc >= r) return idx[i]; }
    return idx[npc - 1];
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
    if (K < 1) K = 1; if (K > 32) K = 32;
    if (pf_gemm && !m->has_mtp) {                       /* batched-verify prefill */
        ds4f_alloc_prefill_batch(m, K);
        size_t hcC = (size_t)m->cfg.hc_mult * C;
        float *Xin = (float *)aligned_alloc(64, (size_t)K*C*4), *vhc = (float *)aligned_alloc(64, (size_t)K*hcC*4);
        int lastM = 0;
        for (int base = pf_from; base < np; base += K) {
            int M = np - base < K ? np - base : K; lastM = M;
            for (int mm = 0; mm < M; mm++) embed_lookup(m, pids[base+mm], Xin + (size_t)mm*C);
            int ot[32]; ds4f_forward_verify(m, Xin, M, base, ot, vhc); pf_last = ot[M-1];
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
    int *hist = (int *)malloc((size_t)(np + max_new) * sizeof(int));   /* penalty history: prompt + gen */
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
static long ds4f_read_seq(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return 0;
    long v = 0; if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); return v;
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
    char *buf = (char *)malloc(sz); if (!buf) return -1;
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
    char *buf = (char *)malloc(sz); if (!buf) { fclose(f); return -1; }
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
      "  --preset decode     bundle: FP8_BF16+Q8_DENSE+HC_PAR+HC_RMSPAR+TIERB2+MHC+OPROJ_FUSE+ATTN_SVE+TP_HEAD\n"
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

int main(int argc,char**argv){
    ds4f_cli(argc,argv);
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
        int cap = 1024; prompt_ids = (int *)malloc((size_t)cap*sizeof(int));
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

    /* FJ mpiexec drops per-rank stderr -> capture to a file so diagnostics survive */
    {   char en[64]; snprintf(en, sizeof en, "ds4f_ep_stderr_rank%02d.txt", MyRank);
        if (!freopen(en, "w", stderr)) { /* keep going */ }
        setvbuf(stderr, NULL, _IOLBF, 0);
    }
    if (MyRank == 0) g_log = fopen("ds4f_ep_rank00.txt", "w");

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
    size_t arena_est = ds4f_arena_size(&cfg, ep_rank, ep_size, dense_bf16,
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
    double ta0 = now_sec();
    ds4f_model *m = real_weights
        ? ds4f_load_real(cfg, ep_rank, ep_size, blob_dir, n_threads, n_cmgs)
        : ds4f_alloc_synth(cfg, ep_rank, ep_size, n_threads, n_cmgs);
    if (!m) { fprintf(stderr, "rank %d: model alloc/load failed\n", MyRank); exit(1); }
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
        float *Xb  = (float *)aligned_alloc(64, (size_t)maxM*C*4);
        float *hcb = (float *)aligned_alloc(64, (size_t)maxM*(size_t)hc*C*4);
        int *cur = (int *)malloc((size_t)maxM*sizeof(int)), *pos = (int *)malloc((size_t)maxM*sizeof(int));
        int *ot  = (int *)malloc((size_t)maxM*sizeof(int));
        if (MyRank == 0) logmsg("DECODE_BATCH throughput sweep (ND=%d steps/M):\n", ND);
        for (int mi = 0; mi < nM; mi++) {
            int M = Ms[mi];
            ds4f_serve_reset(m); ds4f_free_decode_batch(m);   /* free prior M's cache sets (no leak/thrash) */
            for (int k = 0; k < M; k++) cur[k] = 100 + k*1000;
            for (int k = 0; k < M; k++) { embed_lookup(m, cur[k], Xb + (size_t)k*C); pos[k] = 0; }
            ds4f_forward_decode_batch(m, Xb, pos, M, ot, hcb);        /* warm (alloc caches/buffers) */
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
        float *Xb  = (float *)aligned_alloc(64, (size_t)2*C*4);
        float *hcb = (float *)aligned_alloc(64, (size_t)2*(size_t)hc*C*4);
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
    float *x = (float *)aligned_alloc(256, (size_t)C * 4);

    /* ---- HTTP serve loop: load once, then loop on requests from the (python) frontend via shared-FS
     * files. Rank 0 polls a request-seq counter; a barrier releases all ranks together; all read the
     * same prompt (shared FS) -> lockstep, no broadcast. Loops until killed. ---- */
    if (envi("DS4F_SERVE", 0)) {
        const char *reqf = getenv("DS4F_SERVE_REQ"), *respf = getenv("DS4F_SERVE_RESP");
        const char *reqseqf = getenv("DS4F_SERVE_REQSEQ"), *respseqf = getenv("DS4F_SERVE_RESPSEQ");
        if (!reqf || !respf || !reqseqf || !respseqf) die("DS4F_SERVE needs DS4F_SERVE_{REQ,RESP,REQSEQ,RESPSEQ}", -1);
        int maxpos = envi("DS4F_MAXPOS", 4096);
        int *pids = (int *)malloc((size_t)maxpos * sizeof(int));
        int *oids = (int *)malloc((size_t)(maxpos + 1) * sizeof(int));
        int *fids = (int *)malloc((size_t)(maxpos + 1) * sizeof(int));   /* scratch for disk load */
        /* prefix cache (context management): each slot's ids[0,len) is the token sequence held in the
         * KV/compressor caches. A request whose prompt EXTENDS its slot's sequence skips re-prefilling
         * the shared prefix and continues from len (the multi-turn TTFT win). Divergence -> full reset. */
        int prefix_cache = envi("DS4F_SERVE_PREFIX_CACHE", 1);
        /* multi-context slots: DS4F_SERVE_SLOTS independent conversations, one live in the caches at a
         * time. A request's slot id context-switches by snapshotting the live caches to the old slot and
         * restoring the new slot's snapshot (ds4f_ctx_snap). Each slot keeps its own token sequence. */
        int nslots = envi("DS4F_SERVE_SLOTS", 1); if (nslots < 1) nslots = 1; if (nslots > 64) nslots = 64;
        typedef struct { char *snap; int *ids; int len; int used; } serve_slot;
        serve_slot *slots = (serve_slot *)calloc((size_t)nslots, sizeof(serve_slot));
        for (int i = 0; i < nslots; i++) slots[i].ids = (int *)malloc((size_t)(maxpos + 1) * sizeof(int));
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
    float *xe = mtp_on ? (float *)aligned_alloc(64, (size_t)C * 4) : NULL;
    int mtp_prev_draft = -1, mtp_hits = 0, mtp_total = 0;
    sm_state = 0xD5F00D;   /* SAME seed on every rank -> replicated dense + valid all-reduce */
    if (prefill_batch > 0 && prefill > 0) {
        ds4f_alloc_prefill_batch(m, ar_mtile);
        float *X  = (float *)aligned_alloc(256, (size_t)ar_mtile * C * 4);
        int   *bt = (int *)malloc((size_t)ar_mtile * sizeof(int));
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
        /* DS4F_PREFILL_GEMM: batch the gen prefill through the MHC+Tier-B2-capable verify path
         * (chunks of K<=8) -> the per-layer EP all-reduce fires once per K tokens (comm ÷K) and the
         * dense projections become an M=K GEMM instead of K matvecs. Attn/tb2/mHC stay per-position
         * (looped, causal). COHERENT not bit-identical to token-by-token (GEMM reassoc, like GEMM-decode);
         * seeds the same decode. Not with MTP (per-token MTP KV maintenance). */
        int pf_gemm = gen_mode && !mtp_on && envi("DS4F_PREFILL_GEMM", 0);
        if (pf_gemm) {
            int K = envi("DS4F_PREFILL_K", 32); if (K < 1) K = 1; if (K > 32) K = 32;
            ds4f_alloc_prefill_batch(m, K);
            size_t hcC = (size_t)m->cfg.hc_mult * C;
            float *Xin = (float *)aligned_alloc(64, (size_t)K * C * 4);
            float *vhc = (float *)aligned_alloc(64, (size_t)K * hcC * 4);
            int Mlast = 1;
            for (int base = 0; base < prefill; base += K) {
                int M = prefill - base < K ? prefill - base : K; Mlast = M;
                for (int mm = 0; mm < M; mm++) embed_lookup(m, prompt_ids[base + mm], Xin + (size_t)mm * C);
                m->bytes_read = 0; int ot[8];
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
    memset(m->prof, 0, sizeof(m->prof));
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
        gen_ids = (int *)malloc((size_t)(max_new + 2) * sizeof(int));
        char *snap = (char *)malloc(ds4f_tb2_snap_bytes(m));
        float *hc_save = (float *)aligned_alloc(64, hcb), *Xin = NULL, *vhc = NULL;
        if (spec_batch) { ds4f_alloc_prefill_batch(m, 2); Xin = (float *)aligned_alloc(64,(size_t)2*C*4); vhc = (float *)aligned_alloc(64, 2*hcb); }
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
        gen_ids = (int *)malloc((size_t)(max_new + 1) * sizeof(int));
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
