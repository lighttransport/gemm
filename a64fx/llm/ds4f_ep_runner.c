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
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <utofu.h>

#include "ds4f.h"
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

#define MAX_NODES 32
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

/* ---- topology (tofu_topo.txt; written once by tofu_topo_helper) ---- */
static int read_topo(uint8_t coords[][TOFU_NCOORDS]) {
    FILE *f = fopen(TOPO_PATH, "r");
    if (!f) { perror("cannot open " TOPO_PATH); fprintf(stderr, "  (run tofu_topo_helper first)\n"); exit(1); }
    int n = 0; char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n >= MAX_NODES) { fprintf(stderr, "too many nodes\n"); exit(1); }
        unsigned r, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u", &r, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) != 7)
            { fprintf(stderr, "malformed line: %s", line); exit(1); }
        if ((int)r != n) { fprintf(stderr, "%s ranks out of order\n", TOPO_PATH); exit(1); }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    if (n < 1) { fprintf(stderr, "%s lists %d node(s)\n", TOPO_PATH, n); exit(1); }
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

int main(void) {
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
    if (MyRank == -1) { fprintf(stderr, "my coords not in %s\n", TOPO_PATH); exit(1); }

    /* FJ mpiexec drops per-rank stderr -> capture to a file so diagnostics survive */
    {   char en[64]; snprintf(en, sizeof en, "ds4f_ep_stderr_rank%02d.txt", MyRank);
        if (!freopen(en, "w", stderr)) { /* keep going */ }
        setvbuf(stderr, NULL, _IOLBF, 0);
    }
    if (MyRank == 0) g_log = fopen("ds4f_ep_rank00.txt", "w");

    int ep_rank = MyRank, ep_size = N;
    ds4f_config cfg = ds4f_default_config();
    cfg.max_pos = maxpos;
    if (layers > 0) cfg.n_layers = layers;
    if (prefill + maxgen > cfg.max_pos) die("prefill+maxgen exceeds max_pos", -1);
    if (ctx_warm + maxgen > cfg.max_pos) die("ctx_warm+maxgen exceeds max_pos", -1);

    int dense_bf16 = envi("DS4F_FP8_BF16", 0);
    const char *pv_e = getenv("DS4F_BF16_PV");          /* auto-on with predequant unless explicitly set */
    int bf16_pv = (pv_e && *pv_e) ? (atoi(pv_e) != 0) : dense_bf16;
    int no = ds4f_n_owned(cfg.n_experts, ep_rank, ep_size);
    size_t arena_est = ds4f_arena_size(&cfg, ep_rank, ep_size, dense_bf16);
    if (MyRank == 0)
        logmsg("=== DS4F EP synthetic harness (Stage 2): %d ranks ===\n"
               "layers=%d hidden=%d experts=%d active=%d  owned~%d/layer  dense=%s\n"
               "threads=%d prefill=%d maxgen=%d max_pos=%d ctx_warm=%d  arena~%.2f GB/node\n",
               N, cfg.n_layers, cfg.hidden, cfg.n_experts, cfg.n_active, no,
               dense_bf16 ? (bf16_pv ? "BF16(predequant,pv)" : "BF16(predequant)") : "FP8(on-demand)",
               n_threads, prefill, maxgen, maxpos, ctx_warm, arena_est/(1024.0*1024.0*1024.0));

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
    if (prefill_batch > 0 && !dense_bf16 && MyRank == 0)
        logmsg("WARN: DS4F_PREFILL_BATCH without DS4F_FP8_BF16=1 -> dense falls back to per-token matvec (no speedup)\n");
    int ar_mtile = (prefill_batch > 0 && prefill > 0)
                 ? (prefill_batch < prefill ? prefill_batch : prefill) : 1;

    /* ---- all-reduce comm region (hidden*ar_mtile floats) + wire model hook ---- */
    static tp_comm comm;
    if (tp_comm_init(&comm, Vcq, PeerVcq, MyRank, N, cfg.hidden * ar_mtile, barrier) != 0)
        die("tp_comm_init", -1);
    m->ar_cb = ep_ar_callback; m->ar_ctx = &comm;

    barrier_robust(1);   /* everyone past load + comm init (robust: tolerate skew) */
    if (MyRank == 0) logmsg("all %d ranks past bootstrap barrier; starting prefill%s\n",
                            N, prefill_batch > 0 ? " [batched M-token GEMM]" : "");

    int C = cfg.hidden;
    float *x = (float *)aligned_alloc(256, (size_t)C * 4);

    /* ---- prefill (synthetic, identical activations on every rank) ----
     * Batched path (prefill_batch>0): ds4f_forward_prefill processes M tokens per call
     * so each weight is read from HBM once per M-tile (compute-bound) and the per-layer
     * EP all-reduce fires ONCE per tile instead of once per token (latency amortized
     * M-fold). The sm_next() draw order is identical to the sequential path, so X is
     * bit-identical on every rank -> replicated dense + lockstep argmax preserved. ---- */
    double t_pf0 = now_sec(); size_t pf_bytes = 0; g_ar_secs = 0; g_ar_calls = 0;
    int nan_count = 0; double xnorm = 0.0; int pf_last_tok = -1;
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
        for (int p = 0; p < prefill; p++) {
            for (int i = 0; i < C; i++) x[i] = (float)(sm_next() * 2.0 - 1.0);
            m->bytes_read = 0;
            pf_last_tok = ds4f_forward_token(m, x, p);
            pf_bytes += m->bytes_read;
        }
        for (int i = 0; i < C; i++) { if (!(x[i] == x[i])) nan_count++; xnorm += (double)x[i]*x[i]; }
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
        ds4f_warm_tb2(m, ctx_warm);
        dec_base = ctx_warm;
        if (MyRank == 0) logmsg("warmed synthetic KV+compressed caches to ctx=%d; decoding from there\n", ctx_warm);
    }

    /* ---- decode (M=1, token-at-a-time) ---- */
    memset(m->prof, 0, sizeof(m->prof));
    double t_dec0 = now_sec(); size_t dec_bytes = 0; g_ar_secs = 0; g_ar_calls = 0;
    int last_tok = 0;
    for (int g = 0; g < maxgen; g++) {
        int pos = dec_base + g;
        for (int i = 0; i < C; i++) x[i] = (float)(sm_next() * 2.0 - 1.0);
        m->bytes_read = 0;
        last_tok = ds4f_forward_token(m, x, pos);
        dec_bytes += m->bytes_read;
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
        double psum = 0; for (int i = 0; i < DS4F_NPHASE; i++) psum += m->prof[i];
        if (psum > 0 && maxgen > 0) {
            logmsg("per-phase decode (ms/tok):\n");
            for (int i = 0; i < DS4F_NPHASE; i++) {
                double ms = m->prof[i]/maxgen*1e3; if (ms <= 0) continue;
                logmsg("  %-9s %7.3f ms  %5.1f%%\n", ds4f_prof_names[i], ms, 100.0*m->prof[i]/psum);
            }
        }
    }

    free(x);
    ds4f_free(m);
    return 0;
}
