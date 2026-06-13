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

#define MAX_NODES 96   /* DS4P targets 48-64 ranks (was 32 for the 11-node DS4F runs) */
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
    if (MyRank == -1) { fprintf(stderr, "my coords not in %s\n", TOPO_PATH); exit(1); }

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
