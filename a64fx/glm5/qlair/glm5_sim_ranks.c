/* glm5_sim_ranks.c — multi-rank GLM-5.2 decode under the qlair A64FX simulator.
 *
 * Runs R EP ranks as pthreads in ONE process (qlair --ranks R maps thread ->
 * rank; the TofuSimulator services the uTofu API the production tp_allreduce.h
 * uses). Each rank owns a full glm5_model (synthetic weights, reduced config)
 * — the same forward code, expert-parallel sharding, and all-reduce protocol
 * as the Fugaku runner, minus the runner's process-global bootstrap.
 *
 * Purpose (all locally, no Fugaku job):
 *   TEST 1  lockstep single-stream decode — every rank must emit the identical
 *           token stream (the EP-lockstep invariant).
 *   TEST 2  batched-vs-single at M>1 — glm5_forward_batch_decode_mla must
 *           reproduce M independent single-stream generations bit-exactly
 *           (per-stream KV + routing independence; the gate the batched-decode
 *           plan could not run without a job).
 *   Also prints per-allreduce simulated ns (feeds decode_sim.recalibrate).
 *
 * Config (env): GLM5_SIM_RANKS(8) GLM5_LAYERS(4) GLM5_DENSE(3) GLM5_EXPERTS(16)
 *   GLM5_VOCAB(8192) GLM5_MAXPOS(64) GLM5_PREFILL(4) GLM5_DECODE(4)
 *   GLM5_BATCH_M(2) GLM5_TP(1: shard attn/head/etc across ranks — the
 *   production decode config) TP_AR_ROBUST/TP_AR_BF16 as in production.
 *
 * Build (x86 host, cross gcc — NO OpenMP; qlair threads are the ranks):
 *   aarch64-linux-gnu-gcc -O2 -static -march=armv8.2-a+sve -fno-math-errno \
 *     -I . -I ../../../common -I ../../utofu-tests \
 *     glm5_sim_ranks.c utofu_stubs.c -o glm5_sim.elf -lm -lpthread
 * Run:
 *   QLAIR_ALIGNED_HEAP_MB=24576 GLM5_SIM_RANKS=8 GLM5_TP=1 \
 *     qlair --cores 8 --ranks 8 -n 200G glm5_sim.elf
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
/* unbuffered progress marker (qlair's host printf is block-buffered under pipes/files) */
#define MK(s) do { ssize_t w_ = write(2, s, sizeof(s) - 1); (void)w_; } while (0)

#include "utofu.h"                 /* qlair shim; also satisfies <utofu.h> */
#include "tp_allreduce.h"          /* the PRODUCTION all-reduce under test */

#include "glm5.h"
#include "glm5_impl.h"

#define MAXR   16
#define MAXTOK 64
#define MAXM   8

static int R = 8;

/* ---- cross-rank bootstrap + verification (threads share the address space) */
static utofu_vcq_id_t    g_vcq_id[MAXR];
static pthread_barrier_t g_bar;
static void bar(void) { pthread_barrier_wait(&g_bar); }

static int  g_gen[MAXR][MAXTOK];            /* TEST 1: per-rank token stream  */
static int  g_single[MAXM][MAXTOK];         /* TEST 2: reference generations  */
static int  g_batch[MAXM][MAXTOK];          /*         batched generations    */
static volatile int g_fail = 0;

/* per-AR simulated-ns accounting (qlair CNTVCT == simulated ns) */
static uint64_t g_ar_ns[MAXR];
static long     g_ar_calls[MAXR];

typedef struct {
    int          rank;
    glm5_model  *m;
    utofu_vcq_hdl_t vcq;
    tp_comm      comm;
} simrank_t;
static simrank_t g_ctx[MAXR];

static int envi_(const char *k, int d) {
    const char *e = getenv(k);
    return (e && *e) ? atoi(e) : d;
}

/* ---- EP callbacks: fragmented over max_count exactly like glm5_ep_runner ---- */
static void sim_ar_cb(float *buf, int count, void *ctx) {
    simrank_t *sc = (simrank_t *)ctx;
    tp_comm *c = &sc->comm;
    int mc = c->max_count > 0 ? c->max_count : count;
    uint64_t t0 = qlair_rd_cyc();
    for (int off = 0; off < count;) {
        int n = count - off; if (n > mc) n = mc;
        tp_allreduce_sum(c, buf + off, n); off += n;
    }
    g_ar_ns[sc->rank] += qlair_rd_cyc() - t0;
    g_ar_calls[sc->rank]++;
}
static void sim_argmax_cb(float *val, int32_t *idx, void *ctx) {
    simrank_t *sc = (simrank_t *)ctx;
    uint64_t t0 = qlair_rd_cyc();
    tp_allreduce_argmax(&sc->comm, val, idx);
    g_ar_ns[sc->rank] += qlair_rd_cyc() - t0;
    g_ar_calls[sc->rank]++;
}
static void sim_argmax_n_cb(float *vi, int n, void *ctx) {
    simrank_t *sc = (simrank_t *)ctx;
    uint64_t t0 = qlair_rd_cyc();
    tp_allreduce_argmax_n(&sc->comm, vi, n);
    g_ar_ns[sc->rank] += qlair_rd_cyc() - t0;
    g_ar_calls[sc->rank]++;
}

/* token id -> embedding (TP_EMBED-aware, same as glm5_ep_runner.c) */
static void sim_embed(glm5_model *m, int tok, float *x) {
    int H = m->cfg.hidden;
    if (tok < 0 || tok >= m->cfg.vocab) tok = 0;
    if (m->emb_rows < m->cfg.vocab) {
        for (int i = 0; i < H; i++) x[i] = 0.f;
        if (tok >= m->emb_r0 && tok < m->emb_r0 + m->emb_rows) {
            const uint16_t *row = m->embed + (size_t)(tok - m->emb_r0) * H;
            for (int i = 0; i < H; i++) x[i] = glm5_bf2f(row[i]);
        }
        if (m->ar_cb) m->ar_cb(x, H, m->ar_ctx);
        return;
    }
    const uint16_t *row = m->embed + (size_t)tok * H;
    for (int i = 0; i < H; i++) x[i] = glm5_bf2f(row[i]);
}

static void *rank_main(void *arg) {
    simrank_t *sc = (simrank_t *)arg;
    const int rank = sc->rank;
    glm5_model *m = sc->m;
    const int H = m->cfg.hidden;
    const int P = envi_("GLM5_PREFILL", 4);
    const int D = envi_("GLM5_DECODE", 4);
    const int M = envi_("GLM5_BATCH_M", 2);

    /* --- uTofu bootstrap (per-rank VCQ; vcq ids published via shared array) */
    utofu_tni_id_t *tnis; size_t ntni;
    if (utofu_get_onesided_tnis(&tnis, &ntni) != UTOFU_SUCCESS || ntni == 0) {
        printf("rank %d: no TNIs (not under qlair?)\n", rank); g_fail = 1; return 0;
    }
    if (utofu_create_vcq(tnis[rank % ntni], 0, &sc->vcq) != UTOFU_SUCCESS) {
        printf("rank %d: create_vcq failed\n", rank); g_fail = 1; return 0;
    }
    utofu_query_vcq_id(sc->vcq, &g_vcq_id[rank]);
    bar();

    int ar_tokens = envi_("GLM5_AR_TOKENS", MAXM);   /* slot covers [M,hidden] */
    if (tp_comm_init(&sc->comm, sc->vcq, g_vcq_id, rank, R, H * ar_tokens, bar) != 0) {
        printf("rank %d: tp_comm_init failed\n", rank); g_fail = 1; return 0;
    }
    m->ar_cb = sim_ar_cb;          m->ar_ctx = sc;
    m->ar_argmax_cb = sim_argmax_cb; m->ar_argmax_ctx = sc;
    m->ar_argmax_n_cb = sim_argmax_n_cb; m->ar_argmax_n_ctx = sc;

    float *x = glm5_amalloc((size_t)H * 4);
    float *X = glm5_amalloc((size_t)MAXM * H * 4);

    /* ================= TEST 1: lockstep single-stream decode ================ */
    {
        int tok = 1, nt = 0;
        for (int p = 0; p < P + D && nt < MAXTOK; p++) {
            sim_embed(m, tok, x);
            int out = glm5_forward_token(m, x, p);
            g_gen[rank][nt++] = out;
            tok = (p < P - 1) ? (2 + p) % m->cfg.vocab : out;  /* prompt then greedy */
        }
        bar();
        if (memcmp(g_gen[rank], g_gen[0], sizeof g_gen[0]) != 0) {
            printf("rank %d: TEST1 LOCKSTEP MISMATCH\n", rank); g_fail = 1;
        }
        bar();
        if (rank == 0) {
            printf("TEST1 lockstep decode (%d prefill + %d decode): %s [",
                   P, D, g_fail ? "FAIL" : "PASS");
            for (int i = 0; i < P + D && i < MAXTOK; i++) printf(" %d", g_gen[0][i]);
            printf(" ]\n");
        }
    }

    /* ============ TEST 2: batched M-stream vs M single-stream runs =========== */
    /* Reference: M independent single-stream generations. Restarting from pos 0
     * simply overwrites the model KV from position 0 — attention reads [0..pos],
     * so each stream sees only its own history. Streams use distinct prompts. */
    {
        MK("[t2ref]");
        for (int s = 0; s < M; s++) {
            int tok = 1 + s, nt = 0;
            for (int p = 0; p < P + D && nt < MAXTOK; p++) {
                sim_embed(m, tok, x);
                int out = glm5_forward_token(m, x, p);
                if (rank == 0) g_single[s][nt] = out;
                nt++;
                tok = (p < P - 1) ? (2 + s + p) % m->cfg.vocab : out;
            }
        }
        MK("[t2ref-done]");
        bar();

        /* Batched: drive prompt AND decode through the batch-MLA kernel; the
         * prompt positions fill the per-stream latent KV (ms->kc). */
        if (glm5_alloc_mstream_ex(m, M, 1) != 0) {
            MK("[t2-msalloc-FAIL]");
            printf("rank %d: mstream alloc failed\n", rank); g_fail = 1; return 0;
        }
        MK("[t2ms]");
        int cur[MAXM], pos[MAXM], out[MAXM];
        for (int s = 0; s < M; s++) cur[s] = 1 + s;
        for (int p = 0; p < P + D; p++) {
            for (int s = 0; s < M; s++) {
                sim_embed(m, cur[s], X + (size_t)s * H);
                pos[s] = p;
            }
            MK("[t2f]");
            glm5_forward_batch_decode_mla(m, X, M, pos, out);
            for (int s = 0; s < M; s++) {
                if (rank == 0) g_batch[s][p] = out[s];
                cur[s] = (p < P - 1) ? (2 + s + p) % m->cfg.vocab : out[s];
            }
        }
        MK("[t2done]");
        bar();
        if (rank == 0) {
            int bad = 0;
            for (int s = 0; s < M; s++)
                for (int p = 0; p < P + D; p++)
                    if (g_batch[s][p] != g_single[s][p]) {
                        if (!bad)
                            printf("TEST2 MISMATCH stream %d step %d batch=%d single=%d\n",
                                   s, p, g_batch[s][p], g_single[s][p]);
                        bad++;
                    }
            if (bad) { printf("TEST2 batched-vs-single M=%d: FAIL (%d diffs)\n", M, bad); g_fail = 1; }
            else       printf("TEST2 batched-vs-single M=%d: PASS (%d steps bit-identical)\n", M, P + D);
        }
        bar();
    }

    /* ====== TEST 3: MTP draft + 2-entry verify (mechanics, lockstep, greedy-equivalence) ====== */
    /* Speculation is EXACT under greedy decode: whatever the acceptance pattern, the emitted
     * stream must equal the plain single-stream greedy stream. Also every rank must agree. */
    if (envi_("GLM5_MTP", 1) && m->mtp_layer) {
        const int D2 = envi_("GLM5_MTP_STEPS", 6);
        float *hbuf = glm5_amalloc((size_t)H * 4);
        float *xb2  = glm5_amalloc((size_t)2 * H * 4);
        int ref[MAXTOK]; int nref = 0;
        {   /* reference: plain greedy on the model KV (restart from pos 0 overwrites) */
            int tok = 3;
            for (int p = 0; p < P + D2 && nref < MAXTOK; p++) {
                sim_embed(m, tok, x);
                int o = glm5_forward_token(m, x, p);
                ref[nref++] = o;
                tok = (p < P - 1) ? (4 + p) % m->cfg.vocab : o;
            }
        }
        glm5_mstream *ms = (glm5_mstream *)m->ms;
        static const int sid0[2] = {0, 0};
        int pos2[2], out2[2], acc = 0, rej = 0, nm = 0;
        int cur = 3;
        ms->sid = sid0;
        for (int p = 0; p < P; p++) {              /* prompt via the batch kernel, stream 0 */
            sim_embed(m, cur, X);
            pos2[0] = p;
            glm5_forward_batch_decode_mla(m, X, 1, pos2, out2);
            if (nm < MAXTOK) g_gen[rank][nm++] = out2[0];
            memcpy(hbuf, X, (size_t)H * 4);
            cur = (p < P - 1) ? (4 + p) % m->cfg.vocab : out2[0];
        }
        int p = P - 1;                             /* last KV-filled position */
        while (nm < P + D2 && nm < MAXTOK) {
            int d = glm5_mtp_draft(m, hbuf, cur, p + 1, xb2);
            sim_embed(m, cur, X); sim_embed(m, d, X + H);
            pos2[0] = p + 1; pos2[1] = p + 2;
            glm5_forward_batch_decode_mla(m, X, 2, pos2, out2);
            if (out2[0] == d) {                    /* accept: 2 tokens this forward */
                g_gen[rank][nm++] = out2[0];
                if (nm < P + D2 && nm < MAXTOK) g_gen[rank][nm++] = out2[1];
                memcpy(hbuf, X + (size_t)H, (size_t)H * 4);
                cur = out2[1]; p += 2; acc++;
            } else {                               /* reject: 1 token; wrong p+2 KV is
                                                    * overwritten by next round's entry 0 */
                g_gen[rank][nm++] = out2[0];
                memcpy(hbuf, X, (size_t)H * 4);
                cur = out2[0]; p += 1; rej++;
            }
        }
        ms->sid = NULL;
        bar();
        if (memcmp(g_gen[rank], g_gen[0], sizeof g_gen[0]) != 0) {
            printf("rank %d: TEST3 MTP LOCKSTEP MISMATCH\n", rank); g_fail = 1;
        }
        bar();
        if (rank == 0) {
            int bad = 0;
            for (int i = 0; i < nm && i < nref; i++) if (g_gen[0][i] != ref[i]) bad++;
            if (bad) g_fail = 1;
            printf("TEST3 MTP draft+verify: %s (accept=%d reject=%d, %d tokens, %d diffs vs greedy ref)\n",
                   bad ? "FAIL" : "PASS", acc, rej, nm, bad);
        }
        glm5_afree(hbuf); glm5_afree(xb2);
    }

    /* ====== TEST 4: chunked prefill (glm5_forward_prefill_chunk) vs token-by-token ====== */
    /* The gate for ANY prefill comm rework (query-SP, moe a2a). The chunk path uses batched
     * GEMMs whose fp summation order differs from the matvec path, so the cross-path check is
     * ARGMAX equality (last prompt argmax + D4 greedy continuations decoded from the chunk-built
     * KV); rank LOCKSTEP stays bitwise. Also exercises a two-chunk boundary (p0>0). */
    if (envi_("GLM5_TEST4", 1)) {
        const int P4 = 6, D4 = 3;
        int prompt[P4]; for (int i = 0; i < P4; i++) prompt[i] = (5 + 3 * i) % m->cfg.vocab;
        int ref[P4 + D4], nref = 0;
        {   /* reference: token-by-token prefill + greedy decode on the model KV */
            int last = -1;
            for (int p = 0; p < P4; p++) { sim_embed(m, prompt[p], x); last = glm5_forward_token(m, x, p); }
            ref[nref++] = last;
            for (int d = 0; d < D4; d++) { sim_embed(m, last, x); last = glm5_forward_token(m, x, P4 + d); ref[nref++] = last; }
        }
        if (m->ms) glm5_free_mstream(m);              /* TEST2/3 allocated the decode-mode mstream */
        if (glm5_alloc_mstream_ex(m, P4, 0) != 0) {   /* prefill mode: shared model KV, MSA buffers */
            printf("rank %d: TEST4 mstream alloc failed\n", rank); g_fail = 1; return 0;
        }
        float *Xc = glm5_amalloc((size_t)P4 * H * 4);
        for (int variant = 0; variant < 2; variant++) {   /* 0: one chunk; 1: two chunks (boundary) */
            int got[P4 + D4], ng = 0, a = -1;
            if (variant == 0) {
                for (int t = 0; t < P4; t++) sim_embed(m, prompt[t], Xc + (size_t)t * H);
                a = glm5_forward_prefill_chunk(m, Xc, P4, 0, 1);
            } else {
                int h1 = P4 / 2;
                for (int t = 0; t < h1; t++) sim_embed(m, prompt[t], Xc + (size_t)t * H);
                glm5_forward_prefill_chunk(m, Xc, h1, 0, 0);
                for (int t = h1; t < P4; t++) sim_embed(m, prompt[t], Xc + (size_t)(t - h1) * H);
                a = glm5_forward_prefill_chunk(m, Xc, P4 - h1, h1, 1);
            }
            got[ng++] = a;
            int last = a;                              /* decode continues on the chunk-built KV */
            for (int d = 0; d < D4; d++) { sim_embed(m, last, x); last = glm5_forward_token(m, x, P4 + d); got[ng++] = last; }
            int bad = 0;
            for (int i = 0; i < ng && i < nref; i++) if (got[i] != ref[i]) bad++;
            memcpy(g_gen[rank], got, (size_t)ng * sizeof(int));   /* compare FILLED entries only */
            bar();
            if (memcmp(g_gen[rank], g_gen[0], (size_t)ng * sizeof(int)) != 0) {
                printf("rank %d: TEST4 v%d LOCKSTEP MISMATCH\n", rank, variant); g_fail = 1;
            }
            bar();
            if (rank == 0) {
                if (bad) g_fail = 1;
                printf("TEST4 prefill-chunk vs token-by-token (%s): %s (%d diffs) [",
                       variant ? "2 chunks" : "1 chunk", bad ? "FAIL" : "PASS", bad);
                for (int i = 0; i < ng; i++) printf(" %d", got[i]);
                printf(" ]\n");
            }
        }
        glm5_afree(Xc);
    }

    /* ====== TEST 5: query-SP prefill (glm5_forward_prefill_chunk_sp) vs token-by-token ====== */
    /* Requires REPLICATED attention (run the harness with GLM5_TP=0): the SP path gives every
     * rank all heads over its home query slice; the zero-extended ar_cb gathers must make the
     * result token-equal to the classic paths. Same argmax + D continuation gate as TEST4. */
    if (envi_("GLM5_TEST5", 1) && m->layers[0].qh1 - m->layers[0].qh0 == m->cfg.n_heads) {
        const int P5 = 6, D5 = 3;
        int prompt[P5]; for (int i = 0; i < P5; i++) prompt[i] = (5 + 3 * i) % m->cfg.vocab;
        int ref[P5 + D5], nref = 0;
        {
            int last = -1;
            for (int p = 0; p < P5; p++) { sim_embed(m, prompt[p], x); last = glm5_forward_token(m, x, p); }
            ref[nref++] = last;
            for (int d = 0; d < D5; d++) { sim_embed(m, last, x); last = glm5_forward_token(m, x, P5 + d); ref[nref++] = last; }
        }
        if (!m->ms || ((glm5_mstream *)m->ms)->kc) {   /* need prefill-mode mstream */
            if (m->ms) glm5_free_mstream(m);
            if (glm5_alloc_mstream_ex(m, P5, 0) != 0) {
                printf("rank %d: TEST5 mstream alloc failed\n", rank); g_fail = 1; return 0;
            }
        }
        float *Xc = glm5_amalloc((size_t)P5 * H * 4);
        for (int variant = 0; variant < 2; variant++) {
            int got[P5 + D5], ng = 0, a = -1;
            if (variant == 0) {
                for (int t = 0; t < P5; t++) sim_embed(m, prompt[t], Xc + (size_t)t * H);
                a = glm5_forward_prefill_chunk_sp(m, Xc, P5, 0, 1);
            } else {
                int h1 = P5 / 2;
                for (int t = 0; t < h1; t++) sim_embed(m, prompt[t], Xc + (size_t)t * H);
                glm5_forward_prefill_chunk_sp(m, Xc, h1, 0, 0);
                for (int t = h1; t < P5; t++) sim_embed(m, prompt[t], Xc + (size_t)(t - h1) * H);
                a = glm5_forward_prefill_chunk_sp(m, Xc, P5 - h1, h1, 1);
            }
            if (a < 0) { printf("rank %d: TEST5 SP path rejected (rc=%d)\n", rank, a); g_fail = 1; break; }
            got[ng++] = a;
            int last = a;
            for (int d = 0; d < D5; d++) { sim_embed(m, last, x); last = glm5_forward_token(m, x, P5 + d); got[ng++] = last; }
            int bad = 0;
            for (int i = 0; i < ng && i < nref; i++) if (got[i] != ref[i]) bad++;
            memcpy(g_gen[rank], got, (size_t)ng * sizeof(int));   /* compare FILLED entries only */
            bar();
            if (memcmp(g_gen[rank], g_gen[0], (size_t)ng * sizeof(int)) != 0) {
                printf("rank %d: TEST5 v%d LOCKSTEP MISMATCH\n", rank, variant); g_fail = 1;
            }
            bar();
            if (rank == 0) {
                if (bad) g_fail = 1;
                printf("TEST5 query-SP prefill vs token-by-token (%s): %s (%d diffs) [",
                       variant ? "2 chunks" : "1 chunk", bad ? "FAIL" : "PASS", bad);
                for (int i = 0; i < ng; i++) printf(" %d", got[i]);
                printf(" ]\n");
            }
        }
        glm5_afree(Xc);
    } else if (rank == 0 && envi_("GLM5_TEST5", 1)) {
        printf("TEST5 skipped (needs GLM5_TP=0 / replicated attention)\n");
    }

    if (rank == 0) {
        printf("per-rank all-reduce cost (simulated):\n");
        for (int r = 0; r < R; r++)
            printf("  rank %d: %ld ARs, %llu ns total, %llu ns/AR\n", r,
                   g_ar_calls[r], (unsigned long long)g_ar_ns[r],
                   (unsigned long long)(g_ar_calls[r] ? g_ar_ns[r] / g_ar_calls[r] : 0));
        printf(g_fail ? "FAIL\n" : "PASS\n");
    }
    return 0;
}

int main(void) {
    R = envi_("GLM5_SIM_RANKS", 8);
    if (R < 1 || R > MAXR) R = 8;

    glm5_config cfg = glm5_default_config();
    cfg.max_pos        = envi_("GLM5_MAXPOS", 64);
    cfg.n_layers       = envi_("GLM5_LAYERS", 4);
    cfg.n_dense_layers = envi_("GLM5_DENSE", 3);
    if (cfg.n_dense_layers > cfg.n_layers) cfg.n_dense_layers = cfg.n_layers;
    cfg.n_experts      = envi_("GLM5_EXPERTS", 16);
    if (cfg.n_active > cfg.n_experts) cfg.n_active = cfg.n_experts;
    cfg.vocab          = envi_("GLM5_VOCAB", 8192);
    /* sim-only dimension shrinks: interpretation is ~10-100 MIPS, so cut the big dims to
     * keep runs in minutes. Comm pattern / lockstep / KV addressing are dimension-agnostic. */
    cfg.hidden         = envi_("GLM5_HIDDEN", cfg.hidden);
    cfg.n_heads        = envi_("GLM5_HEADS", cfg.n_heads);
    cfg.moe_inter      = envi_("GLM5_MOE_INTER", cfg.moe_inter);
    cfg.dense_inter    = envi_("GLM5_DENSE_INTER", cfg.dense_inter);
    cfg.q_lora         = envi_("GLM5_QLORA", cfg.q_lora);

    printf("glm5_sim_ranks: R=%d layers=%d(%dd) experts=%d vocab=%d maxpos=%d "
           "TP=%d robust=%s\n",
           R, cfg.n_layers, cfg.n_dense_layers, cfg.n_experts, cfg.vocab,
           cfg.max_pos, envi_("GLM5_TP", 0),
           getenv("TP_AR_ROBUST") ? getenv("TP_AR_ROBUST") : "1(default)");

    /* Serial model alloc, one per rank. glm5_sm (the synthetic-weight PRNG) is
     * process-global; on Fugaku each rank is a separate process with the same
     * stream, so RESET IT before each rank's alloc or replicated tensors
     * (embed, router, norms) diverge across ranks and lockstep breaks. */
    int want_mtp = envi_("GLM5_MTP", 1);
    for (int r = 0; r < R; r++) {
        glm5_sm = 0;
        g_ctx[r].rank = r;
        g_ctx[r].m = glm5_alloc_synth(cfg, r, R, /*threads*/1, /*cmgs*/1);
        if (!g_ctx[r].m) { printf("rank %d: model alloc FAILED (heap too small? "
                                  "set QLAIR_ALIGNED_HEAP_MB)\n", r); return 1; }
        if (want_mtp) {
            glm5_sm = 0x4d545000;      /* distinct-but-identical-per-rank MTP weight stream */
            if (glm5_alloc_mtp_synth(g_ctx[r].m)) { printf("rank %d: MTP alloc FAILED\n", r); return 1; }
        }
        printf("  rank %d model: %lu MB%s\n", r,
               (unsigned long)(g_ctx[r].m->arena_used >> 20), want_mtp ? " +mtp" : "");
    }

    pthread_barrier_init(&g_bar, NULL, (unsigned)R);
    pthread_t th[MAXR];
    for (int r = 1; r < R; r++)
        pthread_create(&th[r], NULL, rank_main, &g_ctx[r]);
    rank_main(&g_ctx[0]);                    /* main thread = core 0 = rank 0 */
    for (int r = 1; r < R; r++) pthread_join(th[r], NULL);

    return g_fail ? 1 : 0;
}
