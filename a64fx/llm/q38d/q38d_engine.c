/* q38d: dedicated single-node Qwen3.8-27B decode engine for A64FX.
 *
 * 48 persistent workers (CPUs 12..59, CMG c = worker / 12) execute every
 * token in lockstep. Projection rows are split into four CMG-resident parts
 * and then into twelve per-worker ranges of 8-row groups. Activations are
 * quantized per 16 columns (A8/A16) or kept in FP32 (reference mode, exact
 * weights). Per layer:
 *   SSM : in-proj (qkv, z, alpha, beta) | per-head conv + gated delta rule +
 *         gated RMSNorm (+ quantize) | out-proj into the residual
 *   attn: in-proj (q+gate, k, v) | per-CMG KV head: norm/RoPE/cache,
 *         12-way position split, merge, sigmoid gate (+ quantize) | out-proj
 *   FFN : gate/up per 16-row unit, SiLU*up, quantize | down-proj
 * Each "|" is a global barrier. RMSNorm+quantize of the residual is done
 * redundantly by every worker after the preceding barrier.
 */
#define _GNU_SOURCE
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"
#include "../qwen38_lowbit_model.h"
#include "q38d_kern.h"
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define NT 48
#define NCMG 4
#define PER 12
#define EMBD 5120
#define NFF 17408
#define NLAYER 64
#define NHEAD 24
#define NKV 4
#define HD 256
#define QKVD 10240
#define DINNER 6144
#define DS 128
#define NGROUP 16
#define NVH 48
#define NATTN 16

/* kch > 1: within each worker's group range, groups are stored K-chunk
 * major (chunk 0 of every group, then chunk 1, ...), each (group, chunk)
 * being a self-contained group of cols/kch columns. Keeps the activation
 * chunk resident in L1 for wide matrices. */
/* unit16: worker ranges are multiples of two groups (16 rows), so a worker
 * owns whole 16-column activation units of this matrix's output. */
typedef struct { int rows, cols, fmt, first[5], kch, unit16; uint8_t *part[4]; } q38d_mat;
static inline void lane_groups(const q38d_mat *m, int c, int l, int *g0, int *g1) {
    int G8 = (m->first[c + 1] - m->first[c] + 7) / 8;
    if (m->unit16 && !(G8 & 1)) { int U = G8 / 2; *g0 = 2 * (U * l / 12); *g1 = 2 * (U * (l + 1) / 12); }
    else { *g0 = G8 * l / 12; *g1 = G8 * (l + 1) / 12; }
}

typedef struct {
    int ssm, ai;
    const float *attn_norm, *post_norm;
    q38d_mat qkv, z, alpha, beta, out;
    float conv_w[QKVD * 4];
    float ssm_a[NVH], dt_bias[NVH], ssm_norm[DS];
    q38d_mat q, k, v, o;
    float q_norm[HD], k_norm[HD];
    q38d_mat gate, up, down;
} q38d_layer;

typedef struct {
    int fmt, arith, max_seq, n_vocab;
    float eps, rope_base, inv_freq[32];
    q38d_layer *L;
    q38d_mat head;
    const float *out_norm;
    const q38_lowbit_matrix *embed_lb; /* FP6 model: original tiles */
    const uint8_t *embed_q6k;          /* FP4 model: raw Q6_K rows */
    size_t embed_row_bytes;
    /* shared runtime state */
    float *x, *qkv, *zb, *ab, *bb, *qg, *kb, *vb, *o, *h, *logits;
    q38d_act act_o, act_h, act_x[NT], act_c[NCMG];
    float *xn[NT], *xn_c[NCMG];
    float *ssm_state[NLAYER][NVH];
    float *conv_state[NLAYER];
    /* per head (worker) CMG-local copies: history [4 slots][q,k,v][128] and
     * conv weights [4 taps][q,k,v][128] of that head's channels */
    float *conv_hist[NLAYER][NVH], *conv_wl[NLAYER][NVH];
    float *kc[NATTN][NKV], *vc[NATTN][NKV];
    float *qh[NCMG];        /* [6][256] per CMG */
    float *apart[NCMG];     /* [12][6][2+256] per CMG */
    float best_val[NT];
    int best_idx[NT];
    int next_token;
    _Atomic int *unit_cnt; /* per 16-row FFN unit, parity counter */
    /* profiling (thread 0) */
    double prof[16];
} q38d_engine;

enum { P_EMBED, P_SSM_IN, P_SSM_CORE, P_SSM_OUT, P_ATT_IN, P_ATT_CORE, P_ATT_OUT, P_FFN_UP, P_FFN_DOWN, P_HEAD, P_N };
static const char *prof_name[P_N] = {"embed", "ssm_in", "ssm_core", "ssm_out", "attn_in", "attn_core",
                                     "attn_out", "ffn_gateup", "ffn_down", "head"};

static uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0,cntvct_el0" : "=r"(v)); return v; }
static double tick_hz(void) { uint64_t v; __asm__ volatile("mrs %0,cntfrq_el0" : "=r"(v)); return (double)v; }
static double now_sec(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* ------------------------------------------------------------------ */
/* barriers                                                             */

typedef struct { _Atomic int v; char pad[252]; } q38d_line;
static q38d_line bar_cmg[NCMG], bar_glob, bar_sense;
static q38d_line cbar_cnt[NCMG], cbar_sense[NCMG];
static int epoch_bar = 0;

static double kprof_wait_acc;
static int busy_on;
static uint64_t busy_start[NT];
static double busy_acc[NT][16];
static int cur_phase[NT];
static void gbarrier_impl(int tid, int *sense);
static void gbarrier(int tid, int *sense) {
    uint64_t t0 = ticks();
    if (busy_on) busy_acc[tid][cur_phase[tid]] += (double)(t0 - busy_start[tid]);
    gbarrier_impl(tid, sense);
    uint64_t t1 = ticks();
    if (!tid) kprof_wait_acc += (double)(t1 - t0);
    busy_start[tid] = t1;
}
/* Epoch barriers without atomic RMW: every worker publishes its epoch in
 * its own line; each CMG leader gathers its lanes, exchanges with the other
 * three leaders, then releases its CMG through a CMG-local line. */
static q38d_line ep_arr[NT], ep_lead[NCMG], ep_rel[NCMG], ec_arr[NT], ec_rel[NCMG];
static int ep_my[NT], ec_my[NT];
static void gbarrier_epoch(int tid) {
    int e = ++ep_my[tid], c = tid / PER, l = tid % PER;
    if (l) {
        atomic_store_explicit(&ep_arr[tid].v, e, memory_order_release);
        while (atomic_load_explicit(&ep_rel[c].v, memory_order_acquire) < e) __asm__ volatile("yield" ::: "memory");
        return;
    }
    for (int j = 1; j < PER; j++)
        while (atomic_load_explicit(&ep_arr[c * PER + j].v, memory_order_acquire) < e) __asm__ volatile("yield" ::: "memory");
    atomic_store_explicit(&ep_lead[c].v, e, memory_order_release);
    for (int k = 0; k < NCMG; k++)
        if (k != c) while (atomic_load_explicit(&ep_lead[k].v, memory_order_acquire) < e) __asm__ volatile("yield" ::: "memory");
    atomic_store_explicit(&ep_rel[c].v, e, memory_order_release);
}
static void cbarrier_epoch(int tid) {
    int e = ++ec_my[tid], c = tid / PER, l = tid % PER;
    if (l) {
        atomic_store_explicit(&ec_arr[tid].v, e, memory_order_release);
        while (atomic_load_explicit(&ec_rel[c].v, memory_order_acquire) < e) __asm__ volatile("yield" ::: "memory");
        return;
    }
    for (int j = 1; j < PER; j++)
        while (atomic_load_explicit(&ec_arr[c * PER + j].v, memory_order_acquire) < e) __asm__ volatile("yield" ::: "memory");
    atomic_store_explicit(&ec_rel[c].v, e, memory_order_release);
}
static void gbarrier_impl(int tid, int *sense) {
    if (epoch_bar) { gbarrier_epoch(tid); return; }
    int s = !*sense;
    *sense = s;
    int c = tid / PER;
    if (atomic_fetch_add_explicit(&bar_cmg[c].v, 1, memory_order_acq_rel) == PER - 1) {
        atomic_store_explicit(&bar_cmg[c].v, 0, memory_order_relaxed);
        if (atomic_fetch_add_explicit(&bar_glob.v, 1, memory_order_acq_rel) == NCMG - 1) {
            atomic_store_explicit(&bar_glob.v, 0, memory_order_relaxed);
            atomic_store_explicit(&bar_sense.v, s, memory_order_release);
            return;
        }
    }
    while (atomic_load_explicit(&bar_sense.v, memory_order_acquire) != s) __asm__ volatile("yield" ::: "memory");
}
static void cbarrier_epoch(int tid);
static void cbarrier(int tid, int *sense) {
    if (epoch_bar) { cbarrier_epoch(tid); return; }
    int c = tid / PER, s = !*sense;
    *sense = s;
    if (atomic_fetch_add_explicit(&cbar_cnt[c].v, 1, memory_order_acq_rel) == PER - 1) {
        atomic_store_explicit(&cbar_cnt[c].v, 0, memory_order_relaxed);
        atomic_store_explicit(&cbar_sense[c].v, s, memory_order_release);
        return;
    }
    while (atomic_load_explicit(&cbar_sense[c].v, memory_order_acquire) != s) __asm__ volatile("yield" ::: "memory");
}

static void pin_cpu(int cpu) {
    cpu_set_t m; CPU_ZERO(&m); CPU_SET(cpu, &m);
    if (sched_setaffinity(0, sizeof(m), &m)) perror("sched_setaffinity");
}

/* CMG-local anonymous allocation; the caller must first-touch from that CMG. */
static void *cmg_alloc(size_t bytes, int cmg) {
    size_t n = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    unsigned long node = 1ul << (4 + cmg);
    if (syscall(SYS_mbind, p, n, 2 /*MPOL_BIND*/, &node, 64ul, 0ul)) perror("mbind");
    return p;
}

/* ------------------------------------------------------------------ */
/* model loading                                                        */

static gguf_context *G;
static int out_kch = 1; /* K chunks for the 6144-column residual projections (3 measured slower) */
static int q6k_expand = 1; /* Q6_K -> exact int8 codes (no decode, +28% bytes) */
static q38_lowbit_model *LB;
static q38d_engine E;

static int find_tensor(const char *name) {
    for (uint64_t i = 0; i < G->n_tensors; i++)
        if (!strcmp(G->tensors[i].name.str, name)) return (int)i;
    return -1;
}
static int need_tensor(const char *fmt, int layer) {
    char name[128];
    snprintf(name, sizeof name, fmt, layer);
    int i = find_tensor(name);
    if (i < 0) { fprintf(stderr, "q38d: missing tensor %s\n", name); exit(1); }
    return i;
}
static float bf16f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static void load_vec(float *dst, int idx, size_t n) {
    const gguf_tensor_info *t = &G->tensors[idx];
    const void *d = gguf_tensor_data(G, idx);
    size_t cnt = 1;
    for (uint32_t k = 0; k < t->n_dims; k++) cnt *= t->dims[k];
    if (cnt != n) { fprintf(stderr, "q38d: %s has %zu elements, want %zu\n", t->name.str, cnt, n); exit(1); }
    if (t->type == GGML_TYPE_F32) memcpy(dst, d, n * 4);
    else if (t->type == GGML_TYPE_BF16) for (size_t i = 0; i < n; i++) dst[i] = bf16f(((const uint16_t *)d)[i]);
    else if (t->type == GGML_TYPE_F16) for (size_t i = 0; i < n; i++) dst[i] = q38d_half_to_float(((const uint16_t *)d)[i]);
    else { fprintf(stderr, "q38d: %s unsupported vector type %u\n", t->name.str, t->type); exit(1); }
}
static const float *vec_ptr(int idx, size_t n) {
    float *p = malloc(n * sizeof(float));
    load_vec(p, idx, n);
    return p;
}

/* Matrices are described now and repacked by the worker pool. */
typedef struct { q38d_mat *m; int idx; int kind; /* 0 lowbit in place, 1 raw Q6_K/Q4_K */ } pending;
static pending *todo;
static int ntodo;

static void describe(q38d_mat *m, int idx) {
    const gguf_tensor_info *t = &G->tensors[idx];
    const q38_lowbit_matrix *lb = q38_lowbit_model_tensor(G, idx);
    m->rows = (int)t->dims[1]; m->cols = (int)t->dims[0];
    todo = realloc(todo, (size_t)(ntodo + 1) * sizeof(*todo));
    if (lb) {
        m->fmt = lb->format == Q38_LB_NVFP4 ? Q38D_F4 : Q38D_F6;
        for (int c = 0; c < 5; c++) m->first[c] = lb->first[c];
        for (int c = 0; c < 4; c++) m->part[c] = lb->part[c];
        todo[ntodo++] = (pending){m, idx, 0};
    } else if (t->type == GGML_TYPE_Q6_K || t->type == GGML_TYPE_Q4_K) {
        m->fmt = t->type == GGML_TYPE_Q6_K ? (q6k_expand ? Q38D_Q8K : Q38D_Q6K) : Q38D_Q4K;
        int groups = (m->rows + 7) / 8;
        for (int c = 0; c <= 4; c++) {
            int r = (int)((int64_t)groups * c / 4) * 8;
            m->first[c] = r < m->rows ? r : m->rows;
        }
        for (int c = 0; c < 4; c++) {
            int gcount = (m->first[c + 1] - m->first[c] + 7) / 8;
            m->part[c] = gcount ? cmg_alloc((size_t)gcount * q38d_group_bytes(m->fmt, m->cols), c) : NULL;
        }
        todo[ntodo++] = (pending){m, idx, 1};
    } else {
        fprintf(stderr, "q38d: %s: unsupported matrix type %u\n", t->name.str, t->type);
        exit(1);
    }
    if (m->cols % 256) { fprintf(stderr, "q38d: %s cols %d not a multiple of 256\n", t->name.str, m->cols); exit(1); }
}

/* Repack CMG c's groups of all pending matrices, lane l of 12. */
static void repack_worker(int c, int l) {
    uint8_t *tmp = NULL;
    size_t tmp_cap = 0;
    for (int i = 0; i < ntodo; i++) {
        q38d_mat *m = todo[i].m;
        int rows = m->first[c + 1] - m->first[c];
        int G8 = (rows + 7) / 8;
        size_t gb = q38d_group_bytes(m->fmt, m->cols);
        if (gb > tmp_cap) { tmp = realloc(tmp, gb); tmp_cap = gb; }
        int g0, g1;
        (void)G8;
        lane_groups(m, c, l, &g0, &g1);
        for (int g = g0; g < g1; g++) {
            uint8_t *dst = m->part[c] + (size_t)g * gb;
            if (todo[i].kind == 0) {
                memcpy(tmp, dst, gb);
                if (m->fmt == Q38D_F4) q38d_repack_f4(dst, tmp, m->cols);
                else q38d_repack_f6(dst, tmp, m->cols);
            } else {
                const gguf_tensor_info *t = &G->tensors[todo[i].idx];
                size_t rb = (size_t)m->cols / 256 * (t->type == GGML_TYPE_Q6_K ? 210 : 144);
                const uint8_t *src = (const uint8_t *)gguf_tensor_data(G, todo[i].idx) +
                                     (size_t)(m->first[c] + g * 8) * rb;
                int nr = rows - g * 8 < 8 ? rows - g * 8 : 8;
                if (m->fmt == Q38D_Q6K) q38d_repack_q6k(dst, src, rb, nr, m->cols);
                else if (m->fmt == Q38D_Q8K) q38d_repack_q8k(dst, src, rb, nr, m->cols);
                else q38d_repack_q4k(dst, src, rb, nr, m->cols);
            }
        }
    }
    free(tmp);
}

/* Item partition for K-chunked matrices: (group, chunk) items in group-major
 * order are split evenly over the CMG's 12 workers; each worker stores its
 * items chunk-major. A group split between two workers is finished by the
 * second one. */
static inline void lane_items(const q38d_mat *m, int c, int l, int *i0, int *i1) {
    int G8 = (m->first[c + 1] - m->first[c] + 7) / 8, N = G8 * m->kch;
    *i0 = N * l / PER; *i1 = N * (l + 1) / PER;
}
static uint8_t *kchunk_buf[NT][NLAYER * 3 + 4];
static void kchunk_copy_or_write(int c, int l, int write) {
    int tid = c * PER + l, slot = 0;
    for (int i = 0; i < ntodo; i++) {
        q38d_mat *m = todo[i].m;
        if (m->kch <= 1) continue;
        int i0, i1;
        lane_items(m, c, l, &i0, &i1);
        int gf = i0 / m->kch, gl = (i1 - 1) / m->kch, ng = gl - gf + 1;
        size_t gb = q38d_group_bytes(m->fmt, m->cols);
        int kc = m->cols / m->kch;
        size_t sb = q38d_group_bytes(m->fmt, kc), np = (size_t)m->cols / 32, npc = (size_t)kc / 32;
        size_t cpp = m->fmt == Q38D_F6 ? 192 : 128, spp = m->fmt == Q38D_F6 ? 8 : 16;
        if (m->fmt != Q38D_F4 && m->fmt != Q38D_F6) { fprintf(stderr, "q38d: K-chunking unsupported for fmt %d\n", m->fmt); exit(1); }
        if (!write) {
            kchunk_buf[tid][slot] = malloc((size_t)ng * gb);
            memcpy(kchunk_buf[tid][slot], m->part[c] + (size_t)gf * gb, (size_t)ng * gb);
        } else {
            uint8_t *buf = kchunk_buf[tid][slot], *dst = m->part[c] + (size_t)i0 * sb;
            for (int k = 0; k < m->kch; k++)
                for (int g = gf; g <= gl; g++) {
                    int it = g * m->kch + k;
                    if (it < i0 || it >= i1) continue;
                    const uint8_t *src = buf + (size_t)(g - gf) * gb;
                    memcpy(dst, src + k * npc * cpp, npc * cpp);
                    memcpy(dst + npc * cpp, src + np * cpp + k * npc * spp, npc * spp);
                    dst += sb;
                }
            free(buf);
        }
        slot++;
    }
}

static void load_engine(void) {
    E.n_vocab = (int)G->tensors[need_tensor("output.weight", 0)].dims[1];
    E.eps = 1e-6f;
    E.rope_base = 10000000.f;
    for (int j = 0; j < 32; j++) E.inv_freq[j] = 1.0f / powf(E.rope_base, (float)(2 * j) / 64.f);
    E.L = calloc(NLAYER, sizeof(q38d_layer));
    for (int l = 0; l < NLAYER; l++) {
        q38d_layer *L = &E.L[l];
        L->ssm = (l % 4) != 3;
        L->ai = l / 4;
        L->attn_norm = vec_ptr(need_tensor("blk.%d.attn_norm.weight", l), EMBD);
        L->post_norm = vec_ptr(need_tensor("blk.%d.post_attention_norm.weight", l), EMBD);
        if (L->ssm) {
            describe(&L->qkv, need_tensor("blk.%d.attn_qkv.weight", l));
            describe(&L->z, need_tensor("blk.%d.attn_gate.weight", l));
            describe(&L->alpha, need_tensor("blk.%d.ssm_alpha.weight", l));
            describe(&L->beta, need_tensor("blk.%d.ssm_beta.weight", l));
            describe(&L->out, need_tensor("blk.%d.ssm_out.weight", l));
            L->out.unit16 = 1;
            L->out.kch = out_kch;
            if (L->out.kch > 1 && L->out.cols % (L->out.kch * 256) == 0) L->out.unit16 = 0; else L->out.kch = 0;
            {
                float *tmpw = malloc(QKVD * 4 * sizeof(float));
                load_vec(tmpw, need_tensor("blk.%d.ssm_conv1d.weight", l), QKVD * 4);
                for (int ch = 0; ch < QKVD; ch++)
                    for (int kk = 0; kk < 4; kk++) L->conv_w[kk * QKVD + ch] = tmpw[ch * 4 + kk];
                free(tmpw);
            }
            load_vec(L->ssm_a, need_tensor("blk.%d.ssm_a", l), NVH);
            load_vec(L->dt_bias, need_tensor("blk.%d.ssm_dt.bias", l), NVH);
            load_vec(L->ssm_norm, need_tensor("blk.%d.ssm_norm.weight", l), DS);
        } else {
            describe(&L->q, need_tensor("blk.%d.attn_q.weight", l));
            describe(&L->k, need_tensor("blk.%d.attn_k.weight", l));
            describe(&L->v, need_tensor("blk.%d.attn_v.weight", l));
            describe(&L->o, need_tensor("blk.%d.attn_output.weight", l));
            L->o.unit16 = 1;
            L->o.kch = out_kch;
            if (L->o.kch > 1 && L->o.cols % (L->o.kch * 256) == 0) L->o.unit16 = 0; else L->o.kch = 0;
            load_vec(L->q_norm, need_tensor("blk.%d.attn_q_norm.weight", l), HD);
            load_vec(L->k_norm, need_tensor("blk.%d.attn_k_norm.weight", l), HD);
        }
        describe(&L->gate, need_tensor("blk.%d.ffn_gate.weight", l));
        describe(&L->up, need_tensor("blk.%d.ffn_up.weight", l));
        describe(&L->down, need_tensor("blk.%d.ffn_down.weight", l));
        L->down.unit16 = 1;
        L->down.kch = getenv("Q38D_DOWN_KCH") ? atoi(getenv("Q38D_DOWN_KCH")) : 4;
        if (L->down.kch < 1 || L->down.cols % (L->down.kch * 256)) L->down.kch = 1;
        if (L->down.kch > 1) L->down.unit16 = 0;
    }
    describe(&E.head, need_tensor("output.weight", 0));
    E.out_norm = vec_ptr(need_tensor("output_norm.weight", 0), EMBD);
    int ei = need_tensor("token_embd.weight", 0);
    E.embed_lb = q38_lowbit_model_tensor(G, ei);
    if (!E.embed_lb) {
        if (G->tensors[ei].type != GGML_TYPE_Q6_K) { fprintf(stderr, "q38d: embedding type %u\n", G->tensors[ei].type); exit(1); }
        E.embed_q6k = gguf_tensor_data(G, ei);
        E.embed_row_bytes = (size_t)EMBD / 256 * 210;
    }
}

/* ------------------------------------------------------------------ */
/* per-worker primitives                                                */

static inline void mat_range(const q38d_mat *m, int tid, int *c, int *g0, int *g1) {
    *c = tid / PER;
    lane_groups(m, *c, tid % PER, g0, g1);
}
static double kprof_kernel, kprof_norm, kprof_wait;
static void mv_chunked(const q38d_mat *m, const q38d_act *a, float *out, int mode, int c, int g0, int g1) {
    int ng = g1 - g0, kc = m->cols / m->kch;
    size_t gb = q38d_group_bytes(m->fmt, m->cols), sb = q38d_group_bytes(m->fmt, kc);
    const uint8_t *range = m->part[c] + (size_t)g0 * gb;
    float part[8], acc[64][8];
    if (ng > 64) { fprintf(stderr, "q38d: chunked range too large\n"); exit(1); }
    for (int k = 0; k < m->kch; k++) {
        size_t p0 = (size_t)k * kc / 32;
        q38d_act v = *a;
        v.cols = kc;
        v.q = a->q + p0 * (a->arith == Q38D_A16 ? 64 : 32);
        v.sc = a->sc + 2 * p0; v.sum = a->sum + 2 * p0;
        v.x = a->x ? a->x + k * kc : NULL;
        for (int g = 0; g < ng; g++) {
            float *o = k ? part : acc[g];
            q38d_gemv_any(o, range + ((size_t)k * ng + g) * sb, m->fmt, &v, 0, 1, 8, 0);
            if (k) for (int r = 0; r < 8; r++) acc[g][r] += part[r];
        }
    }
    int rows = m->first[c + 1] - m->first[c];
    float *dst = out + m->first[c] + (size_t)g0 * 8;
    for (int g = 0; g < ng; g++)
        for (int r = 0; r < 8 && g0 * 8 + g * 8 + r < rows; r++)
            dst[g * 8 + r] = mode ? dst[g * 8 + r] + acc[g][r] : acc[g][r];
}
typedef struct { float part[2][8]; float ssq; _Atomic int cnt; char pad[180]; } q38d_split;
static q38d_split split_slot[NCMG][PER + 1];
static float mv_ssq[NT];
static void ssq_publish(int tid, int r0, int r1);
/* K-chunked residual matvec (mode add into out == E.x) over this worker's
 * items; publishes the sum of squares of the rows it finalizes. */
static void mv_items(const q38d_mat *m, const q38d_act *a, float *out, int tid) {
    int c = tid / PER, l = tid % PER, i0, i1;
    lane_items(m, c, l, &i0, &i1);
    float ss = 0;
    /* boundary slot l is unused when this lane starts on a group edge */
    if (l > 0 && i0 % m->kch == 0) split_slot[c][l].ssq = 0;
    if (i1 > i0) {
        int gf = i0 / m->kch, gl = (i1 - 1) / m->kch, kc = m->cols / m->kch;
        size_t sb = q38d_group_bytes(m->fmt, kc);
        const uint8_t *w = m->part[c] + (size_t)i0 * sb;
        float acc[64][8], part[8];
        memset(acc, 0, sizeof(float) * 8 * (gl - gf + 1));
        for (int k = 0; k < m->kch; k++) {
            size_t p0 = (size_t)k * kc / 32;
            q38d_act v = *a;
            v.cols = kc;
            v.q = a->q + p0 * (a->arith == Q38D_A16 ? 64 : 32);
            v.sc = a->sc + 2 * p0; v.sum = a->sum + 2 * p0;
            v.x = a->x ? a->x + k * kc : NULL;
            for (int g = gf; g <= gl; g++) {
                int it = g * m->kch + k;
                if (it < i0 || it >= i1) continue;
                q38d_gemv_any(part, w, m->fmt, &v, 0, 1, 8, 0);
                w += sb;
                for (int r = 0; r < 8; r++) acc[g - gf][r] += part[r];
            }
        }
        for (int g = gf; g <= gl; g++) {
            float *x = out + m->first[c] + 8 * g;
            int full = g * m->kch >= i0 && g * m->kch + m->kch - 1 < i1;
            if (full) {
                for (int r = 0; r < 8; r++) { x[r] += acc[g - gf][r]; ss += x[r] * x[r]; }
                continue;
            }
            int b = g * m->kch < i0 ? l : l + 1, side = g * m->kch < i0 ? 1 : 0;
            q38d_split *sp = &split_slot[c][b];
            memcpy(sp->part[side], acc[g - gf], sizeof(float) * 8);
            int old = atomic_fetch_add_explicit(&sp->cnt, 1, memory_order_acq_rel);
            if (old & 1) {
                /* squares go to the boundary slot, not to whichever worker
                 * finished second, so the norm's sum order is fixed */
                float bs = 0;
                for (int r = 0; r < 8; r++) { x[r] += sp->part[0][r] + sp->part[1][r]; bs += x[r] * x[r]; }
                sp->ssq = bs;
            }
        }
    }
    mv_ssq[tid] = ss;
}
static void mv(const q38d_mat *m, const q38d_act *a, float *out, int mode, int tid) {
    int c, g0, g1;
    mat_range(m, tid, &c, &g0, &g1);
    uint64_t t0 = tid ? 0 : ticks();
    if (m->kch > 1 && out == E.x && mode) { mv_items(m, a, out, tid); if (!tid) kprof_kernel += (double)(ticks() - t0); return; }
    if (g0 < g1) {
        if (m->kch > 1) mv_chunked(m, a, out, mode, c, g0, g1);
        else q38d_gemv_any(out + m->first[c], m->part[c], m->fmt, a, g0, g1, m->first[c + 1] - m->first[c], mode);
    }
    if (!tid) kprof_kernel += (double)(ticks() - t0);
}

/* Balanced multi-matrix phase plans: per CMG, the concatenated group list
 * of all matrices of a phase is split into 12 contiguous ranges of equal
 * estimated cost; a range may span matrix boundaries. */
typedef struct { signed char mi, mi2; int g0, g1; } q38d_seg; /* mi2 >= 0: dual with matrix mi2 */
typedef struct { int nseg; q38d_seg s[6]; } q38d_tplan;
typedef struct { int nm; const q38d_mat *m[4]; float *out[4]; q38d_tplan t[NT]; } q38d_plan;
static q38d_plan *plan_ssm, *plan_att;
static int use_plan = 1;

/* Estimated cycles of one group (measured A16 cycles per pair, L1). */
static double cost_q4k = 50, cost_q6k = 70, cost_q8k = 40, cost_f6 = 29;
static double group_cost(const q38d_mat *m) {
    double cpp = m->fmt == Q38D_F4 ? 18.3 : m->fmt == Q38D_F6 ? cost_f6 : m->fmt == Q38D_Q6K ? cost_q6k :
                 m->fmt == Q38D_Q8K ? cost_q8k : cost_q4k;
    return cpp * (m->cols / 32);
}
#define PLAN_STARTUP 1500.0 /* cycles of cold start per segment */
static void plan_add2(q38d_tplan *tp, int i, int i2, int g) {
    if (tp->nseg && tp->s[tp->nseg - 1].mi == i && tp->s[tp->nseg - 1].mi2 == i2 && tp->s[tp->nseg - 1].g1 == g) {
        tp->s[tp->nseg - 1].g1 = g + 1; return;
    }
    if (tp->nseg == 6) { fprintf(stderr, "q38d: plan overflow\n"); exit(1); }
    tp->s[tp->nseg++] = (q38d_seg){(signed char)i, (signed char)i2, g, g + 1};
}
static void plan_add(q38d_tplan *tp, int i, int g) {
    if (tp->nseg && tp->s[tp->nseg - 1].mi == i && tp->s[tp->nseg - 1].mi2 < 0 && tp->s[tp->nseg - 1].g1 == g) { tp->s[tp->nseg - 1].g1 = g + 1; return; }
    if (tp->nseg == 6) { fprintf(stderr, "q38d: plan overflow\n"); exit(1); }
    tp->s[tp->nseg++] = (q38d_seg){(signed char)i, -1, g, g + 1};
}
static void build_plan(q38d_plan *P) {
    for (int c = 0; c < NCMG; c++) {
        double cost[PER] = {0}, total = 0;
        for (int l = 0; l < PER; l++) P->t[c * PER + l].nseg = 0;
        int next = PER - 1;
        /* small matrices first, one group per lane from the top lane down */
        for (int i = 0; i < P->nm; i++) {
            int G8 = (P->m[i]->first[c + 1] - P->m[i]->first[c] + 7) / 8;
            if (G8 > 4) continue;
            for (int g = 0; g < G8; g++) {
                plan_add(&P->t[c * PER + next], i, g);
                cost[next] += group_cost(P->m[i]) + PLAN_STARTUP;
                next = (next + PER - 1) % PER;
            }
        }
        for (int l = 0; l < PER; l++) total += cost[l];
        for (int i = 0; i < P->nm; i++) {
            int G8 = (P->m[i]->first[c + 1] - P->m[i]->first[c] + 7) / 8;
            if (G8 > 4) total += G8 * group_cost(P->m[i]) + PLAN_STARTUP;
        }
        double target = total / PER;
        int lane = 0;
        for (int i = 0; i < P->nm; i++) {
            int G8 = (P->m[i]->first[c + 1] - P->m[i]->first[c] + 7) / 8;
            if (G8 <= 4) continue;
            double gc = group_cost(P->m[i]);
            int started = 0;
            for (int g = 0; g < G8; g++) {
                while (lane < PER - 1 && cost[lane] + 0.5 * gc > target) { lane++; started = 0; }
                if (!started) { cost[lane] += PLAN_STARTUP; started = 1; }
                plan_add(&P->t[c * PER + lane], i, g);
                cost[lane] += gc;
            }
        }
    }
}
static void scale_rows(float *p, int n, float f);
static double dual_cost = 1.6;
/* SSM in-proj with (qkv g, z g) dual items for g < rows(z)/8. */
static void build_plan_ssm_dual(q38d_plan *P) {
    for (int c = 0; c < NCMG; c++) {
        double cost[PER] = {0}, total = 0;
        for (int l = 0; l < PER; l++) P->t[c * PER + l].nseg = 0;
        int Gq = (P->m[0]->first[c + 1] - P->m[0]->first[c] + 7) / 8;
        int Gz = (P->m[1]->first[c + 1] - P->m[1]->first[c] + 7) / 8;
        double gq = group_cost(P->m[0]);
        int next = PER - 1;
        for (int i = 2; i < 4; i++) {
            int G8 = (P->m[i]->first[c + 1] - P->m[i]->first[c] + 7) / 8;
            for (int g = 0; g < G8; g++) {
                plan_add(&P->t[c * PER + next], i, g);
                cost[next] += group_cost(P->m[i]) + PLAN_STARTUP;
                next = (next + PER - 1) % PER;
            }
        }
        for (int l = 0; l < PER; l++) total += cost[l];
        int nd = Gz < Gq ? Gz : Gq;
        total += nd * dual_cost * gq + (Gq - nd) * gq + 2 * PLAN_STARTUP;
        double target = total / PER;
        int lane = 0, started = 0;
        for (int g = 0; g < nd; g++) {
            while (lane < PER - 1 && cost[lane] + 0.5 * dual_cost * gq > target) { lane++; started = 0; }
            if (!started) { cost[lane] += PLAN_STARTUP; started = 1; }
            plan_add2(&P->t[c * PER + lane], 0, 1, g);
            cost[lane] += dual_cost * gq;
        }
        started = 0;
        for (int g = nd; g < Gq; g++) {
            while (lane < PER - 1 && cost[lane] + 0.5 * gq > target) { lane++; started = 0; }
            if (!started) { cost[lane] += PLAN_STARTUP; started = 1; }
            plan_add(&P->t[c * PER + lane], 0, g);
            cost[lane] += gq;
        }
        for (int g = nd; g < Gz; g++) { /* z longer than qkv: not the case for Qwen3.8 */
            plan_add(&P->t[c * PER + PER - 1], 1, g);
        }
    }
}

static void run_plan(const q38d_plan *P, const q38d_act *a, int tid, float omul) {
    int c = tid / PER;
    const q38d_tplan *tp = &P->t[tid];
    for (int k = 0; k < tp->nseg; k++) {
        const q38d_mat *m = P->m[tp->s[k].mi];
        int rows = m->first[c + 1] - m->first[c];
        float *o = P->out[tp->s[k].mi] + m->first[c];
        int g0 = tp->s[k].g0, g1 = tp->s[k].g1;
        if (tp->s[k].mi2 >= 0) {
            const q38d_mat *m2 = P->m[tp->s[k].mi2];
            float *o2 = P->out[tp->s[k].mi2] + m2->first[c];
            if (a->arith != Q38D_F32) q38d_gemv_dual_fmt(o, o2, m->part[c], m2->part[c], a, g0, g1, m->fmt);
            else {
                q38d_gemv_any(o, m->part[c], m->fmt, a, g0, g1, rows, 0);
                q38d_gemv_any(o2, m2->part[c], m2->fmt, a, g0, g1, m2->first[c + 1] - m2->first[c], 0);
            }
            if (omul != 1.0f) scale_rows(o2 + 8 * g0, 8 * (g1 - g0), omul);
        } else q38d_gemv_any(o, m->part[c], m->fmt, a, g0, g1, rows, 0);
        if (omul != 1.0f) {
            int r0 = 8 * g0, r1 = 8 * g1 < rows ? 8 * g1 : rows;
            scale_rows(o + r0, r1 - r0, omul);
        }
    }
}

static float sumsq(const float *x, int n) {
    const svbool_t pf = svptrue_b32();
    svfloat32_t a0 = svdup_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        svfloat32_t v0 = svld1_f32(pf, x + i), v1 = svld1_f32(pf, x + i + 16);
        svfloat32_t v2 = svld1_f32(pf, x + i + 32), v3 = svld1_f32(pf, x + i + 48);
        a0 = svmla_f32_x(pf, a0, v0, v0); a1 = svmla_f32_x(pf, a1, v1, v1);
        a2 = svmla_f32_x(pf, a2, v2, v2); a3 = svmla_f32_x(pf, a3, v3, v3);
    }
    for (; i < n; i += 16) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, x + i);
        a0 = svmla_f32_m(pg, a0, v, v);
    }
    return svaddv_f32(pf, svadd_f32_x(pf, svadd_f32_x(pf, a0, a1), svadd_f32_x(pf, a2, a3)));
}
static float dot(const float *a, const float *b, int n) {
    const svbool_t pf = svptrue_b32();
    svfloat32_t a0 = svdup_n_f32(0), a1 = a0;
    for (int i = 0; i < n; i += 32) {
        a0 = svmla_f32_x(pf, a0, svld1_f32(pf, a + i), svld1_f32(pf, b + i));
        a1 = svmla_f32_x(pf, a1, svld1_f32(pf, a + i + 16), svld1_f32(pf, b + i + 16));
    }
    return svaddv_f32(pf, svadd_f32_x(pf, a0, a1));
}

/* RMSNorm of the residual with weight w into this worker's activation. */
static const q38d_act *norm_act_impl(int tid, const float *w);
static const q38d_act *norm_act(int tid, const float *w) {
    uint64_t t0 = tid ? 0 : ticks();
    const q38d_act *r = norm_act_impl(tid, w);
    if (!tid) kprof_norm += (double)(ticks() - t0);
    return r;
}
static int norm_cmg = 1;
static double norm_cbar_t;
static int *norm_csense[NT];
static float ssq_part[NCMG][64] __attribute__((aligned(256)));
static int ssq_split_valid; /* last residual update went through mv_items */
static int ssq_valid; /* partial sums of squares of E.x are current */
/* After this worker updated residual rows [r0, r1): publish their sum of squares. */
static void ssq_publish(int tid, int r0, int r1) {
    ssq_part[tid / PER][tid % PER] = r1 > r0 ? sumsq(E.x + r0, r1 - r0) : 0.f;
}
/* Producer-side normalization: the worker that updated residual rows of
 * matrix m quantizes x*w for those rows into every CMG's activation copy
 * (per-16 quantization is invariant to the RMS factor, which consumers
 * apply to their outputs) and publishes the rows' sum of squares. */
static int prod_norm = 0;
static int use_dual = 1;
static int ffn_group_split = 1;
static int prod_copies = 1; /* 1: one shared act (act_c[0]) read by all CMGs; 4: per-CMG copies */
static void x_produce(const q38d_mat *m, int tid, const float *w) {
    int c, g0, g1;
    mat_range(m, tid, &c, &g0, &g1);
    int r0 = m->first[c] + 8 * g0, r1 = m->first[c] + 8 * g1;
    if (r1 > m->first[c + 1]) r1 = m->first[c + 1];
    if (r1 < r0) r1 = r0;
    ssq_publish(tid, r0, r1);
    const svbool_t pf = svptrue_b32();
    for (int r = r0; r < r1; r += 16) {
        float v[16] __attribute__((aligned(64)));
        svst1_f32(pf, v, svmul_f32_x(pf, svld1_f32(pf, E.x + r), svld1_f32(pf, w + r)));
        for (int cc = 0; cc < prod_copies; cc++) {
            if (E.arith == Q38D_F32) memcpy(E.xn_c[cc] + r, v, sizeof v);
            else q38d_prepare_unit(&E.act_c[cc], r / 32, (r / 16) & 1, v);
        }
    }
}
static void ssq_publish_rows(const q38d_mat *m, int tid) {
    if (m->kch > 1) { ssq_part[tid / PER][tid % PER] = mv_ssq[tid]; return; }
    int c, g0, g1;
    mat_range(m, tid, &c, &g0, &g1);
    int r0 = m->first[c] + 8 * g0, r1 = m->first[c] + 8 * g1;
    if (r1 > m->first[c + 1]) r1 = m->first[c + 1];
    ssq_publish(tid, r0, r1 > r0 ? r1 : r0);
}
/* Consumer side: RMS factor from the published partial sums. */
static float x_inv(void) {
    float ss = 0;
    for (int c = 0; c < NCMG; c++) for (int l = 0; l < PER; l++) ss += ssq_part[c][l];
    return 1.0f / sqrtf(ss / EMBD + E.eps);
}
static void scale_rows(float *p, int n, float f) {
    const svbool_t pf = svptrue_b32();
    for (int i = 0; i < n; i += 16) {
        svbool_t pg = svwhilelt_b32(i, n);
        svst1_f32(pg, p + i, svmul_n_f32_x(pg, svld1_f32(pg, p + i), f));
    }
}
static const q38d_act *norm_act_impl(int tid, const float *w) {
    float ss;
    if (ssq_valid) {
        ss = 0;
        for (int c = 0; c < NCMG; c++) for (int l = 0; l < PER; l++) ss += ssq_part[c][l];
        if (ssq_split_valid)
            for (int c = 0; c < NCMG; c++) for (int b = 1; b < PER; b++) ss += split_slot[c][b].ssq;
    } else ss = sumsq(E.x, EMBD);
    float inv = 1.0f / sqrtf(ss / EMBD + E.eps);
    if (norm_cmg) {
        /* each CMG lane normalizes/quantizes 1/12 of the pairs, then CMG barrier */
        int c = tid / PER, l = tid % PER, np = EMBD / 32;
        int p0 = np * l / PER, p1 = np * (l + 1) / PER;
        q38d_act *a = &E.act_c[c];
        if (E.arith == Q38D_F32) {
            float *xn = E.xn_c[c];
            const svbool_t pf = svptrue_b32();
            for (int i = 32 * p0; i < 32 * p1; i += 16)
                svst1_f32(pf, xn + i, svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, E.x + i), inv), svld1_f32(pf, w + i)));
            a->x = xn;
        } else q38d_prepare_pairs(a, E.x, inv, w, p0, p1);
        uint64_t Tq = tid ? 0 : ticks();
        cbarrier(tid, norm_csense[tid]);
        if (!tid) norm_cbar_t += (double)(ticks() - Tq);
        return a;
    }
    q38d_act *a = &E.act_x[tid];
    if (E.arith == Q38D_F32) {
        float *xn = E.xn[tid];
        const svbool_t pf = svptrue_b32();
        for (int i = 0; i < EMBD; i += 16)
            svst1_f32(pf, xn + i, svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, E.x + i), inv), svld1_f32(pf, w + i)));
        a->x = xn;
    } else q38d_prepare_sve(a, E.x, inv, w);
    return a;
}

/* ------------------------------------------------------------------ */
/* SSM head                                                             */

static int conv_local = 1;
static void ssm_prefetch_state(int layer, int h);
/* part: 0 q, 1 k, 2 v of head h; channels start at ch0 in the qkv vector */
static void conv_silu_local(int layer, int h, int part, int ch0, int pos, float *out) {
    float *hs = E.conv_hist[layer][h];
    const float *w = E.conv_wl[layer][h] + part * 128;
    const float *in = E.qkv + ch0;
    const float *h1 = hs + (size_t)((pos + 3) & 3) * 384 + part * 128;
    const float *h2 = hs + (size_t)((pos + 2) & 3) * 384 + part * 128;
    const float *h3 = hs + (size_t)((pos + 1) & 3) * 384 + part * 128;
    const svbool_t pf = svptrue_b32();
    for (int j = 0; j < 128; j += 16) {
        svfloat32_t v = svmul_f32_x(pf, svld1_f32(pf, w + j), svld1_f32(pf, h3 + j));
        v = svmla_f32_x(pf, v, svld1_f32(pf, w + 384 + j), svld1_f32(pf, h2 + j));
        v = svmla_f32_x(pf, v, svld1_f32(pf, w + 768 + j), svld1_f32(pf, h1 + j));
        v = svmla_f32_x(pf, v, svld1_f32(pf, w + 1152 + j), svld1_f32(pf, in + j));
        svst1_f32(pf, out + j, svmul_f32_x(pf, v, q38d_sigmoid_sve(pf, v)));
    }
    memcpy(hs + (size_t)(pos & 3) * 384 + part * 128, in, 128 * sizeof(float));
}
static void conv_silu(const q38d_layer *L, int layer, int ch0, int pos, int write, float *out) {
    float *cs = E.conv_state[layer];
    const float *in = E.qkv + ch0;
    const float *h1 = cs + (size_t)((pos + 3) & 3) * QKVD + ch0;
    const float *h2 = cs + (size_t)((pos + 2) & 3) * QKVD + ch0;
    const float *h3 = cs + (size_t)((pos + 1) & 3) * QKVD + ch0;
    const float *w = L->conv_w; /* transposed at load: [4][QKVD] */
    const svbool_t pf = svptrue_b32();
    for (int j = 0; j < 128; j += 16) {
        svfloat32_t v = svmul_f32_x(pf, svld1_f32(pf, w + ch0 + j), svld1_f32(pf, h3 + j));
        v = svmla_f32_x(pf, v, svld1_f32(pf, w + QKVD + ch0 + j), svld1_f32(pf, h2 + j));
        v = svmla_f32_x(pf, v, svld1_f32(pf, w + 2 * QKVD + ch0 + j), svld1_f32(pf, h1 + j));
        v = svmla_f32_x(pf, v, svld1_f32(pf, w + 3 * QKVD + ch0 + j), svld1_f32(pf, in + j));
        svst1_f32(pf, out + j, svmul_f32_x(pf, v, q38d_sigmoid_sve(pf, v)));
    }
    if (write) memcpy(cs + (size_t)(pos & 3) * QKVD + ch0, in, 128 * sizeof(float));
}

static void l2norm128(float *v) {
    float inv = 1.0f / sqrtf(sumsq(v, 128) + E.eps);
    for (int i = 0; i < 128; i++) v[i] *= inv;
}

/* State stored transposed, St[c][r] = S[r][c] (c: key dim, r: value dim), so
 * S k, the rank-1 update and S q are column sweeps without reductions. */
#define SV8(X) svfloat32_t X##0, X##1, X##2, X##3, X##4, X##5, X##6, X##7
#define SV8_EACH(M) M(0); M(1); M(2); M(3); M(4); M(5); M(6); M(7)
static double ssm_t[4];
static int ssm_pf = 0;
static void ssm_head(int layer, int h, int pos) {
    uint64_t T0 = h ? 0 : ticks();
    const q38d_layer *L = &E.L[layer];
    int g = h % NGROUP;
    float q[128], k[128], v[128], o[128];
    if (ssm_pf == 3) ssm_prefetch_state(layer, h);
    if (conv_local) {
        conv_silu_local(layer, h, 0, g * DS, pos, q);
        conv_silu_local(layer, h, 1, NGROUP * DS + g * DS, pos, k);
        conv_silu_local(layer, h, 2, 2 * NGROUP * DS + h * DS, pos, v);
    } else {
        conv_silu(L, layer, g * DS, pos, h < NGROUP, q);
        conv_silu(L, layer, NGROUP * DS + g * DS, pos, h < NGROUP, k);
        conv_silu(L, layer, 2 * NGROUP * DS + h * DS, pos, 1, v);
    }
    l2norm128(q); l2norm128(k);
    const float qs = 1.0f / sqrtf((float)DS);
    for (int i = 0; i < 128; i++) q[i] *= qs;
    float val = E.ab[h] + L->dt_bias[h];
    float sp = val > 20.0f ? val : logf(1.0f + expf(val));
    float decay = expf(sp * L->ssm_a[h]);
    float beta = 1.0f / (1.0f + expf(-E.bb[h]));
    float *St = E.ssm_state[layer][h];
    const svbool_t pf = svptrue_b32();
    uint64_t T1 = h ? 0 : ticks();
    SV8(sk); SV8(dl); SV8(oo);
#define Z0(j) sk##j = svdup_n_f32(0); oo##j = sk##j
    SV8_EACH(Z0);
    for (int c = 0; c < DS; c++) {
        float *row = St + (size_t)c * DS;
        svfloat32_t kc = svdup_n_f32(k[c]);
#define P1(j) { svfloat32_t s_ = svmul_n_f32_x(pf, svld1_f32(pf, row + 16 * j), decay); \
                svst1_f32(pf, row + 16 * j, s_); sk##j = svmla_f32_x(pf, sk##j, s_, kc); }
        SV8_EACH(P1);
    }
#define DL(j) dl##j = svmul_n_f32_x(pf, svsub_f32_x(pf, svld1_f32(pf, v + 16 * j), sk##j), beta)
    SV8_EACH(DL);
    uint64_t T2 = h ? 0 : ticks();
    for (int c = 0; c < DS; c++) {
        float *row = St + (size_t)c * DS;
        svfloat32_t kc = svdup_n_f32(k[c]), qc = svdup_n_f32(q[c]);
#define P2(j) { svfloat32_t s_ = svmla_f32_x(pf, svld1_f32(pf, row + 16 * j), dl##j, kc); \
                svst1_f32(pf, row + 16 * j, s_); oo##j = svmla_f32_x(pf, oo##j, s_, qc); }
        SV8_EACH(P2);
    }
#define OS(j) svst1_f32(pf, o + 16 * j, oo##j)
    SV8_EACH(OS);
    uint64_t T3 = h ? 0 : ticks();
    float inv = 1.0f / sqrtf(sumsq(o, DS) / DS + E.eps);
    const float *z = E.zb + (size_t)h * DS;
    float *dst = E.o + (size_t)h * DS;
    for (int i = 0; i < DS; i += 16) {
        svfloat32_t zv = svld1_f32(pf, z + i);
        svfloat32_t sz = svmul_f32_x(pf, zv, q38d_sigmoid_sve(pf, zv));
        svfloat32_t ov = svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, o + i), inv), svld1_f32(pf, L->ssm_norm + i));
        svst1_f32(pf, dst + i, svmul_f32_x(pf, ov, sz));
    }
    if (E.arith != Q38D_F32) q38d_prepare_range(&E.act_o, E.o, h * 4, h * 4 + 4);
    if (!h) { uint64_t T4 = ticks(); ssm_t[0] += T1 - T0; ssm_t[1] += T2 - T1; ssm_t[2] += T3 - T2; ssm_t[3] += T4 - T3; }
}
static void ssm_prefetch_state(int layer, int h) {
    const char *p = (const char *)E.ssm_state[layer][h];
    for (int i = 0; i < DS * DS * 4; i += 256) __builtin_prefetch(p + i, 1, 2);
}

/* ------------------------------------------------------------------ */
/* attention (CMG c owns KV head c and query heads 6c..6c+5)            */

static void head_rmsnorm_rope(float *v, const float *w, int pos) {
    float inv = 1.0f / sqrtf(sumsq(v, HD) / HD + E.eps);
    for (int i = 0; i < HD; i++) v[i] = v[i] * inv * w[i];
    for (int j = 0; j < 32; j++) {
        float th = (float)pos * E.inv_freq[j];
        float cs = cosf(th), sn = sinf(th);
        float a = v[j], b = v[j + 32];
        v[j] = a * cs - b * sn;
        v[j + 32] = a * sn + b * cs;
    }
}

/* Scores s[h][t-t0] = scale * q_h . K[t] for 6 heads, four positions per
 * block with 24 register accumulators (each K row read once for all heads). */
#define QK_ACC(h) svfloat32_t a##h##0 = svdup_n_f32(0), a##h##1 = a##h##0, a##h##2 = a##h##0, a##h##3 = a##h##0
#define QK_FMA(h) do { svfloat32_t qv = svld1_f32(pf, qh + (h) * HD + 16 * j);                       \
    a##h##0 = svmla_f32_x(pf, a##h##0, qv, k0); a##h##1 = svmla_f32_x(pf, a##h##1, qv, k1);          \
    a##h##2 = svmla_f32_x(pf, a##h##2, qv, k2); a##h##3 = svmla_f32_x(pf, a##h##3, qv, k3); } while (0)
#define QK_OUT(h) do { sc[(h) * nt + i + 0] = svaddv_f32(pf, a##h##0) * scale; sc[(h) * nt + i + 1] = svaddv_f32(pf, a##h##1) * scale; \
    sc[(h) * nt + i + 2] = svaddv_f32(pf, a##h##2) * scale; sc[(h) * nt + i + 3] = svaddv_f32(pf, a##h##3) * scale; } while (0)
static void attn_scores(const float *qh, const float *K, int t0, int t1, float *sc, int nt, float scale) {
    const svbool_t pf = svptrue_b32();
    int i = 0;
    for (; i + 4 <= nt; i += 4) {
        const float *k = K + (size_t)(t0 + i) * HD;
        QK_ACC(0); QK_ACC(1); QK_ACC(2); QK_ACC(3); QK_ACC(4); QK_ACC(5);
        for (int j = 0; j < HD / 16; j++) {
            svfloat32_t k0 = svld1_f32(pf, k + 16 * j), k1 = svld1_f32(pf, k + HD + 16 * j);
            svfloat32_t k2 = svld1_f32(pf, k + 2 * HD + 16 * j), k3 = svld1_f32(pf, k + 3 * HD + 16 * j);
            QK_FMA(0); QK_FMA(1); QK_FMA(2); QK_FMA(3); QK_FMA(4); QK_FMA(5);
        }
        QK_OUT(0); QK_OUT(1); QK_OUT(2); QK_OUT(3); QK_OUT(4); QK_OUT(5);
    }
    for (; i < nt; i++)
        for (int h = 0; h < 6; h++) sc[h * nt + i] = dot(qh + h * HD, K + (size_t)(t0 + i) * HD, HD) * scale;
}
/* o[256] = sum_t p[t-t0] V[t], sixteen register accumulators. */
#define PV_ACC(j) svfloat32_t o##j = svdup_n_f32(0)
#define PV_FMA(j) o##j = svmla_f32_x(pf, o##j, svld1_f32(pf, v + 16 * (j)), pv)
#define PV_ST(j) svst1_f32(pf, out + 16 * (j), o##j)
static void attn_pv(const float *V, int t0, int t1, const float *p, float *out) {
    const svbool_t pf = svptrue_b32();
    PV_ACC(0); PV_ACC(1); PV_ACC(2); PV_ACC(3); PV_ACC(4); PV_ACC(5); PV_ACC(6); PV_ACC(7);
    PV_ACC(8); PV_ACC(9); PV_ACC(10); PV_ACC(11); PV_ACC(12); PV_ACC(13); PV_ACC(14); PV_ACC(15);
    for (int t = t0; t < t1; t++) {
        const float *v = V + (size_t)t * HD;
        svfloat32_t pv = svdup_n_f32(p[t - t0]);
        PV_FMA(0); PV_FMA(1); PV_FMA(2); PV_FMA(3); PV_FMA(4); PV_FMA(5); PV_FMA(6); PV_FMA(7);
        PV_FMA(8); PV_FMA(9); PV_FMA(10); PV_FMA(11); PV_FMA(12); PV_FMA(13); PV_FMA(14); PV_FMA(15);
    }
    PV_ST(0); PV_ST(1); PV_ST(2); PV_ST(3); PV_ST(4); PV_ST(5); PV_ST(6); PV_ST(7);
    PV_ST(8); PV_ST(9); PV_ST(10); PV_ST(11); PV_ST(12); PV_ST(13); PV_ST(14); PV_ST(15);
}

static double att_t[4];
static void attention(int layer, int tid, int pos, int *csense) {
    uint64_t T0 = tid ? 0 : ticks();
    const q38d_layer *L = &E.L[layer];
    int c = tid / PER, l = tid % PER, ai = L->ai;
    float *qh = E.qh[c];
    if (l < 6) {
        int hq = 6 * c + l;
        memcpy(qh + l * HD, E.qg + (size_t)hq * 2 * HD, HD * sizeof(float));
        head_rmsnorm_rope(qh + l * HD, L->q_norm, pos);
    } else if (l == 6) {
        float kv[HD];
        memcpy(kv, E.kb + c * HD, sizeof kv);
        head_rmsnorm_rope(kv, L->k_norm, pos);
        memcpy(E.kc[ai][c] + (size_t)pos * HD, kv, sizeof kv);
        memcpy(E.vc[ai][c] + (size_t)pos * HD, E.vb + c * HD, HD * sizeof(float));
    }
    uint64_t T1 = tid ? 0 : ticks();
    cbarrier(tid, csense);
    uint64_t T2 = tid ? 0 : ticks();
    int n = pos + 1, t0 = (int)((int64_t)n * l / PER), t1 = (int)((int64_t)n * (l + 1) / PER);
    const float scale = 1.0f / 16.0f;
    const svbool_t pf = svptrue_b32();
    float *part = E.apart[c] + (size_t)l * 6 * (2 + HD);
    const float *K = E.kc[ai][c], *V = E.vc[ai][c];
    int nt = t1 - t0;
    float sc[6 * (nt > 0 ? nt : 1)];
    if (nt <= 0) {
        for (int hh = 0; hh < 6; hh++) {
            float *pp = part + hh * (2 + HD);
            pp[0] = -INFINITY; pp[1] = 0; memset(pp + 2, 0, HD * 4);
        }
    } else {
        attn_scores(qh, K, t0, t1, sc, nt, scale);
        for (int hh = 0; hh < 6; hh++) {
            float *s_h = sc + hh * nt, *pp = part + hh * (2 + HD);
            float m = -INFINITY;
            for (int i = 0; i < nt; i++) if (s_h[i] > m) m = s_h[i];
            svfloat32_t lsv = svdup_n_f32(0);
            for (int i = 0; i < nt; i += 16) {
                svbool_t pg = svwhilelt_b32(i, nt);
                svfloat32_t e = q38d_exp_sve(pg, svsub_n_f32_x(pg, svld1_f32(pg, s_h + i), m));
                svst1_f32(pg, s_h + i, e);
                lsv = svadd_f32_m(pg, lsv, e);
            }
            pp[0] = m; pp[1] = svaddv_f32(pf, lsv);
            attn_pv(V, t0, t1, s_h, pp + 2);
        }
    }
    uint64_t T3 = tid ? 0 : ticks();
    cbarrier(tid, csense);
    if (!tid) { uint64_t T4 = ticks(); att_t[0] += T1 - T0; att_t[1] += T2 - T1; att_t[2] += T3 - T2; att_t[3] += T4 - T3; }
    if (l < 6) {
        int hq = 6 * c + l;
        float M = -INFINITY;
        for (int i = 0; i < PER; i++) {
            float mi = E.apart[c][(size_t)i * 6 * (2 + HD) + l * (2 + HD)];
            if (mi > M) M = mi;
        }
        float o[HD], den = 0;
        memset(o, 0, sizeof o);
        for (int i = 0; i < PER; i++) {
            const float *pp = E.apart[c] + (size_t)i * 6 * (2 + HD) + l * (2 + HD);
            if (pp[1] == 0) continue;
            float w = q38d_expf(pp[0] - M);
            den += w * pp[1];
            for (int j = 0; j < HD; j += 16)
                svst1_f32(pf, o + j, svmla_n_f32_x(pf, svld1_f32(pf, o + j), svld1_f32(pf, pp + 2 + j), w));
        }
        const float *gate = E.qg + (size_t)hq * 2 * HD + HD;
        float *dst = E.o + (size_t)hq * HD;
        float rden = 1.0f / den;
        for (int j = 0; j < HD; j += 16) {
            svfloat32_t gv = q38d_sigmoid_sve(pf, svld1_f32(pf, gate + j));
            svst1_f32(pf, dst + j, svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, o + j), rden), gv));
        }
        if (E.arith != Q38D_F32) q38d_prepare_range(&E.act_o, E.o, hq * 8, hq * 8 + 8);
    }
}

/* ------------------------------------------------------------------ */
/* token step                                                           */

/* Columns [k0, k1) of row `row` from the original FP6 lowbit tiles
 * (qwen38_lowbit layout: low[4][64], high[4][32], scale[2][8] per 8x64). */
static void embed_fp6_cols(float *dst, const q38_lowbit_matrix *m, int row, int k0, int k1) {
    int c = 0;
    while (c < 3 && row >= m->first[c + 1]) c++;
    int rr = row - m->first[c];
    const uint8_t *tiles = (const uint8_t *)m->part[c];
    size_t nb = (size_t)m->cols / 64;
    for (int k = k0; k < k1; k++) {
        const uint8_t *t = tiles + ((size_t)(rr / 8) * nb + k / 64) * 400;
        int r8 = rr % 8, kk = k % 64, sb = kk / 16, lane = r8 * 8 + kk % 8, half = (kk % 16) / 8;
        unsigned lo = (t[sb * 64 + lane] >> (half * 4)) & 15;
        unsigned hi = (t[256 + sb * 32 + lane / 2] >> ((lane & 1) * 4 + half * 2)) & 3;
        dst[k] = ldexpf((float)q38d_lut_f6[lo | hi << 4], (int)t[384 + (kk / 32) * 8 + r8] - 130);
    }
}

static void embed_token(int tid, int token) {
    if (E.embed_q6k) {
        int blocks = EMBD / 256;
        if (tid < blocks)
            dequant_row(GGML_TYPE_Q6_K, E.embed_q6k + (size_t)token * E.embed_row_bytes + (size_t)tid * 210,
                        E.x + tid * 256, 256);
    } else if (E.embed_lb->format == Q38_LB_FP6_E2M3) {
        int k0 = EMBD * tid / NT, k1 = EMBD * (tid + 1) / NT;
        embed_fp6_cols(E.x, E.embed_lb, token, k0, k1);
    } else if (tid == 0) {
        q38_lowbit_matrix_row(E.x, E.embed_lb, token);
    }
}

static void prof_mark(int tid, uint64_t *t, int phase) {
    if (tid) return;
    uint64_t n = ticks();
    E.prof[phase] += (double)(n - *t);
    *t = n;
}

/* Warm the start of upcoming weight streams into L2 while waiting. */
static size_t pf_bytes = 48 * 1024;
static inline void pf_l2(const void *p, size_t bytes) {
    for (size_t o = 0; o < bytes; o += 256) __builtin_prefetch((const char *)p + o, 0, 2);
}
static void pf_mat(const q38d_mat *m, int tid) {
    if (m->kch > 1) {
        int c = tid / PER, i0, i1;
        lane_items(m, c, tid % PER, &i0, &i1);
        if (i1 > i0) pf_l2(m->part[c] + (size_t)i0 * q38d_group_bytes(m->fmt, m->cols / m->kch), pf_bytes);
        return;
    }
    int c, g0, g1;
    mat_range(m, tid, &c, &g0, &g1);
    if (g0 < g1) pf_l2(m->part[c] + (size_t)g0 * q38d_group_bytes(m->fmt, m->cols), pf_bytes);
}
static void pf_plan(const q38d_plan *P, int tid) {
    int c = tid / PER;
    const q38d_tplan *tp = &P->t[tid];
    for (int k = 0; k < tp->nseg; k++) {
        const q38d_mat *m = P->m[tp->s[k].mi];
        pf_l2(m->part[c] + (size_t)tp->s[k].g0 * q38d_group_bytes(m->fmt, m->cols), pf_bytes / 2);
    }
}
static void pf_ffn(const q38d_layer *L, int tid, int which) {
    int c = tid / PER, l = tid % PER;
    int rows = L->gate.first[c + 1] - L->gate.first[c];
    int g0 = rows / 8 * l / PER;
    size_t gb = q38d_group_bytes(L->gate.fmt, L->gate.cols);
    if (which & 1) pf_l2(L->gate.part[c] + (size_t)g0 * gb, pf_bytes);
    if (which & 2) pf_l2(L->up.part[c] + (size_t)g0 * gb, pf_bytes);
}
static int pf_kv;
static void pf_attn_kv(int tid, int layer);
/* Prefetch what this worker streams after the barrier that ends `phase`. */
static void pf_next(int tid, int layer, int phase) {
    if (!pf_bytes) return;
    const q38d_layer *L = &E.L[layer];
    switch (phase) {
    case P_SSM_IN: pf_mat(&L->out, tid); break;
    case P_ATT_IN: pf_mat(&L->o, tid); if (pf_kv) pf_attn_kv(tid, layer); break;
    case P_SSM_OUT: case P_ATT_OUT: pf_ffn(L, tid, 3); break;
    case P_FFN_UP: pf_mat(&L->down, tid); break;
    case P_FFN_DOWN:
        if (layer + 1 < NLAYER) {
            if (E.L[layer + 1].ssm) pf_plan(&plan_ssm[layer + 1], tid);
            else pf_plan(&plan_att[layer + 1], tid);
        } else pf_mat(&E.head, tid);
        break;
    default: break;
    }
}
static int pf_layer[NT];
static int pf_pos;
static int pf_kv = 1;
static void pf_attn_kv(int tid, int layer) {
    const q38d_layer *L = &E.L[layer];
    int c = tid / PER, l = tid % PER, n = pf_pos + 1;
    int t0 = (int)((int64_t)n * l / PER), t1 = (int)((int64_t)n * (l + 1) / PER);
    if (t1 > t0) {
        pf_l2(E.kc[L->ai][c] + (size_t)t0 * HD, (size_t)(t1 - t0) * HD * 4);
        pf_l2(E.vc[L->ai][c] + (size_t)t0 * HD, (size_t)(t1 - t0) * HD * 4);
    }
}

static void phase_end(int tid, int *gs, uint64_t *t, int phase) {
    pf_next(tid, pf_layer[tid], phase);
    cur_phase[tid] = phase;
    gbarrier(tid, gs);
    prof_mark(tid, t, phase);
}

static void step(int tid, int token, int pos, int want_head, int *gs, int *cs) {
    uint64_t t = ticks();
    pf_pos = pos;
    ssq_valid = 0;
    embed_token(tid, token);
    phase_end(tid, gs, &t, P_EMBED);
    for (int layer = 0; layer < NLAYER; layer++) {
        const q38d_layer *L = &E.L[layer];
        pf_layer[tid] = layer;
        const q38d_act *a;
        float omul = 1.0f;
        if (prod_norm && layer > 0) { int cc = prod_copies > 1 ? tid / PER : 0; a = &E.act_c[cc]; E.act_c[cc].x = E.xn_c[cc]; omul = x_inv(); }
        else a = norm_act(tid, L->attn_norm);
        if (L->ssm) {
            if (ssm_pf == 1) ssm_prefetch_state(layer, tid);
            if (use_plan) run_plan(&plan_ssm[layer], a, tid, omul);
            else { mv(&L->qkv, a, E.qkv, 0, tid); mv(&L->z, a, E.zb, 0, tid);
                   mv(&L->alpha, a, E.ab, 0, tid); mv(&L->beta, a, E.bb, 0, tid); }
            phase_end(tid, gs, &t, P_SSM_IN);
            if (ssm_pf == 2) ssm_prefetch_state(layer, tid);
            ssm_head(layer, tid, pos);
            if (E.arith == Q38D_F32) E.act_o.x = E.o;
            phase_end(tid, gs, &t, P_SSM_CORE);
            mv(&L->out, &E.act_o, E.x, 1, tid);
            ssq_valid = 1;
            ssq_split_valid = L->out.kch > 1;
            if (prod_norm) x_produce(&L->out, tid, L->post_norm);
            else ssq_publish_rows(&L->out, tid);
            phase_end(tid, gs, &t, P_SSM_OUT);
        } else {
            if (use_plan) run_plan(&plan_att[layer], a, tid, omul);
            else { mv(&L->q, a, E.qg, 0, tid); mv(&L->k, a, E.kb, 0, tid); mv(&L->v, a, E.vb, 0, tid); }
            phase_end(tid, gs, &t, P_ATT_IN);
            attention(layer, tid, pos, cs);
            if (E.arith == Q38D_F32) E.act_o.x = E.o;
            phase_end(tid, gs, &t, P_ATT_CORE);
            mv(&L->o, &E.act_o, E.x, 1, tid);
            ssq_valid = 1;
            ssq_split_valid = L->o.kch > 1;
            if (prod_norm) x_produce(&L->o, tid, L->post_norm);
            else ssq_publish_rows(&L->o, tid);
            phase_end(tid, gs, &t, P_ATT_OUT);
        }
        if (prod_norm) { int cc = prod_copies > 1 ? tid / PER : 0; a = &E.act_c[cc]; E.act_c[cc].x = E.xn_c[cc]; omul = x_inv(); }
        else { a = norm_act(tid, L->post_norm); omul = 1.0f; }
        {
            int c = tid / PER, l = tid % PER;
            int rows = L->gate.first[c + 1] - L->gate.first[c];
            /* 8-row group partition (+-1 group); a 16-row activation unit
             * split between two workers is quantized by whichever finishes
             * second (per-unit counter parity, no reset needed). */
            int G8 = rows / 8, g0 = G8 * l / PER, g1 = G8 * (l + 1) / PER;
            if (!ffn_group_split) { int units = rows / 16; g0 = 2 * (units * l / PER); g1 = 2 * (units * (l + 1) / PER); }
            if (g0 < g1) {
                int base = L->gate.first[c];
                if (use_dual && a->arith != Q38D_F32 && L->gate.fmt == L->up.fmt &&
                    (L->gate.fmt == Q38D_F4 || L->gate.fmt == Q38D_F6))
                    q38d_gemv_dual_fmt(E.h + base, E.qkv + base, L->gate.part[c], L->up.part[c], a, g0, g1, L->gate.fmt);
                else {
                    q38d_gemv_any(E.h + base, L->gate.part[c], L->gate.fmt, a, g0, g1, rows, 0);
                    q38d_gemv_any(E.qkv + base, L->up.part[c], L->up.fmt, a, g0, g1, rows, 0);
                }
                if (omul != 1.0f) {
                    scale_rows(E.h + base + 8 * g0, 8 * (g1 - g0), omul);
                    scale_rows(E.qkv + base + 8 * g0, 8 * (g1 - g0), omul);
                }
                const svbool_t pf = svptrue_b32();
                const svbool_t p8 = svptrue_pat_b32(SV_VL8);
                for (int g = g0; g < g1; g++) {
                    int r = base + 8 * g;
                    svfloat32_t gv = svld1_f32(p8, E.h + r);
                    svfloat32_t hv = svmul_f32_x(p8, svmul_f32_x(p8, gv, q38d_sigmoid_sve(p8, gv)), svld1_f32(p8, E.qkv + r));
                    svst1_f32(p8, E.h + r, hv);
                }
                (void)pf;
                if (E.arith != Q38D_F32) {
                    int u_first = g0 / 2, u_last = (g1 + 1) / 2;
                    for (int u = u_first; u < u_last; u++) {
                        int whole = 2 * u >= g0 && 2 * u + 1 < g1;
                        int r = base + 16 * u;
                        if (!whole) {
                            /* shared unit: second finisher quantizes */
                            int old = atomic_fetch_add_explicit(&E.unit_cnt[r / 16], 1, memory_order_acq_rel);
                            if (!(old & 1)) continue;
                        }
                        q38d_prepare_unit(&E.act_h, r / 32, (r / 16) & 1, E.h + r);
                    }
                }
            }
        }
        if (E.arith == Q38D_F32) E.act_h.x = E.h;
        phase_end(tid, gs, &t, P_FFN_UP);
        mv(&L->down, &E.act_h, E.x, 1, tid);
        ssq_split_valid = L->down.kch > 1;
        if (prod_norm) x_produce(&L->down, tid, layer + 1 < NLAYER ? E.L[layer + 1].attn_norm : E.out_norm);
        else ssq_publish_rows(&L->down, tid);
        phase_end(tid, gs, &t, P_FFN_DOWN);
    }
    if (!want_head) return;
    const q38d_act *a;
    float omul = 1.0f;
    if (prod_norm) { int cc = prod_copies > 1 ? tid / PER : 0; a = &E.act_c[cc]; E.act_c[cc].x = E.xn_c[cc]; omul = x_inv(); }
    else a = norm_act(tid, E.out_norm);
    mv(&E.head, a, E.logits, 0, tid);
    if (omul != 1.0f) {
        int c, g0, g1;
        mat_range(&E.head, tid, &c, &g0, &g1);
        int r0 = E.head.first[c] + 8 * g0, r1 = E.head.first[c] + 8 * g1;
        if (r1 > E.head.first[c + 1]) r1 = E.head.first[c + 1];
        if (r1 > r0) scale_rows(E.logits + r0, r1 - r0, omul);
    }
    {
        int c, g0, g1;
        mat_range(&E.head, tid, &c, &g0, &g1);
        int r0 = E.head.first[c] + 8 * g0, r1 = E.head.first[c] + 8 * g1;
        if (r1 > E.head.first[c + 1]) r1 = E.head.first[c + 1];
        float best = -INFINITY; int bi = -1;
        for (int r = r0; r < r1; r++) if (E.logits[r] > best) { best = E.logits[r]; bi = r; }
        E.best_val[tid] = best; E.best_idx[tid] = bi;
    }
    gbarrier(tid, gs);
    if (tid == 0) {
        float best = -INFINITY; int bi = 0;
        for (int i = 0; i < NT; i++) if (E.best_idx[i] >= 0 && E.best_val[i] > best) { best = E.best_val[i]; bi = E.best_idx[i]; }
        E.next_token = bi;
    }
    prof_mark(tid, &t, P_HEAD);
    gbarrier(tid, gs);
}

/* ------------------------------------------------------------------ */
/* run control                                                          */

typedef struct {
    int32_t *tok;
    int n_prompt, n_gen, dump;
    double t_prefill, t_decode;
    float *trace_logit;
} q38d_job;
static q38d_job JOB;
static int bench_mv;

static void *worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    pin_cpu(12 + tid);
    int gs = 0, cs = 0;
    /* first touch of per-worker and per-head state on the owning CMG */
    E.xn[tid] = aligned_alloc(256, EMBD * sizeof(float));
    memset(E.xn[tid], 0, EMBD * sizeof(float));
    q38d_act *a = &E.act_x[tid];
    a->cols = EMBD; a->arith = E.arith;
    a->q = aligned_alloc(256, q38d_act_qbytes(EMBD, Q38D_A16) + 256);
    a->sc = aligned_alloc(256, EMBD / 16 * 4 + 256);
    a->sum = aligned_alloc(256, EMBD / 16 * 4 + 256);
    for (int layer = 0; layer < NLAYER; layer++)
        if (E.L[layer].ssm) {
            E.ssm_state[layer][tid] = aligned_alloc(256, DS * DS * sizeof(float));
            memset(E.ssm_state[layer][tid], 0, DS * DS * sizeof(float));
            E.conv_hist[layer][tid] = aligned_alloc(256, 4 * 384 * sizeof(float));
            memset(E.conv_hist[layer][tid], 0, 4 * 384 * sizeof(float));
            float *wl = aligned_alloc(256, 4 * 384 * sizeof(float));
            int g = tid % NGROUP, ch[3] = {g * DS, NGROUP * DS + g * DS, 2 * NGROUP * DS + tid * DS};
            for (int kk = 0; kk < 4; kk++)
                for (int part = 0; part < 3; part++)
                    memcpy(wl + kk * 384 + part * 128, E.L[layer].conv_w + (size_t)kk * QKVD + ch[part], 128 * sizeof(float));
            E.conv_wl[layer][tid] = wl;
        }
    int c = tid / PER, l = tid % PER;
    if (l == 0) {
        for (int ai = 0; ai < NATTN; ai++) {
            E.kc[ai][c] = cmg_alloc((size_t)E.max_seq * HD * 4, c);
            E.vc[ai][c] = cmg_alloc((size_t)E.max_seq * HD * 4, c);
            memset(E.kc[ai][c], 0, (size_t)E.max_seq * HD * 4);
            memset(E.vc[ai][c], 0, (size_t)E.max_seq * HD * 4);
        }
        E.qh[c] = aligned_alloc(256, 6 * HD * 4);
        E.xn_c[c] = aligned_alloc(256, EMBD * sizeof(float));
        memset(E.xn_c[c], 0, EMBD * sizeof(float));
        q38d_act *ac = &E.act_c[c];
        ac->cols = EMBD; ac->arith = E.arith;
        ac->q = aligned_alloc(256, q38d_act_qbytes(EMBD, Q38D_A16) + 256);
        ac->sc = aligned_alloc(256, EMBD / 16 * 4 + 256);
        ac->sum = aligned_alloc(256, EMBD / 16 * 4 + 256);
        memset(ac->q, 0, q38d_act_qbytes(EMBD, Q38D_A16));
        E.apart[c] = aligned_alloc(256, (size_t)PER * 6 * (2 + HD) * 4);
    }
    norm_csense[tid] = &cs;
    if (getenv("Q38D_NORM_CMG")) norm_cmg = atoi(getenv("Q38D_NORM_CMG"));
    repack_worker(c, l);
    gbarrier(tid, &gs);
    kchunk_copy_or_write(c, l, 0);
    gbarrier(tid, &gs);
    kchunk_copy_or_write(c, l, 1);
    gbarrier(tid, &gs);
    if (getenv("Q38D_BENCH_BAR")) {
        int n = atoi(getenv("Q38D_BENCH_BAR"));
        gbarrier(tid, &gs);
        uint64_t t0 = ticks();
        for (int i = 0; i < n; i++) gbarrier(tid, &gs);
        uint64_t t1 = ticks();
        for (int i = 0; i < n; i++) cbarrier(tid, &cs);
        uint64_t t2 = ticks();
        if (!tid) fprintf(stderr, "q38d: gbarrier %.3f us, cbarrier %.3f us\n",
                          (t1 - t0) / tick_hz() * 1e6 / n, (t2 - t1) / tick_hz() * 1e6 / n);
        return NULL;
    }
    if (bench_mv) {
        /* stream every FFN gate matrix: per-matrix barrier vs one barrier */
        const q38d_act *a = norm_act(tid, E.L[0].attn_norm);
        for (int mode = 0; mode < 3; mode++) {
            gbarrier(tid, &gs);
            uint64_t t0 = ticks();
            size_t bytes = 0;
            for (int rep = 0; rep < bench_mv; rep++)
                for (int layer = 0; layer < NLAYER; layer++) {
                    const q38d_mat *m = &E.L[layer].gate;
                    if (mode == 2) { norm_act(tid, E.L[layer].post_norm); }
                    else mv(m, a, E.h, 0, tid);
                    if (mode != 1) gbarrier(tid, &gs);
                    bytes += (size_t)m->rows / 8 * q38d_group_bytes(m->fmt, m->cols);
                }
            gbarrier(tid, &gs);
            if (tid == 0) {
                double sec = (double)(ticks() - t0) / tick_hz();
                if (mode < 2) fprintf(stderr, "q38d: bench-mv %s: %.3f ms per 64 gate matrices, %.1f GB/s\n",
                                      mode ? "no-barrier" : "barrier", sec * 1e3 / bench_mv, bytes / sec * 1e-9);
                else fprintf(stderr, "q38d: bench norm_act: %.2f us per call\n", sec * 1e6 / bench_mv / NLAYER);
            }
        }
        return NULL;
    }
    double t0 = 0;
    int pn = JOB.n_prompt, gn = JOB.n_gen;
    if (tid == 0) t0 = now_sec();
    for (int pos = 0; pos < pn; pos++) step(tid, JOB.tok[pos], pos, pos == pn - 1, &gs, &cs);
    if (tid == 0) {
        JOB.t_prefill = now_sec() - t0;
        memset(E.prof, 0, sizeof E.prof);
        kprof_kernel = kprof_norm = kprof_wait_acc = 0;
        memset(ssm_t, 0, sizeof ssm_t);
        memset(att_t, 0, sizeof att_t);
        norm_cbar_t = 0;
    }
    gbarrier(tid, &gs);
    if (tid == 0) {
        memset(busy_acc, 0, sizeof busy_acc);
        busy_on = 1;
        t0 = now_sec();
    }
    for (int n = 0; n < gn; n++) {
        int cur = E.next_token;
        if (tid == 0) { JOB.tok[pn + n] = cur; JOB.trace_logit[n] = E.logits[cur]; }
        step(tid, cur, pn + n, 1, &gs, &cs);
    }
    if (tid == 0) JOB.t_decode = now_sec() - t0;
    return NULL;
}

static void usage(const char *p) {
    fprintf(stderr, "usage: %s MODEL.gguf --fmt fp4|fp6 [--image PATH | --write-image PATH]\n"
                    "       [--act f32|a8|a16] [--prompt TEXT] [--prompt-tokens N] [--gen N]\n", p);
}

int main(int argc, char **argv) {
    const char *path = NULL, *image = NULL, *write_image = NULL, *prompt = "Explain why the sky is blue.";
    int fmt = Q38D_F4, arith = Q38D_A16, pn = 1024, gn = 256;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fmt") && i + 1 < argc) { i++; fmt = !strcmp(argv[i], "fp6") ? Q38D_F6 : Q38D_F4; }
        else if (!strcmp(argv[i], "--image") && i + 1 < argc) image = argv[++i];
        else if (!strcmp(argv[i], "--write-image") && i + 1 < argc) write_image = argv[++i];
        else if (!strcmp(argv[i], "--act") && i + 1 < argc) {
            i++; arith = !strcmp(argv[i], "f32") ? Q38D_F32 : !strcmp(argv[i], "a8") ? Q38D_A8 : Q38D_A16;
        }
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--prompt-tokens") && i + 1 < argc) pn = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gn = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench-mv") && i + 1 < argc) bench_mv = atoi(argv[++i]);
        else if (argv[i][0] != '-' && !path) path = argv[i];
        else { usage(argv[0]); return 2; }
    }
    if (!path || pn < 1 || gn < 1) { usage(argv[0]); return 2; }
    E.fmt = fmt; E.arith = arith; E.max_seq = pn + gn + 1;
    if (getenv("Q38D_OUT_KCH")) out_kch = atoi(getenv("Q38D_OUT_KCH"));
    if (getenv("Q38D_Q6K_EXPAND")) q6k_expand = atoi(getenv("Q38D_Q6K_EXPAND"));
    if (getenv("Q38D_ASM")) q38d_asm_variant = atoi(getenv("Q38D_ASM"));
    double t_load = now_sec();
    setenv("GGUF_LAZY_MMAP", "1", 1);
    G = gguf_open_multi(path, 1);
    if (!G) return 1;
    LB = image ? q38_lowbit_model_load_image(G, fmt == Q38D_F4 ? Q38_LB_NVFP4 : Q38_LB_FP6_E2M3, 0, 1,
                                             (size_t)4 << 30, image)
               : q38_lowbit_model_load(G, fmt == Q38D_F4 ? Q38_LB_NVFP4 : Q38_LB_FP6_E2M3, 0, 1, (size_t)4 << 30);
    if (!LB) { fprintf(stderr, "q38d: low-bit load failed\n"); return 1; }
    if (write_image && !q38_lowbit_model_save_image(LB, write_image)) { fprintf(stderr, "q38d: image write failed\n"); return 1; }
    bpe_vocab *vocab = bpe_vocab_load(G);
    if (!vocab) return 1;
    load_engine();
    /* shared buffers (first touched by main thread; small) */
    E.x = aligned_alloc(256, EMBD * 4);
    E.qkv = aligned_alloc(256, NFF * 4);
    E.zb = aligned_alloc(256, DINNER * 4);
    E.ab = aligned_alloc(256, 64 * 4); E.bb = aligned_alloc(256, 64 * 4);
    E.qg = aligned_alloc(256, 2 * NHEAD * HD * 4);
    E.kb = aligned_alloc(256, NKV * HD * 4); E.vb = aligned_alloc(256, NKV * HD * 4);
    E.o = aligned_alloc(256, DINNER * 4);
    E.h = aligned_alloc(256, NFF * 4);
    E.unit_cnt = calloc(NFF / 16, sizeof(*E.unit_cnt));
    E.logits = aligned_alloc(256, (size_t)E.n_vocab * 4);
    E.act_o = (q38d_act){DINNER, arith, aligned_alloc(256, q38d_act_qbytes(DINNER, Q38D_A16)),
                         aligned_alloc(256, DINNER / 16 * 4), aligned_alloc(256, DINNER / 16 * 4), E.o};
    E.act_h = (q38d_act){NFF, arith, aligned_alloc(256, q38d_act_qbytes(NFF, Q38D_A16)),
                         aligned_alloc(256, NFF / 16 * 4), aligned_alloc(256, NFF / 16 * 4), E.h};
    for (int l = 0; l < NLAYER; l++) {
        E.conv_state[l] = calloc((size_t)4 * QKVD, sizeof(float));
    }
    if (getenv("Q38D_PLAN")) use_plan = atoi(getenv("Q38D_PLAN"));
    if (getenv("Q38D_SSM_PF")) ssm_pf = atoi(getenv("Q38D_SSM_PF"));
    if (getenv("Q38D_PF_BYTES")) pf_bytes = (size_t)atoi(getenv("Q38D_PF_BYTES"));
    if (getenv("Q38D_PROD_NORM")) prod_norm = atoi(getenv("Q38D_PROD_NORM"));
    if (getenv("Q38D_DUAL")) use_dual = atoi(getenv("Q38D_DUAL"));
    if (getenv("Q38D_FFN_SPLIT")) ffn_group_split = atoi(getenv("Q38D_FFN_SPLIT"));
    if (getenv("Q38D_PAIR")) q38d_pair_groups = atoi(getenv("Q38D_PAIR"));
    if (getenv("Q38D_CONV_LOCAL")) conv_local = atoi(getenv("Q38D_CONV_LOCAL"));
    if (getenv("Q38D_DUAL_COST")) dual_cost = atof(getenv("Q38D_DUAL_COST"));
    if (getenv("Q38D_PF_KV")) pf_kv = atoi(getenv("Q38D_PF_KV"));
    if (getenv("Q38D_EPOCH_BAR")) epoch_bar = atoi(getenv("Q38D_EPOCH_BAR"));
    if (getenv("Q38D_PROD_COPIES")) prod_copies = atoi(getenv("Q38D_PROD_COPIES"));
    if (getenv("Q38D_COST_Q4K")) cost_q4k = atof(getenv("Q38D_COST_Q4K"));
    if (getenv("Q38D_COST_Q6K")) cost_q6k = atof(getenv("Q38D_COST_Q6K"));
    if (getenv("Q38D_COST_Q8K")) cost_q8k = atof(getenv("Q38D_COST_Q8K"));
    plan_ssm = calloc(NLAYER, sizeof(q38d_plan));
    plan_att = calloc(NLAYER, sizeof(q38d_plan));
    for (int l = 0; l < NLAYER; l++) {
        q38d_layer *L = &E.L[l];
        if (L->ssm) {
            q38d_plan *P = &plan_ssm[l];
            P->nm = 4;
            P->m[0] = &L->qkv; P->out[0] = E.qkv; P->m[1] = &L->z; P->out[1] = E.zb;
            P->m[2] = &L->alpha; P->out[2] = E.ab; P->m[3] = &L->beta; P->out[3] = E.bb;
            if (use_dual && L->qkv.fmt == L->z.fmt && (L->qkv.fmt == Q38D_F4 || L->qkv.fmt == Q38D_F6) &&
                L->qkv.cols == L->z.cols) build_plan_ssm_dual(P);
            else build_plan(P);
        } else {
            q38d_plan *P = &plan_att[l];
            P->nm = 3;
            P->m[0] = &L->q; P->out[0] = E.qg; P->m[1] = &L->k; P->out[1] = E.kb; P->m[2] = &L->v; P->out[2] = E.vb;
            build_plan(P);
        }
    }
    /* prompt */
    int32_t base[4096];
    int bn = bpe_tokenize(vocab, prompt, -1, base, 4096);
    if (bn <= 0) { fprintf(stderr, "q38d: prompt did not tokenize\n"); return 1; }
    JOB.tok = malloc((size_t)(pn + gn + 1) * sizeof(int32_t));
    JOB.trace_logit = malloc((size_t)gn * sizeof(float));
    for (int i = 0; i < pn; i++) JOB.tok[i] = base[i % bn];
    JOB.n_prompt = pn; JOB.n_gen = gn;
    fprintf(stderr, "q38d: fmt=%s act=%s prompt=%d gen=%d matrices=%d load=%.1fs\n",
            fmt == Q38D_F4 ? "fp4" : "fp6", arith == Q38D_F32 ? "f32" : arith == Q38D_A8 ? "a8" : "a16",
            pn, gn, ntodo, now_sec() - t_load);
    pthread_t th[NT];
    for (int i = 1; i < NT; i++) pthread_create(&th[i], NULL, worker, (void *)(intptr_t)i);
    worker((void *)(intptr_t)0);
    for (int i = 1; i < NT; i++) pthread_join(th[i], NULL);
    for (int n = 0; n < gn; n++)
        fprintf(stderr, "q38d: token n=%d pos=%d id=%d logit=%a\n", n, pn + n, JOB.tok[pn + n], JOB.trace_logit[n]);
    double hz = tick_hz();
    fprintf(stderr, "q38d: prefill %d tok %.3f s (%.3f tok/s); decode %d tok %.3f s = %.3f tok/s (%.3f ms/tok)\n",
            pn, JOB.t_prefill, pn / JOB.t_prefill, gn, JOB.t_decode, gn / JOB.t_decode, 1e3 * JOB.t_decode / gn);
    fprintf(stderr, "q38d: stages ms/tok:");
    for (int p = 0; p < P_N; p++) fprintf(stderr, " %s=%.3f", prof_name[p], E.prof[p] / hz * 1e3 / gn);
    fprintf(stderr, "\n");
    fprintf(stderr, "q38d: busy ms/tok (mean/max over workers):");
    for (int p = 0; p < P_N; p++) {
        double mx = 0, mean = 0;
        for (int i = 0; i < NT; i++) { mean += busy_acc[i][p]; if (busy_acc[i][p] > mx) mx = busy_acc[i][p]; }
        fprintf(stderr, " %s=%.3f/%.3f", prof_name[p], mean / NT / hz * 1e3 / gn, mx / hz * 1e3 / gn);
    }
    fprintf(stderr, "\n");
    for (int p = P_SSM_IN; p <= P_FFN_DOWN; p++) {
        fprintf(stderr, "q38d: busy[%s] us/tok per worker:", prof_name[p]);
        for (int i = 0; i < NT; i++) fprintf(stderr, "%s%.0f", i % 12 ? " " : " | ", busy_acc[i][p] / hz * 1e6 / gn);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "q38d: ssm_head worker0 us/layer: prep=%.2f pass1=%.2f pass2=%.2f finish=%.2f\n",
            ssm_t[0] / hz * 1e6 / gn / 48, ssm_t[1] / hz * 1e6 / gn / 48, ssm_t[2] / hz * 1e6 / gn / 48, ssm_t[3] / hz * 1e6 / gn / 48);
    fprintf(stderr, "q38d: norm_act cbarrier part worker0 ms/tok=%.3f\n", norm_cbar_t / hz * 1e3 / gn);
    fprintf(stderr, "q38d: attention worker0 us/layer: prep=%.2f cbar1=%.2f scores+pv=%.2f cbar2=%.2f\n",
            att_t[0] / hz * 1e6 / gn / 16, att_t[1] / hz * 1e6 / gn / 16, att_t[2] / hz * 1e6 / gn / 16, att_t[3] / hz * 1e6 / gn / 16);
    fprintf(stderr, "q38d: worker0 ms/tok: kernel=%.3f norm_act=%.3f barrier_wait=%.3f\n",
            kprof_kernel / hz * 1e3 / gn, kprof_norm / hz * 1e3 / gn, kprof_wait_acc / hz * 1e3 / gn);
    printf("RESULT fmt=%s act=%s prompt=%d gen=%d decode_tok_s=%.3f\n", fmt == Q38D_F4 ? "fp4" : "fp6",
           arith == Q38D_F32 ? "f32" : arith == Q38D_A8 ? "a8" : "a16", pn, gn, gn / JOB.t_decode);
    return 0;
}
