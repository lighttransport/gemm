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
typedef struct { int rows, cols, fmt, first[5], kch, unit16; uint8_t *part[4]; const uint8_t *src; } q38d_mat;
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
    float *conv_state[NLAYER];
    /* per head (worker) CMG-local copies: history [4 slots][q,k,v][128] and
     * conv weights [4 taps][q,k,v][128] of that head's channels */
    float *conv_hist[NLAYER][NVH], *conv_wl[NLAYER][NVH];
    float *kc[NATTN + 1][NKV], *vc[NATTN + 1][NKV]; /* [NATTN]: MTP layer */
    float *qh[NCMG];        /* [6][256] per CMG */
    float *apart[NCMG];     /* [12][6][2+256] per CMG */
    float best_val[NT];
    int best_idx[NT];
    int next_token;
    _Atomic int *unit_cnt; /* per 16-row FFN unit, parity counter */
    /* NextN/MTP drafter (Q38D_MTP=1, measurement): layer E.L[NLAYER] */
    q38d_mat eh, dhead;   /* dhead: first draft_v rows of the head (MTP drafts) */
    const float *enorm, *hnorm, *shnorm;
    float *mtp_in, *x_mtp, *h_mtp;
    q38d_act act_mtp;
    int mtp_draft;
    /* profiling (thread 0) */
    double prof[16];
} q38d_engine;
static int mtp_on, draft_v, draft_f4;

/* Per-call buffers of the SSM and attention cores (single-token path: the
 * E.* buffers; verification pass: token i's buffers). SSM state of
 * (layer, head) lives in a ring of SSM_RING buffers; the single-token path
 * updates slot 0 in place, token i of a verification pass reads slot i and
 * writes slot i+1, and accepting A tokens advances the ring by A. */
#define TMAX 4
typedef struct {
    const float *qkv, *zb, *ab, *bb, *qg, *kb, *vb;
    float *o;
    q38d_act *act_o;
    int slot;            /* SSM: read ring slot `slot`, write `slot + 1` (or in place: -1) */
} q38d_io;
static int ssm_ring = 1, ssm_cur;
static float **ssm_buf[NLAYER];   /* [NVH * ssm_ring] */
static inline float *ssm_st(int layer, int h, int i) { return ssm_buf[layer][h * ssm_ring + (ssm_cur + i) % ssm_ring]; }

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
typedef struct { q38d_mat *m; int idx; int kind; /* 0 lowbit in place, 1 raw Q6_K/Q4_K, 2 raw -> FP4 */ } pending;
static int describe_f4;   /* describe raw K-quant matrices as requantized FP4 (drafter) */
static int describe_rows; /* > 0: use only the first rows of the tensor */
static pending *todo;
static int ntodo;

static void describe(q38d_mat *m, int idx) {
    const gguf_tensor_info *t = &G->tensors[idx];
    const q38_lowbit_matrix *lb = q38_lowbit_model_tensor(G, idx);
    m->rows = describe_rows > 0 ? describe_rows : (int)t->dims[1]; m->cols = (int)t->dims[0];
    todo = realloc(todo, (size_t)(ntodo + 1) * sizeof(*todo));
    if (lb) {
        m->fmt = lb->format == Q38_LB_NVFP4 ? Q38D_F4 : Q38D_F6;
        for (int c = 0; c < 5; c++) m->first[c] = lb->first[c];
        for (int c = 0; c < 4; c++) m->part[c] = lb->part[c];
        todo[ntodo++] = (pending){m, idx, 0};
    } else if (t->type == GGML_TYPE_Q6_K || t->type == GGML_TYPE_Q4_K) {
        m->fmt = describe_f4 ? Q38D_F4 : t->type == GGML_TYPE_Q6_K ? (q6k_expand ? Q38D_Q8K : Q38D_Q6K) : Q38D_Q4K;
        int groups = (m->rows + 7) / 8;
        for (int c = 0; c <= 4; c++) {
            int r = (int)((int64_t)groups * c / 4) * 8;
            m->first[c] = r < m->rows ? r : m->rows;
        }
        for (int c = 0; c < 4; c++) {
            int gcount = (m->first[c + 1] - m->first[c] + 7) / 8;
            m->part[c] = gcount ? cmg_alloc((size_t)gcount * q38d_group_bytes(m->fmt, m->cols), c) : NULL;
        }
        todo[ntodo++] = (pending){m, idx, describe_f4 ? 2 : 1};
    } else {
        fprintf(stderr, "q38d: %s: unsupported matrix type %u\n", t->name.str, t->type);
        exit(1);
    }
    if (m->cols % 256) { fprintf(stderr, "q38d: %s cols %d not a multiple of 256\n", t->name.str, m->cols); exit(1); }
}

/* FP4 requantization of up to eight raw K-quant rows (drafter matrices:
 * E2M1 codes, per-16 E5M3 scale = smallest code >= amax / 6). The kernel's
 * weight is e2m1(code) * dec(u), dec(u) = (8 + (u & 7)) * 2^((u >> 3) - 20). */
static uint8_t e5m3_ceil(float s) {
    if (!(s > 0)) return 0;
    int e = ilogbf(s) + 17;                      /* s * 2^(20-e) in [8, 16) */
    int m = (int)ceilf(ldexpf(s, 20 - e)) - 8;
    if (m >= 8) { m = 0; e++; }
    if (e < 0) return 1;
    if (e > 31) { e = 31; m = 7; }
    return (uint8_t)(e << 3 | m);
}
static float e5m3_dec(uint8_t u) { return u ? ldexpf((float)(8 + (u & 7)), (u >> 3) - 20) : 0.f; }
static void f4_quant_group(uint8_t *dst, const uint8_t *src, size_t rb, int type, int nr, int cols) {
    static const float mag[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    size_t np = (size_t)cols / 32;
    memset(dst, 0, q38d_group_bytes(Q38D_F4, cols));
    uint8_t *sc = dst + np * 128;
    float *row = malloc((size_t)cols * sizeof(float));
    for (int r = 0; r < nr; r++) {
        dequant_row(type, src + (size_t)r * rb, row, cols);
        for (int b = 0; b < cols / 16; b++) {
            const float *v = row + 16 * b;
            float amax = 0;
            for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(v[j]));
            uint8_t u = e5m3_ceil(amax / 6.0f);
            float d = e5m3_dec(u), inv = d > 0 ? 1.0f / d : 0.f;
            for (int j = 0; j < 16; j++) {
                float x = fabsf(v[j]) * inv;
                int best = 0;
                for (int k = 1; k < 8; k++) if (fabsf(mag[k] - x) < fabsf(mag[best] - x)) best = k;
                unsigned code = (unsigned)best | (v[j] < 0 && best ? 8u : 0u);
                q38d_put_code(dst, Q38D_F4, cols, r, 16 * b + j, code);
            }
            sc[(size_t)(b / 2) * 16 + 2 * r + (b & 1)] = u;
        }
    }
    free(row);
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
            } else if (todo[i].kind == 2) {
                const gguf_tensor_info *t = &G->tensors[todo[i].idx];
                size_t rb = (size_t)m->cols / 256 * (t->type == GGML_TYPE_Q6_K ? 210 : 144);
                const uint8_t *src = (m->src ? m->src : (const uint8_t *)gguf_tensor_data(G, todo[i].idx)) +
                                     (size_t)(m->first[c] + g * 8) * rb;
                int nr = rows - g * 8 < 8 ? rows - g * 8 : 8;
                f4_quant_group(dst, src, rb, t->type, nr, m->cols);
            } else {
                const gguf_tensor_info *t = &G->tensors[todo[i].idx];
                size_t rb = (size_t)m->cols / 256 * (t->type == GGML_TYPE_Q6_K ? 210 : 144);
                const uint8_t *src = (m->src ? m->src : (const uint8_t *)gguf_tensor_data(G, todo[i].idx)) +
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

/* CMG-aligned SSM heads (ssm_perm): CMG c owns key groups 4c..4c+3 and the
 * value heads {g, g+16, g+32} of those groups, lane l running head
 * 4c + (l & 3) + 16 (l >> 2). The in-projection rows are permuted so that
 * CMG c's partition holds exactly its heads' q, k, v, z, alpha and beta
 * rows (qkv per CMG: q[4], k[4], v[12] blocks of 128), which lets the
 * in-proj -> core step use a CMG barrier instead of a global one. */
static int ssm_perm = 1;
static inline int ssm_head_of(int tid) {
    int c = tid / PER, l = tid % PER;
    return ssm_perm ? 4 * c + (l & 3) + 16 * (l >> 2) : tid;
}
static inline int ssm_slot(int h) { return ((h % NGROUP) / 4) * PER + (h & 3) + 4 * (h / NGROUP); }
static inline int ssm_qoff(int g) { return ssm_perm ? (QKVD / 4) * (g / 4) + (g % 4) * DS : g * DS; }
static inline int ssm_koff(int g) { return ssm_perm ? (QKVD / 4) * (g / 4) + 4 * DS + (g % 4) * DS : NGROUP * DS + g * DS; }
static inline int ssm_voff(int h) {
    if (!ssm_perm) return 2 * NGROUP * DS + h * DS;
    int sl = ssm_slot(h);
    return (QKVD / 4) * (sl / PER) + 8 * DS + (sl % PER) * DS;
}
static inline int ssm_zoff(int h) { return ssm_perm ? ssm_slot(h) * DS : h * DS; }
static inline int ssm_abi(int h) { return ssm_perm ? ssm_slot(h) : h; }
/* Move 128-row blocks of a low-bit matrix: new block b takes old block
 * src_blk[b]; blocks are 16 whole groups, so CMG parts stay in place. */
static void permute_blocks(q38d_mat *m, const int *src_blk, int nblk) {
    size_t gb = q38d_group_bytes(m->fmt, m->cols), bb = 16 * gb;
    uint8_t *tmp = malloc((size_t)nblk * bb);
    for (int b = 0; b < nblk; b++) {
        int G = b * 16, c = 0;
        while (G >= m->first[c + 1] / 8) c++;
        memcpy(tmp + (size_t)b * bb, m->part[c] + (size_t)(G - m->first[c] / 8) * gb, bb);
    }
    for (int b = 0; b < nblk; b++) {
        int G = b * 16, c = 0;
        while (G >= m->first[c + 1] / 8) c++;
        memcpy(m->part[c] + (size_t)(G - m->first[c] / 8) * gb, tmp + (size_t)src_blk[b] * bb, bb);
    }
    free(tmp);
}
static int ssm_perm_ok(const q38d_layer *L) {
    for (int c = 0; c <= 4; c++)
        if (L->qkv.first[c] != c * QKVD / 4 || L->z.first[c] != c * DINNER / 4) return 0;
    /* alpha/beta are re-gathered from raw K-quant rows (12 per CMG) */
    for (int k = 0; k < 2; k++) {
        const q38d_mat *m = k ? &L->beta : &L->alpha;
        if (m->rows != NVH || (m->fmt != Q38D_Q4K && m->fmt != Q38D_Q6K && m->fmt != Q38D_Q8K)) return 0;
    }
    return L->qkv.fmt == Q38D_F4 || L->qkv.fmt == Q38D_F6 ? (L->z.fmt == L->qkv.fmt) : 0;
}
static void ssm_permute_layer(q38d_layer *L, int layer) {
    int bq[QKVD / DS], bz[NVH];
    for (int c = 0; c < 4; c++)
        for (int j = 0; j < 20; j++) {
            int b = 20 * c + j;
            if (j < 4) bq[b] = 4 * c + j;                               /* q group */
            else if (j < 8) bq[b] = NGROUP + 4 * c + (j - 4);          /* k group */
            else bq[b] = 2 * NGROUP + ssm_head_of(c * PER + (j - 8));  /* v head  */
        }
    for (int t = 0; t < NVH; t++) bz[t] = ssm_head_of(t);
    permute_blocks(&L->qkv, bq, QKVD / DS);
    permute_blocks(&L->z, bz, NVH);
    /* alpha, beta (raw Q4_K/Q6_K rows): 12 rows per CMG in slot order */
    q38d_mat *ab[2] = {&L->alpha, &L->beta};
    const char *nm[2] = {"blk.%d.ssm_alpha.weight", "blk.%d.ssm_beta.weight"};
    for (int k = 0; k < 2; k++) {
        q38d_mat *m = ab[k];
        int idx = need_tensor(nm[k], layer);
        const gguf_tensor_info *t = &G->tensors[idx];
        size_t rb = (size_t)m->cols / 256 * (t->type == GGML_TYPE_Q6_K ? 210 : 144);
        uint8_t *buf = malloc((size_t)NVH * rb);
        const uint8_t *base = gguf_tensor_data(G, idx);
        for (int t2 = 0; t2 < NVH; t2++) memcpy(buf + (size_t)t2 * rb, base + (size_t)ssm_head_of(t2) * rb, rb);
        m->src = buf;
        for (int c = 0; c <= 4; c++) m->first[c] = c * NVH / 4;
        for (int c = 0; c < 4; c++)
            m->part[c] = cmg_alloc((size_t)((NVH / 4 + 7) / 8) * q38d_group_bytes(m->fmt, m->cols), c);
    }
}

static void load_engine(void) {
    E.n_vocab = (int)G->tensors[need_tensor("output.weight", 0)].dims[1];
    E.eps = 1e-6f;
    E.rope_base = 10000000.f;
    for (int j = 0; j < 32; j++) E.inv_freq[j] = 1.0f / powf(E.rope_base, (float)(2 * j) / 64.f);
    E.L = calloc(NLAYER + 1, sizeof(q38d_layer));
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
    if (ssm_perm) {
        for (int l = 0; l < NLAYER; l++)
            if (E.L[l].ssm && !ssm_perm_ok(&E.L[l])) { ssm_perm = 0; break; }
        if (!ssm_perm) fprintf(stderr, "q38d: SSM in-proj layout not permutable, ssm_perm off\n");
        for (int l = 0; ssm_perm && l < NLAYER; l++)
            if (E.L[l].ssm) ssm_permute_layer(&E.L[l], l);
    }
    if (mtp_on) {
        int l = NLAYER;
        q38d_layer *L = &E.L[l];
        L->ssm = 0;
        L->ai = NATTN;
        L->attn_norm = vec_ptr(need_tensor("blk.%d.attn_norm.weight", l), EMBD);
        L->post_norm = vec_ptr(need_tensor("blk.%d.post_attention_norm.weight", l), EMBD);
        describe_f4 = draft_f4;
        describe(&L->q, need_tensor("blk.%d.attn_q.weight", l));
        describe(&L->k, need_tensor("blk.%d.attn_k.weight", l));
        describe(&L->v, need_tensor("blk.%d.attn_v.weight", l));
        describe(&L->o, need_tensor("blk.%d.attn_output.weight", l));
        describe_f4 = 0;
        L->o.unit16 = 1;
        L->o.kch = 0;
        load_vec(L->q_norm, need_tensor("blk.%d.attn_q_norm.weight", l), HD);
        load_vec(L->k_norm, need_tensor("blk.%d.attn_k_norm.weight", l), HD);
        describe(&L->gate, need_tensor("blk.%d.ffn_gate.weight", l));
        describe(&L->up, need_tensor("blk.%d.ffn_up.weight", l));
        describe(&L->down, need_tensor("blk.%d.ffn_down.weight", l));
        L->down.unit16 = 1;
        L->down.kch = 4;
        if (L->down.cols % (L->down.kch * 256)) L->down.kch = 1;
        if (L->down.kch > 1) L->down.unit16 = 0;
        describe_f4 = draft_f4;
        describe(&E.eh, need_tensor("blk.%d.nextn.eh_proj.weight", l));
        describe_f4 = 0;
        E.enorm = vec_ptr(need_tensor("blk.%d.nextn.enorm.weight", l), EMBD);
        E.hnorm = vec_ptr(need_tensor("blk.%d.nextn.hnorm.weight", l), EMBD);
        E.shnorm = vec_ptr(need_tensor("blk.%d.nextn.shared_head_norm.weight", l), EMBD);
    }
    describe(&E.head, need_tensor("output.weight", 0));
    if (mtp_on && draft_f4) {
        /* FP4 draft head over the first draft_v vocabulary rows (all if 0) */
        describe_f4 = 1;
        describe_rows = draft_v ? draft_v : E.n_vocab;
        describe(&E.dhead, need_tensor("output.weight", 0));
        describe_rows = 0;
        describe_f4 = 0;
    }
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
static q38d_plan *plan_ssm, *plan_att, *plan_ssm_mt, *plan_att_mt;
static int use_plan = 1;

/* Estimated cycles of one group (measured A16 cycles per pair, L1). */
static double cost_q4k = 50, cost_q6k = 70, cost_q8k = 40, cost_f6 = 29;
/* cost multipliers for the plans of a T-token verification pass: FP4 goes
 * through the multi-token kernels, K-quant matrices run T single passes */
static double mt_cost_f4 = 1, mt_cost_kq = 1;
static double group_cost(const q38d_mat *m) {
    double cpp = m->fmt == Q38D_F4 ? 18.3 : m->fmt == Q38D_F6 ? cost_f6 : m->fmt == Q38D_Q6K ? cost_q6k :
                 m->fmt == Q38D_Q8K ? cost_q8k : cost_q4k;
    return cpp * (m->cols / 32) * (m->fmt == Q38D_F4 ? mt_cost_f4 : mt_cost_kq);
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
static int norm_cmg = 1, norm_pf = 1;
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
        if (norm_pf)
            for (int o = 32 * p0 * 4; o < 32 * p1 * 4; o += 256) __builtin_prefetch((const char *)E.x + o, 0, 3);
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

static q38d_io io_single_v;
static const q38d_io *io_single(void) {
    io_single_v = (q38d_io){E.qkv, E.zb, E.ab, E.bb, E.qg, E.kb, E.vb, E.o, &E.act_o, -1};
    return &io_single_v;
}
static int conv_local = 1;
static void ssm_prefetch_state(int layer, int h);
/* part: 0 q, 1 k, 2 v of head h; channels start at ch0 in the qkv vector */
/* history ring of 8 positions so that rejected speculative positions never
 * overwrite the three accepted ones the next token reads */
static void conv_silu_local(int layer, int h, int part, int ch0, int pos, float *out, const float *qkv) {
    float *hs = E.conv_hist[layer][h];
    const float *w = E.conv_wl[layer][h] + part * 128;
    const float *in = qkv + ch0;
    const float *h1 = hs + (size_t)((pos + 7) & 7) * 384 + part * 128;
    const float *h2 = hs + (size_t)((pos + 6) & 7) * 384 + part * 128;
    const float *h3 = hs + (size_t)((pos + 5) & 7) * 384 + part * 128;
    const svbool_t pf = svptrue_b32();
    for (int j = 0; j < 128; j += 16) {
        svfloat32_t v = svmul_f32_x(pf, svld1_f32(pf, w + j), svld1_f32(pf, h3 + j));
        v = svmla_f32_x(pf, v, svld1_f32(pf, w + 384 + j), svld1_f32(pf, h2 + j));
        v = svmla_f32_x(pf, v, svld1_f32(pf, w + 768 + j), svld1_f32(pf, h1 + j));
        v = svmla_f32_x(pf, v, svld1_f32(pf, w + 1152 + j), svld1_f32(pf, in + j));
        svst1_f32(pf, out + j, svmul_f32_x(pf, v, q38d_sigmoid_sve(pf, v)));
    }
    memcpy(hs + (size_t)(pos & 7) * 384 + part * 128, in, 128 * sizeof(float));
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
static int ssm_lazy = 1, ssm_rowpf = 8;
static void ssm_finish(const q38d_layer *L, int h, const float *o, const q38d_io *io);
static void ssm_head_io(int layer, int h, int pos, const q38d_io *io);
static const q38d_io *io_single(void);
static void ssm_head(int layer, int h, int pos) { ssm_head_io(layer, h, pos, io_single()); }
static void ssm_head_io(int layer, int h, int pos, const q38d_io *io) {
    uint64_t T0 = h ? 0 : ticks();
    const q38d_layer *L = &E.L[layer];
    int g = h % NGROUP;
    float q[128], k[128], v[128], o[128];
    if (ssm_pf == 3) ssm_prefetch_state(layer, h);
    if (conv_local) {
        conv_silu_local(layer, h, 0, ssm_qoff(g), pos, q, io->qkv);
        conv_silu_local(layer, h, 1, ssm_koff(g), pos, k, io->qkv);
        conv_silu_local(layer, h, 2, ssm_voff(h), pos, v, io->qkv);
    } else {
        conv_silu(L, layer, g * DS, pos, h < NGROUP, q);
        conv_silu(L, layer, NGROUP * DS + g * DS, pos, h < NGROUP, k);
        conv_silu(L, layer, 2 * NGROUP * DS + h * DS, pos, 1, v);
    }
    l2norm128(q); l2norm128(k);
    const float qs = 1.0f / sqrtf((float)DS);
    for (int i = 0; i < 128; i++) q[i] *= qs;
    float val = io->ab[ssm_abi(h)] + L->dt_bias[h];
    float sp = val > 20.0f ? val : logf(1.0f + expf(val));
    float decay = expf(sp * L->ssm_a[h]);
    float beta = 1.0f / (1.0f + expf(-io->bb[ssm_abi(h)]));
    float *St = ssm_st(layer, h, io->slot < 0 ? 0 : io->slot);
    float *Sd = io->slot < 0 ? St : ssm_st(layer, h, io->slot + 1);
    const svbool_t pf = svptrue_b32();
    uint64_t T1 = h ? 0 : ticks();
    if (ssm_lazy) {
        /* One sweep: the stored state is A = decay S_prev without the last
         * rank-1 update, kept as (pk, pd) after the matrix. Materialize
         * S_prev = A + pd pk^T row by row, decay it, and accumulate A k and
         * A q; then o = A q + delta (k . q). */
        float *pk = St + DS * DS, *pd = pk + DS;
        float *pkd = Sd + DS * DS, *pdd = pkd + DS;
        SV8(sk); SV8(oa); SV8(pv);
#define LZ0(j) sk##j = svdup_n_f32(0); oa##j = sk##j; pv##j = svld1_f32(pf, pd + 16 * j)
        SV8_EACH(LZ0);
        for (int c = 0; c < DS; c++) {
            float *row = St + (size_t)c * DS, *drow = Sd + (size_t)c * DS;
            if (ssm_rowpf && c + ssm_rowpf < DS) {
                __builtin_prefetch(row + ssm_rowpf * DS, 1, 3);
                __builtin_prefetch(row + ssm_rowpf * DS + 64, 1, 3);
            }
            svfloat32_t pc = svdup_n_f32(pk[c]), kc = svdup_n_f32(k[c]), qc = svdup_n_f32(q[c]);
#define LZ1(j) { svfloat32_t s_ = svmul_n_f32_x(pf, svmla_f32_x(pf, svld1_f32(pf, row + 16 * j), pv##j, pc), decay); \
                svst1_f32(pf, drow + 16 * j, s_); sk##j = svmla_f32_x(pf, sk##j, s_, kc); oa##j = svmla_f32_x(pf, oa##j, s_, qc); }
            SV8_EACH(LZ1);
        }
        uint64_t T2l = h ? 0 : ticks();
        float kq = dot(k, q, DS);
#define LZ2(j) { svfloat32_t d_ = svmul_n_f32_x(pf, svsub_f32_x(pf, svld1_f32(pf, v + 16 * j), sk##j), beta); \
                svst1_f32(pf, pdd + 16 * j, d_); svst1_f32(pf, o + 16 * j, svmla_n_f32_x(pf, oa##j, d_, kq)); }
        SV8_EACH(LZ2);
        memcpy(pkd, k, DS * sizeof(float));
        ssm_finish(L, h, o, io);
        if (!h) { uint64_t T4 = ticks(); ssm_t[0] += T1 - T0; ssm_t[1] += T2l - T1; ssm_t[3] += T4 - T2l; }
        return;
    }
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
    ssm_finish(L, h, o, io);
    if (!h) { uint64_t T4 = ticks(); ssm_t[0] += T1 - T0; ssm_t[1] += T2 - T1; ssm_t[2] += T3 - T2; ssm_t[3] += T4 - T3; }
}
/* gated RMSNorm of head h's output into E.o, quantized for the out-proj */
static void ssm_finish(const q38d_layer *L, int h, const float *o, const q38d_io *io) {
    const svbool_t pf = svptrue_b32();
    float inv = 1.0f / sqrtf(sumsq(o, DS) / DS + E.eps);
    const float *z = io->zb + ssm_zoff(h);
    float *dst = io->o + (size_t)h * DS;
    for (int i = 0; i < DS; i += 16) {
        svfloat32_t zv = svld1_f32(pf, z + i);
        svfloat32_t sz = svmul_f32_x(pf, zv, q38d_sigmoid_sve(pf, zv));
        svfloat32_t ov = svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, o + i), inv), svld1_f32(pf, L->ssm_norm + i));
        svst1_f32(pf, dst + i, svmul_f32_x(pf, ov, sz));
    }
    if (E.arith != Q38D_F32) q38d_prepare_range(io->act_o, io->o, h * 4, h * 4 + 4);
}
static void ssm_prefetch_state(int layer, int h) {
    const char *p = (const char *)ssm_st(layer, h, 0);
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
static void attention_io(int layer, int tid, int pos, int *csense, const q38d_io *io);
static void attention(int layer, int tid, int pos, int *csense) { attention_io(layer, tid, pos, csense, io_single()); }
static void attention_io(int layer, int tid, int pos, int *csense, const q38d_io *io) {
    uint64_t T0 = tid ? 0 : ticks();
    const q38d_layer *L = &E.L[layer];
    int c = tid / PER, l = tid % PER, ai = L->ai;
    float *qh = E.qh[c];
    if (l < 6) {
        int hq = 6 * c + l;
        memcpy(qh + l * HD, io->qg + (size_t)hq * 2 * HD, HD * sizeof(float));
        head_rmsnorm_rope(qh + l * HD, L->q_norm, pos);
    } else if (l == 6) {
        float kv[HD];
        memcpy(kv, io->kb + c * HD, sizeof kv);
        head_rmsnorm_rope(kv, L->k_norm, pos);
        memcpy(E.kc[ai][c] + (size_t)pos * HD, kv, sizeof kv);
        memcpy(E.vc[ai][c] + (size_t)pos * HD, io->vb + c * HD, HD * sizeof(float));
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
        const float *gate = io->qg + (size_t)hq * 2 * HD + HD;
        float *dst = io->o + (size_t)hq * HD;
        float rden = 1.0f / den;
        for (int j = 0; j < HD; j += 16) {
            svfloat32_t gv = q38d_sigmoid_sve(pf, svld1_f32(pf, gate + j));
            svst1_f32(pf, dst + j, svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, o + j), rden), gv));
        }
        if (E.arith != Q38D_F32) q38d_prepare_range(io->act_o, io->o, hq * 8, hq * 8 + 8);
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

/* Phase end with a CMG barrier: the next phase reads only CMG-local data. */
static int inproj_cbar = 1, attn_cmg_ok = 1;
static void phase_end_c(int tid, int *cs, uint64_t *t, int phase) {
    pf_next(tid, pf_layer[tid], phase);
    cur_phase[tid] = phase;
    uint64_t t0 = ticks();
    if (busy_on) busy_acc[tid][phase] += (double)(t0 - busy_start[tid]);
    cbarrier(tid, cs);
    uint64_t t1 = ticks();
    if (!tid) kprof_wait_acc += (double)(t1 - t0);
    busy_start[tid] = t1;
    prof_mark(tid, t, phase);
}

/* experiment: A8 activations for the FFN gate/up input / down input */
static int ffn_a8, down_a8;
static void layer_body(int tid, int layer, int pos, int *gs, int *cs, uint64_t *tp) {
#define t (*tp)
        if (ffn_a8 && E.arith == Q38D_A16) E.act_c[tid / PER].arith = Q38D_A16;
        const q38d_layer *L = &E.L[layer];
        pf_layer[tid] = layer;
        const q38d_act *a;
        float omul = 1.0f;
        if (prod_norm && layer > 0) { int cc = prod_copies > 1 ? tid / PER : 0; a = &E.act_c[cc]; E.act_c[cc].x = E.xn_c[cc]; omul = x_inv(); }
        else a = norm_act(tid, L->attn_norm);
        if (L->ssm) {
            int hh = ssm_head_of(tid);
            if (ssm_pf == 1) ssm_prefetch_state(layer, hh);
            if (use_plan) run_plan(&plan_ssm[layer], a, tid, omul);
            else { mv(&L->qkv, a, E.qkv, 0, tid); mv(&L->z, a, E.zb, 0, tid);
                   mv(&L->alpha, a, E.ab, 0, tid); mv(&L->beta, a, E.bb, 0, tid); }
            if (inproj_cbar && ssm_perm && use_plan) phase_end_c(tid, cs, &t, P_SSM_IN);
            else phase_end(tid, gs, &t, P_SSM_IN);
            if (ssm_pf == 2) ssm_prefetch_state(layer, hh);
            ssm_head(layer, hh, pos);
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
            if (inproj_cbar && attn_cmg_ok && use_plan) phase_end_c(tid, cs, &t, P_ATT_IN);
            else phase_end(tid, gs, &t, P_ATT_IN);
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
        if (ffn_a8 && E.arith == Q38D_A16) E.act_c[tid / PER].arith = Q38D_A8;
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
    #undef t
}

/* final norm (weight nw), output head and global argmax -> E.next_token */
static void head_argmax_m(int tid, const float *nw, const q38d_mat *hm, int *gs, uint64_t *tp);
static void head_argmax(int tid, const float *nw, int *gs, uint64_t *tp) { head_argmax_m(tid, nw, &E.head, gs, tp); }
static void head_argmax_m(int tid, const float *nw, const q38d_mat *hm, int *gs, uint64_t *tp) {
#define t (*tp)
    const q38d_act *a;
    float omul = 1.0f;
    if (prod_norm) { int cc = prod_copies > 1 ? tid / PER : 0; a = &E.act_c[cc]; E.act_c[cc].x = E.xn_c[cc]; omul = x_inv(); }
    else a = norm_act(tid, nw);
    mv(&(*hm), a, E.logits, 0, tid);
    if (omul != 1.0f) {
        int c, g0, g1;
        mat_range(&(*hm), tid, &c, &g0, &g1);
        int r0 = (*hm).first[c] + 8 * g0, r1 = (*hm).first[c] + 8 * g1;
        if (r1 > (*hm).first[c + 1]) r1 = (*hm).first[c + 1];
        if (r1 > r0) scale_rows(E.logits + r0, r1 - r0, omul);
    }
    {
        int c, g0, g1;
        mat_range(&(*hm), tid, &c, &g0, &g1);
        int r0 = (*hm).first[c] + 8 * g0, r1 = (*hm).first[c] + 8 * g1;
        if (r1 > (*hm).first[c + 1]) r1 = (*hm).first[c + 1];
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
#undef t
}

static void step(int tid, int token, int pos, int want_head, int *gs, int *cs) {
    uint64_t t = ticks();
    pf_pos = pos;
    ssq_valid = 0;
    embed_token(tid, token);
    phase_end(tid, gs, &t, P_EMBED);
    for (int layer = 0; layer < NLAYER; layer++) layer_body(tid, layer, pos, gs, cs, &t);
    if (!want_head) return;
    head_argmax(tid, E.out_norm, gs, &t);
}

/* NextN/MTP drafter: from hidden h (main final residual at position pos, or
 * the drafter's previous output) and the embedding of `token` (= x_{pos+1}),
 * one attention layer at position pos predicts x_{pos+2} (E.mtp_draft when
 * want_head). The drafter's output hidden is kept in E.h_mtp for chaining. */
static void embed_token_to(int tid, int token, float *dst);
static void rmsnorm_to(float *dst, const float *x, const float *w) {
    double ss = 0;
    for (int i = 0; i < EMBD; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / EMBD) + E.eps);
    for (int i = 0; i < EMBD; i++) dst[i] = x[i] * inv * w[i];
}
static void mtp_step(int tid, int token, int pos, const float *hidden, int want_head, int *gs, int *cs) {
    uint64_t t = ticks();
    static float *save_x;
    static int save_next;
    /* input [enorm(emb(token)); hnorm(hidden)], prepared by all workers:
     * embedding blocks, then every worker takes both norms' sums of squares
     * (fixed order) and quantizes its share of the 320 pairs */
    embed_token_to(tid, token, E.x_mtp);
    if (tid == 0) { save_x = E.x; save_next = E.next_token; }
    gbarrier(tid, gs);
    {
        float inv_e = 1.0f / sqrtf(sumsq(E.x_mtp, EMBD) / EMBD + E.eps);
        float inv_h = 1.0f / sqrtf(sumsq(hidden, EMBD) / EMBD + E.eps);
        int np = EMBD / 32, p0 = 2 * np * tid / NT, p1 = 2 * np * (tid + 1) / NT;
        if (E.arith == Q38D_F32) {
            for (int i = 32 * p0; i < 32 * p1; i++)
                E.mtp_in[i] = i < EMBD ? E.x_mtp[i] * inv_e * E.enorm[i] : hidden[i - EMBD] * inv_h * E.hnorm[i - EMBD];
        } else {
            /* pairs [p0, p1) of the 10240-column activation, in two halves */
            q38d_act *a = &E.act_mtp;
            if (p0 < np) {
                q38d_act ae = *a;
                ae.cols = EMBD;
                q38d_prepare_pairs(&ae, E.x_mtp, inv_e, E.enorm, p0, p1 < np ? p1 : np);
            }
            if (p1 > np) {
                q38d_act ah = *a;
                ah.cols = EMBD;
                ah.q = a->q + (size_t)np * 64; ah.sc = a->sc + 2 * np; ah.sum = a->sum + 2 * np;
                q38d_prepare_pairs(&ah, hidden, inv_h, E.hnorm, p0 > np ? p0 - np : 0, p1 - np);
            }
        }
    }
    gbarrier(tid, gs);
    mv(&E.eh, &E.act_mtp, E.x_mtp, 0, tid);
    gbarrier(tid, gs);
    if (tid == 0) E.x = E.x_mtp;
    ssq_valid = 0;
    gbarrier(tid, gs);
    layer_body(tid, NLAYER, pos, gs, cs, &t);
    if (tid == 0) memcpy(E.h_mtp, E.x_mtp, EMBD * sizeof(float));
    if (want_head) head_argmax_m(tid, E.shnorm, draft_v || draft_f4 ? &E.dhead : &E.head, gs, &t);
    gbarrier(tid, gs);
    if (tid == 0) {
        E.x = save_x;
        if (want_head) E.mtp_draft = E.next_token;
        E.next_token = save_next;
    }
    ssq_valid = 0;
    gbarrier(tid, gs);
}
/* drafts[i][k]: depth-k+1 draft for x_{pn-1+i+2+k}, made after x_{pn+i} is known */
static int (*mtp_chain)[3];
static void mtp_drafts(int tid, int i, int pos, int *gs, int *cs) {
    int tok = E.next_token;
    mtp_step(tid, tok, pos, E.x, 1, gs, cs);
    int d1 = E.mtp_draft;
    mtp_step(tid, d1, pos + 1, E.h_mtp, 1, gs, cs);
    int d2 = E.mtp_draft;
    mtp_step(tid, d2, pos + 2, E.h_mtp, 1, gs, cs);
    if (tid == 0) { mtp_chain[i][0] = d1; mtp_chain[i][1] = d2; mtp_chain[i][2] = E.mtp_draft; }
}

/* Projection-only verification benchmark (Q38D_VBENCH=T, T = 1, 2, 4):
 * every layer's FP4 projections for T tokens with the engine's partitions
 * and barrier structure; SSM/attention cores are skipped, K-quant matrices
 * run T single-token passes. Results go to scratch. */
static int vbench_T;
void q38d_asmg2s2_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *, const float *, long, float *, const int8_t *);
void q38d_asmg2s4_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *, const float *, long, float *, const int8_t *);
void q38d_asmn4_f4_a16(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *, const float *, long, float *, const int8_t *);
static int8_t *vb_q[NCMG][2];   /* interleaved activations: [0] EMBD/DINNER cols, [1] NFF cols */
static float *vb_s[NCMG][2];
static void vb_pair(int T, const uint8_t *wa, const uint8_t *wb, int np, const int8_t *aq, const float *as, float *acc) {
    if (T == 1) q38d_asm8_f4_a16(wa, wb, wa + (size_t)np * 128, aq, as, np, acc, q38d_lut_f4, wb + (size_t)np * 128);
    else (T == 4 ? q38d_asmg2s4_f4_a16 : q38d_asmg2s2_f4_a16)(wa, (long)(wb - wa), wa + (size_t)np * 128, aq, as, np, acc, q38d_lut_f4);
}
static void vb_one(int T, const uint8_t *w, int np, const int8_t *aq, const float *as, float *acc) {
    if (T == 4) q38d_asmn4_f4_a16(w, NULL, w + (size_t)np * 128, aq, as, np, acc, q38d_lut_f4);
    else vb_pair(T, w, w, np, aq, as, acc);
}
/* groups [g0, g1) of one matrix (pairs of consecutive groups) or of two
 * matrices side by side (dual) */
static void vb_groups(int T, const q38d_mat *m, const q38d_mat *m2, int c, int g0, int g1, int np,
                      const int8_t *aq, const float *as, float *acc) {
    size_t gb = q38d_group_bytes(m->fmt, np * 32);
    if (m2) {
        for (int g = g0; g < g1; g++) vb_pair(T, m->part[c] + (size_t)g * gb, m2->part[c] + (size_t)g * gb, np, aq, as, acc);
        return;
    }
    int g = g0;
    for (; g + 1 < g1; g += 2) vb_pair(T, m->part[c] + (size_t)g * gb, m->part[c] + (size_t)(g + 1) * gb, np, aq, as, acc);
    if (g < g1) vb_one(T, m->part[c] + (size_t)g * gb, np, aq, as, acc);
}
static void vb_kquant(int T, const q38d_mat *m, int tid, float *scratch) {
    int c, g0, g1;
    mat_range(m, tid, &c, &g0, &g1);
    const q38d_act *a = &E.act_c[c];
    for (int t = 0; t < T; t++)
        if (g0 < g1) q38d_gemv_any(scratch, m->part[c], m->fmt, a, g0, g1, m->first[c + 1] - m->first[c], 0);
}
static void vbench(int tid, int *gs, int *cs) {
    int T = vbench_T, c = tid / PER, l = tid % PER;
    float acc[2 * 4 * 16] __attribute__((aligned(256)));
    static float *scratch_all;
    if (tid == 0) scratch_all = aligned_alloc(256, (size_t)NT * NFF * sizeof(float));
    if (l == 0) {
        /* replicate this CMG's quantized x (and h) T times, interleaved per pair */
        const q38d_act *srcs[2] = {&E.act_c[c], &E.act_h};
        int cols[2] = {DINNER > EMBD ? DINNER : EMBD, NFF};
        for (int k = 0; k < 2; k++) {
            int np = cols[k] / 32, npa = srcs[k]->cols / 32;
            vb_q[c][k] = aligned_alloc(256, (size_t)np * T * 64);
            vb_s[c][k] = aligned_alloc(256, (size_t)np * T * 8);
            for (int p = 0; p < np; p++)
                for (int t = 0; t < T; t++) {
                    memcpy(vb_q[c][k] + ((size_t)p * T + t) * 64, srcs[k]->q + (size_t)(p % npa) * 64, 64);
                    memcpy(vb_s[c][k] + ((size_t)p * T + t) * 2, srcs[k]->sc + (size_t)(p % npa) * 2, 8);
                }
        }
    }
    gbarrier(tid, gs);
    float *scratch = scratch_all + (size_t)tid * NFF;
    int reps = 8;
    double t0 = now_sec();
    for (int rep = 0; rep < reps; rep++)
        for (int layer = 0; layer < NLAYER; layer++) {
            const q38d_layer *L = &E.L[layer];
            const int8_t *aq = vb_q[c][0];
            const float *as = vb_s[c][0];
            const q38d_tplan *tp = L->ssm ? &plan_ssm[layer].t[tid] : &plan_att[layer].t[tid];
            const q38d_plan *P = L->ssm ? &plan_ssm[layer] : &plan_att[layer];
            for (int k = 0; k < tp->nseg; k++) {
                const q38d_mat *m = P->m[tp->s[k].mi];
                const q38d_mat *m2 = tp->s[k].mi2 >= 0 ? P->m[tp->s[k].mi2] : NULL;
                if (m->fmt == Q38D_F4) vb_groups(T, m, m2, c, tp->s[k].g0, tp->s[k].g1, m->cols / 32, aq, as, acc);
                else for (int t = 0; t < T; t++)
                    q38d_gemv_any(scratch, m->part[c], m->fmt, &E.act_c[c], tp->s[k].g0, tp->s[k].g1, m->first[c + 1] - m->first[c], 0);
            }
            gbarrier(tid, gs);
            gbarrier(tid, gs);                              /* core phase */
            const q38d_mat *o = L->ssm ? &L->out : &L->o;
            { int cc, g0, g1; mat_range(o, tid, &cc, &g0, &g1); if (g0 < g1) vb_groups(T, o, NULL, c, g0, g1, o->cols / 32, aq, as, acc); }
            gbarrier(tid, gs);
            {   /* FFN gate/up: 8-row partition, dual */
                int rows = L->gate.first[c + 1] - L->gate.first[c], G8 = rows / 8;
                vb_groups(T, &L->gate, &L->up, c, G8 * l / PER, G8 * (l + 1) / PER, L->gate.cols / 32, aq, as, acc);
            }
            gbarrier(tid, gs);
            {   /* down: K-chunked items, chunk-major per worker */
                const q38d_mat *m = &L->down;
                int i0, i1;
                lane_items(m, c, l, &i0, &i1);
                if (i1 > i0) {
                    int kc = m->cols / m->kch, npc = kc / 32;
                    size_t sb = q38d_group_bytes(m->fmt, kc);
                    const uint8_t *w = m->part[c] + (size_t)i0 * sb;
                    int gf = i0 / m->kch, gl = (i1 - 1) / m->kch;
                    for (int kk = 0; kk < m->kch; kk++) {
                        int n = 0;
                        for (int g = gf; g <= gl; g++) { int it = g * m->kch + kk; if (it >= i0 && it < i1) n++; }
                        const int8_t *aqk = vb_q[c][1] + (size_t)kk * npc * T * 64;
                        const float *ask = vb_s[c][1] + (size_t)kk * npc * T * 2;
                        int g = 0;
                        for (; g + 1 < n; g += 2) vb_pair(T, w + (size_t)g * sb, w + (size_t)(g + 1) * sb, npc, aqk, ask, acc);
                        if (g < n) vb_one(T, w + (size_t)g * sb, npc, aqk, ask, acc);
                        w += (size_t)n * sb;
                    }
                }
            }
            gbarrier(tid, gs);
        }
    if (tid == 0) {
        double ms = (now_sec() - t0) * 1e3 / reps;
        fprintf(stderr, "q38d: vbench T=%d: %.2f ms per pass (projections + 5 barriers/layer), %.2f ms per token\n", T, ms, ms / T);
    }
    gbarrier(tid, gs);
    (void)cs; (void)vb_kquant;
}

/* ------------------------------------------------------------------ */
/* speculative decoding: T-token verification pass (Q38D_SPEC=k, T = k + 1) */

static int spec_k;
typedef struct {
    float *x[TMAX], *qkv[TMAX], *zb[TMAX], *ab[TMAX], *bb[TMAX], *qg[TMAX], *kb[TMAX], *vb[TMAX], *o[TMAX], *h[TMAX];
    q38d_act act_o[TMAX], act_h[TMAX];
    q38d_act act_c[NCMG][TMAX];
    _Atomic int *unit_cnt[TMAX];
    int argmax[TMAX];
    float logit[TMAX];
} q38d_mtbuf;
static q38d_mtbuf MT;
static float ssp_part[NCMG][PER][TMAX] __attribute__((aligned(256)));
typedef void (*g2p_fn)(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *,
                       const int8_t *);
void q38d_asmg2p2_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *, const int8_t *);
void q38d_asmg2p3_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *, const int8_t *);
void q38d_asmg2p4_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *, const int8_t *);
void q38d_asmg2j3_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *, const int8_t *);
void q38d_asmg2j4_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *, const int8_t *);
static int g2_variant = 1;   /* 1: decode-interleaved kernels for T = 3, 4 */
static g2p_fn g2p_for(int T) {
    if (g2_variant && T == 3) return q38d_asmg2j3_f4_a16;
    if (g2_variant && T == 4) return q38d_asmg2j4_f4_a16;
    return T == 2 ? q38d_asmg2p2_f4_a16 : T == 3 ? q38d_asmg2p3_f4_a16 : q38d_asmg2p4_f4_a16;
}
typedef void (*g2q_fn)(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *,
                       const float *);
void q38d_asmg2p2_q8k_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *, const float *);
void q38d_asmg2p3_q8k_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *, const float *);
void q38d_asmg2p4_q8k_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *, const float *);
static g2q_fn g2q_for(int T) { return T == 2 ? q38d_asmg2p2_q8k_a16 : T == 3 ? q38d_asmg2p3_q8k_a16 : q38d_asmg2p4_q8k_a16; }
static float *mt_logits[TMAX];
static float mt_best_val[NT][TMAX];
static int mt_best_idx[NT][TMAX];

static inline void mt_out(float *o, const float *acc, int mode) {
    for (int r = 0; r < 8; r++) {
        float v = (acc[2 * r] + acc[2 * r + 1]) * 0x1p45f;
        o[r] = mode ? o[r] + v : v;
    }
}
/* Groups [g0, g1) of CMG c for T tokens: dual (m2, same group index) or
 * consecutive group pairs of one matrix. outA/outB[t]: row 0 of the CMG. */
static void mt_mat(int T, const q38d_mat *m, const q38d_mat *m2, int c, int g0, int g1, const q38d_act *acts,
                   float *const *outA, float *const *outB, int mode) {
    if (g0 >= g1) return;
    if (m->fmt != Q38D_F4 || T < 2 || E.arith != Q38D_A16 || (m2 && m2->fmt != Q38D_F4)) {
        for (int t = 0; t < T; t++) {
            q38d_gemv_any(outA[t], m->part[c], m->fmt, &acts[t], g0, g1, m->first[c + 1] - m->first[c], mode);
            if (m2) q38d_gemv_any(outB[t], m2->part[c], m2->fmt, &acts[t], g0, g1, m2->first[c + 1] - m2->first[c], mode);
        }
        return;
    }
    const int np = m->cols / 32;
    const size_t gb = q38d_group_bytes(Q38D_F4, m->cols);
    const int8_t *aq[TMAX];
    const float *as[TMAX];
    for (int t = 0; t < T; t++) { aq[t] = acts[t].q; as[t] = acts[t].sc; }
    float acc[2 * TMAX * 16] __attribute__((aligned(256)));
    g2p_fn f = g2p_for(T);
    for (int g = g0; g < g1;) {
        const uint8_t *wa = m->part[c] + (size_t)g * gb;
        int gA = g, gB;
        float *const *oB;
        long gs;
        if (m2) { gs = (long)(m2->part[c] - m->part[c]); gB = g; oB = outB; g += 1; }
        else if (g + 1 < g1) { gs = (long)gb; gB = g + 1; oB = outA; g += 2; }
        else { gs = 0; gB = -1; oB = NULL; g += 1; }
        f(wa, gs, wa + (size_t)np * 128, aq, as, np, acc, q38d_lut_f4);
        for (int t = 0; t < T; t++) {
            mt_out(outA[t] + 8 * gA, acc + 16 * t, mode);
            if (gB >= 0) mt_out(oB[t] + 8 * gB, acc + 16 * (T + t), mode);
        }
    }
}
static void run_plan_mt(const q38d_plan *P, int T, const q38d_act *acts, float *(*outs)[TMAX], int tid) {
    int c = tid / PER;
    const q38d_tplan *tp = &P->t[tid];
    for (int k = 0; k < tp->nseg; k++) {
        int mi = tp->s[k].mi, mi2 = tp->s[k].mi2;
        const q38d_mat *m = P->m[mi], *m2 = mi2 >= 0 ? P->m[mi2] : NULL;
        float *oa[TMAX], *ob[TMAX];
        for (int t = 0; t < T; t++) {
            oa[t] = outs[mi][t] + m->first[c];
            ob[t] = m2 ? outs[mi2][t] + m2->first[c] : NULL;
        }
        mt_mat(T, m, m2, c, tp->s[k].g0, tp->s[k].g1, acts, oa, m2 ? ob : NULL, 0);
    }
}
/* RMSNorm of xs[0..T) with weight w into this CMG's act_c[c][t]: lane l sums
 * and quantizes its 1/12 of the pairs; two CMG barriers. */
static double nm_t[4];
static void norm_mt(int tid, float *const *xs, int T, const float *w, int *cs) {
    int c = tid / PER, l = tid % PER, np = EMBD / 32, p0 = np * l / PER, p1 = np * (l + 1) / PER;
    /* the slices were just written on other CMGs: issue all line fetches first */
    if (norm_pf)
        for (int t = 0; t < T; t++)
            for (int o = 32 * p0 * 4; o < 32 * p1 * 4; o += 256) __builtin_prefetch((const char *)xs[t] + o, 0, 3);
    uint64_t n0 = tid ? 0 : ticks();
    for (int t = 0; t < T; t++) ssp_part[c][l][t] = sumsq(xs[t] + 32 * p0, 32 * (p1 - p0));
    uint64_t n1 = tid ? 0 : ticks();
    cbarrier(tid, cs);
    uint64_t n2 = tid ? 0 : ticks();
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int j = 0; j < PER; j++) ss += ssp_part[c][j][t];
        q38d_prepare_pairs(&MT.act_c[c][t], xs[t], 1.0f / sqrtf(ss / EMBD + E.eps), w, p0, p1);
    }
    uint64_t n3 = tid ? 0 : ticks();
    cbarrier(tid, cs);
    if (!tid) { nm_t[0] += n1 - n0; nm_t[1] += n2 - n1; nm_t[2] += n3 - n2; nm_t[3] += ticks() - n3; }
}
typedef struct { float part[2][TMAX][8]; _Atomic int cnt; char pad[252]; } q38d_split_mt;
static q38d_split_mt split_mt[NCMG][PER + 1];
/* K-chunked residual matvec for T tokens (add into MT.x[t]) */
static void mv_items_mt(const q38d_mat *m, int T, int tid, float *const *xs) {
    int c = tid / PER, l = tid % PER, i0, i1;
    lane_items(m, c, l, &i0, &i1);
    if (i1 <= i0) return;
    const int gf = i0 / m->kch, gl = (i1 - 1) / m->kch, kc = m->cols / m->kch, npc = kc / 32;
    const size_t sb = q38d_group_bytes(m->fmt, kc);
    const uint8_t *w = m->part[c] + (size_t)i0 * sb;
    float accg[64][TMAX][8];
    memset(accg, 0, sizeof(float) * TMAX * 8 * (gl - gf + 1));
    float acc[2 * TMAX * 16] __attribute__((aligned(256)));
    const int Tk = T < 2 ? 2 : T;   /* one token: two-token kernel on a duplicated pointer */
    g2p_fn f = g2p_for(Tk);
    for (int k = 0; k < m->kch; k++) {
        int ga = -1, n = 0;
        for (int g = gf; g <= gl; g++) { int it = g * m->kch + k; if (it >= i0 && it < i1) { if (ga < 0) ga = g; n++; } }
        if (!n) continue;
        const int8_t *aq[TMAX];
        const float *as[TMAX];
        for (int t = 0; t < T; t++) { aq[t] = MT.act_h[t].q + (size_t)k * npc * 64; as[t] = MT.act_h[t].sc + (size_t)k * npc * 2; }
        if (T < 2) { aq[1] = aq[0]; as[1] = as[0]; }
        for (int j = 0; j < n;) {
            int two = j + 1 < n;
            f(w, two ? (long)sb : 0, w + (size_t)npc * 128, aq, as, npc, acc, q38d_lut_f4);
            for (int t = 0; t < T; t++)
                for (int r = 0; r < 8; r++) {
                    accg[ga + j - gf][t][r] += (acc[16 * t + 2 * r] + acc[16 * t + 2 * r + 1]) * 0x1p45f;
                    if (two) accg[ga + j + 1 - gf][t][r] += (acc[16 * (Tk + t) + 2 * r] + acc[16 * (Tk + t) + 2 * r + 1]) * 0x1p45f;
                }
            w += (size_t)(two ? 2 : 1) * sb;
            j += two ? 2 : 1;
        }
    }
    for (int g = gf; g <= gl; g++) {
        int full = g * m->kch >= i0 && g * m->kch + m->kch - 1 < i1;
        int row = m->first[c] + 8 * g;
        if (full) {
            for (int t = 0; t < T; t++) for (int r = 0; r < 8; r++) xs[t][row + r] += accg[g - gf][t][r];
            continue;
        }
        int b = g * m->kch < i0 ? l : l + 1, side = g * m->kch < i0 ? 1 : 0;
        q38d_split_mt *sp = &split_mt[c][b];
        memcpy(sp->part[side], accg[g - gf], sizeof(float) * TMAX * 8);
        int old = atomic_fetch_add_explicit(&sp->cnt, 1, memory_order_acq_rel);
        if (old & 1)
            for (int t = 0; t < T; t++)
                for (int r = 0; r < 8; r++) xs[t][row + r] += sp->part[0][t][r] + sp->part[1][t][r];
    }
}
static void embed_token_to(int tid, int token, float *dst) {
    if (E.embed_q6k) {
        int blocks = EMBD / 256;
        if (tid < blocks)
            dequant_row(GGML_TYPE_Q6_K, E.embed_q6k + (size_t)token * E.embed_row_bytes + (size_t)tid * 210, dst + tid * 256, 256);
    } else if (E.embed_lb->format == Q38_LB_FP6_E2M3) {
        int k0 = EMBD * tid / NT, k1 = EMBD * (tid + 1) / NT;
        embed_fp6_cols(dst, E.embed_lb, token, k0, k1);
    } else if (tid == 0) {
        q38_lowbit_matrix_row(dst, E.embed_lb, token);
    }
}
/* argmax of the head for residual x -> *id, *logit (all workers) */
static void head_token_m(int tid, float *x, const float *nw, const q38d_mat *hm, int *gs, int *cs, int *id, float *logit) {
    int c = tid / PER;
    float *xs[1] = {x};
    norm_mt(tid, xs, 1, nw, cs);
    mv(hm, &MT.act_c[c][0], E.logits, 0, tid);
    {
        int cc, g0, g1;
        mat_range(hm, tid, &cc, &g0, &g1);
        int r0 = hm->first[cc] + 8 * g0, r1 = hm->first[cc] + 8 * g1;
        if (r1 > hm->first[cc + 1]) r1 = hm->first[cc + 1];
        float best = -INFINITY; int bi = -1;
        for (int r = r0; r < r1; r++) if (E.logits[r] > best) { best = E.logits[r]; bi = r; }
        E.best_val[tid] = best; E.best_idx[tid] = bi;
    }
    gbarrier(tid, gs);
    if (tid == 0) {
        float best = -INFINITY; int bi = 0;
        for (int i = 0; i < NT; i++) if (E.best_idx[i] >= 0 && E.best_val[i] > best) { best = E.best_val[i]; bi = E.best_idx[i]; }
        *id = bi; *logit = best;
    }
    gbarrier(tid, gs);
}
static void head_token(int tid, float *x, int *gs, int *cs, int *id, float *logit) {
    head_token_m(tid, x, E.out_norm, &E.head, gs, cs, id, logit);
}
/* output head for T tokens (Q8K multi-token kernel) -> MT.argmax/logit */
static void head_mt(int tid, int T, int *gs, int *cs) {
    const q38d_mat *m = &E.head;
    if (m->fmt != Q38D_Q8K) {
        for (int i = 0; i < T; i++) head_token(tid, MT.x[i], gs, cs, &MT.argmax[i], &MT.logit[i]);
        return;
    }
    int c, g0, g1;
    norm_mt(tid, MT.x, T, E.out_norm, cs);
    mat_range(m, tid, &c, &g0, &g1);
    const int np = m->cols / 32, Tk = T < 2 ? 2 : T;
    const size_t gb = q38d_group_bytes(m->fmt, m->cols);
    const int8_t *aq[TMAX];
    const float *as[TMAX];
    for (int t = 0; t < T; t++) { aq[t] = MT.act_c[c][t].q; as[t] = MT.act_c[c][t].sc; }
    if (T < 2) { aq[1] = aq[0]; as[1] = as[0]; }
    float acc[2 * TMAX * 16] __attribute__((aligned(256)));
    g2q_fn f = g2q_for(Tk);
    const float e = q38d_out_scale(Q38D_Q8K);
    for (int g = g0; g < g1;) {
        const uint8_t *w = m->part[c] + (size_t)g * gb;
        int two = g + 1 < g1;
        f(w, two ? (long)gb : 0, w + (size_t)np * 256, aq, as, np, acc, (const float *)(w + (size_t)np * 272));
        for (int t = 0; t < T; t++) {
            float *o = mt_logits[t] + m->first[c] + 8 * g;
            for (int r = 0; r < 8; r++) {
                o[r] = (acc[16 * t + 2 * r] + acc[16 * t + 2 * r + 1]) * e;
                if (two) o[8 + r] = (acc[16 * (Tk + t) + 2 * r] + acc[16 * (Tk + t) + 2 * r + 1]) * e;
            }
        }
        g += two ? 2 : 1;
    }
    int r0 = m->first[c] + 8 * g0, r1 = m->first[c] + 8 * g1;
    if (r1 > m->first[c + 1]) r1 = m->first[c + 1];
    for (int t = 0; t < T; t++) {
        float best = -INFINITY; int bi = -1;
        for (int r = r0; r < r1; r++) if (mt_logits[t][r] > best) { best = mt_logits[t][r]; bi = r; }
        mt_best_val[tid][t] = best; mt_best_idx[tid][t] = bi;
    }
    gbarrier(tid, gs);
    if (tid == 0)
        for (int t = 0; t < T; t++) {
            float best = -INFINITY; int bi = 0;
            for (int i = 0; i < NT; i++) if (mt_best_idx[i][t] >= 0 && mt_best_val[i][t] > best) { best = mt_best_val[i][t]; bi = mt_best_idx[i][t]; }
            MT.argmax[t] = bi; MT.logit[t] = best;
        }
    gbarrier(tid, gs);
}
static int head_multi = 1, mt_plans = 0;
/* worker-0 time split of speculative decoding (ticks): 0 norm, 1 proj,
 * 2 cores, 3 head, 4 phase barriers, 5 MTP catch-up + drafts */
static double sp_t[6];
#define SP_T(k, ...) do { uint64_t a_ = tid ? 0 : ticks(); __VA_ARGS__; if (!tid) sp_t[k] += (double)(ticks() - a_); } while (0)
/* Attention for T consecutive query tokens (positions pos0..pos0+T-1) of
 * one layer: every lane scores its position range for all queries (the K/V
 * rows come from HBM once, then from L2), and the 6*T (head, token) merges
 * are spread over the CMG's 12 lanes. */
static float *qh_mt[NCMG], *apart_mt[NCMG];
static int attn_multi = 1;
static void attention_mt(int layer, int tid, int pos0, int T, int *cs) {
    const q38d_layer *L = &E.L[layer];
    const int c = tid / PER, l = tid % PER, ai = L->ai;
    const size_t PS = 2 + HD;   /* partial: max, sum, o[HD] */
    for (int i = 0; i < T; i++) {
        if (l < 6) {
            float *q = qh_mt[c] + ((size_t)i * 6 + l) * HD;
            memcpy(q, MT.qg[i] + (size_t)(6 * c + l) * 2 * HD, HD * sizeof(float));
            head_rmsnorm_rope(q, L->q_norm, pos0 + i);
        } else if (l == 6) {
            float kv[HD];
            memcpy(kv, MT.kb[i] + c * HD, sizeof kv);
            head_rmsnorm_rope(kv, L->k_norm, pos0 + i);
            memcpy(E.kc[ai][c] + (size_t)(pos0 + i) * HD, kv, sizeof kv);
            memcpy(E.vc[ai][c] + (size_t)(pos0 + i) * HD, MT.vb[i] + c * HD, HD * sizeof(float));
        }
    }
    cbarrier(tid, cs);
    const int n = pos0 + T, t0 = (int)((int64_t)n * l / PER), t1 = (int)((int64_t)n * (l + 1) / PER);
    const float scale = 1.0f / 16.0f;
    const svbool_t pf = svptrue_b32();
    const float *K = E.kc[ai][c], *V = E.vc[ai][c];
    float sc[6 * (t1 - t0 > 0 ? t1 - t0 : 1)];
    for (int i = 0; i < T; i++) {
        int b = t1 < pos0 + i + 1 ? t1 : pos0 + i + 1, nt = b - t0;
        float *part = apart_mt[c] + ((size_t)i * PER + l) * 6 * PS;
        if (nt <= 0) {
            for (int hh = 0; hh < 6; hh++) { float *pp = part + hh * PS; pp[0] = -INFINITY; pp[1] = 0; memset(pp + 2, 0, HD * 4); }
            continue;
        }
        attn_scores(qh_mt[c] + (size_t)i * 6 * HD, K, t0, b, sc, nt, scale);
        for (int hh = 0; hh < 6; hh++) {
            float *s_h = sc + hh * nt, *pp = part + hh * PS;
            float m = -INFINITY;
            for (int j = 0; j < nt; j++) if (s_h[j] > m) m = s_h[j];
            svfloat32_t lsv = svdup_n_f32(0);
            for (int j = 0; j < nt; j += 16) {
                svbool_t pg = svwhilelt_b32(j, nt);
                svfloat32_t ev = q38d_exp_sve(pg, svsub_n_f32_x(pg, svld1_f32(pg, s_h + j), m));
                svst1_f32(pg, s_h + j, ev);
                lsv = svadd_f32_m(pg, lsv, ev);
            }
            pp[0] = m; pp[1] = svaddv_f32(pf, lsv);
            attn_pv(V, t0, b, s_h, pp + 2);
        }
    }
    cbarrier(tid, cs);
    for (int j = l; j < 6 * T; j += PER) {
        int i = j / 6, hh = j % 6, hq = 6 * c + hh;
        float M = -INFINITY;
        for (int k = 0; k < PER; k++) {
            float mk = apart_mt[c][((size_t)i * PER + k) * 6 * PS + hh * PS];
            if (mk > M) M = mk;
        }
        float o[HD], den = 0;
        memset(o, 0, sizeof o);
        for (int k = 0; k < PER; k++) {
            const float *pp = apart_mt[c] + ((size_t)i * PER + k) * 6 * PS + hh * PS;
            if (pp[1] == 0) continue;
            float w = q38d_expf(pp[0] - M);
            den += w * pp[1];
            for (int d = 0; d < HD; d += 16)
                svst1_f32(pf, o + d, svmla_n_f32_x(pf, svld1_f32(pf, o + d), svld1_f32(pf, pp + 2 + d), w));
        }
        const float *gate = MT.qg[i] + (size_t)hq * 2 * HD + HD;
        float *dst = MT.o[i] + (size_t)hq * HD;
        float rden = 1.0f / den;
        for (int d = 0; d < HD; d += 16) {
            svfloat32_t gv = q38d_sigmoid_sve(pf, svld1_f32(pf, gate + d));
            svst1_f32(pf, dst + d, svmul_f32_x(pf, svmul_n_f32_x(pf, svld1_f32(pf, o + d), rden), gv));
        }
        q38d_prepare_range(&MT.act_o[i], MT.o[i], hq * 8, hq * 8 + 8);
    }
}
/* one layer for T tokens with residuals xs[0..T) */
static void layer_mt(int tid, int layer, int pos0, int T, float *const *xs, int *gs, int *cs, uint64_t *tp) {
#define t (*tp)
    const int c = tid / PER, l = tid % PER;
        const q38d_layer *L = &E.L[layer];
        pf_layer[tid] = layer;
        SP_T(0, norm_mt(tid, xs, T, L->attn_norm, cs));
        if (L->ssm) {
            float *(outs[4])[TMAX];
            for (int i = 0; i < T; i++) { outs[0][i] = MT.qkv[i]; outs[1][i] = MT.zb[i]; outs[2][i] = MT.ab[i]; outs[3][i] = MT.bb[i]; }
            SP_T(1, run_plan_mt(plan_ssm_mt && mt_plans ? &plan_ssm_mt[layer] : &plan_ssm[layer], T, MT.act_c[c], outs, tid));
            if (inproj_cbar && ssm_perm) SP_T(4, phase_end_c(tid, cs, &t, P_SSM_IN));
            else phase_end(tid, gs, &t, P_SSM_IN);
            int hh = ssm_head_of(tid);
            SP_T(2, for (int i = 0; i < T; i++) {
                q38d_io io = {MT.qkv[i], MT.zb[i], MT.ab[i], MT.bb[i], NULL, NULL, NULL, MT.o[i], &MT.act_o[i], i};
                ssm_head_io(layer, hh, pos0 + i, &io);
            });
            SP_T(4, phase_end(tid, gs, &t, P_SSM_CORE));
            {
                int cc, g0, g1;
                mat_range(&L->out, tid, &cc, &g0, &g1);
                float *oa[TMAX];
                for (int i = 0; i < T; i++) oa[i] = xs[i] + L->out.first[c];
                SP_T(1, mt_mat(T, &L->out, NULL, c, g0, g1, MT.act_o, oa, NULL, 1));
            }
            SP_T(4, phase_end(tid, gs, &t, P_SSM_OUT));
        } else {
            float *(outs[3])[TMAX];
            for (int i = 0; i < T; i++) { outs[0][i] = MT.qg[i]; outs[1][i] = MT.kb[i]; outs[2][i] = MT.vb[i]; }
            SP_T(1, run_plan_mt(plan_att_mt && mt_plans ? &plan_att_mt[layer] : &plan_att[layer], T, MT.act_c[c], outs, tid));
            if (inproj_cbar && attn_cmg_ok) SP_T(4, phase_end_c(tid, cs, &t, P_ATT_IN));
            else phase_end(tid, gs, &t, P_ATT_IN);
            if (attn_multi) SP_T(2, attention_mt(layer, tid, pos0, T, cs));
            else SP_T(2, for (int i = 0; i < T; i++) {
                q38d_io io = {NULL, NULL, NULL, NULL, MT.qg[i], MT.kb[i], MT.vb[i], MT.o[i], &MT.act_o[i], -1};
                attention_io(layer, tid, pos0 + i, cs, &io);
            });
            SP_T(4, phase_end(tid, gs, &t, P_ATT_CORE));
            {
                int cc, g0, g1;
                mat_range(&L->o, tid, &cc, &g0, &g1);
                float *oa[TMAX];
                for (int i = 0; i < T; i++) oa[i] = xs[i] + L->o.first[c];
                SP_T(1, mt_mat(T, &L->o, NULL, c, g0, g1, MT.act_o, oa, NULL, 1));
            }
            SP_T(4, phase_end(tid, gs, &t, P_ATT_OUT));
        }
        SP_T(0, norm_mt(tid, xs, T, L->post_norm, cs));
        {
            int rows = L->gate.first[c + 1] - L->gate.first[c];
            int G8 = rows / 8, g0 = G8 * l / PER, g1 = G8 * (l + 1) / PER;
            int base = L->gate.first[c];
            float *oa[TMAX], *ob[TMAX];
            for (int i = 0; i < T; i++) { oa[i] = MT.h[i] + base; ob[i] = MT.qkv[i] + base; }
            SP_T(1, mt_mat(T, &L->gate, &L->up, c, g0, g1, MT.act_c[c], oa, ob, 0));
            const svbool_t p8 = svptrue_pat_b32(SV_VL8);
            for (int i = 0; i < T; i++) {
                for (int g = g0; g < g1; g++) {
                    int r = base + 8 * g;
                    svfloat32_t gv = svld1_f32(p8, MT.h[i] + r);
                    svst1_f32(p8, MT.h[i] + r, svmul_f32_x(p8, svmul_f32_x(p8, gv, q38d_sigmoid_sve(p8, gv)), svld1_f32(p8, MT.qkv[i] + r)));
                }
                int u_first = g0 / 2, u_last = (g1 + 1) / 2;
                for (int u = u_first; u < u_last; u++) {
                    int whole = 2 * u >= g0 && 2 * u + 1 < g1;
                    int r = base + 16 * u;
                    if (!whole) {
                        int old = atomic_fetch_add_explicit(&MT.unit_cnt[i][r / 16], 1, memory_order_acq_rel);
                        if (!(old & 1)) continue;
                    }
                    q38d_prepare_unit(&MT.act_h[i], r / 32, (r / 16) & 1, MT.h[i] + r);
                }
            }
        }
        SP_T(4, phase_end(tid, gs, &t, P_FFN_UP));
        SP_T(1, mv_items_mt(&L->down, T, tid, xs));
        SP_T(4, phase_end(tid, gs, &t, P_FFN_DOWN));
    #undef t
}
static void step_mt(int tid, const int *tok, int pos0, int T, int *gs, int *cs) {
    uint64_t t = ticks();
    for (int i = 0; i < T; i++) embed_token_to(tid, tok[i], MT.x[i]);
    phase_end(tid, gs, &t, P_EMBED);
    for (int layer = 0; layer < NLAYER; layer++) layer_mt(tid, layer, pos0, T, MT.x, gs, cs, &t);
    if (head_multi) SP_T(3, head_mt(tid, T, gs, cs));
    else for (int i = 0; i < T; i++) head_token(tid, MT.x[i], gs, cs, &MT.argmax[i], &MT.logit[i]);
    prof_mark(tid, &t, P_HEAD);
}
/* NextN drafter over the A positions accepted by a verification pass, as one
 * A-token pass: inputs (hidden MT.x[i], token nxt[i] = x_{pos+i+1}) at
 * positions pos+i; the last position's output hidden goes to E.h_mtp and its
 * draft (x_{pos+A+1}) to E.mtp_draft. */
static float *mtp_em[TMAX], *mtp_xm[TMAX];
static q38d_act mtp_act_t[TMAX];
static int mtp_batch = 1;
static void mtp_catchup_mt(int tid, const int *nxt, int pos, int A, int *gs, int *cs) {
    uint64_t t = ticks();
    for (int i = 0; i < A; i++) embed_token_to(tid, nxt[i], mtp_em[i]);
    gbarrier(tid, gs);
    const int np = EMBD / 32, p0 = 2 * np * tid / NT, p1 = 2 * np * (tid + 1) / NT;
    for (int i = 0; i < A; i++) {
        float inv_e = 1.0f / sqrtf(sumsq(mtp_em[i], EMBD) / EMBD + E.eps);
        float inv_h = 1.0f / sqrtf(sumsq(MT.x[i], EMBD) / EMBD + E.eps);
        q38d_act *a = &mtp_act_t[i];
        if (p0 < np) {
            q38d_act ae = *a;
            ae.cols = EMBD;
            q38d_prepare_pairs(&ae, mtp_em[i], inv_e, E.enorm, p0, p1 < np ? p1 : np);
        }
        if (p1 > np) {
            q38d_act ah = *a;
            ah.cols = EMBD;
            ah.q = a->q + (size_t)np * 64; ah.sc = a->sc + 2 * np; ah.sum = a->sum + 2 * np;
            q38d_prepare_pairs(&ah, MT.x[i], inv_h, E.hnorm, p0 > np ? p0 - np : 0, p1 - np);
        }
    }
    gbarrier(tid, gs);
    {
        int c, g0, g1;
        mat_range(&E.eh, tid, &c, &g0, &g1);
        float *oa[TMAX];
        for (int i = 0; i < A; i++) oa[i] = mtp_xm[i] + E.eh.first[c];
        mt_mat(A, &E.eh, NULL, c, g0, g1, mtp_act_t, oa, NULL, 0);
    }
    gbarrier(tid, gs);
    layer_mt(tid, NLAYER, pos, A, mtp_xm, gs, cs, &t);
    if (tid == 0) memcpy(E.h_mtp, mtp_xm[A - 1], EMBD * sizeof(float));
    int id = 0;
    float lg = 0;
    head_token_m(tid, mtp_xm[A - 1], E.shnorm, draft_v || draft_f4 ? &E.dhead : &E.head, gs, cs, &id, &lg);
    if (tid == 0) E.mtp_draft = id;
    gbarrier(tid, gs);
}
static void mt_alloc(void) {
    for (int c = 0; c < NCMG; c++) {
        qh_mt[c] = aligned_alloc(256, (size_t)TMAX * 6 * HD * 4);
        apart_mt[c] = aligned_alloc(256, (size_t)TMAX * PER * 6 * (2 + HD) * 4);
    }
    for (int t = 0; t < TMAX; t++) {
        mtp_em[t] = aligned_alloc(256, EMBD * 4);
        mtp_xm[t] = aligned_alloc(256, EMBD * 4);
        mtp_act_t[t] = (q38d_act){2 * EMBD, E.arith, aligned_alloc(256, q38d_act_qbytes(2 * EMBD, Q38D_A16)),
                                  aligned_alloc(256, 2 * EMBD / 16 * 4), aligned_alloc(256, 2 * EMBD / 16 * 4), NULL};
    }
    for (int t = 0; t < TMAX; t++) {
        mt_logits[t] = aligned_alloc(256, (size_t)E.n_vocab * 4);
        MT.x[t] = aligned_alloc(256, EMBD * 4);
        MT.qkv[t] = aligned_alloc(256, NFF * 4);
        MT.zb[t] = aligned_alloc(256, DINNER * 4);
        MT.ab[t] = aligned_alloc(256, 64 * 4); MT.bb[t] = aligned_alloc(256, 64 * 4);
        MT.qg[t] = aligned_alloc(256, 2 * NHEAD * HD * 4);
        MT.kb[t] = aligned_alloc(256, NKV * HD * 4); MT.vb[t] = aligned_alloc(256, NKV * HD * 4);
        MT.o[t] = aligned_alloc(256, DINNER * 4);
        MT.h[t] = aligned_alloc(256, NFF * 4);
        MT.unit_cnt[t] = calloc(NFF / 16, sizeof(*MT.unit_cnt[t]));
        MT.act_o[t] = (q38d_act){DINNER, E.arith, aligned_alloc(256, q38d_act_qbytes(DINNER, Q38D_A16)),
                                 aligned_alloc(256, DINNER / 16 * 4), aligned_alloc(256, DINNER / 16 * 4), MT.o[t]};
        MT.act_h[t] = (q38d_act){NFF, E.arith, aligned_alloc(256, q38d_act_qbytes(NFF, Q38D_A16)),
                                 aligned_alloc(256, NFF / 16 * 4), aligned_alloc(256, NFF / 16 * 4), MT.h[t]};
        for (int c = 0; c < NCMG; c++)
            MT.act_c[c][t] = (q38d_act){EMBD, E.arith, aligned_alloc(256, q38d_act_qbytes(EMBD, Q38D_A16) + 256),
                                        aligned_alloc(256, EMBD / 16 * 4 + 256), aligned_alloc(256, EMBD / 16 * 4 + 256), NULL};
    }
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
            int hh = ssm_head_of(tid);
            for (int r = 0; r < ssm_ring; r++) {
                float *b = aligned_alloc(256, (DS * DS + 2 * DS) * sizeof(float));
                memset(b, 0, (DS * DS + 2 * DS) * sizeof(float));
                ssm_buf[layer][hh * ssm_ring + r] = b;
            }
            E.conv_hist[layer][hh] = aligned_alloc(256, 8 * 384 * sizeof(float));
            memset(E.conv_hist[layer][hh], 0, 8 * 384 * sizeof(float));
            float *wl = aligned_alloc(256, 4 * 384 * sizeof(float));
            int g = hh % NGROUP, ch[3] = {g * DS, NGROUP * DS + g * DS, 2 * NGROUP * DS + hh * DS};
            for (int kk = 0; kk < 4; kk++)
                for (int part = 0; part < 3; part++)
                    memcpy(wl + kk * 384 + part * 128, E.L[layer].conv_w + (size_t)kk * QKVD + ch[part], 128 * sizeof(float));
            E.conv_wl[layer][hh] = wl;
        }
    int c = tid / PER, l = tid % PER;
    if (l == 0) {
        for (int ai = 0; ai < NATTN + mtp_on; ai++) {
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
    if (mtp_on && draft_v && !draft_f4) {
        /* draft head: rows [first'[c], first'[c+1]) copied from the head's groups */
        const q38d_mat *h = &E.head;
        size_t gb = q38d_group_bytes(h->fmt, h->cols);
        int G = (E.dhead.first[c + 1] - E.dhead.first[c]) / 8, a0 = G * l / PER, a1 = G * (l + 1) / PER;
        for (int g = a0; g < a1; g++) {
            int row = E.dhead.first[c] + 8 * g, sc = 0;
            while (sc < 3 && row >= h->first[sc + 1]) sc++;
            memcpy(E.dhead.part[c] + (size_t)g * gb, h->part[sc] + (size_t)((row - h->first[sc]) / 8) * gb, gb);
        }
        gbarrier(tid, &gs);
    }
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
    for (int pos = 0; pos < pn; pos++) {
        step(tid, JOB.tok[pos], pos, pos == pn - 1, &gs, &cs);
        if (mtp_on && pos + 1 < pn) mtp_step(tid, JOB.tok[pos + 1], pos, E.x, 0, &gs, &cs);
    }
    static float spec_logit0;
    if (tid == 0) spec_logit0 = E.logits[E.next_token];
    if (mtp_on) mtp_drafts(tid, 0, pn - 1, &gs, &cs);
    if (vbench_T) { vbench(tid, &gs, &cs); return NULL; }
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
    if (spec_k) {
        /* verify [cur, d1..dk] in one pass; accept the matching draft prefix
         * plus the pass's own next token; advance the SSM ring; catch the
         * MTP drafter up over the accepted positions and draft again */
        int n = 0, pos = pn, cur = E.next_token, passes = 0;
        static int hist[TMAX + 1];
        float cur_logit = spec_logit0;
        int dr[3] = {mtp_chain[0][0], mtp_chain[0][1], mtp_chain[0][2]};
        while (n < gn) {
            int T = 1 + spec_k;
            if (T > gn - n) T = gn - n;
            int tok[TMAX] = {cur, dr[0], dr[1], dr[2]};
            step_mt(tid, tok, pos, T, &gs, &cs);
            int A = 1;
            while (A < T && MT.argmax[A - 1] == tok[A]) A++;
            int bonus = MT.argmax[A - 1];
            if (tid == 0) {
                for (int i = 0; i < A; i++) { JOB.tok[pn + n + i] = tok[i]; JOB.trace_logit[n + i] = i ? MT.logit[i - 1] : cur_logit; }
                cur_logit = MT.logit[A - 1];
                ssm_cur = (ssm_cur + A) % ssm_ring;
                hist[A]++;
            }
            gbarrier(tid, &gs);
            uint64_t tm0 = tid ? 0 : ticks();
            if (n + A < gn) {
                if (mtp_batch) {
                    int nxt[TMAX];
                    for (int i = 0; i < A; i++) nxt[i] = i + 1 < A ? tok[i + 1] : bonus;
                    mtp_catchup_mt(tid, nxt, pos, A, &gs, &cs);
                } else
                    for (int i = 0; i < A; i++)
                        mtp_step(tid, i + 1 < A ? tok[i + 1] : bonus, pos + i, MT.x[i], i == A - 1, &gs, &cs);
                dr[0] = E.mtp_draft;
                for (int k = 1; k < spec_k; k++) {
                    mtp_step(tid, dr[k - 1], pos + A - 1 + k, E.h_mtp, 1, &gs, &cs);
                    dr[k] = E.mtp_draft;
                }
            }
            if (!tid) sp_t[5] += (double)(ticks() - tm0);
            n += A; pos += A; cur = bonus; passes++;
        }
        if (tid == 0) {
            JOB.t_decode = now_sec() - t0;
            fprintf(stderr, "q38d: spec k=%d: %d passes, %.3f tokens/pass, accepted-length histogram:", spec_k, passes, (double)gn / passes);
            for (int a = 1; a <= spec_k + 1; a++) fprintf(stderr, " %d:%d", a, hist[a]);
            fprintf(stderr, "\n");
            double hz = tick_hz();
            fprintf(stderr, "q38d: norm_mt worker0 us/call-set: sum=%.3f cbar1=%.3f quant=%.3f cbar2=%.3f ms/pass\n",
                    nm_t[0] / hz * 1e3 / passes, nm_t[1] / hz * 1e3 / passes, nm_t[2] / hz * 1e3 / passes, nm_t[3] / hz * 1e3 / passes);
            fprintf(stderr, "q38d: spec worker0 ms/pass: norm=%.2f proj=%.2f cores=%.2f head=%.2f phase_wait=%.2f mtp=%.2f\n",
                    sp_t[0] / hz * 1e3 / passes, sp_t[1] / hz * 1e3 / passes, sp_t[2] / hz * 1e3 / passes,
                    sp_t[3] / hz * 1e3 / passes, sp_t[4] / hz * 1e3 / passes, sp_t[5] / hz * 1e3 / passes);
        }
        return NULL;
    }
    for (int n = 0; n < gn; n++) {
        int cur = E.next_token;
        if (tid == 0) { JOB.tok[pn + n] = cur; JOB.trace_logit[n] = E.logits[cur]; }
        step(tid, cur, pn + n, 1, &gs, &cs);
        if (mtp_on) mtp_drafts(tid, n + 1, pn + n, &gs, &cs);
    }
    if (mtp_on && tid == 0) {
        /* x_{pn+j} = JOB.tok[pn+j], j < gn. Chain i predicts x_{pn+i+1..pn+i+3}. */
        int hit[3] = {0}, tot[3] = {0};
        for (int i = 0; i + 1 < gn; i++)
            for (int k = 0; k < 3 && i + 1 + k < gn; k++) {
                int ok = 1;
                for (int j = 0; j <= k; j++) ok &= mtp_chain[i][j] == JOB.tok[pn + i + 1 + j];
                tot[k]++; hit[k] += ok;
            }
        fprintf(stderr, "q38d: MTP prefix acceptance: d1 %d/%d (%.3f), d1..2 %d/%d (%.3f), d1..3 %d/%d (%.3f)\n",
                hit[0], tot[0], (double)hit[0] / tot[0], hit[1], tot[1], (double)hit[1] / tot[1], hit[2], tot[2], (double)hit[2] / tot[2]);
        for (int K = 1; K <= 3; K++) {
            int p = 0, passes = 0;   /* verify pass at chain index p accepts L drafts, advances 1 + L */
            while (p + 1 < gn) {
                int L = 0;
                while (L < K && p + 1 + L < gn && mtp_chain[p][L] == JOB.tok[pn + p + 1 + L]) L++;
                passes++; p += 1 + L;
            }
            fprintf(stderr, "q38d: MTP k=%d drafts: %.3f tokens per verify pass\n", K, (double)(gn - 1) / passes);
        }
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
    E.fmt = fmt; E.arith = arith; E.max_seq = pn + gn + 8;
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
    if (getenv("Q38D_VBENCH")) vbench_T = atoi(getenv("Q38D_VBENCH"));
    if (getenv("Q38D_MTP")) mtp_on = atoi(getenv("Q38D_MTP")) != 0;
    if (getenv("Q38D_SPEC")) spec_k = atoi(getenv("Q38D_SPEC"));
    if (getenv("Q38D_FFN_A8")) ffn_a8 = atoi(getenv("Q38D_FFN_A8"));
    if (getenv("Q38D_DOWN_A8")) down_a8 = atoi(getenv("Q38D_DOWN_A8"));
    if (getenv("Q38D_G2")) g2_variant = atoi(getenv("Q38D_G2"));
    if (getenv("Q38D_NORM_PF")) norm_pf = atoi(getenv("Q38D_NORM_PF"));
    if (getenv("Q38D_ATTN_MULTI")) attn_multi = atoi(getenv("Q38D_ATTN_MULTI"));
    if (getenv("Q38D_MT_PLANS")) mt_plans = atoi(getenv("Q38D_MT_PLANS"));
    if (getenv("Q38D_MTP_BATCH")) mtp_batch = atoi(getenv("Q38D_MTP_BATCH"));
    if (getenv("Q38D_DRAFT_F4")) draft_f4 = atoi(getenv("Q38D_DRAFT_F4"));
    if (getenv("Q38D_DRAFT_V")) draft_v = atoi(getenv("Q38D_DRAFT_V"));
    if (getenv("Q38D_HEAD_MULTI")) head_multi = atoi(getenv("Q38D_HEAD_MULTI"));
    if (spec_k) {
        if (spec_k < 1 || spec_k > TMAX - 1 || arith != Q38D_A16 || fmt != Q38D_F4) {
            fprintf(stderr, "q38d: Q38D_SPEC needs 1..%d drafts, --act a16, --fmt fp4\n", TMAX - 1);
            return 1;
        }
        mtp_on = 1;
        ssm_ring = spec_k + 2;
    }
    if (getenv("Q38D_SSM_PERM")) ssm_perm = atoi(getenv("Q38D_SSM_PERM"));
    if (getenv("Q38D_CONV_LOCAL") && !atoi(getenv("Q38D_CONV_LOCAL"))) ssm_perm = 0;
    load_engine();
    for (int l = 0; l < NLAYER + mtp_on; l++) {
        const q38d_layer *L = &E.L[l];
        if (L->ssm) continue;
        for (int c = 0; c <= 4; c++)
            if (L->q.first[c] != c * (L->q.rows / 4) || L->k.first[c] != c * HD || L->v.first[c] != c * HD) attn_cmg_ok = 0;
    }
    if (!attn_cmg_ok) fprintf(stderr, "q38d: attention in-proj not CMG-aligned, global barrier kept\n");
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
    if (down_a8 && arith == Q38D_A16) E.act_h.arith = Q38D_A8;
    for (int l = 0; l < NLAYER; l++) {
        E.conv_state[l] = calloc((size_t)4 * QKVD, sizeof(float));
        ssm_buf[l] = calloc((size_t)NVH * ssm_ring, sizeof(float *));
    }
    if (mtp_on) {
        mtp_chain = calloc((size_t)gn + 2, sizeof(*mtp_chain));
        E.mtp_in = aligned_alloc(256, 2 * EMBD * 4);
        E.x_mtp = aligned_alloc(256, EMBD * 4);
        E.h_mtp = aligned_alloc(256, EMBD * 4);
        E.act_mtp = (q38d_act){2 * EMBD, arith, aligned_alloc(256, q38d_act_qbytes(2 * EMBD, Q38D_A16)),
                               aligned_alloc(256, 2 * EMBD / 16 * 4), aligned_alloc(256, 2 * EMBD / 16 * 4), E.mtp_in};
    }
    if (mtp_on && draft_v && !draft_f4) {
        const q38d_mat *h = &E.head;
        if (draft_v % 32 || draft_v > h->rows) { fprintf(stderr, "q38d: Q38D_DRAFT_V must be a multiple of 32 <= %d\n", h->rows); return 1; }
        E.dhead = *h;
        E.dhead.rows = draft_v;
        size_t gb = q38d_group_bytes(h->fmt, h->cols);
        for (int c = 0; c <= 4; c++) E.dhead.first[c] = c * (draft_v / 4);
        for (int c = 0; c < 4; c++) E.dhead.part[c] = cmg_alloc((size_t)(draft_v / 32) * gb, c);
    }
    if (spec_k) {
        if (!ssm_lazy || !conv_local) { fprintf(stderr, "q38d: Q38D_SPEC needs ssm_lazy and conv_local\n"); return 1; }
        mt_alloc();
    }
    if (getenv("Q38D_PLAN")) use_plan = atoi(getenv("Q38D_PLAN"));
    if (getenv("Q38D_SSM_PF")) ssm_pf = atoi(getenv("Q38D_SSM_PF"));
    if (getenv("Q38D_PF_BYTES")) pf_bytes = (size_t)atoi(getenv("Q38D_PF_BYTES"));
    if (getenv("Q38D_PROD_NORM")) prod_norm = atoi(getenv("Q38D_PROD_NORM"));
    if (getenv("Q38D_DUAL")) use_dual = atoi(getenv("Q38D_DUAL"));
    if (getenv("Q38D_FFN_SPLIT")) ffn_group_split = atoi(getenv("Q38D_FFN_SPLIT"));
    if (getenv("Q38D_PAIR")) q38d_pair_groups = atoi(getenv("Q38D_PAIR"));
    if (getenv("Q38D_SSM_ROWPF")) ssm_rowpf = atoi(getenv("Q38D_SSM_ROWPF"));
    if (getenv("Q38D_SSM_LAZY")) ssm_lazy = atoi(getenv("Q38D_SSM_LAZY"));
    if (getenv("Q38D_INPROJ_CBAR")) inproj_cbar = atoi(getenv("Q38D_INPROJ_CBAR"));
    if (getenv("Q38D_CONV_LOCAL")) conv_local = atoi(getenv("Q38D_CONV_LOCAL"));
    if (getenv("Q38D_DUAL_COST")) dual_cost = atof(getenv("Q38D_DUAL_COST"));
    if (getenv("Q38D_PF_KV")) pf_kv = atoi(getenv("Q38D_PF_KV"));
    if (getenv("Q38D_EPOCH_BAR")) epoch_bar = atoi(getenv("Q38D_EPOCH_BAR"));
    if (getenv("Q38D_PROD_COPIES")) prod_copies = atoi(getenv("Q38D_PROD_COPIES"));
    if (getenv("Q38D_COST_Q4K")) cost_q4k = atof(getenv("Q38D_COST_Q4K"));
    if (getenv("Q38D_COST_Q6K")) cost_q6k = atof(getenv("Q38D_COST_Q6K"));
    if (getenv("Q38D_COST_Q8K")) cost_q8k = atof(getenv("Q38D_COST_Q8K"));
    plan_ssm = calloc(NLAYER, sizeof(q38d_plan));
    plan_att = calloc(NLAYER + 1, sizeof(q38d_plan));
    for (int l = 0; l < NLAYER + mtp_on; l++) {
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
    if (spec_k) {
        /* verification-pass plans: T = spec_k + 1 tokens */
        int T = spec_k + 1;
        double sv_dual = dual_cost;
        mt_cost_f4 = T == 2 ? 1.72 : T == 3 ? 2.0 : 2.3;
        mt_cost_kq = T;
        dual_cost = 2.0;
        if (getenv("Q38D_MT_COST_F4")) mt_cost_f4 = atof(getenv("Q38D_MT_COST_F4"));
        plan_ssm_mt = calloc(NLAYER + 1, sizeof(q38d_plan));
        plan_att_mt = calloc(NLAYER + 1, sizeof(q38d_plan));
        for (int l = 0; l < NLAYER + mtp_on; l++) {
            q38d_layer *L = &E.L[l];
            if (L->ssm) {
                q38d_plan *P = &plan_ssm_mt[l];
                *P = plan_ssm[l];
                if (use_dual && L->qkv.fmt == L->z.fmt && L->qkv.fmt == Q38D_F4 && L->qkv.cols == L->z.cols) build_plan_ssm_dual(P);
                else build_plan(P);
            } else {
                q38d_plan *P = &plan_att_mt[l];
                *P = plan_att[l];
                build_plan(P);
            }
        }
        mt_cost_f4 = mt_cost_kq = 1;
        dual_cost = sv_dual;
    }
    if (getenv("Q38D_PLAN_DUMP")) {
        for (int k = 0; k < 2; k++) {
            int l = 0;
            while (l < NLAYER && E.L[l].ssm != !k) l++;
            q38d_plan *P = k ? &plan_att[l] : &plan_ssm[l];
            for (int i = 0; i < P->nm; i++)
                fprintf(stderr, "q38d: plan %s m%d fmt=%d rows/cmg=%d cols=%d gcost=%.0f\n", k ? "att" : "ssm", i,
                        P->m[i]->fmt, P->m[i]->first[1] - P->m[i]->first[0], P->m[i]->cols, group_cost(P->m[i]));
            for (int t = 0; t < PER; t++) {
                fprintf(stderr, "q38d: plan %s lane %2d:", k ? "att" : "ssm", t);
                for (int j = 0; j < P->t[t].nseg; j++)
                    fprintf(stderr, " m%d%s[%d,%d)", P->t[t].s[j].mi, P->t[t].s[j].mi2 >= 0 ? "+" : "", P->t[t].s[j].g0, P->t[t].s[j].g1);
                fprintf(stderr, "\n");
            }
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
