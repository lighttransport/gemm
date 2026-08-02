/* bench_expert_bw.c — S0 roofline: what decode rate can this CPU sustain on
 * DeepSeek-V4-Flash's routed-expert gather?
 *
 * The hetero plan puts the FP8 dense path (MLA + shared expert, 5.7 GB total)
 * on the RX 9070 XT and leaves only the MXFP4 routed experts on the CPU. Per
 * decoded token that is:
 *
 *     6 active experts x 43 layers x 13.35 MB = 3.44 GB of DRAM traffic
 *
 * (per expert: w1 4 MiB + scale 256 KiB, w3 likewise, w2 4 MiB + 256 KiB).
 * So the achievable decode rate is bounded by sustained_gather_BW / 3.44 GB.
 * 10 tok/s needs ~34 GB/s. This benchmark measures that number directly, with
 * the real access pattern: random 13.35 MB expert blocks pulled out of a large
 * region that does not fit in cache, and the real MXFP4 SwiGLU math.
 *
 * Modes:
 *   --mode stream   read-only sum over the same gathered blocks (BW ceiling,
 *                   no dequant/FMA cost)
 *   --mode matvec   full expert: w1/w3 (4096->2048) + SwiGLU + w2 (2048->4096)
 *   --mode triad    plain STREAM triad over an anonymous buffer (DRAM ceiling)
 *   --mode verify   AVX2 kernel vs scalar reference, then exit
 *
 * Build:  make bench_expert_bw
 * Run:    ./build/bench_expert_bw --gib 64 --threads 16 --tokens 8
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "mxfp4_avx2.h"

static int g_f32seq = 0;   /* --f32seq: exact f32 activation, sequential rows */
#define MV_F32(...) (g_f32seq ? ds4f_matvec_mxfp4_8row_f32seq(__VA_ARGS__) \
                              : ds4f_matvec_mxfp4_8row(__VA_ARGS__))

/* --i8 selects the 128-bit W4A8 kernel, --i8x2 the 256-bit paired-block one. */
static int g_i8v = 0;   /* 0 = 128-bit, 1 = 256-bit paired, 2 = hoisted scales */
#define MV_I8(...)    (g_i8v == 1 ? ds4f_matvec_mxfp4_8row_i8x2(__VA_ARGS__) \
                     : g_i8v == 2 ? ds4f_matvec_mxfp4_8row_i8h(__VA_ARGS__)  \
                     : g_i8v == 3 ? ds4f_matvec_mxfp4_8row_i8seq(__VA_ARGS__) \
                                  : ds4f_matvec_mxfp4_8row_i8(__VA_ARGS__))
#define QUANT_I8(...) (g_i8v == 1 ? ds4f_quant_act_i8_pairs(__VA_ARGS__) \
                                  : ds4f_quant_act_i8(__VA_ARGS__))

/* ---- DS4F-0731 routed-expert geometry (from the safetensors headers) ---- */
#define HIDDEN     4096
#define MOE_INTER  2048
#define N_LAYERS     43
#define N_ACTIVE      6
/* w1: [MOE_INTER rows, HIDDEN k]; w3 same; w2: [HIDDEN rows, MOE_INTER k] */
#define W13_BYTES  ((size_t)MOE_INTER * HIDDEN / 2)          /* 4 MiB */
#define W13_SCALE  ((size_t)MOE_INTER * (HIDDEN / 32))       /* 256 KiB */
#define W2_BYTES   ((size_t)HIDDEN * MOE_INTER / 2)          /* 4 MiB */
#define W2_SCALE   ((size_t)HIDDEN * (MOE_INTER / 32))       /* 256 KiB */
#define EXPERT_BYTES (2 * (W13_BYTES + W13_SCALE) + W2_BYTES + W2_SCALE)

/* One expert block, laid out contiguously the way a packed model file would. */
typedef struct {
    uint8_t *w1, *s1, *w3, *s3, *w2, *s2;
} expert_view;

static void expert_view_of(uint8_t *base, expert_view *v) {
    uint8_t *p = base;
    v->w1 = p; p += W13_BYTES;  v->s1 = p; p += W13_SCALE;
    v->w3 = p; p += W13_BYTES;  v->s3 = p; p += W13_SCALE;
    v->w2 = p; p += W2_BYTES;   v->s2 = p;
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ---------------- pinned spin-barrier thread pool ---------------- */
typedef void (*pool_fn)(void *arg, int tid, int nthr);

typedef struct {
    int nthr;
    pthread_t *th;
    _Atomic int seq;                 /* incremented to release a job */
    _Atomic int done;
    pool_fn fn;
    void *arg;
    _Atomic int stop;
} pool_t;

typedef struct { pool_t *p; int tid; } worker_arg;

static int pin_thread(int tid, int nthr) {
    /* Prefer one thread per physical core (Zen1 SMT siblings share the L1/L2
     * and the load/store unit, so they do not add gather bandwidth). With <=16
     * threads take even CPU ids, which are the physical cores on this box. */
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    int cpu = (nthr <= ncpu / 2) ? (tid * 2) : tid;
    if (cpu >= ncpu) cpu = tid % ncpu;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void *pool_worker(void *v) {
    worker_arg *wa = (worker_arg *)v;
    pool_t *p = wa->p;
    int tid = wa->tid;
    pin_thread(tid, p->nthr);
    int last = 0;
    for (;;) {
        while (atomic_load_explicit(&p->seq, memory_order_acquire) == last) {
            if (atomic_load_explicit(&p->stop, memory_order_relaxed)) return NULL;
            __builtin_ia32_pause();
        }
        last = atomic_load_explicit(&p->seq, memory_order_acquire);
        p->fn(p->arg, tid, p->nthr);
        atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
    }
}

static pool_t *pool_start(int nthr) {
    pool_t *p = calloc(1, sizeof(*p));
    p->nthr = nthr;
    p->th = calloc(nthr, sizeof(pthread_t));
    atomic_store(&p->seq, 0); atomic_store(&p->done, 0); atomic_store(&p->stop, 0);
    for (int i = 1; i < nthr; i++) {
        worker_arg *wa = malloc(sizeof(*wa));
        wa->p = p; wa->tid = i;
        pthread_create(&p->th[i], NULL, pool_worker, wa);
    }
    pin_thread(0, nthr);
    return p;
}

static void pool_run(pool_t *p, pool_fn fn, void *arg) {
    p->fn = fn; p->arg = arg;
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->seq, 1, memory_order_release);
    fn(arg, 0, p->nthr);
    while (atomic_load_explicit(&p->done, memory_order_acquire) < p->nthr - 1)
        __builtin_ia32_pause();
}

static void pool_stop(pool_t *p) {
    atomic_store_explicit(&p->stop, 1, memory_order_release);
    atomic_fetch_add_explicit(&p->seq, 1, memory_order_release);
    for (int i = 1; i < p->nthr; i++) pthread_join(p->th[i], NULL);
    free(p->th); free(p);
}

/* ---------------- jobs ---------------- */
typedef struct {
    expert_view ev[N_ACTIVE];
    int n_expert;
    const float *x;        /* [HIDDEN] layer input */
    float *g;              /* [N_ACTIVE][MOE_INTER] w1 out (gate) */
    float *u;              /* [N_ACTIVE][MOE_INTER] w3 out (up)   */
    float *y;              /* [N_ACTIVE][HIDDEN]    w2 out        */
    /* W4A8: activation quantized once per stage, shared by every row/expert */
    int use_i8;
    int8_t *xq;  float *xs;  float *xc;              /* stage 1 input, K=HIDDEN */
    int8_t *gq;  float *gs;  float *gc;              /* stage 2 input, K=MOE_INTER, per expert */
    _Atomic uint64_t checksum;
} moe_job;

/* Split N rows across nthr threads in 8-row groups. */
static void row_range(int N, int tid, int nthr, int *r0, int *r1) {
    int groups = N / 8;
    int per = groups / nthr, rem = groups % nthr;
    int g0 = tid * per + (tid < rem ? tid : rem);
    int gn = per + (tid < rem ? 1 : 0);
    *r0 = g0 * 8; *r1 = (g0 + gn) * 8;
}

/* Stage 1: all experts' w1 and w3 (both read the same activation x). */
static void job_gateup(void *arg, int tid, int nthr) {
    moe_job *j = (moe_job *)arg;
    int r0, r1; row_range(MOE_INTER, tid, nthr, &r0, &r1);
    const size_t rw = HIDDEN / 2, rs = HIDDEN / 32;
    for (int e = 0; e < j->n_expert; e++) {
        const expert_view *v = &j->ev[e];
        for (int r = r0; r < r1; r += 8) {
            const uint8_t *w = v->w1 + (size_t)r * rw, *s = v->s1 + (size_t)r * rs;
            #define MV13(DST, WB, SB) do {                                        \
                const uint8_t *w_ = (WB), *s_ = (SB);                             \
                if (j->use_i8)                                                    \
                    MV_I8(DST,                                \
                        w_, w_+rw, w_+2*rw, w_+3*rw, w_+4*rw, w_+5*rw, w_+6*rw, w_+7*rw, \
                        s_, s_+rs, s_+2*rs, s_+3*rs, s_+4*rs, s_+5*rs, s_+6*rs, s_+7*rs, \
                        j->xq, j->xs, j->xc, HIDDEN);                             \
                else                                                              \
                    MV_F32(DST,                                                   \
                        w_, w_+rw, w_+2*rw, w_+3*rw, w_+4*rw, w_+5*rw, w_+6*rw, w_+7*rw, \
                        s_, s_+rs, s_+2*rs, s_+3*rs, s_+4*rs, s_+5*rs, s_+6*rs, s_+7*rs, \
                        j->x, HIDDEN);                                            \
            } while (0)
            MV13(j->g + (size_t)e * MOE_INTER + r, w, s);
            MV13(j->u + (size_t)e * MOE_INTER + r, v->w3 + (size_t)r * rw, v->s3 + (size_t)r * rs);
            #undef MV13
        }
        /* SwiGLU on this thread's slice, in place into g */
        for (int r = r0; r < r1; r++) {
            float gv = j->g[(size_t)e * MOE_INTER + r];
            j->g[(size_t)e * MOE_INTER + r] = (gv / (1.f + expf(-gv))) * j->u[(size_t)e * MOE_INTER + r];
        }
    }
}

/* Stage 2: all experts' w2 over the SwiGLU output. */
static void job_down(void *arg, int tid, int nthr) {
    moe_job *j = (moe_job *)arg;
    int r0, r1; row_range(HIDDEN, tid, nthr, &r0, &r1);
    const size_t rw = MOE_INTER / 2, rs = MOE_INTER / 32;
    for (int e = 0; e < j->n_expert; e++) {
        const expert_view *v = &j->ev[e];
        const float *xin = j->g + (size_t)e * MOE_INTER;
        const size_t nb2 = MOE_INTER / 32;
        for (int r = r0; r < r1; r += 8) {
            const uint8_t *w = v->w2 + (size_t)r * rw, *s = v->s2 + (size_t)r * rs;
            if (j->use_i8)
                MV_I8(j->y + (size_t)e * HIDDEN + r,
                    w, w+rw, w+2*rw, w+3*rw, w+4*rw, w+5*rw, w+6*rw, w+7*rw,
                    s, s+rs, s+2*rs, s+3*rs, s+4*rs, s+5*rs, s+6*rs, s+7*rs,
                    j->gq + (size_t)e * MOE_INTER, j->gs + (size_t)e * nb2,
                    j->gc + (size_t)e * nb2, MOE_INTER);
            else
                MV_F32(j->y + (size_t)e * HIDDEN + r,
                    w, w+rw, w+2*rw, w+3*rw, w+4*rw, w+5*rw, w+6*rw, w+7*rw,
                    s, s+rs, s+2*rs, s+3*rs, s+4*rs, s+5*rs, s+6*rs, s+7*rs,
                    xin, MOE_INTER);
        }
    }
}

/* Read-only sum over the same bytes: the pure-bandwidth ceiling. */
static void job_stream(void *arg, int tid, int nthr) {
    moe_job *j = (moe_job *)arg;
    uint64_t acc = 0;
    for (int e = 0; e < j->n_expert; e++) {
        const uint8_t *base = j->ev[e].w1;   /* block is contiguous from w1 */
        size_t per = EXPERT_BYTES / nthr & ~(size_t)63;
        size_t off = (size_t)tid * per;
        size_t end = (tid == nthr - 1) ? EXPERT_BYTES : off + per;
        const uint64_t *p = (const uint64_t *)(base + off);
        size_t n = (end - off) / 8;
        for (size_t i = 0; i < n; i += 8) {
            acc += p[i] + p[i+1] + p[i+2] + p[i+3] + p[i+4] + p[i+5] + p[i+6] + p[i+7];
        }
    }
    atomic_fetch_add_explicit(&j->checksum, acc, memory_order_relaxed);
}

/* ---------------- STREAM triad ---------------- */
typedef struct { float *a, *b, *c; size_t n; } triad_job;
static void job_triad(void *arg, int tid, int nthr) {
    triad_job *t = (triad_job *)arg;
    size_t per = (t->n / nthr) & ~(size_t)15;
    size_t i0 = (size_t)tid * per, i1 = (tid == nthr - 1) ? t->n : i0 + per;
    for (size_t i = i0; i < i1; i++) t->a[i] = t->b[i] + 3.0f * t->c[i];
}

/* ---------------- verify ---------------- */
static int run_verify(void) {
    const int K = 4096;
    uint8_t *w = malloc((size_t)8 * K / 2);
    uint8_t *s = malloc((size_t)8 * K / 32);
    float *x = malloc(sizeof(float) * K);
    uint32_t st = 12345;
    #define RND() (st = st * 1664525u + 1013904223u)
    for (size_t i = 0; i < (size_t)8 * K / 2; i++) w[i] = (uint8_t)(RND() >> 24);
    /* keep E8M0 exponents near 1.0 so the reference sum stays well-conditioned */
    for (size_t i = 0; i < (size_t)8 * K / 32; i++) s[i] = (uint8_t)(120 + (RND() >> 28));
    for (int i = 0; i < K; i++) x[i] = ((float)(RND() >> 8) / 8388608.0f) - 1.0f;

    float got[8], want[8];
    const size_t rw = K / 2, rs = K / 32;
    ds4f_matvec_mxfp4_8row(got, w, w+rw, w+2*rw, w+3*rw, w+4*rw, w+5*rw, w+6*rw, w+7*rw,
                           s, s+rs, s+2*rs, s+3*rs, s+4*rs, s+5*rs, s+6*rs, s+7*rs, x, K);
    for (int r = 0; r < 8; r++)
        ds4f_matvec_mxfp4_ref(&want[r], w + (size_t)r * rw, s + (size_t)r * rs, x, K);

    int bad = 0;
    for (int r = 0; r < 8; r++) {
        float d = fabsf(got[r] - want[r]);
        float tol = 1e-4f * fmaxf(1.f, fabsf(want[r]));
        printf("  row %d: avx2 %14.6f  ref %14.6f  |d| %.3e %s\n",
               r, got[r], want[r], d, d <= tol ? "ok" : "FAIL");
        if (d > tol) bad = 1;
    }
    /* W4A8 path: activation is int8-quantized per 32-block, so this is checked
     * against the same f32 reference on RELATIVE error, not bit equality. */
    int8_t *xq = malloc(K);
    float *xs = malloc(sizeof(float) * K / 32), *xc = malloc(sizeof(float) * K / 32);
    QUANT_I8(x, K, xq, xs, xc);
    float gi8[8];
    MV_I8(gi8, w, w+rw, w+2*rw, w+3*rw, w+4*rw, w+5*rw, w+6*rw, w+7*rw,
                              s, s+rs, s+2*rs, s+3*rs, s+4*rs, s+5*rs, s+6*rs, s+7*rs,
                              xq, xs, xc, K);
    /* Scale of a K=4096 random dot product, for a meaningful relative error. */
    float norm = 0.f;
    for (int r = 0; r < 8; r++) norm += fabsf(want[r]);
    norm /= 8.f;
    /* Smoke test only: uniformly random nibbles make the true dot a random walk,
     * which is the worst case for activation quantization. The real accuracy
     * gate is a model-level logit comparison in S1, not this. */
    printf("\nW4A8 (int8 activation, random-weight smoke test):\n");
    float relmax = 0.f;
    for (int r = 0; r < 8; r++) {
        float rel = fabsf(gi8[r] - want[r]) / norm;
        if (rel > relmax) relmax = rel;
        printf("  row %d: i8 %14.6f  ref %14.6f  rel %.3e\n", r, gi8[r], want[r], rel);
    }
    printf("  max rel %.3e %s\n", relmax, relmax <= 3e-2f ? "ok" : "FAIL");
    if (relmax > 3e-2f) bad = 1;
    free(xq); free(xs); free(xc);
    free(w); free(s); free(x);
    printf("verify: %s\n", bad ? "FAIL" : "PASS");
    return bad;
}

/* ---------------- main ---------------- */
int main(int argc, char **argv) {
    const char *mode = "matvec";
    double gib = 64.0;
    int nthr = 16, tokens = 8, use_i8 = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--i8"))                     use_i8 = 1;
        else if (!strcmp(argv[i], "--f32seq"))            g_f32seq = 1;
        else if (!strcmp(argv[i], "--i8x2"))               { use_i8 = 1; g_i8v = 1; }
        else if (!strcmp(argv[i], "--i8h"))                { use_i8 = 1; g_i8v = 2; }
        else if (!strcmp(argv[i], "--i8seq"))              { use_i8 = 1; g_i8v = 3; }
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc)    mode = argv[++i];
        else if (!strcmp(argv[i], "--gib") && i + 1 < argc)     gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nthr = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc)   tokens = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--mode stream|matvec|triad|verify] "
                               "[--gib N] [--threads N] [--tokens N]\n", argv[0]); return 2; }
    }
    if (!strcmp(mode, "verify")) return run_verify();

    printf("DS4F expert-gather roofline\n");
    printf("  expert block   %.2f MiB   (w1 %zu + s %zu, w3 same, w2 %zu + s %zu)\n",
           EXPERT_BYTES / 1048576.0, W13_BYTES, W13_SCALE, W2_BYTES, W2_SCALE);
    printf("  per token      %d layers x %d experts = %.2f GB\n",
           N_LAYERS, N_ACTIVE, N_LAYERS * N_ACTIVE * EXPERT_BYTES / 1e9);
    printf("  mode %s  region %.0f GiB  threads %d  tokens %d\n\n",
           mode, gib, nthr, tokens);

    pool_t *pool = pool_start(nthr);

    if (!strcmp(mode, "triad")) {
        size_t n = (size_t)(gib * (1u << 30) / (3 * sizeof(float)));
        triad_job t = { NULL, NULL, NULL, n };
        t.a = aligned_alloc(64, n * 4); t.b = aligned_alloc(64, n * 4); t.c = aligned_alloc(64, n * 4);
        if (!t.a || !t.b || !t.c) { fprintf(stderr, "alloc failed\n"); return 1; }
        for (size_t i = 0; i < n; i++) { t.b[i] = 1.f; t.c[i] = 2.f; t.a[i] = 0.f; }
        pool_run(pool, job_triad, &t);                      /* warm */
        double t0 = now_s();
        for (int r = 0; r < 5; r++) pool_run(pool, job_triad, &t);
        double dt = now_s() - t0;
        printf("STREAM triad: %.1f GB/s\n", 5.0 * n * 3 * 4 / dt / 1e9);
        pool_stop(pool);
        return 0;
    }

    /* Allocate and touch the expert region. */
    size_t n_experts = (size_t)(gib * (1u << 30)) / EXPERT_BYTES;
    if (n_experts < 64) { fprintf(stderr, "--gib too small (need >= 1 GiB)\n"); return 1; }
    size_t region = n_experts * EXPERT_BYTES;
    printf("allocating %zu experts = %.1f GiB ...\n", n_experts, region / 1073741824.0);
    uint8_t *region_p = mmap(NULL, region, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (region_p == MAP_FAILED) { perror("mmap"); return 1; }
    /* Fill with pseudo-random nibbles; E8M0 scale bytes near 127 so the matvec
     * math stays in a sane range and the dequant work is representative. */
    double tf = now_s();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < region; i += 4096) {
        uint32_t st = (uint32_t)(i >> 12) * 2654435761u + 1u;
        size_t end = i + 4096 < region ? i + 4096 : region;
        for (size_t k = i; k < end; k++) { st = st * 1664525u + 1013904223u; region_p[k] = (uint8_t)(st >> 24); }
    }
    for (size_t e = 0; e < n_experts; e++) {   /* fix up the scale planes */
        expert_view v; expert_view_of(region_p + e * EXPERT_BYTES, &v);
        memset(v.s1, 127, W13_SCALE); memset(v.s3, 127, W13_SCALE); memset(v.s2, 127, W2_SCALE);
    }
    printf("filled in %.1f s\n\n", now_s() - tf);

    float *x = aligned_alloc(64, HIDDEN * sizeof(float));
    for (int i = 0; i < HIDDEN; i++) x[i] = 0.01f * ((i % 17) - 8);
    moe_job job;
    memset(&job, 0, sizeof(job));
    job.n_expert = N_ACTIVE;
    job.x = x;
    job.g = aligned_alloc(64, (size_t)N_ACTIVE * MOE_INTER * sizeof(float));
    job.u = aligned_alloc(64, (size_t)N_ACTIVE * MOE_INTER * sizeof(float));
    job.y = aligned_alloc(64, (size_t)N_ACTIVE * HIDDEN * sizeof(float));
    job.use_i8 = use_i8;
    const size_t nb1 = HIDDEN / 32, nb2 = MOE_INTER / 32;
    job.xq = aligned_alloc(64, HIDDEN);
    job.xs = aligned_alloc(64, nb1 * sizeof(float));
    job.xc = aligned_alloc(64, nb1 * sizeof(float));
    job.gq = aligned_alloc(64, (size_t)N_ACTIVE * MOE_INTER);
    job.gs = aligned_alloc(64, (size_t)N_ACTIVE * nb2 * sizeof(float));
    job.gc = aligned_alloc(64, (size_t)N_ACTIVE * nb2 * sizeof(float));
    if (use_i8) QUANT_I8(x, HIDDEN, job.xq, job.xs, job.xc);

    int stream = !strcmp(mode, "stream");
    uint32_t rng = 0xC0FFEE;
    double best = 0;
    for (int t = 0; t < tokens; t++) {
        double t0 = now_s();
        for (int L = 0; L < N_LAYERS; L++) {
            for (int e = 0; e < N_ACTIVE; e++) {
                rng = rng * 1664525u + 1013904223u;
                expert_view_of(region_p + (size_t)(rng % n_experts) * EXPERT_BYTES, &job.ev[e]);
            }
            if (stream) {
                pool_run(pool, job_stream, &job);
            } else {
                pool_run(pool, job_gateup, &job);
                if (use_i8)
                    for (int e = 0; e < N_ACTIVE; e++)
                        QUANT_I8(job.g + (size_t)e * MOE_INTER, MOE_INTER,
                                          job.gq + (size_t)e * MOE_INTER,
                                          job.gs + (size_t)e * nb2, job.gc + (size_t)e * nb2);
                pool_run(pool, job_down, &job);
            }
        }
        double dt = now_s() - t0;
        double bytes = (double)N_LAYERS * N_ACTIVE * EXPERT_BYTES;
        double gbps = bytes / dt / 1e9;
        if (gbps > best) best = gbps;
        printf("  token %2d: %7.1f ms   %6.1f GB/s   => %5.2f tok/s\n",
               t, dt * 1e3, gbps, 1.0 / dt);
    }
    double per_token_gb = (double)N_LAYERS * N_ACTIVE * EXPERT_BYTES / 1e9;
    printf("\nbest %.1f GB/s => %.2f tok/s (expert path only)\n", best, best / per_token_gb);
    printf("gate: need >= %.1f GB/s for 10 tok/s -- %s\n",
           per_token_gb * 10.0, best >= per_token_gb * 10.0 ? "PASS" : "not yet");
    if (stream) printf("checksum %llu\n", (unsigned long long)atomic_load(&job.checksum));
    else        printf("y[0] %.6f  y[last] %.6f\n", job.y[0], job.y[N_ACTIVE * HIDDEN - 1]);

    pool_stop(pool);
    return 0;
}
