/*
 * DS4F decode-bandwidth bench — single node, NO MPI, NO model alloc.
 *
 * PURPOSE
 *   Decode in DS4F is M=1 matvec, every weight read cold from HBM once per token.
 *   That is the HBM-bandwidth regime. This bench measures, per real decode matvec
 *   shape, the achievable streaming throughput of each weight representation/kernel
 *   so we can decide: is decode bandwidth-bound (FP8 halves the bytes -> ~2x and
 *   -6 GB resident, and faster decode-dequant buys nothing beyond the raw-read
 *   ceiling) or dequant/issue-bound (faster decode helps, double-buffer might)?
 *
 *   It reproduces production exactly: the custom PINNED spin-barrier pool (cores
 *   12..59, 12/CMG, copied verbatim from common/ds4f_impl.h), ds4f_rowsplit8 row
 *   ownership, one barrier per matvec sweep. It runs DIRECTLY on this node (no
 *   pjsub, no mpiexec) and leaves cores 0..11 + memory free for the session.
 *
 * TWO CONFOUNDS THE OLD ds4f_fp8_fast_bench.c HAD (both fixed here):
 *   (a) serial first-touch -> all weight pages on ONE CMG -> 48 readers hammer one
 *       HBM controller -> large-N BW capped ~75-90 GB/s instead of the ~650 GB/s
 *       node. FIX: fill each weight pool THROUGH the pool, every thread first-
 *       touches the rows (ds4f_rowsplit8) it later reads -> pages spread over 4 CMGs.
 *   (b) inter-iteration cache reuse -> small matrices go L2-resident -> a fake
 *       "bf16 wins at small N". FIX: a POOL of P distinct matrices (>=256 MB) cycled
 *       per "token"; revisits are >2x aggregate-L2 apart so every read is cold.
 *
 * VARIANTS (per dense FP8 shape unless noted):
 *   R           raw-read ceiling (svld1_u8 + eor, no decode) = empirical node BW
 *   bf16        matvec_bf16_8row over a bf16 predequant copy (2 B/elem)
 *   fp8-gather  matvec_fp8e4m3_8row          (256-LUT gather)
 *   fp8-magic   matvec_fp8e4m3_8row_magic    (denormal magic-multiply, needs FTZ)
 *   fp8-scalar  NEW: scalar ldrb+LUT+fmadd, vectorize-disabled (latency probe)
 *   fp8-neon    NEW: 128-bit NEON magic decode (short-vector probe)
 *   mxfp4       matvec_mxfp4_8row (expert shapes only) + its own R ceiling
 *
 * NUMERIC GATE (hard rule: NEVER bit-equality): each kernel vs an f32 double-accum
 *   reference of its OWN representation -> argmax identical + top-5 set identical +
 *   max relative error reported. magic maps exp==15 to a finite large (not NaN);
 *   the gate uses a NaN/exp15-free fill so magic must be argmax-exact vs the LUT ref.
 *
 * Build (add `ds4f_decode_bw` to a64fx/llm/Makefile):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -I../../common \
 *       -o build/ds4f_decode_bw ds4f_decode_bw_bench.c -lm -lpthread
 * Run directly on this node (redirect, then grep — context hygiene):
 *   ./build/ds4f_decode_bw > ~/tmp/decbw.log 2>&1 ; grep -E 'SUMMARY|gate|\] R ' ~/tmp/decbw.log
 * Env: DS4F_POOL_MB (per-shape cold budget, default 384), DS4F_DENSE_GB +
 *   DS4F_DENSE_FRAC (optional: illustrative dense-read tok/s from best 48T magic BW).
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <arm_sve.h>
#include <arm_neon.h>

#include "ggml_dequant.h"   /* matvec_{bf16,fp8e4m3,fp8e4m3_magic,mxfp4}_8row + LUT/e8m0 */

/* ============================ pinned pool ============================ *
 * Verbatim shape of common/ds4f_impl.h's pool so the measured path == prod:
 * spin-barrier, main = tid 0, bpin spreads `nthr` threads over `n_cmgs`. */
typedef void (*bench_fn)(void *arg, int tid, int nthr);
typedef struct {
    int nthr, n_cmgs;
    pthread_t *threads;
    _Atomic int seq, done, stop;
    bench_fn fn; void *arg;
} bpool;

static int bpin(int tid, int nthr, int n_cmgs) {
    if (nthr < 1) nthr = 1; if (n_cmgs < 1) n_cmgs = 1; if (n_cmgs > 4) n_cmgs = 4;
    int cmg = (int)((long)tid * n_cmgs / nthr);
    int cmg_first = (int)(((long)cmg * nthr + n_cmgs - 1) / n_cmgs);
    int local = tid - cmg_first; if (local < 0) local = 0; if (local > 11) local = 11;
    int core = 12 + cmg * 12 + local;   /* A64FX compute cores 12..59, 12/CMG */
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
static inline void brelax(void){ __asm__ __volatile__("yield" ::: "memory"); }
typedef struct { bpool *p; int tid; } bwctx;
static void *bworker(void *v) {
    bwctx *w = (bwctx *)v; bpool *p = w->p; int tid = w->tid;
    bpin(tid, p->nthr, p->n_cmgs);
    int last = 0;
    for (;;) {
        while (atomic_load_explicit(&p->seq, memory_order_acquire) == last &&
               !atomic_load_explicit(&p->stop, memory_order_acquire)) brelax();
        if (atomic_load_explicit(&p->stop, memory_order_acquire)) break;
        last = atomic_load_explicit(&p->seq, memory_order_acquire);
        if (p->fn) p->fn(p->arg, tid, p->nthr);
        atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
    }
    free(w); return NULL;
}
static bpool *bpool_start(int nthr, int n_cmgs) {
    bpool *p = (bpool *)calloc(1, sizeof(*p));
    p->nthr = nthr; p->n_cmgs = n_cmgs;
    atomic_store(&p->seq, 0); atomic_store(&p->done, 0); atomic_store(&p->stop, 0);
    p->threads = (pthread_t *)calloc(nthr, sizeof(pthread_t));
    for (int t = 1; t < nthr; t++) {
        bwctx *w = (bwctx *)malloc(sizeof(*w)); w->p = p; w->tid = t;
        pthread_create(&p->threads[t], NULL, bworker, w);
    }
    bpin(0, nthr, n_cmgs);   /* main pinned to core 12 */
    return p;
}
static void bpool_run(bpool *p, bench_fn fn, void *arg) {
    p->fn = fn; p->arg = arg;
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->seq, 1, memory_order_release);
    fn(arg, 0, p->nthr);
    while (atomic_load_explicit(&p->done, memory_order_acquire) < p->nthr - 1) brelax();
}
static void bpool_stop(bpool *p) {
    atomic_store_explicit(&p->stop, 1, memory_order_release);
    for (int t = 1; t < p->nthr; t++) pthread_join(p->threads[t], NULL);
    free(p->threads); free(p);
}
static inline void rowsplit8(int rows, int nthr, int tid, int *r0, int *r1) {
    int blk = (rows + 7) / 8, per = blk / nthr, extra = blk % nthr;
    int g0 = per * tid + (tid < extra ? tid : extra);
    int g1 = g0 + per + (tid < extra ? 1 : 0);
    *r0 = g0 * 8; *r1 = g1 * 8; if (*r1 > rows) *r1 = rows;
}

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ __volatile__("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ __volatile__("mrs %0, cntfrq_el0":"=r"(v)); return v; }
static inline void set_ftz(void){ uint64_t f; __asm__ __volatile__("mrs %0, fpcr":"=r"(f));
    f |= (1ull<<24); __asm__ __volatile__("msr fpcr, %0"::"r"(f)); }

/* Fresh anonymous pages every call: glibc's aligned_alloc/free RECYCLES freed
 * heap pages that are already faulted-in on whatever CMG first touched them, so a
 * later per-thread "first-touch" is a no-op and the whole pool stays on one CMG
 * (-> the 48T read ceiling collapses to single-CMG ~90 GB/s). mmap+munmap forces
 * genuinely unfaulted pages so the pooled first-touch actually spreads over 4 CMGs. */
static void *xmap(size_t n){ void *p=mmap(NULL,n,PROT_READ|PROT_WRITE,
        MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED){ fprintf(stderr,"mmap %zu MB fail\n",n>>20); exit(1);} return p; }
static void xunmap(void *p,size_t n){ munmap(p,n); }

static uint64_t sm; static inline uint64_t smn(void){
    uint64_t z=(sm+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static inline float rand_unit(void){ return (float)((int64_t)(smn()&0xffffff)-0x800000)/(float)0x800000; }
/* NaN/exp15-free E4M3 byte (exp field never all-ones -> never NaN, never exp==15). */
static inline uint8_t rand_fp8(void){ uint8_t b=(uint8_t)(smn()&0xff); if((b&0x78)==0x78) b&=~0x08; return b; }
static inline float bits2f(uint32_t u){ float f; memcpy(&f,&u,4); return f; }
static inline float bf16f(uint16_t b){ return bits2f((uint32_t)b<<16); }

/* ===================== NEW variant: scalar fp8 decode ===================== *
 * Genuinely scalar (vectorize disabled): LUT + scalar fmadd, 8 rows. Probes the
 * "cores are free in decode, scalar has shorter latency" hypothesis. */
static float g_lutf[256];
__attribute__((noinline))
static void matvec_fp8e4m3_8row_scalar(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *escale, const float *x, int K) {
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (int c0=0;c0<K;c0+=128){
        float s = ggml_e8m0_to_fp32(escale[c0>>7]);
        _Pragma("clang loop vectorize(disable) interleave(disable)")
        for (int c=c0;c<c0+128;c++){
            float xc = x[c]*s;
            a0 += g_lutf[w0[c]]*xc; a1 += g_lutf[w1[c]]*xc;
            a2 += g_lutf[w2[c]]*xc; a3 += g_lutf[w3[c]]*xc;
            a4 += g_lutf[w4[c]]*xc; a5 += g_lutf[w5[c]]*xc;
            a6 += g_lutf[w6[c]]*xc; a7 += g_lutf[w7[c]]*xc;
        }
    }
    dst[0]=a0;dst[1]=a1;dst[2]=a2;dst[3]=a3;dst[4]=a4;dst[5]=a5;dst[6]=a6;dst[7]=a7;
}

/* ===================== NEW variant: NEON (128-bit) magic fp8 decode ===================== */
static inline float32x4_t neon_fp8_magic(uint32x4_t b){
    uint32x4_t sgn = vshlq_n_u32(vandq_u32(b, vdupq_n_u32(0x80u)), 24);
    uint32x4_t mag = vshlq_n_u32(vandq_u32(b, vdupq_n_u32(0x7Fu)), 20);
    float32x4_t f  = vmulq_f32(vreinterpretq_f32_u32(mag), vdupq_n_f32(0x1.0p+120f));
    return vreinterpretq_f32_u32(vorrq_u32(sgn, vreinterpretq_u32_f32(f)));
}
static void matvec_fp8e4m3_8row_neon(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *escale, const float *x, int K) {
    float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0),a2=vdupq_n_f32(0),a3=vdupq_n_f32(0);
    float32x4_t a4=vdupq_n_f32(0),a5=vdupq_n_f32(0),a6=vdupq_n_f32(0),a7=vdupq_n_f32(0);
    for (int c0=0;c0<K;c0+=128){
        float s = ggml_e8m0_to_fp32(escale[c0>>7]);
        for (int c=c0;c<c0+128;c+=16){
            float32x4_t vx0=vmulq_n_f32(vld1q_f32(x+c),    s);
            float32x4_t vx1=vmulq_n_f32(vld1q_f32(x+c+4),  s);
            float32x4_t vx2=vmulq_n_f32(vld1q_f32(x+c+8),  s);
            float32x4_t vx3=vmulq_n_f32(vld1q_f32(x+c+12), s);
            #define NEON_ROW(W,ACC) do {                                         \
                uint8x16_t b=vld1q_u8((W)+c);                                     \
                uint16x8_t lo=vmovl_u8(vget_low_u8(b)),hi=vmovl_u8(vget_high_u8(b)); \
                uint32x4_t u0=vmovl_u16(vget_low_u16(lo)),u1=vmovl_u16(vget_high_u16(lo)); \
                uint32x4_t u2=vmovl_u16(vget_low_u16(hi)),u3=vmovl_u16(vget_high_u16(hi)); \
                ACC=vfmaq_f32(ACC, neon_fp8_magic(u0), vx0);                      \
                ACC=vfmaq_f32(ACC, neon_fp8_magic(u1), vx1);                      \
                ACC=vfmaq_f32(ACC, neon_fp8_magic(u2), vx2);                      \
                ACC=vfmaq_f32(ACC, neon_fp8_magic(u3), vx3);                      \
            } while(0)
            NEON_ROW(w0,a0);NEON_ROW(w1,a1);NEON_ROW(w2,a2);NEON_ROW(w3,a3);
            NEON_ROW(w4,a4);NEON_ROW(w5,a5);NEON_ROW(w6,a6);NEON_ROW(w7,a7);
            #undef NEON_ROW
        }
    }
    dst[0]=vaddvq_f32(a0);dst[1]=vaddvq_f32(a1);dst[2]=vaddvq_f32(a2);dst[3]=vaddvq_f32(a3);
    dst[4]=vaddvq_f32(a4);dst[5]=vaddvq_f32(a5);dst[6]=vaddvq_f32(a6);dst[7]=vaddvq_f32(a7);
}

/* ============================ variants ============================ */
enum { V_RAW, V_BF16, V_FP8_GATHER, V_FP8_MAGIC, V_FP8_SCALAR, V_FP8_NEON, V_MXFP4 };
static const char *vname[]={"R","bf16","fp8-gather","fp8-magic","fp8-scalar","fp8-neon","mxfp4"};

/* ============================ shared bench context ============================ */
typedef struct {
    int variant, nrow, K, active, P, mat_idx;
    uint8_t *wbase;          /* weight pool: P matrices, mat_stride bytes apart */
    size_t   mat_stride;
    int      row_bytes;      /* weight bytes/row (fp8 K, bf16 2K, mxfp4 K/2, raw=K or K/2) */
    uint8_t *ESp; size_t es_stride; int nblk;   /* FP8 per-(group,block) E8M0 */
    uint8_t *MSp; size_t ms_stride; int msblk;  /* MXFP4 per-row E8M0 (K/32) */
    const uint32_t *lut;
    const float *X; float *DST;
    double sink[64];         /* per-thread raw sink (one cacheline each: 64 doubles) */
} bctx;
static bctx g;

/* NUMA-correct first-touch: each thread writes the weight rows it will later read,
 * across all P pool matrices -> pages land on the reading thread's CMG. */
static void fill_weight_worker(void *arg, int tid, int nthr){
    (void)arg;(void)nthr;
    if (tid >= g.active) return;
    int r0,r1; rowsplit8(g.nrow, g.active, tid, &r0,&r1);
    size_t lo=(size_t)r0*g.row_bytes, hi=(size_t)r1*g.row_bytes;
    for (int p=0;p<g.P;p++){
        uint8_t *base = g.wbase + (size_t)p*g.mat_stride;
        for (size_t i=lo;i<hi;i++) base[i]=(uint8_t)(i*131u+7u+(unsigned)p+(unsigned)tid);
    }
}

/* one matvec sweep of the current cold matrix; tid owns its ds4f_rowsplit8 rows */
static void mv_worker(void *arg, int tid, int nthr){
    (void)arg;(void)nthr;
    if (tid >= g.active) return;
    if (g.variant==V_FP8_MAGIC || g.variant==V_FP8_NEON) set_ftz();
    int r0,r1; rowsplit8(g.nrow, g.active, tid, &r0,&r1);
    uint8_t *mb = g.wbase + (size_t)g.mat_idx*g.mat_stride;
    if (g.variant==V_RAW){
        /* True read ceiling: 8 INDEPENDENT eor accumulators (one loop-carried chain
         * serializes at the ~4-cyc eor latency and undercounts BW badly). Streams
         * 8x64B per step, mirroring the 8-row kernels' load pattern. */
        svbool_t pb=svptrue_b8(), pg=svptrue_b32();
        svuint32_t a0=svdup_u32(0),a1=svdup_u32(0),a2=svdup_u32(0),a3=svdup_u32(0);
        svuint32_t a4=svdup_u32(0),a5=svdup_u32(0),a6=svdup_u32(0),a7=svdup_u32(0);
        size_t lo=(size_t)r0*g.row_bytes, hi=(size_t)r1*g.row_bytes, o=lo;
        for (; o+512<=hi; o+=512){
            a0=sveor_u32_x(pg,a0,svreinterpret_u32_u8(svld1_u8(pb,mb+o)));
            a1=sveor_u32_x(pg,a1,svreinterpret_u32_u8(svld1_u8(pb,mb+o+64)));
            a2=sveor_u32_x(pg,a2,svreinterpret_u32_u8(svld1_u8(pb,mb+o+128)));
            a3=sveor_u32_x(pg,a3,svreinterpret_u32_u8(svld1_u8(pb,mb+o+192)));
            a4=sveor_u32_x(pg,a4,svreinterpret_u32_u8(svld1_u8(pb,mb+o+256)));
            a5=sveor_u32_x(pg,a5,svreinterpret_u32_u8(svld1_u8(pb,mb+o+320)));
            a6=sveor_u32_x(pg,a6,svreinterpret_u32_u8(svld1_u8(pb,mb+o+384)));
            a7=sveor_u32_x(pg,a7,svreinterpret_u32_u8(svld1_u8(pb,mb+o+448)));
        }
        for (; o+64<=hi; o+=64) a0=sveor_u32_x(pg,a0,svreinterpret_u32_u8(svld1_u8(pb,mb+o)));
        a0=sveor_u32_x(pg,sveor_u32_x(pg,a0,a1),sveor_u32_x(pg,a2,a3));
        a4=sveor_u32_x(pg,sveor_u32_x(pg,a4,a5),sveor_u32_x(pg,a6,a7));
        a0=sveor_u32_x(pg,a0,a4);
        uint32_t tmp[64]; int vl=(int)svcntw(); svst1_u32(pg,tmp,a0);
        double s=0; for(int i=0;i<vl;i++) s+=tmp[i]; g.sink[tid]+=s;
        return;
    }
    for (int g0=r0; g0<r1; g0+=8){
        if (g.variant==V_BF16){
            const uint16_t *wb=(const uint16_t*)mb;
            matvec_bf16_8row(g.DST+g0,
                wb+(size_t)(g0+0)*g.K,wb+(size_t)(g0+1)*g.K,wb+(size_t)(g0+2)*g.K,wb+(size_t)(g0+3)*g.K,
                wb+(size_t)(g0+4)*g.K,wb+(size_t)(g0+5)*g.K,wb+(size_t)(g0+6)*g.K,wb+(size_t)(g0+7)*g.K,
                g.X,g.K);
        } else if (g.variant==V_MXFP4){
            const uint8_t *ms=g.MSp+(size_t)g.mat_idx*g.ms_stride;
            matvec_mxfp4_8row(g.DST+g0,
                mb+(size_t)(g0+0)*g.row_bytes,mb+(size_t)(g0+1)*g.row_bytes,
                mb+(size_t)(g0+2)*g.row_bytes,mb+(size_t)(g0+3)*g.row_bytes,
                mb+(size_t)(g0+4)*g.row_bytes,mb+(size_t)(g0+5)*g.row_bytes,
                mb+(size_t)(g0+6)*g.row_bytes,mb+(size_t)(g0+7)*g.row_bytes,
                ms+(size_t)(g0+0)*g.msblk,ms+(size_t)(g0+1)*g.msblk,
                ms+(size_t)(g0+2)*g.msblk,ms+(size_t)(g0+3)*g.msblk,
                ms+(size_t)(g0+4)*g.msblk,ms+(size_t)(g0+5)*g.msblk,
                ms+(size_t)(g0+6)*g.msblk,ms+(size_t)(g0+7)*g.msblk,
                g.X,g.K);
        } else {
            const uint8_t *w0=mb+(size_t)(g0+0)*g.K,*w1=mb+(size_t)(g0+1)*g.K;
            const uint8_t *w2=mb+(size_t)(g0+2)*g.K,*w3=mb+(size_t)(g0+3)*g.K;
            const uint8_t *w4=mb+(size_t)(g0+4)*g.K,*w5=mb+(size_t)(g0+5)*g.K;
            const uint8_t *w6=mb+(size_t)(g0+6)*g.K,*w7=mb+(size_t)(g0+7)*g.K;
            const uint8_t *es=g.ESp+(size_t)g.mat_idx*g.es_stride+(size_t)(g0/8)*g.nblk;
            if (g.variant==V_FP8_GATHER)
                matvec_fp8e4m3_8row(g.DST+g0,w0,w1,w2,w3,w4,w5,w6,w7,es,g.lut,g.X,g.K);
            else if (g.variant==V_FP8_MAGIC)
                matvec_fp8e4m3_8row_magic(g.DST+g0,w0,w1,w2,w3,w4,w5,w6,w7,es,g.X,g.K);
            else if (g.variant==V_FP8_SCALAR)
                matvec_fp8e4m3_8row_scalar(g.DST+g0,w0,w1,w2,w3,w4,w5,w6,w7,es,g.X,g.K);
            else /* V_FP8_NEON */
                matvec_fp8e4m3_8row_neon(g.DST+g0,w0,w1,w2,w3,w4,w5,w6,w7,es,g.X,g.K);
        }
    }
}

static double measure(bpool *pool, int iters){
    for(int w=0;w<2;w++){ g.mat_idx=w%g.P; bpool_run(pool,mv_worker,NULL); }  /* warmup */
    for(int t=0;t<64;t++) g.sink[t]=0;
    uint64_t t0=rdcyc();
    for(int it=0;it<iters;it++){ g.mat_idx=it%g.P; bpool_run(pool,mv_worker,NULL); }
    uint64_t t1=rdcyc();
    return (double)(t1-t0)/(double)rdfreq();
}

/* ============================ one (shape,variant,nthr) measurement ============================ *
 * Allocates a fresh P-matrix pool sized to `pool_bytes`, fills NUMA-correct via the
 * pool, times `iters` cold sweeps, prints, frees. Returns effective GB/s. r_gbps:
 * the same-shape R ceiling (pass 0 for the R run itself; we print %-of-R otherwise). */
static double bench_one(bpool *pool, int variant, int nrow, int K, int mxfp4_shape,
                        size_t pool_bytes, double r_gbps){
    int ngrp=nrow/8, nblk=K/128;
    g.variant=variant; g.nrow=nrow; g.K=K; g.active=pool->nthr; g.nblk=nblk;
    g.ESp=NULL; g.MSp=NULL; g.es_stride=0; g.ms_stride=0; g.msblk=K/32;
    int row_bytes, mac_per_sweep=nrow*K; size_t scale_pool=0;
    switch(variant){
        case V_BF16:       row_bytes=2*K; break;
        case V_MXFP4:      row_bytes=K/2; break;
        case V_RAW:        row_bytes=mxfp4_shape?K/2:K; break;
        default:           row_bytes=K;   break;  /* fp8-* */
    }
    size_t mat=(size_t)nrow*row_bytes;
    int P=(int)(pool_bytes/mat); if(P<1)P=1;
    g.P=P; g.mat_stride=mat; g.row_bytes=row_bytes;
    size_t wbytes=(size_t)P*mat;
    g.wbase=(uint8_t*)xmap(wbytes);   /* fresh pages -> pooled first-touch spreads over CMGs */
    /* per-variant scale pools (tiny, serial fill — placement irrelevant to BW) */
    if (variant==V_FP8_GATHER||variant==V_FP8_MAGIC||variant==V_FP8_SCALAR||variant==V_FP8_NEON){
        g.es_stride=(size_t)ngrp*nblk; scale_pool=(size_t)P*g.es_stride;
        g.ESp=(uint8_t*)aligned_alloc(256,scale_pool);
        for(size_t i=0;i<scale_pool;i++) g.ESp[i]=(uint8_t)(125+(smn()%5));
    } else if (variant==V_MXFP4){
        g.ms_stride=(size_t)nrow*g.msblk; scale_pool=(size_t)P*g.ms_stride;
        g.MSp=(uint8_t*)aligned_alloc(256,scale_pool);
        for(size_t i=0;i<scale_pool;i++) g.MSp[i]=(uint8_t)(125+(smn()%5));
    }
    bpool_run(pool,fill_weight_worker,NULL);          /* NUMA first-touch at this nthr */

    int iters=(int)(2.0e9/(double)mat); if(iters<5)iters=5; if(iters>200)iters=200;
    double sec=measure(pool,iters);
    double bytes=(double)mat;                          /* weight bytes / sweep (HBM traffic) */
    double gbps=bytes*iters/sec/1e9;
    double ms=sec/iters*1e3;
    double gmacs=(double)mac_per_sweep*iters/sec/1e9;
    double sink=0; for(int t=0;t<pool->nthr;t++) sink+=g.sink[t];
    if (variant==V_RAW)
        printf("    [%-10s] %7.1f GB/s  %8.3f ms  (%zu MB/mat x%d, sink=%.3g)\n",
               vname[variant],gbps,ms,mat>>20,P,sink+g.DST[0]);
    else
        printf("    [%-10s] %7.1f GB/s  %7.1f Gmac/s  %8.3f ms  %s%5.1f%% of R\n",
               vname[variant],gbps,gmacs,ms,
               (variant==V_BF16?"(2B) ":""), r_gbps>0?100.0*gbps/r_gbps:0.0);
    xunmap(g.wbase,wbytes); free(g.ESp); free(g.MSp); g.ESp=NULL; g.MSp=NULL;
    return gbps;
}

/* ============================ numeric gate ============================ */
static void argmax_top5(const float *v,int n,int top[5]){
    int idx[5]; float val[5]; for(int k=0;k<5;k++){idx[k]=-1;val[k]=-INFINITY;}
    for(int i=0;i<n;i++){ float x=v[i];
        for(int k=0;k<5;k++) if(x>val[k]){ for(int j=4;j>k;j--){val[j]=val[j-1];idx[j]=idx[j-1];}
            val[k]=x; idx[k]=i; break; } }
    for(int k=0;k<5;k++) top[k]=idx[k];
}
static int set_eq5(const int *a,const int *b){ for(int i=0;i<5;i++){ int f=0;
    for(int j=0;j<5;j++) if(a[i]==b[j]){f=1;break;} if(!f) return 0;} return 1; }
static int gate_report(const char *tag,const float *ref,const float *got,int n){
    int tr[5],tg[5]; argmax_top5(ref,n,tr); argmax_top5(got,n,tg);
    double mr=0; for(int i=0;i<n;i++){ double d=fabs((double)got[i]-(double)ref[i]);
        double den=fabs((double)ref[i])+1e-6; if(d/den>mr) mr=d/den; }
    int am=(tr[0]==tg[0]), t5=set_eq5(tr,tg);
    printf("    [gate %-10s] argmax:%s top5:%s max-rel:%.1e\n",
           tag, am?"PASS":"FAIL", t5?"PASS":"FAIL", mr);
    return am && t5;
}
/* gate one shape single-threaded over nrow_g rows (own-repr f32 double-accum ref) */
static int gate_shape(const char *shape,int nrow,int K,int mxfp4_shape,
                      const int *vars,int nv,const uint32_t *lut){
    int nrow_g = nrow>2048?2048:nrow; if(nrow_g%8) nrow_g-=nrow_g%8;
    int ngrp=nrow_g/8, nblk=K/128, msblk=K/32;
    float *x=(float*)aligned_alloc(256,(size_t)K*4); for(int i=0;i<K;i++) x[i]=rand_unit();
    float *ref=(float*)aligned_alloc(256,(size_t)nrow_g*4);
    float *got=(float*)aligned_alloc(256,(size_t)nrow_g*4);
    int ok=1;
    printf("  gate %s [%d x %d, %d rows]\n",shape,nrow,K,nrow_g);
    if (mxfp4_shape){
        uint8_t *W=(uint8_t*)aligned_alloc(256,(size_t)nrow_g*(K/2));
        uint8_t *S=(uint8_t*)aligned_alloc(256,(size_t)nrow_g*msblk);
        for(size_t i=0;i<(size_t)nrow_g*(K/2);i++) W[i]=(uint8_t)(smn()&0xff);
        for(size_t i=0;i<(size_t)nrow_g*msblk;i++) S[i]=(uint8_t)(125+(smn()%5));
        for(int r=0;r<nrow_g;r++){ const uint8_t *w=W+(size_t)r*(K/2),*s=S+(size_t)r*msblk;
            double a=0; int nb=K/32;
            for(int b=0;b<nb;b++){ float sc=ggml_e8m0_to_fp32(s[b]);
                for(int j=0;j<16;j++){ uint8_t by=w[(size_t)b*16+j];
                    a+=(double)(ds4f_kvalues_mxfp4_f32[by&0xf]*sc)*(double)x[b*32+j];
                    a+=(double)(ds4f_kvalues_mxfp4_f32[(by>>4)&0xf]*sc)*(double)x[b*32+j+16]; } }
            ref[r]=(float)a; }
        for(int g0=0;g0<nrow_g;g0+=8){ const uint8_t *w[8],*s[8];
            for(int r=0;r<8;r++){ w[r]=W+(size_t)(g0+r)*(K/2); s[r]=S+(size_t)(g0+r)*msblk; }
            matvec_mxfp4_8row(got+g0,w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],
                s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],x,K); }
        ok &= gate_report("mxfp4",ref,got,nrow_g);
        free(W);free(S);
    } else {
        uint8_t  *W =(uint8_t*) aligned_alloc(256,(size_t)nrow_g*K);
        uint8_t  *ES=(uint8_t*) aligned_alloc(256,(size_t)ngrp*nblk);
        uint16_t *WB=(uint16_t*)aligned_alloc(256,(size_t)nrow_g*K*2);
        for(size_t i=0;i<(size_t)ngrp*nblk;i++) ES[i]=(uint8_t)(125+(smn()%5));
        for(size_t i=0;i<(size_t)nrow_g*K;i++) W[i]=rand_fp8();
        /* bf16 predequant: fold the group E8M0 into the weight */
        for(int gg=0;gg<ngrp;gg++) for(int r=0;r<8;r++){
            const uint8_t *wr=W+((size_t)gg*8+r)*K; uint16_t *wb=WB+((size_t)gg*8+r)*K;
            for(int c=0;c<K;c++){ float f=bits2f(ds4f_fp8_e4m3_to_fp32_bits(wr[c]));
                f*=ggml_e8m0_to_fp32(ES[(size_t)gg*nblk+(c>>7)]);
                uint32_t u; memcpy(&u,&f,4); wb[c]=(uint16_t)(u>>16); } }
        for(int v=0;v<nv;v++){ int var=vars[v]; if(var==V_RAW) continue;
            if(var==V_BF16){
                for(int r=0;r<nrow_g;r++){ const uint16_t *wb=WB+(size_t)r*K; double a=0;
                    for(int c=0;c<K;c++) a+=(double)bf16f(wb[c])*(double)x[c]; ref[r]=(float)a; }
                for(int g0=0;g0<nrow_g;g0+=8){ const uint16_t *w[8];
                    for(int r=0;r<8;r++) w[r]=WB+(size_t)(g0+r)*K;
                    matvec_bf16_8row(got+g0,w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],x,K); }
                ok &= gate_report("bf16",ref,got,nrow_g);
            } else { /* fp8-* share the FP8 f32 ref */
                for(int r=0;r<nrow_g;r++){ const uint8_t *wr=W+(size_t)r*K; double a=0;
                    for(int c=0;c<K;c++) a+=(double)(bits2f(ds4f_fp8_e4m3_to_fp32_bits(wr[c]))
                        *ggml_e8m0_to_fp32(ES[(size_t)(r/8)*nblk+(c>>7)]))*(double)x[c];
                    ref[r]=(float)a; }
                set_ftz();
                for(int g0=0;g0<nrow_g;g0+=8){ const uint8_t *w[8];
                    for(int r=0;r<8;r++) w[r]=W+(size_t)(g0+r)*K;
                    const uint8_t *es=ES+(size_t)(g0/8)*nblk;
                    if(var==V_FP8_GATHER) matvec_fp8e4m3_8row(got+g0,w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],es,lut,x,K);
                    else if(var==V_FP8_MAGIC) matvec_fp8e4m3_8row_magic(got+g0,w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],es,x,K);
                    else if(var==V_FP8_SCALAR) matvec_fp8e4m3_8row_scalar(got+g0,w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],es,x,K);
                    else matvec_fp8e4m3_8row_neon(got+g0,w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],es,x,K); }
                ok &= gate_report(vname[var],ref,got,nrow_g);
            }
        }
        free(W);free(ES);free(WB);
    }
    free(x);free(ref);free(got);
    return ok;
}

typedef struct { const char *name; int nrow, K, mxfp4; } shape_t;

int main(void){
    sm=0xD5F0F00D00000001ULL;
    uint32_t lut[256]; ds4f_init_fp8_e4m3_lut(lut); g.lut=lut;
    for(int i=0;i<256;i++) g_lutf[i]=bits2f(lut[i]);
    size_t pool_bytes=((size_t)(getenv("DS4F_POOL_MB")?atoi(getenv("DS4F_POOL_MB")):384))<<20;
    printf("DS4F decode-bw bench  pool=%zu MB/shape  freq=%.1f MHz\n",
           pool_bytes>>20,(double)rdfreq()/1e6);
    g.X  =(float*)aligned_alloc(256,(size_t)8192*4);
    g.DST=(float*)aligned_alloc(256,(size_t)32768*4);
    for(int i=0;i<8192;i++) ((float*)g.X)[i]=rand_unit();

    /* real DS4F decode matvec shapes (rows, K=hidden); last two are MXFP4 experts */
    shape_t dense[]={
        {"wkv   [512,4096]",   512,4096,0}, {"wq_a  [1024,4096]", 1024,4096,0},
        {"shared[2048,4096]", 2048,4096,0}, {"wo_b  [4096,4096]", 4096,4096,0},
        {"wo_a  [8192,4096]", 8192,4096,0}, {"wq_b  [32768,1024]",32768,1024,0},
        {"bigK  [4096,8192]", 4096,8192,0},
    };
    shape_t expert[]={ {"expert[2048,4096]",2048,4096,1}, {"expert[4096,8192]",4096,8192,1} };
    int ndense=(int)(sizeof(dense)/sizeof(dense[0]));
    int nexp=(int)(sizeof(expert)/sizeof(expert[0]));

    /* ---------- numeric gate (correctness, single-thread) ---------- */
    printf("\n===== NUMERIC GATE (argmax + top5 vs own-repr f32 ref) =====\n");
    int gate_all=1;
    int dvars[]={V_BF16,V_FP8_GATHER,V_FP8_MAGIC,V_FP8_SCALAR,V_FP8_NEON};
    int evars[]={V_MXFP4};
    for(int s=0;s<ndense;s++)
        gate_all &= gate_shape(dense[s].name,dense[s].nrow,dense[s].K,0,dvars,5,lut);
    for(int s=0;s<nexp;s++)
        gate_all &= gate_shape(expert[s].name,expert[s].nrow,expert[s].K,1,evars,1,lut);
    printf("  GATE OVERALL: %s\n", gate_all?"PASS":"FAIL");

    /* ---------- Phase A: thread-scaling roofline (representative shapes) ---------- */
    printf("\n===== PHASE A: thread-scaling roofline =====\n");
    int nthrs[]={1,12,24,48};
    shape_t scaleshapes[]={ dense[4] /*wo_a 8192x4096*/, expert[1] /*4096x8192*/ };
    int sav[]={V_RAW,V_BF16,V_FP8_GATHER,V_FP8_MAGIC};
    for(int ss=0; ss<2; ss++){
        shape_t s=scaleshapes[ss];
        printf("  -- %s --\n", s.name);
        for(int ni=0;ni<4;ni++){
            int nthr=nthrs[ni], ncmg=nthr<4?nthr:4;
            bpool *pool=bpool_start(nthr,ncmg);
            double rg=0;
            for(int vi=0; vi<(s.mxfp4?2:4); vi++){
                int var = s.mxfp4 ? (vi==0?V_RAW:V_MXFP4) : sav[vi];
                printf("   nthr=%2d",nthr);
                double gb=bench_one(pool,var,s.nrow,s.K,s.mxfp4,pool_bytes,rg);
                if(var==V_RAW) rg=gb;
            }
            bpool_stop(pool);
        }
    }

    /* ---------- Phase B: per-shape @ 48 threads (the production decode width) ---------- */
    printf("\n===== PHASE B: per-shape @ 48 threads =====\n");
    double best_magic=0;
    bpool *pool=bpool_start(48,4);
    int bvars[]={V_RAW,V_BF16,V_FP8_GATHER,V_FP8_MAGIC,V_FP8_SCALAR,V_FP8_NEON};
    for(int s=0;s<ndense;s++){
        printf("  -- %s --\n", dense[s].name);
        double rg=0,magic=0,gather=0,bf=0;
        for(int v=0;v<6;v++){
            double gb=bench_one(pool,bvars[v],dense[s].nrow,dense[s].K,0,pool_bytes,rg);
            if(bvars[v]==V_RAW) rg=gb; else if(bvars[v]==V_FP8_MAGIC) magic=gb;
            else if(bvars[v]==V_FP8_GATHER) gather=gb; else if(bvars[v]==V_BF16) bf=gb;
        }
        if(magic>best_magic) best_magic=magic;
        printf("  SUMMARY %s: magic %.1f GB/s vs gather %.1f (%.2fx) vs bf16 %.1f GB/s "
               "(magic/bf16 ms ratio %.2fx)\n", dense[s].name, magic,gather,
               gather>0?magic/gather:0, bf, (bf>0&&magic>0)?(bf/2.0)/(magic):0);
    }
    for(int s=0;s<nexp;s++){
        printf("  -- %s --\n", expert[s].name);
        double rg=0;
        double r=bench_one(pool,V_RAW,expert[s].nrow,expert[s].K,1,pool_bytes,0); rg=r;
        double m=bench_one(pool,V_MXFP4,expert[s].nrow,expert[s].K,1,pool_bytes,rg);
        printf("  SUMMARY %s: mxfp4 %.1f GB/s (%.1f%% of R %.1f)\n",
               expert[s].name, m, rg>0?100*m/rg:0, rg);
    }
    bpool_stop(pool);

    /* ---------- optional illustrative dense-read tok/s projection ---------- */
    if (getenv("DS4F_DENSE_GB")){
        double dgb=atof(getenv("DS4F_DENSE_GB"));
        double df=getenv("DS4F_DENSE_FRAC")?atof(getenv("DS4F_DENSE_FRAC")):0.60;
        double dense_s=dgb/best_magic;        /* s to read all dense FP8 weights once */
        printf("\n===== PROJECTION (illustrative) =====\n");
        printf("  best 48T fp8-magic = %.1f GB/s ; dense FP8 = %.2f GB/token\n", best_magic,dgb);
        printf("  dense-read-only: %.1f tok/s ; folding dense_fraction=%.2f: ~%.1f tok/s\n",
               1.0/dense_s, df, df/dense_s);
        printf("  (weight-dequant lever is Amdahl-capped at the dense+expert fraction;\n"
               "   indexer/attn activations are a SEPARATE lever — see longctx memory)\n");
    }
    printf("\nDONE  (gate %s)\n", gate_all?"PASS":"FAIL");
    return 0;
}
