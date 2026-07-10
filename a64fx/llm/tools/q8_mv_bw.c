/* Q8_PV vs BF16_PV M=1 matvec bandwidth probe (single node, no alloc).
 *
 * Question (D2): production decode dense matvecs run ~340 GB/s aggregate while the
 * bf16-pv roofline (ds4f_decode_bw_bench, reader-local first-touch) is ~610 GB/s.
 * Q8_LOCAL (reader-local q8 placement) was NEUTRAL on 11n -> is ~340 the int8-sdot
 * KERNEL ceiling, or a placement/model artifact? This probe runs the exact
 * production dispatch (ds4f_matvec -> mv_worker) over a POOL of distinct cold
 * tensors (defeats L2 caching), first-touched by the SAME pool split the reader
 * uses (via ds4f_repack / parallel fill), on compute cores 12-59.
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *       -I../../common -o build/q8_mv_bw tools/q8_mv_bw.c -lm -lpthread -lhwb
 *   DS4F_FLAGBAR=1 taskset -c 12-59 ./build/q8_mv_bw [nthr=48] [iters=40]
 */
#include "ds4f.h"
#include <math.h>

static uint32_t rng = 0xFEED5EEDu;
static inline uint32_t nr(void){ rng = rng*1664525u + 1013904223u; return rng; }
static inline float frand(void){ return ((float)(nr()>>8)/(float)(1u<<24))*2.f-1.f; }

static void pack_pv(uint16_t *dst, const uint16_t *Wrm, int rows, int cols){
    for(int i=0;i<rows;i++){ size_t gb=(size_t)(i/8)*8*cols; int loc=i&7,pair=loc>>1,slot=loc&1;
        uint16_t *pb=dst+gb+(size_t)pair*2*cols; for(int j=0;j<cols;j++) pb[2*j+slot]=Wrm[(size_t)i*cols+j]; }
}
typedef struct { uint16_t *dst; const uint16_t *src; size_t n; } fill_task;
static void fill_worker(void *arg, int tid, int nthr){    /* parallel copy = distributed first-touch */
    fill_task *T = (fill_task*)arg;
    size_t i0 = T->n*tid/nthr, i1 = T->n*(tid+1)/nthr;
    memcpy(T->dst+i0, T->src+i0, (i1-i0)*2);
}


/* v2 sdot kernel prototype: replace the 8 scalar fp16->f32 scale converts + 8 svdup per
 * block with one 8-half fcvt + xs-mul + two ld1rq lane-broadcasts + svmla_lane (index
 * selects within each 128-bit segment -> scv_lo lanes 0..3 = rows 0..3, scv_hi = 4..7).
 * BIT-EXACT: same IEEE mul (ws*xs) and same FMA per lane as the svdup path. */
static inline void matvec_sdot_8row_v2(float *dst, const uint8_t *group,
                                       const int8_t *xq, const float *xscale, int K) {
    svbool_t pg = svptrue_b32();
    svbool_t pb = svptrue_b8();
    svbool_t p8h = svwhilelt_b16(0, 8);
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
    int nb = K / 64;
    float scb[8] __attribute__((aligned(32)));
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = group + (size_t)b * 528;
        const int8_t *qs   = (const int8_t *)(blk + 16);
        /* 8 fp16 row-scales -> f32, * xs, to stack, then ld1rq segment-broadcast */
        svfloat16_t h = svld1_f16(p8h, (const float16_t *)blk);
        svfloat32_t sc = svcvt_f32_f16_x(p8h, svzip1_f16(h, h));   /* widen even lanes: zip then cvt keeps order? */
        (void)sc;
        /* simpler + surely-exact: scalar convert loop kept but vector mul (compiler may vectorize) */
        for (int r = 0; r < 8; r++) scb[r] = ggml_fp16_to_fp32(((const uint16_t *)blk)[r]) * xscale[b];
        svfloat32_t sclo = svld1rq_f32(pg, scb);       /* rows 0..3 per segment */
        svfloat32_t schi = svld1rq_f32(pg, scb + 4);   /* rows 4..7 per segment */
        svint8_t xv = svld1_s8(pb, xq + (size_t)b * 64);
        #define SDOT_ROW2(R, ACC, SCV, LN)                                     \
            do {                                                               \
                svint8_t wv = svld1_s8(pb, qs + (size_t)(R) * 64);             \
                svint32_t d = svdot_s32(svdup_s32(0), wv, xv);                 \
                ACC = svmla_lane_f32(ACC, svcvt_f32_s32_x(pg, d), SCV, LN);    \
            } while (0)
        SDOT_ROW2(0, a0, sclo, 0); SDOT_ROW2(1, a1, sclo, 1);
        SDOT_ROW2(2, a2, sclo, 2); SDOT_ROW2(3, a3, sclo, 3);
        SDOT_ROW2(4, a4, schi, 0); SDOT_ROW2(5, a5, schi, 1);
        SDOT_ROW2(6, a6, schi, 2); SDOT_ROW2(7, a7, schi, 3);
        #undef SDOT_ROW2
    }
    dst[0]=svaddv(pg,a0); dst[1]=svaddv(pg,a1); dst[2]=svaddv(pg,a2); dst[3]=svaddv(pg,a3);
    dst[4]=svaddv(pg,a4); dst[5]=svaddv(pg,a5); dst[6]=svaddv(pg,a6); dst[7]=svaddv(pg,a7);
}
typedef struct { float *dst; const ds4f_tensor *t; const float *x; } v2_task;
static __thread int8_t v2_xq[8192]; static __thread float v2_xs[128];
static void v2_worker(void *arg, int tid, int nthr) {
    v2_task *T = (v2_task*)arg; const ds4f_tensor *t = T->t;
    int K = t->cols; int r0, r1; ds4f_rowsplit8(t->rows, nthr, tid, &r0, &r1);
    const uint8_t *base = (const uint8_t *)t->w; size_t gb = (size_t)(K/64)*528;
    ds4f_quant_x_sdot_into(T->x, K, v2_xq, v2_xs);
    for (int i = r0; i + 7 < r1; i += 8)
        matvec_sdot_8row_v2(T->dst + i, base + (size_t)(i/8)*gb, v2_xq, v2_xs, K);
}

int main(int argc, char **argv) {
    int nthr = argc>1?atoi(argv[1]):48, iters = argc>2?atoi(argv[2]):40;
    ds4f_model m; memset(&m,0,sizeof(m)); m.n_threads=nthr; m.n_cmgs=4;
    m.pool = ds4f_pool_start(nthr, 4); ds4f_init_fp8_e4m3_lut(m.fp8_lut);
    int rows = 8192, K = 4096, P = 8;   /* wo_a shape; 8 distinct tensors = 537 MB bf16 pool (cold) */
    printf("q8_mv_bw nthr=%d iters=%d shape=[%d x %d] pool=%d\n", nthr, iters, rows, K, P);

    uint16_t *Wrm = malloc((size_t)rows*K*2);
    for (size_t i = 0; i < (size_t)rows*K; i++) Wrm[i] = ds4f_f32_bf16(frand());
    ds4f_tensor bf[8], q8[8];
    size_t wb = ds4f_wbytes(DS4F_BF16_PV, rows, K);
    for (int p = 0; p < P; p++) {
        memset(&bf[p], 0, sizeof(ds4f_tensor));
        bf[p].type = DS4F_BF16_PV; bf[p].rows = rows; bf[p].cols = K;
        bf[p].w = mmap(NULL, wb, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        uint16_t *tmp = malloc(wb); pack_pv(tmp, Wrm, rows, K);
        fill_task T = { (uint16_t*)bf[p].w, tmp, wb/2 };
        ds4f_pool_run(m.pool, fill_worker, &T);          /* first-touch distributed like a reader */
        free(tmp);
        q8[p] = bf[p];
        q8[p].w = mmap(NULL, wb, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        memcpy(q8[p].w, bf[p].w, wb);                    /* repack reads this; repack buffer is fresh mmap */
        ds4f_repack_bf16pv_to_q8pv(&m, &q8[p]);          /* worker-split fill = reader-local first-touch */
    }
    float *x = aligned_alloc(256, (size_t)K*4);
    for (int i = 0; i < K; i++) x[i] = frand();
    float *y = aligned_alloc(256, (size_t)rows*4);

    size_t q8b = ds4f_wbytes(DS4F_Q8_PV, rows, K);
    struct { const char *name; ds4f_tensor *t; size_t bytes; } V[2] = {
        { "bf16_pv", bf, wb }, { "q8_sdot ", q8, q8b } };
    for (int v = 0; v < 2; v++) {
        for (int p = 0; p < P; p++) ds4f_matvec(&m, y, &V[v].t[p], x);   /* warm dispatch path */
        double best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            double t0 = ds4f_now();
            for (int it = 0; it < iters; it++) ds4f_matvec(&m, y, &V[v].t[it % P], x);
            double dt = ds4f_now() - t0; if (dt < best) best = dt;
        }
        double per = best/iters;
        printf("  %s : %7.3f ms/matvec  %6.1f GB/s (weights)  %6.1f Gmac/s\n",
               V[v].name, per*1e3, V[v].bytes/per/1e9, (double)rows*K/per/1e9);
    }

    /* v2 kernel: ld1rq + svmla_lane scales (bit-exact vs v1) */
    { float *yref = aligned_alloc(256, (size_t)rows*4);
      ds4f_matvec(&m, yref, &q8[0], x);
      v2_task T = { y, &q8[0], x };
      ds4f_pool_run(m.pool, v2_worker, &T);
      int nbit = 0; for (int i = 0; i < rows; i++) if (y[i] == yref[i]) nbit++;
      double best = 1e30;
      for (int rep = 0; rep < 5; rep++) {
          double t0 = ds4f_now();
          for (int it = 0; it < iters; it++) { v2_task T2 = { y, &q8[it % P], x }; ds4f_pool_run(m.pool, v2_worker, &T2); }
          double dt = ds4f_now() - t0; if (dt < best) best = dt;
      }
      double per = best/iters;
      printf("  q8_v2    : %7.3f ms/matvec  %6.1f GB/s (weights)  %6.1f Gmac/s  bitexact=%d/%d\n",
             per*1e3, q8b/per/1e9, (double)rows*K/per/1e9, nbit, rows);
    }
    ds4f_pool_stop(m.pool);
    return 0;
}
