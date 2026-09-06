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

/* ---- v3: per-row weight scale + per-token activation scale, FULL-int32 K accumulation ----
 * The pinned ~390 Gmac/s ceiling is scale APPLICATION (per-64-block: 8 fp16->f32 + 8 mul +
 * 8 svmla PER BLOCK). If BOTH scales are per-row/per-token, out[r] = ws[r]*xs*int32_dot(all K),
 * so the per-block loop is just svdot (no fp/scale ops) and scaling moves to once-per-row.
 * No int32 overflow: K<=8192, |dot| <= 127*127*8192 = 132M < 2^31. LOSSY (per-token activation
 * scale is coarser than per-64-block) -> gate on real-gen. Layout "q8r": per 8-row group =
 * 8 fp32 row-scales (32 B) + 8 rows x K int8 row-major (8*K B) ~= 1.0 B/elem. */
typedef struct { const uint16_t *Wrm; uint8_t *dst; int rows, K; } q8r_repack_task;
static void q8r_repack_worker(void *arg, int tid, int nthr) {
    q8r_repack_task *T = (q8r_repack_task*)arg;
    int rows = T->rows, K = T->K, groups = rows/8;
    int per = groups/nthr, extra = groups%nthr;
    int g0 = per*tid + (tid<extra?tid:extra), g1 = g0 + per + (tid<extra?1:0);
    size_t gb = (size_t)32 + (size_t)8*K;   /* group bytes */
    for (int g = g0; g < g1; g++) {
        uint8_t *gp = T->dst + (size_t)g*gb;
        float *sc = (float*)gp; int8_t *qs = (int8_t*)(gp + 32);
        for (int r = 0; r < 8; r++) {
            const uint16_t *wr = T->Wrm + (size_t)(g*8 + r)*K;
            float amax = 0.f;
            for (int j = 0; j < K; j++) { float v = bf16_to_f32_scalar(wr[j]); v = v<0?-v:v; if (v>amax) amax=v; }
            float s = amax>0 ? amax/127.f : 0.f, inv = amax>0 ? 127.f/amax : 0.f;
            sc[r] = s;
            int8_t *qr = qs + (size_t)r*K;
            for (int j = 0; j < K; j++) { int q = (int)lrintf(bf16_to_f32_scalar(wr[j])*inv); qr[j] = (int8_t)(q>127?127:(q<-127?-127:q)); }
        }
    }
}
/* per-token int8 quant (ONE scale for the whole K vector). */
static void quant_x_pertoken(const float *x, int K, int8_t *xq, float *xs) {
    float amax = 0.f; for (int i = 0; i < K; i++) { float v = x[i]<0?-x[i]:x[i]; if (v>amax) amax=v; }
    float inv = amax>0 ? 127.f/amax : 0.f; *xs = amax>0 ? amax/127.f : 0.f;
    for (int i = 0; i < K; i++) { int q = (int)lrintf(x[i]*inv); xq[i] = (int8_t)(q>127?127:(q<-127?-127:q)); }
}
static inline void matvec_q8r_8row(float *dst, const uint8_t *group, const int8_t *xq, float xs, int K) {
    const float *sc = (const float*)group; const int8_t *qs = (const int8_t*)(group + 32);
    svbool_t pb = svptrue_b8(), p32 = svptrue_b32(); int bl = (int)svcntb();
    svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0);
    svint32_t a4=svdup_s32(0),a5=svdup_s32(0),a6=svdup_s32(0),a7=svdup_s32(0);
    for (int i = 0; i < K; i += bl) {
        svint8_t xv = svld1_s8(pb, xq + i);
        a0=svdot_s32(a0,svld1_s8(pb,qs+0*K+i),xv); a1=svdot_s32(a1,svld1_s8(pb,qs+1*K+i),xv);
        a2=svdot_s32(a2,svld1_s8(pb,qs+2*K+i),xv); a3=svdot_s32(a3,svld1_s8(pb,qs+3*K+i),xv);
        a4=svdot_s32(a4,svld1_s8(pb,qs+4*K+i),xv); a5=svdot_s32(a5,svld1_s8(pb,qs+5*K+i),xv);
        a6=svdot_s32(a6,svld1_s8(pb,qs+6*K+i),xv); a7=svdot_s32(a7,svld1_s8(pb,qs+7*K+i),xv);
    }
    dst[0]=(float)svaddv_s32(p32,a0)*sc[0]*xs; dst[1]=(float)svaddv_s32(p32,a1)*sc[1]*xs;
    dst[2]=(float)svaddv_s32(p32,a2)*sc[2]*xs; dst[3]=(float)svaddv_s32(p32,a3)*sc[3]*xs;
    dst[4]=(float)svaddv_s32(p32,a4)*sc[4]*xs; dst[5]=(float)svaddv_s32(p32,a5)*sc[5]*xs;
    dst[6]=(float)svaddv_s32(p32,a6)*sc[6]*xs; dst[7]=(float)svaddv_s32(p32,a7)*sc[7]*xs;
}
typedef struct { float *dst; const uint8_t *w; int rows, K; const int8_t *xq; float xs; } q8r_task;
static void q8r_worker(void *arg, int tid, int nthr) {
    q8r_task *T = (q8r_task*)arg; int rows = T->rows, K = T->K;
    int r0, r1; ds4f_rowsplit8(rows, nthr, tid, &r0, &r1);
    size_t gb = (size_t)32 + (size_t)8*K;
    for (int i = r0; i + 7 < r1; i += 8)
        matvec_q8r_8row(T->dst + i, T->w + (size_t)(i/8)*gb, T->xq, T->xs, K);
}

/* ---- v4: per-ROW weight scale (deferred to end) + per-64-BLOCK activation scale ----
 * Keeps production activation fidelity (SAFE — the spike test's damage was purely the
 * per-token activation scale; per-row WEIGHT scale was fine at every spike). Same q8r
 * row-major layout as v3. Kernel accumulates 16 f32 lanes per row (svmla by the per-block
 * activation scale, like production) then applies the per-row weight scale once at the end,
 * dropping production's per-block fp16-weight-convert + ws*xs scalar mul. xq/xs = the
 * existing per-64-block quantizer (ds4f_quant_x_sdot_into). */
static inline void matvec_q8r_blk_8row(float *dst, const uint8_t *group,
                                       const int8_t *xq, const float *xs, int K) {
    const float *sc = (const float*)group; const int8_t *qs = (const int8_t*)(group + 32);
    svbool_t pb = svptrue_b8(), pg = svptrue_b32();
    svfloat32_t f0=svdup_f32(0),f1=svdup_f32(0),f2=svdup_f32(0),f3=svdup_f32(0);
    svfloat32_t f4=svdup_f32(0),f5=svdup_f32(0),f6=svdup_f32(0),f7=svdup_f32(0);
    int nb = K/64;
    for (int b = 0; b < nb; b++) {
        int o = b*64; svint8_t xv = svld1_s8(pb, xq + o); svfloat32_t xsb = svdup_f32(xs[b]);
        #define V4ROW(R,F) F=svmla_x(pg,F,svcvt_f32_s32_x(pg,svdot_s32(svdup_s32(0),svld1_s8(pb,qs+(size_t)(R)*K+o),xv)),xsb)
        V4ROW(0,f0);V4ROW(1,f1);V4ROW(2,f2);V4ROW(3,f3);V4ROW(4,f4);V4ROW(5,f5);V4ROW(6,f6);V4ROW(7,f7);
        #undef V4ROW
    }
    dst[0]=svaddv(pg,f0)*sc[0]; dst[1]=svaddv(pg,f1)*sc[1]; dst[2]=svaddv(pg,f2)*sc[2]; dst[3]=svaddv(pg,f3)*sc[3];
    dst[4]=svaddv(pg,f4)*sc[4]; dst[5]=svaddv(pg,f5)*sc[5]; dst[6]=svaddv(pg,f6)*sc[6]; dst[7]=svaddv(pg,f7)*sc[7];
}
typedef struct { float *dst; const uint8_t *w; int rows, K; const int8_t *xq; const float *xs; } q8r_blk_task;
static void q8r_blk_worker(void *arg, int tid, int nthr) {
    q8r_blk_task *T = (q8r_blk_task*)arg; int rows = T->rows, K = T->K;
    int r0, r1; ds4f_rowsplit8(rows, nthr, tid, &r0, &r1);
    size_t gb = (size_t)32 + (size_t)8*K;
    for (int i = r0; i + 7 < r1; i += 8)
        matvec_q8r_blk_8row(T->dst + i, T->w + (size_t)(i/8)*gb, T->xq, T->xs, K);
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
    /* v3: per-row-scale + per-token-activation full-int32 kernel (LOSSY, speed + accuracy) */
    { size_t gb = (size_t)32 + (size_t)8*K, q8rb = (size_t)(rows/8)*gb;
      uint8_t *q8r[8];
      for (int p = 0; p < P; p++) {
          q8r[p] = mmap(NULL, q8rb, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
          q8r_repack_task T = { Wrm, q8r[p], rows, K };
          ds4f_pool_run(m.pool, q8r_repack_worker, &T);    /* group-split = reader-local first-touch */
      }
      int8_t *xq = aligned_alloc(64, K); float xs; quant_x_pertoken(x, K, xq, &xs);
      /* f32 reference (full-precision W x full-precision x) for relL2 of BOTH q8 reps */
      float *fref = aligned_alloc(256, (size_t)rows*4);
      for (int r = 0; r < rows; r++) { double a = 0; const uint16_t *wr = Wrm + (size_t)r*K;
          for (int j = 0; j < K; j++) a += (double)bf16_to_f32_scalar(wr[j])*(double)x[j]; fref[r] = (float)a; }
      q8r_task T = { y, q8r[0], rows, K, xq, xs };
      ds4f_pool_run(m.pool, q8r_worker, &T);
      double e=0,n=0; for (int r=0;r<rows;r++){ double d=y[r]-fref[r]; e+=d*d; n+=(double)fref[r]*fref[r]; }
      double rel_v3 = sqrt(e/n);
      /* also report the production per-64-block q8's relL2 vs the same f32 ref */
      ds4f_matvec(&m, y, &q8[0], x);
      double e2=0,n2=0; for (int r=0;r<rows;r++){ double d=y[r]-fref[r]; e2+=d*d; n2+=(double)fref[r]*fref[r]; }
      double rel_blk = sqrt(e2/n2);
      double best = 1e30;
      for (int rep = 0; rep < 5; rep++) {
          double t0 = ds4f_now();
          for (int it = 0; it < iters; it++) { q8r_task T2 = { y, q8r[it%P], rows, K, xq, xs }; ds4f_pool_run(m.pool, q8r_worker, &T2); }
          double dt = ds4f_now() - t0; if (dt < best) best = dt;
      }
      double per = best/iters;
      printf("  q8_v3    : %7.3f ms/matvec  %6.1f GB/s (weights)  %6.1f Gmac/s  relL2=%.2e (per-64blk q8 relL2=%.2e)\n",
             per*1e3, q8rb/per/1e9, (double)rows*K/per/1e9, rel_v3, rel_blk);
    }
    /* v4: per-row weight + per-64-block activation (SAFE). Speed + spike accuracy. */
    { size_t gb = (size_t)32 + (size_t)8*K, q8rb = (size_t)(rows/8)*gb;
      uint8_t *q8r[8];
      for (int p = 0; p < P; p++) { q8r[p] = mmap(NULL,q8rb,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
          q8r_repack_task T = { Wrm, q8r[p], rows, K }; ds4f_pool_run(m.pool, q8r_repack_worker, &T); }
      int8_t *xq = aligned_alloc(64, K); float xsb[128]; ds4f_quant_x_sdot_into(x, K, xq, xsb);
      float *fref = aligned_alloc(256, (size_t)rows*4);
      for (int r=0;r<rows;r++){ double a=0; const uint16_t *wr=Wrm+(size_t)r*K;
          for (int j=0;j<K;j++) a+=(double)bf16_to_f32_scalar(wr[j])*(double)x[j]; fref[r]=(float)a; }
      q8r_blk_task T = { y, q8r[0], rows, K, xq, xsb }; ds4f_pool_run(m.pool, q8r_blk_worker, &T);
      double e=0,n=0; for(int r=0;r<rows;r++){ double d=y[r]-fref[r]; e+=d*d; n+=(double)fref[r]*fref[r]; }
      double best=1e30;
      for (int rep=0; rep<5; rep++){ double t0=ds4f_now();
          for (int it=0; it<iters; it++){ q8r_blk_task T2={y,q8r[it%P],rows,K,xq,xsb}; ds4f_pool_run(m.pool,q8r_blk_worker,&T2);}
          double dt=ds4f_now()-t0; if(dt<best)best=dt; }
      double per=best/iters;
      printf("  q8_v4    : %7.3f ms/matvec  %6.1f GB/s (weights)  %6.1f Gmac/s  relL2=%.2e (per-row wt + per-64blk act, SAFE)\n",
             per*1e3, q8rb/per/1e9, (double)rows*K/per/1e9, sqrt(e/n));
      free(fref); free(xq); for(int p=0;p<P;p++) munmap(q8r[p],q8rb);
    }
    /* v3/v4 accuracy under a massive-activation input (the real risk: per-token scale collapses the
     * O(1) channels when one channel is ~1e3). Re-quant x with a spike, compare v3 vs per-64-block. */
    { size_t gb = (size_t)32 + (size_t)8*K, q8rb = (size_t)(rows/8)*gb;
      uint8_t *q8r = mmap(NULL, q8rb, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
      q8r_repack_task RT = { Wrm, q8r, rows, K }; ds4f_pool_run(m.pool, q8r_repack_worker, &RT);
      float *xs_in = aligned_alloc(256, (size_t)K*4);
      for (float spike = 1.f; spike <= 1e4f; spike *= 100.f) {
          for (int j = 0; j < K; j++) xs_in[j] = frand();
          xs_in[K/3] = spike; xs_in[2*K/3] = -spike*0.7f;   /* 2 massive channels */
          float *fref = aligned_alloc(256, (size_t)rows*4);
          for (int r = 0; r < rows; r++) { double a=0; const uint16_t *wr = Wrm + (size_t)r*K;
              for (int j = 0; j < K; j++) a += (double)bf16_to_f32_scalar(wr[j])*(double)xs_in[j]; fref[r]=(float)a; }
          int8_t *xqt = aligned_alloc(64, K); float xsc; quant_x_pertoken(xs_in, K, xqt, &xsc);
          q8r_task T = { y, q8r, rows, K, xqt, xsc }; ds4f_pool_run(m.pool, q8r_worker, &T);
          double e=0,n=0; for (int r=0;r<rows;r++){ double d=y[r]-fref[r]; e+=d*d; n+=(double)fref[r]*fref[r]; }
          double rel_v3 = sqrt(e/n);
          int8_t *xqb = aligned_alloc(64, K); float xsb[128]; ds4f_quant_x_sdot_into(xs_in, K, xqb, xsb);
          q8r_blk_task T4 = { y, q8r, rows, K, xqb, xsb }; ds4f_pool_run(m.pool, q8r_blk_worker, &T4);
          double e4=0,n4=0; for (int r=0;r<rows;r++){ double d=y[r]-fref[r]; e4+=d*d; n4+=(double)fref[r]*fref[r]; }
          double rel_v4 = sqrt(e4/n4);
          ds4f_tensor bft; memset(&bft,0,sizeof bft); bft.type=DS4F_BF16_PV; bft.rows=rows; bft.cols=K; bft.w=bf[0].w;
          ds4f_tensor q8t = bft; size_t wbb = ds4f_wbytes(DS4F_BF16_PV,rows,K);
          q8t.w = mmap(NULL,wbb,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); memcpy(q8t.w,bf[0].w,wbb);
          ds4f_repack_bf16pv_to_q8pv(&m,&q8t);
          ds4f_matvec(&m, y, &q8t, xs_in);
          double e2=0,n2=0; for (int r=0;r<rows;r++){ double d=y[r]-fref[r]; e2+=d*d; n2+=(double)fref[r]*fref[r]; }
          printf("  spike=%.0e : v3 relL2=%.2e  v4 relL2=%.2e  per-64blk relL2=%.2e\n", spike, rel_v3, rel_v4, sqrt(e2/n2));
          munmap(q8t.w,wbb); free(fref); free(xqt); free(xqb);
      }
    }
    ds4f_pool_stop(m.pool);
    return 0;
}
