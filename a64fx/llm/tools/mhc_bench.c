/* mHC hc_pre cost microbench (single node, no alloc).
 *
 * The 11n decode profile shows mhc_pre = 10.1 ms/tok = 86 calls x ~118 us, while a plain
 * pool dispatch (mhc_post) is only ~8.6 us. This bench isolates where hc_pre's time goes:
 *   - empty dispatch (pool barrier floor)
 *   - hcmix dispatch alone (24-row F32 matvec + ss, HC_RMSPAR path)
 *   - full hc_pre (hcmix + serial sinkhorn/rsq + hccol dispatch [+ fused resid])
 *   - hc_post dispatch (reference)
 * with 86 DISTINCT fn tensors cycled per "token" to reproduce the production cold-HBM
 * weight stream (each layer's 1.5 MB F32 hc_fn read once per token).
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *       -I../../common -o build/mhc_bench tools/mhc_bench.c -lm -lpthread -lhwb
 *   DS4F_HC_PAR=1 DS4F_HC_RMSPAR=1 DS4F_FLAGBAR=1 ./build/mhc_bench [nthr=48] [ntok=32]
 */
#include "ds4f.h"
#include <math.h>

static uint32_t rng = 0xBEEF1234u;
static inline uint32_t nr(void){ rng = rng*1664525u + 1013904223u; return rng; }
static inline float frand(void){ return ((float)(nr()>>8)/(float)(1u<<24))*2.f-1.f; }


/* ==== prototype SVE variants (bench-only; port to ds4f_impl.h if they win) ==== */
/* half-row-split SVE mixes matvec: 2*mix_hc partial dots (32 KB each) across the pool +
 * per-thread ss slice. Caller combines part[2i]+part[2i+1] (fixed order). */
typedef struct { const float *fn, *x4; float *part; double *ssp; int rows, hd; } mixsve_task;
static void mixsve_worker(void *arg, int tid, int nthr) {
    mixsve_task *T = (mixsve_task *)arg;
    int rows = T->rows, hd = T->hd, half = hd/2;
    int units = rows*2;
    int per = units/nthr, extra = units%nthr;
    int u0 = per*tid + (tid<extra?tid:extra), u1 = u0 + per + (tid<extra?1:0);
    for (int u = u0; u < u1; u++) {
        int i = u >> 1, h = u & 1;
        const float *w = T->fn + (size_t)i*hd + (size_t)h*half;
        const float *x = T->x4 + (size_t)h*half;
        svbool_t pg = svptrue_b32();
        svfloat32_t a0 = svdup_f32(0), a1 = svdup_f32(0), a2 = svdup_f32(0), a3 = svdup_f32(0);
        int j = 0, vl = (int)svcntw();
        for (; j + 4*vl <= half; j += 4*vl) {
            a0 = svmla_x(pg, a0, svld1_f32(pg, w+j),        svld1_f32(pg, x+j));
            a1 = svmla_x(pg, a1, svld1_f32(pg, w+j+vl),     svld1_f32(pg, x+j+vl));
            a2 = svmla_x(pg, a2, svld1_f32(pg, w+j+2*vl),   svld1_f32(pg, x+j+2*vl));
            a3 = svmla_x(pg, a3, svld1_f32(pg, w+j+3*vl),   svld1_f32(pg, x+j+3*vl));
        }
        for (; j < half; j += vl) {
            svbool_t p = svwhilelt_b32(j, half);
            a0 = svmla_x(p, a0, svld1_f32(p, w+j), svld1_f32(p, x+j));
        }
        T->part[u] = svaddv(pg, svadd_x(pg, svadd_x(pg, a0, a1), svadd_x(pg, a2, a3)));
    }
    int d0 = (int)((long)hd*tid/nthr), d1 = (int)((long)hd*(tid+1)/nthr);
    double sng = 0.0;
    for (int j = d0; j < d1; j++) { float v = T->x4[j]; sng += (double)v*v; }
    T->ssp[tid] = sng;
}
/* sinkhorn with SVE-vectorized divisions (hc==4): identical adds/order; the 16 elementwise
 * fdivs of each normalize become one svdiv (lane fdiv == scalar fdiv -> BIT-EXACT). */
static void sinkhorn_sve(const float *mixes, const float *scale, const float *base,
                         int hc, int iters, float eps,
                         float *pre, float *post, float *comb) {
    for (int j = 0; j < hc; j++) pre[j]  = ds4f_sigmoidf(mixes[j]*scale[0] + base[j]) + eps;
    for (int j = 0; j < hc; j++) post[j] = 2.0f*ds4f_sigmoidf(mixes[j+hc]*scale[1] + base[j+hc]);
    for (int j = 0; j < hc; j++)
        for (int k = 0; k < hc; k++)
            comb[j*hc+k] = mixes[j*hc + k + 2*hc]*scale[2] + base[j*hc + k + 2*hc];
    for (int j = 0; j < hc; j++) {
        float mx = comb[j*hc];
        for (int k = 1; k < hc; k++) if (comb[j*hc+k] > mx) mx = comb[j*hc+k];
        float sacc = 0.f;
        for (int k = 0; k < hc; k++) { float e = expf(comb[j*hc+k]-mx); comb[j*hc+k] = e; sacc += e; }
        for (int k = 0; k < hc; k++) comb[j*hc+k] = comb[j*hc+k]/sacc + eps;
    }
    if (hc == 4) {
        svbool_t pg16 = svwhilelt_b32(0, 16);
        float den[16];
        /* first col-normalize */
        for (int k = 0; k < 4; k++) {
            float cs = comb[k]+comb[4+k]+comb[8+k]+comb[12+k] + eps;
            den[k]=den[4+k]=den[8+k]=den[12+k]=cs;
        }
        svst1_f32(pg16, comb, svdiv_x(pg16, svld1_f32(pg16, comb), svld1_f32(pg16, den)));
        for (int it = 0; it < iters-1; it++) {
            for (int j = 0; j < 4; j++) {
                float rs = comb[j*4]+comb[j*4+1]+comb[j*4+2]+comb[j*4+3] + eps;
                den[j*4]=den[j*4+1]=den[j*4+2]=den[j*4+3]=rs;
            }
            svst1_f32(pg16, comb, svdiv_x(pg16, svld1_f32(pg16, comb), svld1_f32(pg16, den)));
            for (int k = 0; k < 4; k++) {
                float cs = comb[k]+comb[4+k]+comb[8+k]+comb[12+k] + eps;
                den[k]=den[4+k]=den[8+k]=den[12+k]=cs;
            }
            svst1_f32(pg16, comb, svdiv_x(pg16, svld1_f32(pg16, comb), svld1_f32(pg16, den)));
        }
    } else {
        for (int k = 0; k < hc; k++) {
            float cs = 0.f; for (int j = 0; j < hc; j++) cs += comb[j*hc+k];
            cs += eps; for (int j = 0; j < hc; j++) comb[j*hc+k] /= cs;
        }
        for (int it = 0; it < iters-1; it++) {
            for (int j = 0; j < hc; j++) {
                float rs = 0.f; for (int k = 0; k < hc; k++) rs += comb[j*hc+k];
                rs += eps; for (int k = 0; k < hc; k++) comb[j*hc+k] /= rs;
            }
            for (int k = 0; k < hc; k++) {
                float cs = 0.f; for (int j = 0; j < hc; j++) cs += comb[j*hc+k];
                cs += eps; for (int j = 0; j < hc; j++) comb[j*hc+k] /= cs;
            }
        }
    }
}

typedef struct { float *p; size_t n; } touch_task;
static void touch_worker(void *arg, int tid, int nthr) {
    touch_task *T = (touch_task *)arg;
    size_t i0 = T->n*tid/nthr, i1 = T->n*(tid+1)/nthr;
    memset(T->p + i0, 0, (i1 - i0)*4);
}
static void empty_worker(void *arg, int tid, int nthr){ (void)arg;(void)tid;(void)nthr; }

int main(int argc, char **argv) {
    int nthr = argc > 1 ? atoi(argv[1]) : 48;
    int ntok = argc > 2 ? atoi(argv[2]) : 32;
    ds4f_model m; memset(&m, 0, sizeof(m));
    m.cfg = ds4f_default_config();
    m.n_threads = nthr; m.n_cmgs = nthr > 36 ? 4 : (nthr > 24 ? 3 : (nthr > 12 ? 2 : 1));
    m.pool = ds4f_pool_start(nthr, m.n_cmgs);
    ds4f_config *c = &m.cfg;
    int hc = c->hc_mult, C = c->hidden, hd = hc*C, mix_hc = (2+hc)*hc;
    int NL = 86;   /* 43 layers x 2 (attn+ffn) distinct fn tensors */
    printf("mhc_bench nthr=%d ntok=%d hc=%d C=%d hd=%d mix_hc=%d NL=%d (HC_PAR=%s RMSPAR=%s FLAGBAR=%s)\n",
           nthr, ntok, hc, C, hd, mix_hc, NL,
           getenv("DS4F_HC_PAR")?:"-", getenv("DS4F_HC_RMSPAR")?:"-", getenv("DS4F_FLAGBAR")?:"-");

    float **fn = malloc(NL*sizeof(float*));
    float **sc = malloc(NL*sizeof(float*));
    float **bs = malloc(NL*sizeof(float*));
    for (int l = 0; l < NL; l++) {
        fn[l] = aligned_alloc(256, (size_t)mix_hc*hd*4);
        { touch_task T = { fn[l], (size_t)mix_hc*hd }; ds4f_pool_run(m.pool, touch_worker, &T); }  /* spread pages over CMGs */
        for (size_t i = 0; i < (size_t)mix_hc*hd; i++) fn[l][i] = frand()*0.01f;
        sc[l] = aligned_alloc(64, 4*4); sc[l][0]=sc[l][1]=sc[l][2]=1.f;
        bs[l] = aligned_alloc(64, (size_t)mix_hc*4);
        for (int i = 0; i < mix_hc; i++) bs[l][i] = frand()*0.1f;
    }
    float *x4    = aligned_alloc(256, (size_t)hd*4);
    float *y     = aligned_alloc(256, (size_t)C*4);
    float *resid = aligned_alloc(256, (size_t)hd*4);
    float *fblk  = aligned_alloc(256, (size_t)C*4);
    for (int i = 0; i < hd; i++) x4[i] = frand();
    for (int i = 0; i < C;  i++) fblk[i] = frand();
    float post[16], comb[64];

    /* 1. empty-dispatch floor */
    { double t0 = ds4f_now();
      for (int t = 0; t < ntok; t++) for (int l = 0; l < NL; l++) ds4f_pool_run(m.pool, empty_worker, NULL);
      double dt = ds4f_now()-t0;
      printf("empty dispatch      : %7.2f us/call\n", dt/(ntok*NL)*1e6); }

    /* 2. hcmix dispatch alone (RMSPAR fused matvec+ss) */
    { float mixes[64]; double ssp[64];
      double t0 = ds4f_now();
      for (int t = 0; t < ntok; t++) for (int l = 0; l < NL; l++) {
          ds4f_hcmix_task T = { fn[l], x4, mixes, ssp, mix_hc, hd };
          ds4f_pool_run(m.pool, ds4f_hcmix_worker, &T);
      }
      double dt = ds4f_now()-t0;
      printf("hcmix dispatch      : %7.2f us/call  (%.1f GB/s fn-weight agg)\n",
             dt/(ntok*NL)*1e6, (double)ntok*NL*mix_hc*hd*4/dt/1e9); }

    /* 3. hccol dispatch alone (collapse + fused resid) */
    { float pre[16]; for (int k = 0; k < hc; k++) pre[k] = 0.25f;
      double t0 = ds4f_now();
      for (int t = 0; t < ntok; t++) for (int l = 0; l < NL; l++) {
          ds4f_hccol_task T = { x4, pre, y, resid, hc, C };
          ds4f_pool_run(m.pool, ds4f_hccol_worker, &T);
      }
      double dt = ds4f_now()-t0;
      printf("hccol+resid dispatch: %7.2f us/call\n", dt/(ntok*NL)*1e6); }

    /* 4. full hc_pre (as production) */
    { double t0 = ds4f_now();
      for (int t = 0; t < ntok; t++) for (int l = 0; l < NL; l++)
          ds4f_hc_pre(&m, x4, fn[l], sc[l], bs[l], y, post, comb, resid);
      double dt = ds4f_now()-t0;
      printf("hc_pre full         : %7.2f us/call  -> %5.2f ms/tok at 86 calls\n",
             dt/(ntok*NL)*1e6, dt/(ntok*NL)*1e6*86/1e3); }

    /* 5. hc_post (reference dispatch) */
    { double t0 = ds4f_now();
      for (int t = 0; t < ntok; t++) for (int l = 0; l < NL; l++)
          ds4f_hc_post(&m, x4, resid, fblk, post, comb);
      double dt = ds4f_now()-t0;
      printf("hc_post             : %7.2f us/call\n", dt/(ntok*NL)*1e6); }

    /* 6. serial sinkhorn alone */
    { float mixes[64]; for (int i = 0; i < mix_hc; i++) mixes[i] = frand();
      float pre[16];
      double t0 = ds4f_now();
      for (int it = 0; it < 100000; it++)
          ds4f_hc_sinkhorn(mixes, sc[0], bs[0], hc, c->hc_iters, c->hc_eps, pre, post, comb);
      double dt = ds4f_now()-t0;
      printf("sinkhorn (serial)   : %7.2f us/call\n", dt/100000*1e6); }


    /* 7. SVE half-row hcmix prototype */
    { float part[64]; double ssp[64]; float mixes[64];
      double t0 = ds4f_now();
      for (int t = 0; t < ntok; t++) for (int l = 0; l < NL; l++) {
          mixsve_task T = { fn[l], x4, part, ssp, mix_hc, hd };
          ds4f_pool_run(m.pool, mixsve_worker, &T);
          for (int i = 0; i < mix_hc; i++) mixes[i] = part[2*i] + part[2*i+1];
      }
      double dt = ds4f_now()-t0;
      /* correctness vs scalar */
      { ds4f_hcmix_task R = { fn[0], x4, (float[64]){0}, ssp, mix_hc, hd };
        float ref[64]; R.mixes = ref; ds4f_pool_run(m.pool, ds4f_hcmix_worker, &R);
        mixsve_task T = { fn[0], x4, part, ssp, mix_hc, hd };
        ds4f_pool_run(m.pool, mixsve_worker, &T);
        double mx=0; for (int i = 0; i < mix_hc; i++) { double d=fabs((double)(part[2*i]+part[2*i+1])-ref[i]); double r=fabs(ref[i])+1e-30; if(d/r>mx)mx=d/r; }
        printf("hcmix SVE half-row  : %7.2f us/call  (%.1f GB/s)  relerr vs scalar %.1e\n",
               dt/(ntok*NL)*1e6, (double)ntok*NL*mix_hc*hd*4/dt/1e9, mx); } }

    /* 8. SVE sinkhorn prototype (exactness check + speed) */
    { float mixes[64]; for (int i = 0; i < mix_hc; i++) mixes[i] = frand();
      float pre1[16], post1[16], comb1[64], pre2[16], post2[16], comb2[64];
      ds4f_hc_sinkhorn(mixes, sc[0], bs[0], hc, c->hc_iters, c->hc_eps, pre1, post1, comb1);
      sinkhorn_sve   (mixes, sc[0], bs[0], hc, c->hc_iters, c->hc_eps, pre2, post2, comb2);
      int nbit = 0; for (int i = 0; i < hc*hc; i++) if (comb1[i] == comb2[i]) nbit++;
      double t0 = ds4f_now();
      for (int it = 0; it < 100000; it++)
          sinkhorn_sve(mixes, sc[0], bs[0], hc, c->hc_iters, c->hc_eps, pre2, post2, comb2);
      double dt = ds4f_now()-t0;
      printf("sinkhorn SVE        : %7.2f us/call  comb bit-exact %d/16\n", dt/100000*1e6, nbit); }

    ds4f_pool_stop(m.pool);
    return 0;
}
