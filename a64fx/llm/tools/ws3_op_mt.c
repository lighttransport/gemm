/* WS3 decisive de-risk: multi-threaded, FULL-SHAPE, real weight-streaming compare of the
 * current dot-product 8x3 bf16-pv kernel vs the outer-product 12x32 kernel. This mimics the
 * production GEMM (48 threads, weights streamed once from HBM, no artificial tile reuse) to
 * test whether the single-tile 2x kernel win survives multi-core weight streaming -- the
 * exact question the in-file note (ds4f_impl.h:566) warns about ("weight-streaming/compute-
 * bound; no loop transform helps"). Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I../../common \
 *       -o /tmp/ws3mt tools/ws3_op_mt.c -lm
 *   OMP_NUM_THREADS=48 taskset -c 12-59 /tmp/ws3mt
 */
#include <arm_sve.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

static inline uint16_t f2bf(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }
static inline float bf2f(uint16_t h){ uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,4); return f; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* dot-product 8x3 (current kernel, pv weights, K-in-lanes, svaddv reduce) */
static inline void dot_8x3(float *acc0,float *acc1,float *acc2,
        const uint16_t *pAB,const uint16_t *pCD,const uint16_t *pEF,const uint16_t *pGH,
        const float *x0,const float *x1,const float *x2,int n){
    svbool_t pg=svptrue_b32(); svbool_t pah=svptrue_b16();
    svuint16_t idx=svindex_u16(0,1);
    svbool_t p_odd=svcmpne_n_u16(pah,svand_n_u16_x(pah,idx,1),0);
    int vl=(int)svcntw();
    svfloat32_t a00=svdup_f32(0),a10=svdup_f32(0),a20=svdup_f32(0),a30=svdup_f32(0);
    svfloat32_t a40=svdup_f32(0),a50=svdup_f32(0),a60=svdup_f32(0),a70=svdup_f32(0);
    svfloat32_t a01=svdup_f32(0),a11=svdup_f32(0),a21=svdup_f32(0),a31=svdup_f32(0);
    svfloat32_t a41=svdup_f32(0),a51=svdup_f32(0),a61=svdup_f32(0),a71=svdup_f32(0);
    svfloat32_t a02=svdup_f32(0),a12=svdup_f32(0),a22=svdup_f32(0),a32=svdup_f32(0);
    svfloat32_t a42=svdup_f32(0),a52=svdup_f32(0),a62=svdup_f32(0),a72=svdup_f32(0);
    for(int i=0;i+vl-1<n;i+=vl){
        const uint16_t *ab=pAB+2*i,*cd=pCD+2*i,*ef=pEF+2*i,*gh=pGH+2*i;
        svfloat32_t vx0=svld1(pg,&x0[i]),vx1=svld1(pg,&x1[i]),vx2=svld1(pg,&x2[i]);
        svfloat32_t w;
        w=svreinterpret_f32(svld1_u16(p_odd,ab-1)); a00=svmla_x(pg,a00,w,vx0);a01=svmla_x(pg,a01,w,vx1);a02=svmla_x(pg,a02,w,vx2);
        w=svreinterpret_f32(svld1_u16(p_odd,ab  )); a10=svmla_x(pg,a10,w,vx0);a11=svmla_x(pg,a11,w,vx1);a12=svmla_x(pg,a12,w,vx2);
        w=svreinterpret_f32(svld1_u16(p_odd,cd-1)); a20=svmla_x(pg,a20,w,vx0);a21=svmla_x(pg,a21,w,vx1);a22=svmla_x(pg,a22,w,vx2);
        w=svreinterpret_f32(svld1_u16(p_odd,cd  )); a30=svmla_x(pg,a30,w,vx0);a31=svmla_x(pg,a31,w,vx1);a32=svmla_x(pg,a32,w,vx2);
        w=svreinterpret_f32(svld1_u16(p_odd,ef-1)); a40=svmla_x(pg,a40,w,vx0);a41=svmla_x(pg,a41,w,vx1);a42=svmla_x(pg,a42,w,vx2);
        w=svreinterpret_f32(svld1_u16(p_odd,ef  )); a50=svmla_x(pg,a50,w,vx0);a51=svmla_x(pg,a51,w,vx1);a52=svmla_x(pg,a52,w,vx2);
        w=svreinterpret_f32(svld1_u16(p_odd,gh-1)); a60=svmla_x(pg,a60,w,vx0);a61=svmla_x(pg,a61,w,vx1);a62=svmla_x(pg,a62,w,vx2);
        w=svreinterpret_f32(svld1_u16(p_odd,gh  )); a70=svmla_x(pg,a70,w,vx0);a71=svmla_x(pg,a71,w,vx1);a72=svmla_x(pg,a72,w,vx2);
    }
    acc0[0]+=svaddv(pg,a00);acc0[1]+=svaddv(pg,a10);acc0[2]+=svaddv(pg,a20);acc0[3]+=svaddv(pg,a30);
    acc0[4]+=svaddv(pg,a40);acc0[5]+=svaddv(pg,a50);acc0[6]+=svaddv(pg,a60);acc0[7]+=svaddv(pg,a70);
    acc1[0]+=svaddv(pg,a01);acc1[1]+=svaddv(pg,a11);acc1[2]+=svaddv(pg,a21);acc1[3]+=svaddv(pg,a31);
    acc1[4]+=svaddv(pg,a41);acc1[5]+=svaddv(pg,a51);acc1[6]+=svaddv(pg,a61);acc1[7]+=svaddv(pg,a71);
    acc2[0]+=svaddv(pg,a02);acc2[1]+=svaddv(pg,a12);acc2[2]+=svaddv(pg,a22);acc2[3]+=svaddv(pg,a32);
    acc2[4]+=svaddv(pg,a42);acc2[5]+=svaddv(pg,a52);acc2[6]+=svaddv(pg,a62);acc2[7]+=svaddv(pg,a72);
}

/* outer-product 12x32: Wt[k*32+r] (k-major, 32 rows), Xt[k*12+t] (k-major, 12 tokens),
 * out[t*32+r]. fp32 acc, no svaddv. */
#define Zc(t) svfloat32_t A0_##t=svdup_f32(0), A1_##t=svdup_f32(0);
#define FMx(t) { svfloat32_t xb=svdup_f32(xk[t]); A0_##t=svmla_x(pg,A0_##t,w0,xb); A1_##t=svmla_x(pg,A1_##t,w1,xb); }
#define ST(t) { svst1(pg,out+(size_t)t*32,A0_##t); svst1(pg,out+(size_t)t*32+16,A1_##t); }
#define ALL(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) M(11)
static inline void op_12x32(float *out, const uint16_t *Wt, const float *Xt, int K){
    svbool_t pg=svptrue_b32();
    ALL(Zc)
    for(int k=0;k<K;k++){
        const uint16_t *wk=Wt+(size_t)k*32; const float *xk=Xt+(size_t)k*12;
        svfloat32_t w0=svreinterpret_f32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wk),16));
        svfloat32_t w1=svreinterpret_f32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wk+16),16));
        ALL(FMx)
    }
    ALL(ST)
}

int main(void){
    const int R=getenv("R")?atoi(getenv("R")):8192, K=getenv("K")?atoi(getenv("K")):4096, M=getenv("M")?atoi(getenv("M")):24;
    int vl=(int)svcntw();
    printf("MT full-shape bench: R=%d K=%d M=%d vl=%d  nthr=%d\n", R,K,M,vl,omp_get_max_threads());
    float *Wf=malloc((size_t)R*K*4), *Xf=malloc((size_t)M*K*4);
    srand(7);
    for(size_t i=0;i<(size_t)R*K;i++){ float v=((rand()%2001)-1000)/1000.0f; Wf[i]=bf2f(f2bf(v)); }
    for(size_t i=0;i<(size_t)M*K;i++){ Xf[i]=((rand()%2001)-1000)/1000.0f; }

    /* pv weights (dot path): per 8-row group, 4 pair-bufs */
    uint16_t *pv=malloc((size_t)R*K*2);
    #pragma omp parallel for
    for(int g=0; g<R/8; g++){
        uint16_t *base=pv+(size_t)g*8*K, *pAB=base,*pCD=base+2*K,*pEF=base+4*K,*pGH=base+6*K;
        for(int k=0;k<K;k++){
            pAB[2*k]=f2bf(Wf[(8*g+0)*K+k]); pAB[2*k+1]=f2bf(Wf[(8*g+1)*K+k]);
            pCD[2*k]=f2bf(Wf[(8*g+2)*K+k]); pCD[2*k+1]=f2bf(Wf[(8*g+3)*K+k]);
            pEF[2*k]=f2bf(Wf[(8*g+4)*K+k]); pEF[2*k+1]=f2bf(Wf[(8*g+5)*K+k]);
            pGH[2*k]=f2bf(Wf[(8*g+6)*K+k]); pGH[2*k+1]=f2bf(Wf[(8*g+7)*K+k]);
        }
    }
    /* k-major weights (op path): per 32-row block, Wt[blk][k*32+r] */
    uint16_t *km=malloc((size_t)R*K*2);
    #pragma omp parallel for
    for(int b=0;b<R/32;b++){ uint16_t *Wt=km+(size_t)b*32*K;
        for(int k=0;k<K;k++)for(int r=0;r<32;r++) Wt[(size_t)k*32+r]=f2bf(Wf[(32*b+r)*K+k]); }
    /* transposed x per 12-token tile (op path): xtile[tile] is [k*12+t], built ONCE
     * (real integration transposes x once per GEMM, amortized over all row-blocks). */
    int ntile=M/12;
    float **xtile=malloc(ntile*sizeof(float*));
    for(int tl=0;tl<ntile;tl++){ xtile[tl]=malloc((size_t)K*12*4);
        for(int k=0;k<K;k++)for(int t=0;t<12;t++) xtile[tl][(size_t)k*12+t]=Xf[(size_t)(tl*12+t)*K+k]; }

    float *Ydot=malloc((size_t)M*R*4);   /* [t][r] */
    float *Yop =malloc((size_t)M*R*4);
    double macs=(double)R*K*M;

    /* ---- dot path: group-outer, token-triple-inner (mimics ds4f_gemm_worker) ---- */
    double best_dot=1e30;
    for(int rep=0;rep<4;rep++){ double t0=now();
        #pragma omp parallel for schedule(static)
        for(int g=0; g<R/8; g++){ uint16_t *base=pv+(size_t)g*8*K;
            for(int tt=0;tt<M;tt+=3){ float a0[8]={0},a1[8]={0},a2[8]={0};
                dot_8x3(a0,a1,a2, base,base+2*K,base+4*K,base+6*K,
                        &Xf[(tt+0)*K],&Xf[(tt+1)*K],&Xf[(tt+2)*K], K);
                for(int r=0;r<8;r++){ Ydot[(tt+0)*R+8*g+r]=a0[r]; Ydot[(tt+1)*R+8*g+r]=a1[r]; Ydot[(tt+2)*R+8*g+r]=a2[r]; } } }
        double dt=now()-t0; if(dt<best_dot)best_dot=dt; }
    double gdot=macs/best_dot/1e9;

    /* ---- op path: block-outer, token-tile(12)-inner ---- */
    double best_op=1e30;
    for(int rep=0;rep<4;rep++){ double t0=now();
        #pragma omp parallel for schedule(static)
        for(int b=0;b<R/32;b++){ uint16_t *Wt=km+(size_t)b*32*K;
            float out[12*32];
            for(int tl=0;tl<ntile;tl++){
                op_12x32(out, Wt, xtile[tl], K);
                for(int t=0;t<12;t++)for(int r=0;r<32;r++) Yop[(tl*12+t)*R+32*b+r]=out[t*32+r];
            }
        }
        double dt=now()-t0; if(dt<best_op)best_op=dt; }
    double gop=macs/best_op/1e9;

    /* correctness: op vs dot */
    double e=0,n=0; for(size_t i=0;i<(size_t)M*R;i++){ double d=Yop[i]-Ydot[i]; e+=d*d; n+=(double)Ydot[i]*Ydot[i]; }
    printf("relL2(op vs dot)=%.3e\n", sqrt(e/n));
    printf("AGG Gmac/s @%d thr:  dot8x3=%.1f   outer-product=%.1f   speedup=%.2fx\n",
           omp_get_max_threads(), gdot, gop, gop/gdot);
    return 0;
}
