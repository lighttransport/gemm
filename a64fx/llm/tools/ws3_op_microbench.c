/* WS3 micro-bench: current dot-product bf16-pv 8x3 kernel vs a candidate outer-product
 * (broadcast-x, vector-weight, fp32 accumulate, NO svaddv) kernel. Single-thread, pinned,
 * L2-resident tile. Proves whether the outer-product restructure is worth a full port
 * BEFORE touching the production GEMM. Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -I../../common \
 *       -o /tmp/ws3mb tools/ws3_op_microbench.c -lm
 *   taskset -c 12 /tmp/ws3mb
 */
#include <arm_sve.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

static inline uint16_t f2bf(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }
static inline float bf2f(uint16_t h){ uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,4); return f; }

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ---- current dot-product 8x3 kernel (verbatim structure from ggml_dequant.h) ----
 * pair-interleaved weights: pAB holds rows A,B interleaved [a0,b0,a1,b1,...]. */
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

/* ---- candidate outer-product kernel: tile = TT tokens x 32 output rows ----
 * weights packed column-major in K: Wp[k*32 + r], r in [0,32). x[t*K + k].
 * acc[t][0]=rows0..15, acc[t][1]=rows16..31. NO svaddv (acc IS the output).
 * fp32 accumulate (bf16 widened via ld1uh_u32 + lsl16). */
#define TT 12
/* 24 accumulators: A0_t = rows0..15 for token t, A1_t = rows16..31. (SVE types can't
 * be array elements, so unroll explicitly with macros.) */
#define Zc(t) svfloat32_t A0_##t=svdup_f32(0), A1_##t=svdup_f32(0);
#define FM(t) { svfloat32_t xb=svdup_f32(X[(size_t)t*Ks+k]); \
                A0_##t=svmla_x(pg,A0_##t,w0,xb); A1_##t=svmla_x(pg,A1_##t,w1,xb); }
#define ST(t) { svst1(pg,Y+(size_t)t*32,A0_##t); svst1(pg,Y+(size_t)t*32+16,A1_##t); }
#define ALL(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) M(11)
static inline void op_12x32(float *Y /* [TT*32] row=token */, const uint16_t *Wp,
                            const float *X /* [TT*K] */, int K, int Ks){
    svbool_t pg=svptrue_b32();
    ALL(Zc)
    for(int k=0;k<K;k++){
        const uint16_t *wk=Wp+(size_t)k*32;
        svfloat32_t w0=svreinterpret_f32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wk),16));
        svfloat32_t w1=svreinterpret_f32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wk+16),16));
        ALL(FM)
    }
    ALL(ST)
}

/* outer-product with x PRE-TRANSPOSED to [k][t] (token-contiguous per k -> the 12 broadcasts
 * read a 48-byte L1-hot region, no stride). This is the fair analog of the blueprint (A L1-resident). */
static inline void op_12x32_xt(float *Y, const uint16_t *Wp, const float *Xp /* [K*TT] */, int K){
    svbool_t pg=svptrue_b32();
    ALL(Zc)
    for(int k=0;k<K;k++){
        const uint16_t *wk=Wp+(size_t)k*32;
        const float *xk=Xp+(size_t)k*TT;
        svfloat32_t w0=svreinterpret_f32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wk),16));
        svfloat32_t w1=svreinterpret_f32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wk+16),16));
        #define FMx(t) { svfloat32_t xb=svdup_f32(xk[t]); A0_##t=svmla_x(pg,A0_##t,w0,xb); A1_##t=svmla_x(pg,A1_##t,w1,xb); }
        ALL(FMx)
        #undef FMx
    }
    ALL(ST)
}

int main(void){
    const int K=4096, R=32, T=12;       /* one tile: 32 rows x 12 tokens */
    int vl=(int)svcntw();
    printf("vl(f32 lanes)=%d  tile R=%d T=%d K=%d\n",vl,R,T,K);
    /* random bf16 weights [R][K] and x [T][K] */
    float *Wf=malloc((size_t)R*K*4), *Xf=malloc((size_t)T*K*4);
    srand(1234);
    for(size_t i=0;i<(size_t)R*K;i++){ float v=((rand()%2001)-1000)/1000.0f; Wf[i]=bf2f(f2bf(v)); }
    for(size_t i=0;i<(size_t)T*K;i++){ Xf[i]=((rand()%2001)-1000)/1000.0f; }
    /* double-precision reference Y[t][r] */
    double *Yref=malloc((size_t)T*R*8);
    for(int t=0;t<T;t++)for(int r=0;r<R;r++){ double s=0; for(int k=0;k<K;k++) s+=(double)Xf[t*K+k]*Wf[r*K+k]; Yref[t*R+r]=s; }

    /* pack weights for the OUTER-PRODUCT kernel: Wp[k*32+r] */
    uint16_t *Wp=malloc((size_t)R*K*2);
    for(int k=0;k<K;k++)for(int r=0;r<R;r++) Wp[(size_t)k*32+r]=f2bf(Wf[r*K+k]);

    /* pack weights for the DOT 8x3 kernel: pair-interleaved per 8-row group.
     * pAB for group g holds rows (8g+0,8g+1) interleaved: [a0,b0,a1,b1,...] over K. */
    int ng=R/8;
    uint16_t *pv=malloc((size_t)R*K*2);   /* 4 pair-bufs per group, each 2*K hw -> group stride 8K hw */
    for(int g=0;g<ng;g++){
        uint16_t *base=pv+(size_t)g*8*K;
        uint16_t *pAB=base, *pCD=base+2*K, *pEF=base+4*K, *pGH=base+6*K;
        for(int k=0;k<K;k++){
            pAB[2*k]=f2bf(Wf[(8*g+0)*K+k]); pAB[2*k+1]=f2bf(Wf[(8*g+1)*K+k]);
            pCD[2*k]=f2bf(Wf[(8*g+2)*K+k]); pCD[2*k+1]=f2bf(Wf[(8*g+3)*K+k]);
            pEF[2*k]=f2bf(Wf[(8*g+4)*K+k]); pEF[2*k+1]=f2bf(Wf[(8*g+5)*K+k]);
            pGH[2*k]=f2bf(Wf[(8*g+6)*K+k]); pGH[2*k+1]=f2bf(Wf[(8*g+7)*K+k]);
        }
    }

    float *Yop=malloc((size_t)T*R*4);
    float *Ydot=malloc((size_t)T*R*4);   /* [t][r] */

    /* correctness: outer-product */
    op_12x32(Yop, Wp, Xf, K, K);
    /* correctness: dot 8x3 -> process 32 rows = 4 groups, 12 tokens = 4 token-triples */
    memset(Ydot,0,(size_t)T*R*4);
    for(int g=0;g<ng;g++){
        uint16_t *base=pv+(size_t)g*8*K;
        for(int tt=0;tt<T;tt+=3){
            float acc0[8]={0},acc1[8]={0},acc2[8]={0};
            dot_8x3(acc0,acc1,acc2, base,base+2*K,base+4*K,base+6*K,
                    &Xf[(tt+0)*K],&Xf[(tt+1)*K],&Xf[(tt+2)*K], K);
            for(int r=0;r<8;r++){ Ydot[(tt+0)*R+8*g+r]=acc0[r]; Ydot[(tt+1)*R+8*g+r]=acc1[r]; Ydot[(tt+2)*R+8*g+r]=acc2[r]; }
        }
    }
    double eop=0,edot=0,nrm=0;
    for(int t=0;t<T;t++)for(int r=0;r<R;r++){ double y=Yref[t*R+r];
        double d1=Yop[t*R+r]-y, d2=Ydot[t*R+r]-y; eop+=d1*d1; edot+=d2*d2; nrm+=y*y; }
    printf("relL2: outer-product=%.3e  dot8x3=%.3e\n", sqrt(eop/nrm), sqrt(edot/nrm));

    /* perf: many reps, tile stays L2-resident (R*K*2 = 256KB weights) */
    const long REPS=4000;
    double macs=(double)T*R*K;   /* per tile call */

    /* transposed-x layout Xp[k*TT+t] */
    float *Xp=malloc((size_t)K*T*4);
    for(int k=0;k<K;k++)for(int t=0;t<T;t++) Xp[(size_t)k*T+t]=Xf[(size_t)t*K+k];
    float *Yopx=malloc((size_t)T*R*4);
    op_12x32_xt(Yopx, Wp, Xp, K);
    double eopx=0;
    for(int t=0;t<T;t++)for(int r=0;r<R;r++){ double d=Yopx[t*R+r]-Yref[t*R+r]; eopx+=d*d; }
    printf("relL2: outer-product-xt(transposed x)=%.3e\n", sqrt(eopx/nrm));

    double t0=now();
    for(long rep=0;rep<REPS;rep++) op_12x32(Yop, Wp, Xf, K, K);
    double t1=now();
    double gop = macs*REPS/(t1-t0)/1e9;

    t0=now();
    for(long rep=0;rep<REPS;rep++) op_12x32_xt(Yopx, Wp, Xp, K);
    t1=now();
    double gopx = macs*REPS/(t1-t0)/1e9;

    t0=now();
    for(long rep=0;rep<REPS;rep++){
        for(int g=0;g<ng;g++){ uint16_t *base=pv+(size_t)g*8*K;
            for(int tt=0;tt<T;tt+=3){ float acc0[8]={0},acc1[8]={0},acc2[8]={0};
                dot_8x3(acc0,acc1,acc2, base,base+2*K,base+4*K,base+6*K,
                        &Xf[(tt+0)*K],&Xf[(tt+1)*K],&Xf[(tt+2)*K], K);
                for(int r=0;r<8;r++){ Ydot[(tt+0)*R+8*g+r]=acc0[r]; Ydot[(tt+1)*R+8*g+r]=acc1[r]; Ydot[(tt+2)*R+8*g+r]=acc2[r]; } } }
    }
    t1=now();
    double gdot = macs*REPS/(t1-t0)/1e9;

    printf("single-thread Gmac/s:  dot8x3=%.1f   outer-product(strided x)=%.1f   outer-product(transposed x)=%.1f\n",
           gdot, gop, gopx);
    printf("  outer-product-xt vs dot8x3 speedup = %.2fx\n", gopx/gdot);
    return 0;
}
