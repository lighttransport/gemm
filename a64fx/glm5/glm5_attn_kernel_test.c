/* Prefill attention-score kernel bench for GLM-5.2 absorbed-MLA on A64FX.
 * The QK scores are S[h,j] = Q[h,:]·KV[j,:] over kv_lora=512 (h=64 heads, j=ns keys).
 * Today's runner does this PAIRWISE (per key, a length-512 SVE dot per head) — no register
 * reuse of the Q/KV tiles across heads×keys. This bench compares that baseline against
 * register-blocked fp32 and int16 (svdot_s64) attention-GEMM, which reuse each loaded Q/KV
 * row across the tile. Reports Gop/s + rms-rel vs the f64 reference.
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -o glm5_attn_kernel_test \
 *          glm5_attn_kernel_test.c -lm
 * Run:   OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=48 NS=512 ./glm5_attn_kernel_test */
#include <arm_sve.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif
static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint32_t lcg(uint32_t*s){ *s=*s*1664525u+1013904223u; return *s; }
static int envi(const char*k,int d){ const char*e=getenv(k); return e&&*e?atoi(e):d; }
#define KVL 512   /* kv_lora */

/* ---- baseline: pairwise SVE fp32 dot (the current glm5_dot_f32_opt, 4-acc) ---- */
static inline float dot_sve(const float*a,const float*b,int n){
    int vl=(int)svcntw(); svbool_t pg=svptrue_b32();
    svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0; int i=0;
    for(;i+4*vl<=n;i+=4*vl){
        a0=svmla_f32_x(pg,a0,svld1(pg,a+i),svld1(pg,b+i));
        a1=svmla_f32_x(pg,a1,svld1(pg,a+i+vl),svld1(pg,b+i+vl));
        a2=svmla_f32_x(pg,a2,svld1(pg,a+i+2*vl),svld1(pg,b+i+2*vl));
        a3=svmla_f32_x(pg,a3,svld1(pg,a+i+3*vl),svld1(pg,b+i+3*vl));
    }
    for(;i<n;i+=vl){ svbool_t pt=svwhilelt_b32(i,n); a0=svmla_f32_x(pt,a0,svld1(pt,a+i),svld1(pt,b+i)); }
    a0=svadd_f32_x(pg,svadd_f32_x(pg,a0,a1),svadd_f32_x(pg,a2,a3));
    return svaddv_f32(pg,a0);
}
static void qk_pairwise(const float*Q,const float*KV,float*S,int H,int ns){
    #pragma omp parallel for schedule(static)
    for(int h=0;h<H;h++){ const float*qh=Q+(size_t)h*KVL;
        for(int j=0;j<ns;j++) S[(size_t)h*ns+j]=dot_sve(qh,KV+(size_t)j*KVL,KVL); }
}

/* ---- register-blocked fp32 QK GEMM: MR=4 heads x NR=4 keys tile, kv row reused across the tile ---- */
static void qk_fp32_rb(const float*Q,const float*KV,float*S,int H,int ns){
    int vl=(int)svcntw(); svbool_t pg=svptrue_b32();
    #pragma omp parallel for schedule(static)
    for(int h0=0;h0<H;h0+=4){
        const float*q0=Q+(size_t)(h0+0)*KVL,*q1=Q+(size_t)(h0+1)*KVL,*q2=Q+(size_t)(h0+2)*KVL,*q3=Q+(size_t)(h0+3)*KVL;
        int j=0;
        for(;j+4<=ns;j+=4){
            const float*k0=KV+(size_t)(j+0)*KVL,*k1=KV+(size_t)(j+1)*KVL,*k2=KV+(size_t)(j+2)*KVL,*k3=KV+(size_t)(j+3)*KVL;
            svfloat32_t c00=svdup_f32(0),c01=c00,c02=c00,c03=c00, c10=c00,c11=c00,c12=c00,c13=c00,
                        c20=c00,c21=c00,c22=c00,c23=c00, c30=c00,c31=c00,c32=c00,c33=c00;
            for(int i=0;i<KVL;i+=vl){
                svfloat32_t qa=svld1(pg,q0+i),qb=svld1(pg,q1+i),qc=svld1(pg,q2+i),qd=svld1(pg,q3+i);
                svfloat32_t ka=svld1(pg,k0+i);
                c00=svmla_f32_x(pg,c00,qa,ka); c10=svmla_f32_x(pg,c10,qb,ka); c20=svmla_f32_x(pg,c20,qc,ka); c30=svmla_f32_x(pg,c30,qd,ka);
                svfloat32_t kb=svld1(pg,k1+i);
                c01=svmla_f32_x(pg,c01,qa,kb); c11=svmla_f32_x(pg,c11,qb,kb); c21=svmla_f32_x(pg,c21,qc,kb); c31=svmla_f32_x(pg,c31,qd,kb);
                svfloat32_t kc=svld1(pg,k2+i);
                c02=svmla_f32_x(pg,c02,qa,kc); c12=svmla_f32_x(pg,c12,qb,kc); c22=svmla_f32_x(pg,c22,qc,kc); c32=svmla_f32_x(pg,c32,qd,kc);
                svfloat32_t kd=svld1(pg,k3+i);
                c03=svmla_f32_x(pg,c03,qa,kd); c13=svmla_f32_x(pg,c13,qb,kd); c23=svmla_f32_x(pg,c23,qc,kd); c33=svmla_f32_x(pg,c33,qd,kd);
            }
            float*s0=S+(size_t)(h0+0)*ns+j,*s1=S+(size_t)(h0+1)*ns+j,*s2=S+(size_t)(h0+2)*ns+j,*s3=S+(size_t)(h0+3)*ns+j;
            s0[0]=svaddv_f32(pg,c00);s0[1]=svaddv_f32(pg,c01);s0[2]=svaddv_f32(pg,c02);s0[3]=svaddv_f32(pg,c03);
            s1[0]=svaddv_f32(pg,c10);s1[1]=svaddv_f32(pg,c11);s1[2]=svaddv_f32(pg,c12);s1[3]=svaddv_f32(pg,c13);
            s2[0]=svaddv_f32(pg,c20);s2[1]=svaddv_f32(pg,c21);s2[2]=svaddv_f32(pg,c22);s2[3]=svaddv_f32(pg,c23);
            s3[0]=svaddv_f32(pg,c30);s3[1]=svaddv_f32(pg,c31);s3[2]=svaddv_f32(pg,c32);s3[3]=svaddv_f32(pg,c33);
        }
        for(;j<ns;j++){ const float*kj=KV+(size_t)j*KVL;
            for(int hh=0;hh<4;hh++) S[(size_t)(h0+hh)*ns+j]=dot_sve(Q+(size_t)(h0+hh)*KVL,kj,KVL); }
    }
}

/* ---- int16 QK GEMM: quantize Q,KV rows to int16 (32767/amax), svdot_s64, MR=4 x NR=4 ---- */
static void quant_i16(const float*A,int16_t*Q,float*sc,int rows,int n){
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++){ const float*a=A+(size_t)r*n; float mx=1e-20f;
        for(int i=0;i<n;i++){ float v=fabsf(a[i]); if(v>mx)mx=v; }
        float q=32767.0f/mx, inv=mx/32767.0f; sc[r]=inv; int16_t*o=Q+(size_t)r*n;
        for(int i=0;i<n;i++){ float v=a[i]*q; o[i]=(int16_t)(v<0?v-0.5f:v+0.5f); } }
}
static void qk_int16_rb(const int16_t*Q,const float*qsc,const int16_t*KV,const float*ksc,float*S,int H,int ns){
    int vl=(int)svcnth(); svbool_t pg=svptrue_b16();  /* 16-bit lanes */
    #pragma omp parallel for schedule(static)
    for(int h0=0;h0<H;h0+=4){
        const int16_t*q0=Q+(size_t)(h0+0)*KVL,*q1=Q+(size_t)(h0+1)*KVL,*q2=Q+(size_t)(h0+2)*KVL,*q3=Q+(size_t)(h0+3)*KVL;
        float qs0=qsc[h0],qs1=qsc[h0+1],qs2=qsc[h0+2],qs3=qsc[h0+3];
        int j=0;
        for(;j+4<=ns;j+=4){
            const int16_t*k0=KV+(size_t)(j+0)*KVL,*k1=KV+(size_t)(j+1)*KVL,*k2=KV+(size_t)(j+2)*KVL,*k3=KV+(size_t)(j+3)*KVL;
            svint64_t a00=svdup_s64(0),a01=a00,a02=a00,a03=a00, a10=a00,a11=a00,a12=a00,a13=a00,
                      a20=a00,a21=a00,a22=a00,a23=a00, a30=a00,a31=a00,a32=a00,a33=a00;
            for(int i=0;i<KVL;i+=vl){
                svint16_t qa=svld1_s16(pg,q0+i),qb=svld1_s16(pg,q1+i),qc=svld1_s16(pg,q2+i),qd=svld1_s16(pg,q3+i);
                svint16_t ka=svld1_s16(pg,k0+i);
                a00=svdot_s64(a00,qa,ka); a10=svdot_s64(a10,qb,ka); a20=svdot_s64(a20,qc,ka); a30=svdot_s64(a30,qd,ka);
                svint16_t kb=svld1_s16(pg,k1+i);
                a01=svdot_s64(a01,qa,kb); a11=svdot_s64(a11,qb,kb); a21=svdot_s64(a21,qc,kb); a31=svdot_s64(a31,qd,kb);
                svint16_t kc=svld1_s16(pg,k2+i);
                a02=svdot_s64(a02,qa,kc); a12=svdot_s64(a12,qb,kc); a22=svdot_s64(a22,qc,kc); a32=svdot_s64(a32,qd,kc);
                svint16_t kd=svld1_s16(pg,k3+i);
                a03=svdot_s64(a03,qa,kd); a13=svdot_s64(a13,qb,kd); a23=svdot_s64(a23,qc,kd); a33=svdot_s64(a33,qd,kd);
            }
            svbool_t p64=svptrue_b64();
            float ks0=ksc[j],ks1=ksc[j+1],ks2=ksc[j+2],ks3=ksc[j+3];
            float*s0=S+(size_t)(h0+0)*ns+j,*s1=S+(size_t)(h0+1)*ns+j,*s2=S+(size_t)(h0+2)*ns+j,*s3=S+(size_t)(h0+3)*ns+j;
            s0[0]=(float)svaddv_s64(p64,a00)*qs0*ks0; s0[1]=(float)svaddv_s64(p64,a01)*qs0*ks1; s0[2]=(float)svaddv_s64(p64,a02)*qs0*ks2; s0[3]=(float)svaddv_s64(p64,a03)*qs0*ks3;
            s1[0]=(float)svaddv_s64(p64,a10)*qs1*ks0; s1[1]=(float)svaddv_s64(p64,a11)*qs1*ks1; s1[2]=(float)svaddv_s64(p64,a12)*qs1*ks2; s1[3]=(float)svaddv_s64(p64,a13)*qs1*ks3;
            s2[0]=(float)svaddv_s64(p64,a20)*qs2*ks0; s2[1]=(float)svaddv_s64(p64,a21)*qs2*ks1; s2[2]=(float)svaddv_s64(p64,a22)*qs2*ks2; s2[3]=(float)svaddv_s64(p64,a23)*qs2*ks3;
            s3[0]=(float)svaddv_s64(p64,a30)*qs3*ks0; s3[1]=(float)svaddv_s64(p64,a31)*qs3*ks1; s3[2]=(float)svaddv_s64(p64,a32)*qs3*ks2; s3[3]=(float)svaddv_s64(p64,a33)*qs3*ks3;
        }
        for(;j<ns;j++){ const int16_t*kj=KV+(size_t)j*KVL; float ks=ksc[j]; svbool_t p64=svptrue_b64();
            for(int hh=0;hh<4;hh++){ svint64_t ac=svdup_s64(0); const int16_t*qh=Q+(size_t)(h0+hh)*KVL;
                for(int i=0;i<KVL;i+=vl) ac=svdot_s64(ac,svld1_s16(pg,qh+i),svld1_s16(pg,kj+i));
                S[(size_t)(h0+hh)*ns+j]=(float)svaddv_s64(p64,ac)*qsc[h0+hh]*ks; } }
    }
}

int main(void){
    int H=envi("H",64), ns=envi("NS",512), reps=envi("REPS",200);
    float*Q=aligned_alloc(256,(size_t)H*KVL*4), *KV=aligned_alloc(256,(size_t)ns*KVL*4);
    float*S0=aligned_alloc(256,(size_t)H*ns*4),*S1=aligned_alloc(256,(size_t)H*ns*4),*S2=aligned_alloc(256,(size_t)H*ns*4);
    int16_t*Qi=aligned_alloc(256,(size_t)H*KVL*2),*KVi=aligned_alloc(256,(size_t)ns*KVL*2);
    float*qsc=aligned_alloc(256,(size_t)H*4),*ksc=aligned_alloc(256,(size_t)ns*4);
    uint32_t s=12345; for(size_t i=0;i<(size_t)H*KVL;i++) Q[i]=((int)(lcg(&s)%2001)-1000)*1e-3f;
    for(size_t i=0;i<(size_t)ns*KVL;i++) KV[i]=((int)(lcg(&s)%2001)-1000)*1e-3f;
    quant_i16(Q,Qi,qsc,H,KVL); quant_i16(KV,KVi,ksc,ns,KVL);
    /* reference (f64) */
    double*Sref=malloc((size_t)H*ns*8);
    for(int h=0;h<H;h++)for(int j=0;j<ns;j++){ double d=0; for(int k=0;k<KVL;k++) d+=(double)Q[(size_t)h*KVL+k]*KV[(size_t)j*KVL+k]; Sref[(size_t)h*ns+j]=d; }
    double ops=2.0*reps*H*ns*(double)KVL;
    /* warm + time each */
    #define BENCH(NAME,CALL,SBUF) do{ CALL; double t0=wall(); for(int r=0;r<reps;r++){ CALL; } double dt=wall()-t0; \
        double rms=0,ref=0; for(size_t i=0;i<(size_t)H*ns;i++){ double e=SBUF[i]-Sref[i]; rms+=e*e; ref+=Sref[i]*Sref[i]; } \
        rms=sqrt(rms/ref); printf("  %-14s %7.1f Gop/s   rms_rel=%.2e\n",NAME,ops/dt/1e9,rms); }while(0)
    printf("QK scores: H=%d heads, kv_lora=%d, ns=%d keys, reps=%d, threads=%d\n",H,KVL,ns,reps,
#ifdef _OPENMP
        omp_get_max_threads());
#else
        1);
#endif
    BENCH("pairwise_f32", qk_pairwise(Q,KV,S0,H,ns), S0);
    BENCH("blocked_f32",  qk_fp32_rb(Q,KV,S1,H,ns),  S1);
    BENCH("blocked_int16", qk_int16_rb(Qi,qsc,KVi,ksc,S2,H,ns), S2);
    return 0;
}
