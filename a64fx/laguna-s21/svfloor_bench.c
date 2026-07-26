/* True floor of the attention inner loop (8 loads + 8 FMLA over head_dim=128).
 *
 * NB an earlier version of this benchmark consumed only 2 of the 8 accumulators,
 * so the compiler deleted 6 of the 8 FMLA chains as dead and reported a "floor"
 * of ~9 cyc/key that no real kernel could reach.  Every variant here stores all
 * eight, so nothing can be eliminated.
 *
 * Variants, n keys, kv stride as in the runner:
 *   bf16  8x svld1uh_u32 (widening: fills 16 f32 lanes from 32 bytes) + 8 FMLA
 *   unpk  4x svld1_u16 (full 64-byte loads) + unpack + 8 FMLA
 *   f32   8x svld1_f32 (no widening at all; 2x the bytes) + 8 FMLA
 * plus the qk shapes: per-key svaddv vs a zip-tree over 16 keys.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <arm_sve.h>

enum { HD = 128, KVH = 8 };
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9; }
static uint16_t f2b(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)((u+0x7fff+((u>>16)&1))>>16); }
static inline svfloat32_t ldb(svbool_t pg, const uint16_t *p) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, p), 16));
}
#define ST8 svst1_f32(pt,acc+0*VL,a0);svst1_f32(pt,acc+1*VL,a1); \
            svst1_f32(pt,acc+2*VL,a2);svst1_f32(pt,acc+3*VL,a3); \
            svst1_f32(pt,acc+4*VL,a4);svst1_f32(pt,acc+5*VL,a5); \
            svst1_f32(pt,acc+6*VL,a6);svst1_f32(pt,acc+7*VL,a7)
#define LD8 svfloat32_t a0=svld1_f32(pt,acc+0*VL),a1=svld1_f32(pt,acc+1*VL), \
            a2=svld1_f32(pt,acc+2*VL),a3=svld1_f32(pt,acc+3*VL), \
            a4=svld1_f32(pt,acc+4*VL),a5=svld1_f32(pt,acc+5*VL), \
            a6=svld1_f32(pt,acc+6*VL),a7=svld1_f32(pt,acc+7*VL)

#define QREGS(qp) \
    svfloat32_t q0=svld1_f32(pt,(qp)+0*VL),q1=svld1_f32(pt,(qp)+1*VL), \
                q2=svld1_f32(pt,(qp)+2*VL),q3=svld1_f32(pt,(qp)+3*VL), \
                q4=svld1_f32(pt,(qp)+4*VL),q5=svld1_f32(pt,(qp)+5*VL), \
                q6=svld1_f32(pt,(qp)+6*VL),q7=svld1_f32(pt,(qp)+7*VL)
#define DOT(dst, kp) do { const uint16_t *k_=(kp); \
    svfloat32_t x_=svmul_f32_x(pt,q0,ldb(pt,k_+0*VL)); \
    svfloat32_t y_=svmul_f32_x(pt,q4,ldb(pt,k_+4*VL)); \
    x_=svmla_f32_x(pt,x_,q1,ldb(pt,k_+1*VL)); y_=svmla_f32_x(pt,y_,q5,ldb(pt,k_+5*VL)); \
    x_=svmla_f32_x(pt,x_,q2,ldb(pt,k_+2*VL)); y_=svmla_f32_x(pt,y_,q6,ldb(pt,k_+6*VL)); \
    x_=svmla_f32_x(pt,x_,q3,ldb(pt,k_+3*VL)); y_=svmla_f32_x(pt,y_,q7,ldb(pt,k_+7*VL)); \
    (dst)=svadd_f32_x(pt,x_,y_); } while(0)
#define BFLY(a,b) svadd_f32_x(pt, svuzp1_f32((a),(b)), svuzp2_f32((a),(b)))
#define UNLO(h) svreinterpret_f32_u32(svlsl_n_u32_x(pt,svunpklo_u32(h),16))
#define UNHI(h) svreinterpret_f32_u32(svlsl_n_u32_x(pt,svunpkhi_u32(h),16))
/* same dot with 4 full-width u16 loads instead of 8 widening ones */
#define DOTU(dst, kp) do { const uint16_t *k_=(kp); \
    svuint16_t h0_=svld1_u16(ph,k_+0*2*VL), h1_=svld1_u16(ph,k_+1*2*VL); \
    svuint16_t h2_=svld1_u16(ph,k_+2*2*VL), h3_=svld1_u16(ph,k_+3*2*VL); \
    svfloat32_t x_=svmul_f32_x(pt,q0,UNLO(h0_)); \
    svfloat32_t y_=svmul_f32_x(pt,q4,UNLO(h2_)); \
    x_=svmla_f32_x(pt,x_,q1,UNHI(h0_)); y_=svmla_f32_x(pt,y_,q5,UNHI(h2_)); \
    x_=svmla_f32_x(pt,x_,q2,UNLO(h1_)); y_=svmla_f32_x(pt,y_,q6,UNLO(h3_)); \
    x_=svmla_f32_x(pt,x_,q3,UNHI(h1_)); y_=svmla_f32_x(pt,y_,q7,UNHI(h3_)); \
    (dst)=svadd_f32_x(pt,x_,y_); } while(0)

int main(int argc,char**argv){
    int n=argc>1?atoi(argv[1]):128, reps=argc>2?atoi(argv[2]):60000;
    int stride=KVH*HD, VL=(int)svcntw();
    uint16_t *V=aligned_alloc(256,(size_t)n*stride*2);
    float *Vf=aligned_alloc(256,(size_t)n*stride*4);
    float *w=aligned_alloc(256,(size_t)(n+16)*4), *acc=aligned_alloc(256,HD*4);
    float *q=aligned_alloc(256,HD*4), *out=aligned_alloc(256,(size_t)(n+64)*4);
    uint64_t s=5;
    for(size_t i=0;i<(size_t)n*stride;i++){ s=s*6364136223846793005ull+1;
        float v=(float)((int)((s>>33)%200)-100)/300.0f; V[i]=f2b(v); Vf[i]=v; }
    for(int i=0;i<n;i++) w[i]=0.01f;
    for(int i=0;i<HD;i++){ acc[i]=0; q[i]=(float)((i%13)-6)/7.0f; }
    svbool_t pt=svptrue_b32(); svbool_t ph=svptrue_b16();
    double t; volatile float sink=0;

    t=now_s();
    for(int r=0;r<reps;r++){ LD8;
        for(int i=0;i<n;i++){ const uint16_t *v=V+(size_t)i*stride; svfloat32_t p=svdup_f32(w[i]);
            a0=svmla_f32_x(pt,a0,p,ldb(pt,v+0*VL)); a1=svmla_f32_x(pt,a1,p,ldb(pt,v+1*VL));
            a2=svmla_f32_x(pt,a2,p,ldb(pt,v+2*VL)); a3=svmla_f32_x(pt,a3,p,ldb(pt,v+3*VL));
            a4=svmla_f32_x(pt,a4,p,ldb(pt,v+4*VL)); a5=svmla_f32_x(pt,a5,p,ldb(pt,v+5*VL));
            a6=svmla_f32_x(pt,a6,p,ldb(pt,v+6*VL)); a7=svmla_f32_x(pt,a7,p,ldb(pt,v+7*VL)); }
        ST8; }
    double c_bf16=(now_s()-t)/reps/n*1e9;

    t=now_s();
    for(int r=0;r<reps;r++){ LD8;
        for(int i=0;i<n;i++){ const uint16_t *v=V+(size_t)i*stride; svfloat32_t p=svdup_f32(w[i]);
            svuint16_t h0=svld1_u16(ph,v+0*2*VL), h1=svld1_u16(ph,v+1*2*VL);
            svuint16_t h2=svld1_u16(ph,v+2*2*VL), h3=svld1_u16(ph,v+3*2*VL);
            a0=svmla_f32_x(pt,a0,p,svreinterpret_f32_u32(svlsl_n_u32_x(pt,svunpklo_u32(h0),16)));
            a1=svmla_f32_x(pt,a1,p,svreinterpret_f32_u32(svlsl_n_u32_x(pt,svunpkhi_u32(h0),16)));
            a2=svmla_f32_x(pt,a2,p,svreinterpret_f32_u32(svlsl_n_u32_x(pt,svunpklo_u32(h1),16)));
            a3=svmla_f32_x(pt,a3,p,svreinterpret_f32_u32(svlsl_n_u32_x(pt,svunpkhi_u32(h1),16)));
            a4=svmla_f32_x(pt,a4,p,svreinterpret_f32_u32(svlsl_n_u32_x(pt,svunpklo_u32(h2),16)));
            a5=svmla_f32_x(pt,a5,p,svreinterpret_f32_u32(svlsl_n_u32_x(pt,svunpkhi_u32(h2),16)));
            a6=svmla_f32_x(pt,a6,p,svreinterpret_f32_u32(svlsl_n_u32_x(pt,svunpklo_u32(h3),16)));
            a7=svmla_f32_x(pt,a7,p,svreinterpret_f32_u32(svlsl_n_u32_x(pt,svunpkhi_u32(h3),16))); }
        ST8; }
    double c_unpk=(now_s()-t)/reps/n*1e9;

    t=now_s();
    for(int r=0;r<reps;r++){ LD8;
        for(int i=0;i<n;i++){ const float *v=Vf+(size_t)i*stride; svfloat32_t p=svdup_f32(w[i]);
            a0=svmla_f32_x(pt,a0,p,svld1_f32(pt,v+0*VL)); a1=svmla_f32_x(pt,a1,p,svld1_f32(pt,v+1*VL));
            a2=svmla_f32_x(pt,a2,p,svld1_f32(pt,v+2*VL)); a3=svmla_f32_x(pt,a3,p,svld1_f32(pt,v+3*VL));
            a4=svmla_f32_x(pt,a4,p,svld1_f32(pt,v+4*VL)); a5=svmla_f32_x(pt,a5,p,svld1_f32(pt,v+5*VL));
            a6=svmla_f32_x(pt,a6,p,svld1_f32(pt,v+6*VL)); a7=svmla_f32_x(pt,a7,p,svld1_f32(pt,v+7*VL)); }
        ST8; }
    double c_f32=(now_s()-t)/reps/n*1e9;

    t=now_s();
    { QREGS(q);
      for(int r=0;r<reps;r++){
        for(int i=0;i<n;i++){ svfloat32_t v; DOT(v, V+(size_t)i*stride); out[i]=svaddv_f32(pt,v); }
        sink+=out[0]; } }
    double c_addv=(now_s()-t)/reps/n*1e9;

    t=now_s();
    { QREGS(q);
      for(int r=0;r<reps;r++){
        for(int base=0; base+16<=n; base+=16){
            const uint16_t *kb=V+(size_t)base*stride; svfloat32_t x0,x1;
            DOT(x0,kb+ 0*stride); DOT(x1,kb+ 1*stride); svfloat32_t t0=BFLY(x0,x1);
            DOT(x0,kb+ 2*stride); DOT(x1,kb+ 3*stride); svfloat32_t t1=BFLY(x0,x1);
            DOT(x0,kb+ 4*stride); DOT(x1,kb+ 5*stride); svfloat32_t t2=BFLY(x0,x1);
            DOT(x0,kb+ 6*stride); DOT(x1,kb+ 7*stride); svfloat32_t t3=BFLY(x0,x1);
            DOT(x0,kb+ 8*stride); DOT(x1,kb+ 9*stride); svfloat32_t t4=BFLY(x0,x1);
            DOT(x0,kb+10*stride); DOT(x1,kb+11*stride); svfloat32_t t5=BFLY(x0,x1);
            DOT(x0,kb+12*stride); DOT(x1,kb+13*stride); svfloat32_t t6=BFLY(x0,x1);
            DOT(x0,kb+14*stride); DOT(x1,kb+15*stride); svfloat32_t t7=BFLY(x0,x1);
            svfloat32_t u0=BFLY(t0,t1),u1=BFLY(t2,t3),u2=BFLY(t4,t5),u3=BFLY(t6,t7);
            svst1_f32(pt,out+base,BFLY(BFLY(u0,u1),BFLY(u2,u3)));
        }
        sink+=out[0]; } }
    double c_bfly=(now_s()-t)/reps/n*1e9;

    t=now_s();
    { QREGS(q);
      for(int r=0;r<reps;r++){
        for(int base=0; base+16<=n; base+=16){
            const uint16_t *kb=V+(size_t)base*stride; svfloat32_t x0,x1;
            DOTU(x0,kb+ 0*stride); DOTU(x1,kb+ 1*stride); svfloat32_t t0=BFLY(x0,x1);
            DOTU(x0,kb+ 2*stride); DOTU(x1,kb+ 3*stride); svfloat32_t t1=BFLY(x0,x1);
            DOTU(x0,kb+ 4*stride); DOTU(x1,kb+ 5*stride); svfloat32_t t2=BFLY(x0,x1);
            DOTU(x0,kb+ 6*stride); DOTU(x1,kb+ 7*stride); svfloat32_t t3=BFLY(x0,x1);
            DOTU(x0,kb+ 8*stride); DOTU(x1,kb+ 9*stride); svfloat32_t t4=BFLY(x0,x1);
            DOTU(x0,kb+10*stride); DOTU(x1,kb+11*stride); svfloat32_t t5=BFLY(x0,x1);
            DOTU(x0,kb+12*stride); DOTU(x1,kb+13*stride); svfloat32_t t6=BFLY(x0,x1);
            DOTU(x0,kb+14*stride); DOTU(x1,kb+15*stride); svfloat32_t t7=BFLY(x0,x1);
            svfloat32_t u0=BFLY(t0,t1),u1=BFLY(t2,t3),u2=BFLY(t4,t5),u3=BFLY(t6,t7);
            svst1_f32(pt,out+base,BFLY(BFLY(u0,u1),BFLY(u2,u3)));
        }
        sink+=out[0]; } }
    double c_bfu=(now_s()-t)/reps/n*1e9;

    { double worst=0;
      for(int i=0;i<(n/16)*16;i++){
        double ref=0; const uint16_t *k=V+(size_t)i*stride;
        for(int d=0;d<HD;d++){ uint32_t u=(uint32_t)k[d]<<16; float kv; memcpy(&kv,&u,4);
            ref += (double)q[d]*kv; }
        double e=fabs(ref-out[i]); if(e>worst) worst=e; }
      printf("butterfly max abs err vs scalar ref: %.3e\n", worst); }

    printf("n=%d head_dim=%d  (cycles at 2.0 GHz, all 8 accumulators live)\n",n,HD);
    printf("  av  bf16 widening ld   %6.2f ns  %5.1f cyc/key\n",c_bf16,c_bf16*2);
    printf("  av  full-width + unpk  %6.2f ns  %5.1f cyc/key  %.2fx\n",c_unpk,c_unpk*2,c_bf16/c_unpk);
    printf("  av  f32 (2x bytes)     %6.2f ns  %5.1f cyc/key  %.2fx\n",c_f32,c_f32*2,c_bf16/c_f32);
    printf("  qk  svaddv per key     %6.2f ns  %5.1f cyc/key\n",c_addv,c_addv*2);
    printf("  qk  zip tree /16 keys  %6.2f ns  %5.1f cyc/key  %.2fx\n",c_bfly,c_bfly*2,c_addv/c_bfly);
    printf("  qk  zip tree + unpk    %6.2f ns  %5.1f cyc/key  %.2fx\n",c_bfu,c_bfu*2,c_addv/c_bfu);
    (void)sink; return 0;
}
