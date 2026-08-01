/* attn_decode_bench.c — GLM5.2 decode "absorb" flash-attention inner loop, real dims.
 * Cycle-profiled under qlair (CNTVCT delta == simulated ns). Compares the current single-
 * accumulator dot (glm5_dot_f32_opt) against a 4-accumulator dot that breaks the A64FX FMA
 * dependency chain. The axpy/scale online-softmax accumulation is included for realism.
 *
 * Build:  aarch64-linux-gnu-gcc -O3 -march=armv8.2-a+sve -ffast-math -static \
 *           attn_decode_bench.c -o attn_decode_bench.elf -lm
 * Run:    ~/work/clair/clair/build/qlair attn_decode_bench.elf         # cycle-accurate (default)
 */
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <arm_sve.h>

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ __volatile__("isb; mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t cntfrq(void){ uint64_t f; __asm__ __volatile__("mrs %0, cntfrq_el0":"=r"(f)); return f; }
/* CNTVCT ticks -> ns: qlair sets CNTFRQ=1e9 (ticks==ns); real A64FX ~100 MHz (x10). Report ns so
 * a native 1-node run (K0) and the qlair cycle run are directly comparable -> sets --friction. */
static uint64_t G_FRQ=1;
static inline uint64_t to_ns(uint64_t ticks){ return G_FRQ? ticks*1000000000ull/G_FRQ : ticks; }

#define KVL   512      /* kv_lora  */
#define ROPE  64       /* qk_rope  */
#define NOWN  16       /* heads on this rank */
#define NSEL  128      /* context length (selected KV positions) */
#define REPS  8

static float qa[NOWN*KVL], qr[NOWN*ROPE], kvbuf[NSEL*(KVL+ROPE)], ctx[NOWN*KVL];

/* --- current: single-accumulator SVE dot (mirrors glm5_dot_f32_opt) --- */
static inline float dot1(const float*a,const float*b,int n){
    svfloat32_t acc=svdup_f32(0.f);
    for(int i=0;i<n;i+=svcntw()){ svbool_t pg=svwhilelt_b32(i,n); acc=svmla_f32_x(pg,acc,svld1(pg,a+i),svld1(pg,b+i)); }
    return svaddv_f32(svptrue_b32(),acc);
}
/* --- optimized: 4 independent accumulators (breaks the FMA latency chain) --- */
static inline float dot4(const float*a,const float*b,int n){
    svfloat32_t a0=svdup_f32(0.f),a1=a0,a2=a0,a3=a0; int vl=svcntw(); int i=0;
    for(;i+4*vl<=n;i+=4*vl){
        svbool_t pg=svptrue_b32();
        a0=svmla_f32_x(pg,a0,svld1(pg,a+i+0*vl),svld1(pg,b+i+0*vl));
        a1=svmla_f32_x(pg,a1,svld1(pg,a+i+1*vl),svld1(pg,b+i+1*vl));
        a2=svmla_f32_x(pg,a2,svld1(pg,a+i+2*vl),svld1(pg,b+i+2*vl));
        a3=svmla_f32_x(pg,a3,svld1(pg,a+i+3*vl),svld1(pg,b+i+3*vl));
    }
    for(;i<n;i+=vl){ svbool_t pg=svwhilelt_b32(i,n); a0=svmla_f32_x(pg,a0,svld1(pg,a+i),svld1(pg,b+i)); }
    return svaddv_f32(svptrue_b32(),svadd_f32_x(svptrue_b32(),svadd_f32_x(svptrue_b32(),a0,a1),svadd_f32_x(svptrue_b32(),a2,a3)));
}
/* --- 8 accumulators --- */
static inline float dot8(const float*a,const float*b,int n){
    svfloat32_t z=svdup_f32(0.f),a0=z,a1=z,a2=z,a3=z,a4=z,a5=z,a6=z,a7=z; int vl=svcntw(); int i=0; svbool_t pg=svptrue_b32();
    for(;i+8*vl<=n;i+=8*vl){
        a0=svmla_f32_x(pg,a0,svld1(pg,a+i+0*vl),svld1(pg,b+i+0*vl)); a1=svmla_f32_x(pg,a1,svld1(pg,a+i+1*vl),svld1(pg,b+i+1*vl));
        a2=svmla_f32_x(pg,a2,svld1(pg,a+i+2*vl),svld1(pg,b+i+2*vl)); a3=svmla_f32_x(pg,a3,svld1(pg,a+i+3*vl),svld1(pg,b+i+3*vl));
        a4=svmla_f32_x(pg,a4,svld1(pg,a+i+4*vl),svld1(pg,b+i+4*vl)); a5=svmla_f32_x(pg,a5,svld1(pg,a+i+5*vl),svld1(pg,b+i+5*vl));
        a6=svmla_f32_x(pg,a6,svld1(pg,a+i+6*vl),svld1(pg,b+i+6*vl)); a7=svmla_f32_x(pg,a7,svld1(pg,a+i+7*vl),svld1(pg,b+i+7*vl));
    }
    for(;i<n;i+=vl){ svbool_t q=svwhilelt_b32(i,n); a0=svmla_f32_x(q,a0,svld1(q,a+i),svld1(q,b+i)); }
    a0=svadd_f32_x(pg,svadd_f32_x(pg,svadd_f32_x(pg,a0,a1),svadd_f32_x(pg,a2,a3)),svadd_f32_x(pg,svadd_f32_x(pg,a4,a5),svadd_f32_x(pg,a6,a7)));
    return svaddv_f32(pg,a0);
}
static inline void axpy(float*y,const float*x,float a,int n){
    for(int i=0;i<n;i+=svcntw()){ svbool_t pg=svwhilelt_b32(i,n); svst1(pg,y+i,svmla_n_f32_x(pg,svld1(pg,y+i),svld1(pg,x+i),a)); }
}
/* 4-way unrolled axpy: independent iterations (no reduction), so unrolling only helps the
 * load/store schedule -> test whether it wins in qlair before touching the real kernel. */
static inline void axpy4(float*y,const float*x,float a,int n){
    int vl=svcntw(); svbool_t pg=svptrue_b32(); int i=0;
    for(;i+4*vl<=n;i+=4*vl){
        svst1(pg,y+i,     svmla_n_f32_x(pg,svld1(pg,y+i),     svld1(pg,x+i),a));
        svst1(pg,y+i+vl,  svmla_n_f32_x(pg,svld1(pg,y+i+vl),  svld1(pg,x+i+vl),a));
        svst1(pg,y+i+2*vl,svmla_n_f32_x(pg,svld1(pg,y+i+2*vl),svld1(pg,x+i+2*vl),a));
        svst1(pg,y+i+3*vl,svmla_n_f32_x(pg,svld1(pg,y+i+3*vl),svld1(pg,x+i+3*vl),a));
    }
    for(;i<n;i+=vl){ svbool_t q=svwhilelt_b32(i,n); svst1(q,y+i,svmla_n_f32_x(q,svld1(q,y+i),svld1(q,x+i),a)); }
}
static inline void scal(float*y,float s,int n){
    for(int i=0;i<n;i+=svcntw()){ svbool_t pg=svwhilelt_b32(i,n); svst1(pg,y+i,svmul_n_f32_x(pg,svld1(pg,y+i),s)); }
}

/* one layer of absorb attention: NSEL positions x NOWN heads, dotN selects the dot variant */
static double run(int variant){
    float hmx[NOWN],hse[NOWN]; const float ascale=1.f/sqrtf(256.f);
    for(int hh=0;hh<NOWN;hh++){ hmx[hh]=-1e30f; hse[hh]=0.f; for(int i=0;i<KVL;i++) ctx[hh*KVL+i]=0.f; }
    for(int j=0;j<NSEL;j++){
        const float*kv=kvbuf+j*(KVL+ROPE);
        for(int hh=0;hh<NOWN;hh++){
            float d = variant? dot4(qa+hh*KVL,kv,KVL) : dot1(qa+hh*KVL,kv,KVL);
            const float*qrp=qr+hh*ROPE; for(int i=0;i<ROPE;i++) d+=qrp[i]*kv[KVL+i];
            float s=d*ascale; float*c=ctx+hh*KVL;
            if(s>hmx[hh]){ float r=(hmx[hh]>-1e20f)?expf(hmx[hh]-s):0.f; hse[hh]*=r; scal(c,r,KVL); hmx[hh]=s; }
            float e=expf(s-hmx[hh]); hse[hh]+=e; axpy(c,kv,e,KVL);
        }
    }
    double chk=0; for(int hh=0;hh<NOWN;hh++){ float inv=1.f/(hse[hh]>0?hse[hh]:1); for(int i=0;i<KVL;i++) chk+=ctx[hh*KVL+i]*inv; }
    return chk;
}

int main(void){
    G_FRQ=cntfrq(); printf("cntfrq=%llu Hz (qlair=1e9; A64FX~1e8)\n",(unsigned long long)G_FRQ);
    for(int i=0;i<NOWN*KVL;i++) qa[i]=((i*13)%101-50)/50.f;
    for(int i=0;i<NOWN*ROPE;i++) qr[i]=((i*7)%97-48)/48.f;
    for(int i=0;i<NSEL*(KVL+ROPE);i++) kvbuf[i]=((i*5)%89-44)/44.f;

    double c0=0,c1=0; uint64_t t0,t1,best0=~0ull,best1=~0ull;
    for(int r=0;r<REPS;r++){ t0=rdcyc(); c0=run(0); t1=rdcyc(); if(t1-t0<best0)best0=t1-t0; }
    for(int r=0;r<REPS;r++){ t0=rdcyc(); c1=run(1); t1=rdcyc(); if(t1-t0<best1)best1=t1-t0; }
    printf("attn_decode inner loop (NOWN=%d NSEL=%d KVL=%d, per-layer, 1 rank):\n",NOWN,NSEL,KVL);
    printf("  full loop dot1 single-acc: %llu ns   chk=%.4f\n",(unsigned long long)to_ns(best0),c0);
    printf("  full loop dot4 4-acc     : %llu ns   chk=%.4f\n",(unsigned long long)to_ns(best1),c1);
    printf("  => full-loop speedup: %llu/1000 (x1000)  ok=%d\n",
           (unsigned long long)(best1?best0*1000/best1:0), (c1-c0>-1e-2&&c1-c0<1e-2));

    /* isolated PURE dot: NSEL*NOWN dots, no expf/axpy dilution */
    volatile float sink=0; uint64_t bd1=~0ull,bd4=~0ull,bd8=~0ull;
    for(int r=0;r<REPS;r++){ t0=rdcyc(); for(int j=0;j<NSEL;j++)for(int hh=0;hh<NOWN;hh++) sink+=dot1(qa+hh*KVL,kvbuf+j*(KVL+ROPE),KVL); t1=rdcyc(); if(t1-t0<bd1)bd1=t1-t0; }
    for(int r=0;r<REPS;r++){ t0=rdcyc(); for(int j=0;j<NSEL;j++)for(int hh=0;hh<NOWN;hh++) sink+=dot4(qa+hh*KVL,kvbuf+j*(KVL+ROPE),KVL); t1=rdcyc(); if(t1-t0<bd4)bd4=t1-t0; }
    for(int r=0;r<REPS;r++){ t0=rdcyc(); for(int j=0;j<NSEL;j++)for(int hh=0;hh<NOWN;hh++) sink+=dot8(qa+hh*KVL,kvbuf+j*(KVL+ROPE),KVL); t1=rdcyc(); if(t1-t0<bd8)bd8=t1-t0; }
    printf("isolated pure dot (%d dots x %d dim):\n",NSEL*NOWN,KVL);
    printf("  dot1: %llu ns   dot4: %llu ns (x1000 %llu)   dot8: %llu ns (x1000 %llu)\n",
           (unsigned long long)to_ns(bd1),(unsigned long long)to_ns(bd4),(unsigned long long)(bd4?bd1*1000/bd4:0),
           (unsigned long long)to_ns(bd8),(unsigned long long)(bd8?bd1*1000/bd8:0));

    /* isolated AXPY: does unrolling the online-softmax accumulation help? (no reduction dep) */
    uint64_t ba1=~0ull,ba4=~0ull;
    for(int r=0;r<REPS;r++){ t0=rdcyc(); for(int j=0;j<NSEL;j++)for(int hh=0;hh<NOWN;hh++) axpy(ctx+hh*KVL,kvbuf+j*(KVL+ROPE),1.0001f,KVL); t1=rdcyc(); if(t1-t0<ba1)ba1=t1-t0; }
    for(int r=0;r<REPS;r++){ t0=rdcyc(); for(int j=0;j<NSEL;j++)for(int hh=0;hh<NOWN;hh++) axpy4(ctx+hh*KVL,kvbuf+j*(KVL+ROPE),1.0001f,KVL); t1=rdcyc(); if(t1-t0<ba4)ba4=t1-t0; }
    printf("isolated axpy (%d x %d dim): axpy1 %llu ns   axpy4 %llu ns (x1000 %llu)\n",
           NSEL*NOWN,KVL,(unsigned long long)to_ns(ba1),(unsigned long long)to_ns(ba4),
           (unsigned long long)(ba4?ba1*1000/ba4:0));
    return (int)sink&0;
}
