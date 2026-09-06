/* sdot_roofline.c - Why is single-core int8 SDOT matvec sub-20x?
 *
 * Three measurements on one A64FX core:
 *   1. COMPUTE PEAK : SDOT on register-resident data, 16-way ILP, no memory.
 *                     Should approach 2 SDOT/cyc * 64 MAC * 2 ops * freq.
 *   2. STREAM BW    : SDOT-reduce a large (>L2) int8 buffer. Single-core
 *                     read bandwidth; matvec (intensity 2 op/B) is pinned
 *                     to ~2*BW here.
 *   3. REUSE SWEEP  : keep 16-way ILP fixed, vary how many SDOTs share each
 *                     loaded 64-B weight vector (intensity = 2*reuse op/B).
 *                     This is the matvec->GEMM axis: reuse=1 is matvec
 *                     (BW-bound), reuse>=8 is GEMM (compute-bound -> peak).
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
 *           -D_GNU_SOURCE sdot_roofline.c -lm -o sdot_roofline
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arm_sve.h>

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }

/* One svdot_s32 = 16 int32 lanes * 4 int8 MACs = 64 MACs = 128 int8 ops. */
#define OPS_PER_SDOT 128.0

#define DECL16 svint32_t \
  c0=svdup_s32(0),c1=svdup_s32(0),c2=svdup_s32(0),c3=svdup_s32(0), \
  c4=svdup_s32(0),c5=svdup_s32(0),c6=svdup_s32(0),c7=svdup_s32(0), \
  c8=svdup_s32(0),c9=svdup_s32(0),cA=svdup_s32(0),cB=svdup_s32(0), \
  cC=svdup_s32(0),cD=svdup_s32(0),cE=svdup_s32(0),cF=svdup_s32(0)
#define SINK16 do{ svbool_t pg=svptrue_b32(); \
  svint32_t s=svadd_s32_x(pg, \
    svadd_s32_x(pg,svadd_s32_x(pg,svadd_s32_x(pg,c0,c1),svadd_s32_x(pg,c2,c3)), \
                   svadd_s32_x(pg,svadd_s32_x(pg,c4,c5),svadd_s32_x(pg,c6,c7))), \
    svadd_s32_x(pg,svadd_s32_x(pg,svadd_s32_x(pg,c8,c9),svadd_s32_x(pg,cA,cB)), \
                   svadd_s32_x(pg,svadd_s32_x(pg,cC,cD),svadd_s32_x(pg,cE,cF)))); \
  volatile int sink=svaddv_s32(pg,s);(void)sink; }while(0)

static double peak_giops(double freq){
    svint8_t a=svdup_s8(3), b=svdup_s8(5);
    DECL16;
    const uint64_t ITERS=200ULL*1000*1000;     /* 16 SDOT each = 3.2e9 SDOT */
    uint64_t t0=rdcyc();
    for(uint64_t i=0;i<ITERS;i++){
        c0=svdot_s32(c0,a,b); c1=svdot_s32(c1,a,b); c2=svdot_s32(c2,a,b); c3=svdot_s32(c3,a,b);
        c4=svdot_s32(c4,a,b); c5=svdot_s32(c5,a,b); c6=svdot_s32(c6,a,b); c7=svdot_s32(c7,a,b);
        c8=svdot_s32(c8,a,b); c9=svdot_s32(c9,a,b); cA=svdot_s32(cA,a,b); cB=svdot_s32(cB,a,b);
        cC=svdot_s32(cC,a,b); cD=svdot_s32(cD,a,b); cE=svdot_s32(cE,a,b); cF=svdot_s32(cF,a,b);
    }
    uint64_t t1=rdcyc(); SINK16;
    double sec=(double)(t1-t0)/freq;
    return ITERS*16.0*OPS_PER_SDOT/sec/1e9;
}

/* 16-way ILP SDOT-reduce of a big buffer; intensity 2 op/B (== matvec). */
static void stream_bw(double freq,const int8_t*buf,size_t bytes,int reps,double*giops,double*gbps){
    svint8_t x=svdup_s8(7);
    size_t blocks=(bytes/64)&~(size_t)15;
    DECL16;
    uint64_t t0=rdcyc();
    for(int r=0;r<reps;r++){
        const int8_t*p=buf;
        for(size_t b=0;b<blocks;b+=16,p+=16*64){
            c0=svdot_s32(c0,svld1_s8(svptrue_b8(),p+ 0*64),x); c1=svdot_s32(c1,svld1_s8(svptrue_b8(),p+ 1*64),x);
            c2=svdot_s32(c2,svld1_s8(svptrue_b8(),p+ 2*64),x); c3=svdot_s32(c3,svld1_s8(svptrue_b8(),p+ 3*64),x);
            c4=svdot_s32(c4,svld1_s8(svptrue_b8(),p+ 4*64),x); c5=svdot_s32(c5,svld1_s8(svptrue_b8(),p+ 5*64),x);
            c6=svdot_s32(c6,svld1_s8(svptrue_b8(),p+ 6*64),x); c7=svdot_s32(c7,svld1_s8(svptrue_b8(),p+ 7*64),x);
            c8=svdot_s32(c8,svld1_s8(svptrue_b8(),p+ 8*64),x); c9=svdot_s32(c9,svld1_s8(svptrue_b8(),p+ 9*64),x);
            cA=svdot_s32(cA,svld1_s8(svptrue_b8(),p+10*64),x); cB=svdot_s32(cB,svld1_s8(svptrue_b8(),p+11*64),x);
            cC=svdot_s32(cC,svld1_s8(svptrue_b8(),p+12*64),x); cD=svdot_s32(cD,svld1_s8(svptrue_b8(),p+13*64),x);
            cE=svdot_s32(cE,svld1_s8(svptrue_b8(),p+14*64),x); cF=svdot_s32(cF,svld1_s8(svptrue_b8(),p+15*64),x);
        }
    }
    uint64_t t1=rdcyc(); SINK16;
    double sec=(double)(t1-t0)/freq;
    *giops=(double)blocks*reps*OPS_PER_SDOT/sec/1e9;
    *gbps =(double)blocks*64*reps/sec/1e9;
}

/* Reuse kernels: 16 SDOT per inner block, L distinct 64-B loads (L=16/reuse).
 * Intensity = 16*128/(L*64) = 2*reuse op/B. blocks chosen so total bytes
 * touched ~= the full (>L2) buffer => loads hit HBM in the low-reuse cases. */
#define LD(i) svld1_s8(svptrue_b8(), p+(i)*64)

/* reuse=1: 16 loads, each SDOT its own vector (== pure stream, intensity 2) */
static double reuse1(double freq,const int8_t*buf,size_t bytes,int reps){
    svint8_t x=svdup_s8(7); DECL16;
    size_t blocks=bytes/(16*64); uint64_t t0=rdcyc();
    for(int r=0;r<reps;r++){ const int8_t*p=buf;
        for(size_t b=0;b<blocks;b++,p+=16*64){
            c0=svdot_s32(c0,LD(0),x); c1=svdot_s32(c1,LD(1),x); c2=svdot_s32(c2,LD(2),x); c3=svdot_s32(c3,LD(3),x);
            c4=svdot_s32(c4,LD(4),x); c5=svdot_s32(c5,LD(5),x); c6=svdot_s32(c6,LD(6),x); c7=svdot_s32(c7,LD(7),x);
            c8=svdot_s32(c8,LD(8),x); c9=svdot_s32(c9,LD(9),x); cA=svdot_s32(cA,LD(10),x); cB=svdot_s32(cB,LD(11),x);
            cC=svdot_s32(cC,LD(12),x); cD=svdot_s32(cD,LD(13),x); cE=svdot_s32(cE,LD(14),x); cF=svdot_s32(cF,LD(15),x);
        } }
    uint64_t t1=rdcyc(); SINK16; double sec=(double)(t1-t0)/freq;
    return (double)blocks*16.0*reps*OPS_PER_SDOT/sec/1e9;
}
/* reuse=4: 4 loads, each shared by 4 SDOTs (intensity 8) */
static double reuse4(double freq,const int8_t*buf,size_t bytes,int reps){
    svint8_t x=svdup_s8(7); DECL16;
    size_t blocks=bytes/(4*64); uint64_t t0=rdcyc();
    for(int r=0;r<reps;r++){ const int8_t*p=buf;
        for(size_t b=0;b<blocks;b++,p+=4*64){
            svint8_t w0=LD(0),w1=LD(1),w2=LD(2),w3=LD(3);
            c0=svdot_s32(c0,w0,x); c1=svdot_s32(c1,w0,x); c2=svdot_s32(c2,w0,x); c3=svdot_s32(c3,w0,x);
            c4=svdot_s32(c4,w1,x); c5=svdot_s32(c5,w1,x); c6=svdot_s32(c6,w1,x); c7=svdot_s32(c7,w1,x);
            c8=svdot_s32(c8,w2,x); c9=svdot_s32(c9,w2,x); cA=svdot_s32(cA,w2,x); cB=svdot_s32(cB,w2,x);
            cC=svdot_s32(cC,w3,x); cD=svdot_s32(cD,w3,x); cE=svdot_s32(cE,w3,x); cF=svdot_s32(cF,w3,x);
        } }
    uint64_t t1=rdcyc(); SINK16; double sec=(double)(t1-t0)/freq;
    return (double)blocks*16.0*reps*OPS_PER_SDOT/sec/1e9;
}
/* reuse=8: 2 loads, each shared by 8 SDOTs (intensity 16) */
static double reuse8(double freq,const int8_t*buf,size_t bytes,int reps){
    svint8_t x=svdup_s8(7); DECL16;
    size_t blocks=bytes/(2*64); uint64_t t0=rdcyc();
    for(int r=0;r<reps;r++){ const int8_t*p=buf;
        for(size_t b=0;b<blocks;b++,p+=2*64){
            svint8_t w0=LD(0),w1=LD(1);
            c0=svdot_s32(c0,w0,x); c1=svdot_s32(c1,w0,x); c2=svdot_s32(c2,w0,x); c3=svdot_s32(c3,w0,x);
            c4=svdot_s32(c4,w0,x); c5=svdot_s32(c5,w0,x); c6=svdot_s32(c6,w0,x); c7=svdot_s32(c7,w0,x);
            c8=svdot_s32(c8,w1,x); c9=svdot_s32(c9,w1,x); cA=svdot_s32(cA,w1,x); cB=svdot_s32(cB,w1,x);
            cC=svdot_s32(cC,w1,x); cD=svdot_s32(cD,w1,x); cE=svdot_s32(cE,w1,x); cF=svdot_s32(cF,w1,x);
        } }
    uint64_t t1=rdcyc(); SINK16; double sec=(double)(t1-t0)/freq;
    return (double)blocks*16.0*reps*OPS_PER_SDOT/sec/1e9;
}
/* reuse=16: 1 load shared by all 16 SDOTs (intensity 32) */
static double reuse16(double freq,const int8_t*buf,size_t bytes,int reps){
    svint8_t x=svdup_s8(7); DECL16;
    size_t blocks=bytes/(1*64); uint64_t t0=rdcyc();
    for(int r=0;r<reps;r++){ const int8_t*p=buf;
        for(size_t b=0;b<blocks;b++,p+=1*64){
            svint8_t w0=LD(0);
            c0=svdot_s32(c0,w0,x); c1=svdot_s32(c1,w0,x); c2=svdot_s32(c2,w0,x); c3=svdot_s32(c3,w0,x);
            c4=svdot_s32(c4,w0,x); c5=svdot_s32(c5,w0,x); c6=svdot_s32(c6,w0,x); c7=svdot_s32(c7,w0,x);
            c8=svdot_s32(c8,w0,x); c9=svdot_s32(c9,w0,x); cA=svdot_s32(cA,w0,x); cB=svdot_s32(cB,w0,x);
            cC=svdot_s32(cC,w0,x); cD=svdot_s32(cD,w0,x); cE=svdot_s32(cE,w0,x); cF=svdot_s32(cF,w0,x);
        } }
    uint64_t t1=rdcyc(); SINK16; double sec=(double)(t1-t0)/freq;
    return (double)blocks*16.0*reps*OPS_PER_SDOT/sec/1e9;
}

int main(void){
    double freq=(double)rdfreq();
    printf("cntfrq=%.0f MHz (timer only; CPU runs at 2.0/2.2 GHz)\n", freq/1e6);
    printf("int8 SDOT compute peak: @2.0GHz=512 GIOPS/core, @2.2GHz=563\n\n");

    printf("[1] COMPUTE PEAK (register-resident, 16-way ILP, no memory)\n");
    printf("    %.1f GIOPS/core\n\n", peak_giops(freq));

    size_t BYTES=256ULL*1024*1024;             /* >> 8 MB CMG L2 */
    int8_t *buf=(int8_t*)aligned_alloc(256,BYTES);
    for(size_t i=0;i<BYTES;i++) buf[i]=(int8_t)(i&0x3f);

    double g,bw; stream_bw(freq,buf,BYTES,4,&g,&bw);
    printf("[2] STREAM BW (256 MB read, intensity 2 op/B == matvec M=1)\n");
    printf("    %.1f GB/s single-core read => %.1f GIOPS (roofline 2*BW=%.0f)\n\n", bw,g,2*bw);

    printf("[3] REUSE SWEEP (16-way ILP fixed; weights stream from 256 MB)\n");
    printf("    %-7s %-15s %8s %9s  %s\n","reuse","intensity(op/B)","GIOPS","%peak","note");
    struct{int ru;double(*fn)(double,const int8_t*,size_t,int);}t[]={
        {1,reuse1},{4,reuse4},{8,reuse8},{16,reuse16}};
    for(int k=0;k<4;k++){
        double gi=t[k].fn(freq,buf,BYTES,4);
        const char*note=t[k].ru==1?"matvec (M=1)":(t[k].ru>=8?"GEMM (heavy reuse)":"");
        printf("    %-7d %-15d %8.1f %8.1f%%  %s\n",t[k].ru,2*t[k].ru,gi,gi/512*100,note);
    }
    free(buf);
    return 0;
}
