/* int8_gemv_bench.c — GLM5.2 w8a16 int8 GEMV (glm5_matvec_int8_8row) profiled under qlair + native.
 * This is the dominant decode compute (qkv+shared+router+o_proj ~65 ms/tok). Question: is it
 * FMA-latency-bound (multi-acc helps) or byte-load/dequant-bound? qlair's profile summary answers it.
 *
 * Build: aarch64-linux-gnu-gcc -O2 -march=armv8.2-a+sve -fno-math-errno -fno-tree-vectorize -static \
 *          int8_gemv_bench.c -o int8_gemv_bench.elf -lm
 * Run:   ~/work/clair/clair/build/qlair -n 8G int8_gemv_bench.elf
 */
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "glm5_int8.h"

/* variant: c-loop unrolled 2x with 2 group-accumulators per row (16 independent FMA chains vs 8)
 * to hide the load->cvt->fma latency the baseline leaves exposed. Same group-factored scale/correction. */
static inline void matvec_int8_8row_u2(float*restrict dst,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,
        const float*s0,const float*s1,const float*s2,const float*s3,
        const float*s4,const float*s5,const float*s6,const float*s7,
        int gs,const float*x,int cols){
    svbool_t pt=svptrue_b32(); int vl=(int)svcntw();
    svfloat32_t a0=svdup_f32(0.f),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    float cr0=0,cr1=0,cr2=0,cr3=0,cr4=0,cr5=0,cr6=0,cr7=0;
    for(int b=0;b<cols;b+=gs){
        int bend=b+gs<cols?b+gs:cols, blk=b/gs;
        svfloat32_t g0=svdup_f32(0.f),g1=g0,g2=g0,g3=g0,g4=g0,g5=g0,g6=g0,g7=g0;
        svfloat32_t h0=g0,h1=g0,h2=g0,h3=g0,h4=g0,h5=g0,h6=g0,h7=g0; svfloat32_t sx=g0;
        int c=b;
        #define I8(WP,G,CC) do{ svfloat32_t wv=svcvt_f32_u32_x(pt,svld1ub_u32(pt,&(WP)[CC])); G=svmla_f32_x(pt,G,wv,xv); }while(0)
        for(;c+2*vl<=bend;c+=2*vl){
            svfloat32_t xv=svld1(pt,&x[c]); sx=svadd_f32_x(pt,sx,xv);
            I8(w0,g0,c);I8(w1,g1,c);I8(w2,g2,c);I8(w3,g3,c);I8(w4,g4,c);I8(w5,g5,c);I8(w6,g6,c);I8(w7,g7,c);
            xv=svld1(pt,&x[c+vl]); sx=svadd_f32_x(pt,sx,xv);
            I8(w0,h0,c+vl);I8(w1,h1,c+vl);I8(w2,h2,c+vl);I8(w3,h3,c+vl);I8(w4,h4,c+vl);I8(w5,h5,c+vl);I8(w6,h6,c+vl);I8(w7,h7,c+vl);
        }
        for(;c<bend;c+=vl){ svbool_t pg=svwhilelt_b32(c,bend); svfloat32_t xv=svld1(pg,&x[c]); sx=svadd_f32_m(pg,sx,xv);
            svfloat32_t wv;
            wv=svcvt_f32_u32_x(pg,svld1ub_u32(pg,&w0[c]));g0=svmla_f32_m(pg,g0,wv,xv); wv=svcvt_f32_u32_x(pg,svld1ub_u32(pg,&w1[c]));g1=svmla_f32_m(pg,g1,wv,xv);
            wv=svcvt_f32_u32_x(pg,svld1ub_u32(pg,&w2[c]));g2=svmla_f32_m(pg,g2,wv,xv); wv=svcvt_f32_u32_x(pg,svld1ub_u32(pg,&w3[c]));g3=svmla_f32_m(pg,g3,wv,xv);
            wv=svcvt_f32_u32_x(pg,svld1ub_u32(pg,&w4[c]));g4=svmla_f32_m(pg,g4,wv,xv); wv=svcvt_f32_u32_x(pg,svld1ub_u32(pg,&w5[c]));g5=svmla_f32_m(pg,g5,wv,xv);
            wv=svcvt_f32_u32_x(pg,svld1ub_u32(pg,&w6[c]));g6=svmla_f32_m(pg,g6,wv,xv); wv=svcvt_f32_u32_x(pg,svld1ub_u32(pg,&w7[c]));g7=svmla_f32_m(pg,g7,wv,xv);
        }
        #undef I8
        g0=svadd_f32_x(pt,g0,h0);g1=svadd_f32_x(pt,g1,h1);g2=svadd_f32_x(pt,g2,h2);g3=svadd_f32_x(pt,g3,h3);
        g4=svadd_f32_x(pt,g4,h4);g5=svadd_f32_x(pt,g5,h5);g6=svadd_f32_x(pt,g6,h6);g7=svadd_f32_x(pt,g7,h7);
        float Sx=svaddv_f32(pt,sx);
        float v0=s0[blk],v1=s1[blk],v2=s2[blk],v3=s3[blk],v4=s4[blk],v5=s5[blk],v6=s6[blk],v7=s7[blk];
        a0=svmla_n_f32_x(pt,a0,g0,v0);a1=svmla_n_f32_x(pt,a1,g1,v1);a2=svmla_n_f32_x(pt,a2,g2,v2);a3=svmla_n_f32_x(pt,a3,g3,v3);
        a4=svmla_n_f32_x(pt,a4,g4,v4);a5=svmla_n_f32_x(pt,a5,g5,v5);a6=svmla_n_f32_x(pt,a6,g6,v6);a7=svmla_n_f32_x(pt,a7,g7,v7);
        cr0+=v0*Sx;cr1+=v1*Sx;cr2+=v2*Sx;cr3+=v3*Sx;cr4+=v4*Sx;cr5+=v5*Sx;cr6+=v6*Sx;cr7+=v7*Sx;
    }
    dst[0]=svaddv_f32(pt,a0)-128.f*cr0;dst[1]=svaddv_f32(pt,a1)-128.f*cr1;dst[2]=svaddv_f32(pt,a2)-128.f*cr2;dst[3]=svaddv_f32(pt,a3)-128.f*cr3;
    dst[4]=svaddv_f32(pt,a4)-128.f*cr4;dst[5]=svaddv_f32(pt,a5)-128.f*cr5;dst[6]=svaddv_f32(pt,a6)-128.f*cr6;dst[7]=svaddv_f32(pt,a7)-128.f*cr7;
}

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ __volatile__("isb; mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t cntfrq(void){ uint64_t f; __asm__ __volatile__("mrs %0, cntfrq_el0":"=r"(f)); return f; }
static uint64_t G_FRQ=1;
static inline uint64_t to_ns(uint64_t t){ return G_FRQ? t*1000000000ull/G_FRQ : t; }

#define COLS 6144      /* hidden (wq_a/wkv_a/o_proj/shared input) */
#define GS   128       /* group-128 scale */
#define NG   (COLS/GS)
#define ROWS 256       /* 256*8 = 2048 output rows (q_lora) -> 256 8row calls */
#define REPS 4

static uint8_t W[8*ROWS][COLS];
static float   S[8*ROWS][NG];
static float   X[COLS], DST[8*ROWS];

int main(void){
    G_FRQ=cntfrq(); printf("cntfrq=%llu Hz\n",(unsigned long long)G_FRQ);
    for(int r=0;r<8*ROWS;r++){ for(int c=0;c<COLS;c++) W[r][c]=(uint8_t)((r*7+c*3)&0xff); for(int g=0;g<NG;g++) S[r][g]=((g*5+r)%17-8)/64.f; }
    for(int c=0;c<COLS;c++) X[c]=((c*11)%97-48)/48.f;

    /* one full GEMV = ROWS calls to the 8-row kernel (2048x6144). */
    uint64_t best=~0ull, t0,t1;
    for(int rep=0;rep<REPS;rep++){
        t0=rdcyc();
        for(int r=0;r<ROWS;r++){
            int b=r*8;
            glm5_matvec_int8_8row(DST+b,
                W[b],W[b+1],W[b+2],W[b+3],W[b+4],W[b+5],W[b+6],W[b+7],
                S[b],S[b+1],S[b+2],S[b+3],S[b+4],S[b+5],S[b+6],S[b+7],
                GS,X,COLS);
        }
        t1=rdcyc(); if(t1-t0<best)best=t1-t0;
    }
    float ref0=DST[0];  /* baseline result to compare u2 against */
    uint64_t best2=~0ull;
    for(int rep=0;rep<REPS;rep++){
        t0=rdcyc();
        for(int r=0;r<ROWS;r++){ int b=r*8;
            matvec_int8_8row_u2(DST+b,
                W[b],W[b+1],W[b+2],W[b+3],W[b+4],W[b+5],W[b+6],W[b+7],
                S[b],S[b+1],S[b+2],S[b+3],S[b+4],S[b+5],S[b+6],S[b+7],GS,X,COLS);
        }
        t1=rdcyc(); if(t1-t0<best2)best2=t1-t0;
    }
    float ref=glm5_dot_int8_row(W[0],S[0],GS,X,COLS);
    double relerr=fabs(DST[0]-ref)/(fabs(ref)+1e-6);
    double u2_vs_base=fabs(DST[0]-ref0)/(fabs(ref0)+1e-6);
    printf("int8 GEMV 2048x%d gs=%d (per full GEMV):\n",COLS,GS);
    printf("  baseline 8row : %llu ns   %.1f GB/s\n",(unsigned long long)to_ns(best),(double)(8*ROWS)*COLS/to_ns(best));
    printf("  u2 (2x c-unroll): %llu ns   %.1f GB/s   speedup x1000=%llu\n",
           (unsigned long long)to_ns(best2),(double)(8*ROWS)*COLS/to_ns(best2),(unsigned long long)(best2?best*1000/best2:0));
    printf("  correctness: row0 rel_err(vs scalar)=%.2e  u2_vs_baseline=%.2e  ok=%d\n",
           relerr,u2_vs_base,relerr<1e-4&&u2_vs_base<1e-5);
    return 0;
}
