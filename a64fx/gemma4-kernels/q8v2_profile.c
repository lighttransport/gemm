/* q8v2_profile.c - standalone single-tile microkernel profiler for the Q8v2
 * per-block int8 GEMM kernel (kernel_q8v2_3x4 / _arow). NO model, NO transformer.h,
 * NO gguf - pure synthetic data. Single-threaded, L1-resident, tight rep loop:
 * isolates the microkernel's compute efficiency for qlair / fapp.
 *
 *   C[3 tok][64 col] = sum_b da[b,r]*dw[b,c]*<qa[b,r,:], qw[b,c,:]>   (BLK=32)
 *
 * Build (native run):   make q8v2_profile
 * Build (fapp/qlair):   make q8v2_profile_fj      (Fujitsu -Nnoclang -KSVE + fapp markers)
 * Run:                  ./q8v2_profile [nb=16] [reps=2000] [arow=0]
 *   nb = number of 32-wide K blocks (K = 32*nb). nb=16 -> K=512 (B tile 32KB, L1-resident).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef USE_FAPP
#include "fj_tool/fapp.h"
#endif

void kernel_q8v2_3x4(const int8_t*aq,const float*ad,const int8_t*bq,const float*bd,long nb,float*C,long ldc);
void kernel_q8v2_3x4_arow(const int8_t*aq,const float*ad,const int8_t*bq,const float*bd,long nb,float*C,long ldc);

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }

#define MR 3
#define NR 64
#define BLK 32

int main(int argc,char**argv){
    int nb   = argc>1?atoi(argv[1]):16;     /* K = 32*nb */
    long reps= argc>2?atol(argv[2]):2000;
    int arow = argc>3?atoi(argv[3]):0;
    int K=nb*BLK;
    double cpu_ghz = 2.0;                    /* A64FX normal mode; only affects the GIOPS print */
    double peak = 512.0;                     /* int8 SDOT GIOPS/core (2 sdot/cy * 64 MAC * 2 op * 2GHz) */

    /* synthetic packed operands (centered nibbles in [-8,7] for weights, int8 acts) */
    int8_t*aq=(int8_t*)aligned_alloc(256,(size_t)nb*MR*BLK);
    float *ad=(float*)aligned_alloc(256,(size_t)nb*MR*sizeof(float));
    int8_t*bq=(int8_t*)aligned_alloc(256,(size_t)nb*8*4*64);
    float *bd=(float*)aligned_alloc(256,(size_t)nb*NR*sizeof(float));
    float *C =(float*)aligned_alloc(256,(size_t)MR*NR*sizeof(float));
    for(size_t i=0;i<(size_t)nb*MR*BLK;i++) aq[i]=(int8_t)((int)((i*131+7)%127)-63);
    for(size_t i=0;i<(size_t)nb*MR;i++)     ad[i]=0.002f+0.0001f*(i%17);
    for(size_t i=0;i<(size_t)nb*8*4*64;i++) bq[i]=(int8_t)(((i*97+13)%15)-8);     /* [-8,7] */
    for(size_t i=0;i<(size_t)nb*NR;i++)     bd[i]=0.01f+0.0005f*(i%13);
    if(arow) for(int b=0;b<nb;b++) for(int r=0;r<MR;r++) ad[b*MR+r]=ad[r]; /* per-row: constant across b */

    /* correctness vs naive (row0..2, col0..3) */
    kernel_q8v2_3x4(aq,ad,bq,bd,nb,C,(long)NR*4);
    double maxrel=0;
    for(int r=0;r<MR;r++) for(int c=0;c<4;c++){
        double ref=0;
        for(int b=0;b<nb;b++){ int vec=c/16,col=c%16; double dw=bd[b*NR+vec*16+col],da=ad[b*MR+r];
            for(int k=0;k<BLK;k++){ int g3=k/4,kk=k%4; int qw=bq[((size_t)b*8*4+(size_t)g3*4+vec)*64+col*4+kk];
                int qa=aq[((size_t)b*MR+r)*BLK+k]; ref+=da*dw*(double)qa*(double)qw; } }
        double got=C[r*NR+c]; double e=fabs(got-ref)/(fabs(ref)+1e-9); if(e>maxrel)maxrel=e;
    }

    /* timed rep loop (single thread, L1-resident) */
    void(*kern)(const int8_t*,const float*,const int8_t*,const float*,long,float*,long)=arow?kernel_q8v2_3x4_arow:kernel_q8v2_3x4;
    volatile double sink=0; double freq=(double)rdfreq();
#ifdef USE_FAPP
    fapp_start("q8v2_3x4",1,0);
#endif
    uint64_t c0=rdcyc();
    for(long r=0;r<reps;r++){ kern(aq,ad,bq,bd,nb,C,(long)NR*4); sink+=C[(r*7)%(MR*NR)]; }
    uint64_t c1=rdcyc();
#ifdef USE_FAPP
    fapp_stop("q8v2_3x4",1,0);
#endif
    (void)sink;
    double sec=(double)(c1-c0)/freq;
    double per_call_cyc=(double)(c1-c0)/reps * (cpu_ghz*1e9/freq);  /* counter->core cycles */
    double giops = 2.0*(double)MR*NR*K*reps/sec/1e9;
    printf("q8v2_3x4%s  nb=%d K=%d reps=%ld  | %.1f cyc/call  %.1f GIOPS/core  %.0f%% int8-peak  | maxrel=%.2e\n",
           arow?"_arow":"", nb,K,reps, per_call_cyc, giops, giops/peak*100, maxrel);
    free(aq);free(ad);free(bq);free(bd);free(C);
    return 0;
}
