/* fp16_profile.c - standalone single-tile microkernel profiler for the fp16 12x2
 * GEMM kernel (micro_kernel_fp16_12x2_swp). NO model deps - pure synthetic data.
 * Single-threaded, L1-resident rep loop for qlair / fapp.
 *
 *   C[12 rows][64 cols](fp32) = A[K][12](fp16) x B[K][64](fp16)
 *
 * This is the kernel used by WS3 fused attention (QK^T, P*V). The fp32-shadow
 * driver (init + accum over Kc blocks) is what the full GEMM/attention uses; here
 * we profile a single swp call over K to isolate the microkernel.
 *
 * Build (native):  make fp16_profile     Run: ./fp16_profile [K=512] [reps=2000]
 * Build (fapp):    make fp16_profile_fj
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef USE_FAPP
#include "fj_tool/fapp.h"
#endif

void micro_kernel_fp16_12x2_swp(const _Float16*A,const _Float16*B,float*C,int64_t K,int64_t u,int64_t ldc);

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
#define MR 12
#define NR 64

int main(int argc,char**argv){
    int K    = argc>1?atoi(argv[1]):512;     /* contraction; must be even */
    long reps= argc>2?atol(argv[2]):2000;
    double cpu_ghz=2.0, peak=256.0;          /* fp16 FMA GFLOPS/core */
    if(K&1)K++;

    _Float16*A=(_Float16*)aligned_alloc(256,(size_t)K*MR*sizeof(_Float16)); /* [K][12] */
    _Float16*B=(_Float16*)aligned_alloc(256,(size_t)K*NR*sizeof(_Float16)); /* [K][64] */
    float    *C=(float*)aligned_alloc(256,(size_t)MR*NR*sizeof(float));
    for(size_t i=0;i<(size_t)K*MR;i++) A[i]=(_Float16)(0.03f*(float)((int)(i%31)-15));
    for(size_t i=0;i<(size_t)K*NR;i++) B[i]=(_Float16)(0.025f*(float)((int)(i%29)-14));

    /* correctness vs fp32 ref (cols 0..3 of rows 0..2) */
    micro_kernel_fp16_12x2_swp(A,B,C,K,0,(int64_t)NR*4);
    double maxrel=0;
    for(int r=0;r<3;r++) for(int c=0;c<4;c++){
        double ref=0; for(int k=0;k<K;k++) ref+=(double)(float)A[(size_t)k*MR+r]*(double)(float)B[(size_t)k*NR+c];
        double e=fabs((double)C[r*NR+c]-ref)/(fabs(ref)+1e-9); if(e>maxrel)maxrel=e;
    }

    volatile double sink=0; double freq=(double)rdfreq();
#ifdef USE_FAPP
    fapp_start("fp16_12x2_swp",1,0);
#endif
    uint64_t c0=rdcyc();
    for(long r=0;r<reps;r++){ micro_kernel_fp16_12x2_swp(A,B,C,K,0,(int64_t)NR*4); sink+=C[(r*7)%(MR*NR)]; }
    uint64_t c1=rdcyc();
#ifdef USE_FAPP
    fapp_stop("fp16_12x2_swp",1,0);
#endif
    (void)sink;
    double sec=(double)(c1-c0)/freq;
    double per_call_cyc=(double)(c1-c0)/reps*(cpu_ghz*1e9/freq);
    double gf=2.0*(double)MR*NR*K*reps/sec/1e9;
    printf("fp16_12x2_swp  K=%d reps=%ld  | %.1f cyc/call  %.1f GFLOPS/core  %.0f%% fp16-peak  | maxrel=%.2e\n",
           K,reps, per_call_cyc, gf, gf/peak*100, maxrel);
    free(A);free(B);free(C);
    return 0;
}
