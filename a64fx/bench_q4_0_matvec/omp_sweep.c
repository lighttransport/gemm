/* omp_sweep.c - multi-core int8 SDOT scaling on A64FX.
 *
 *   [A] COMPUTE  : register-resident SDOT (16-way ILP, no memory) per thread.
 *                  If GEMM is cache-blocked + compute-bound, this is the
 *                  aggregate ceiling -> should reach 90%+ of nthreads*512.
 *   [B] STREAM   : each thread SDOT-reduces its OWN 96 MB buffer (first-touch
 *                  local). Aggregate read BW; saturates per CMG (~256 GB/s),
 *                  which is why matvec/HBM-streamed GEMM cannot scale to peak.
 *
 * Peak/core @2.0 GHz = 512 GIOPS. Aggregate peak = nthreads * 512.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *           -D_GNU_SOURCE omp_sweep.c -lm -o omp_sweep
 * Run:   OMP_PROC_BIND=close OMP_PLACES=cores ./omp_sweep
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <arm_sve.h>

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
#define OPS_PER_SDOT 128.0

#define DECL16 svint32_t \
  c0=svdup_s32(0),c1=svdup_s32(0),c2=svdup_s32(0),c3=svdup_s32(0), \
  c4=svdup_s32(0),c5=svdup_s32(0),c6=svdup_s32(0),c7=svdup_s32(0), \
  c8=svdup_s32(0),c9=svdup_s32(0),cA=svdup_s32(0),cB=svdup_s32(0), \
  cC=svdup_s32(0),cD=svdup_s32(0),cE=svdup_s32(0),cF=svdup_s32(0)
#define RED16 (svaddv_s32(svptrue_b32(), svadd_s32_x(svptrue_b32(), \
  svadd_s32_x(svptrue_b32(),svadd_s32_x(svptrue_b32(),svadd_s32_x(svptrue_b32(),c0,c1),svadd_s32_x(svptrue_b32(),c2,c3)), \
                 svadd_s32_x(svptrue_b32(),svadd_s32_x(svptrue_b32(),c4,c5),svadd_s32_x(svptrue_b32(),c6,c7))), \
  svadd_s32_x(svptrue_b32(),svadd_s32_x(svptrue_b32(),svadd_s32_x(svptrue_b32(),c8,c9),svadd_s32_x(svptrue_b32(),cA,cB)), \
                 svadd_s32_x(svptrue_b32(),svadd_s32_x(svptrue_b32(),cC,cD),svadd_s32_x(svptrue_b32(),cE,cF))))))

static long compute_thread(uint64_t iters){
    svint8_t a=svdup_s8(3), b=svdup_s8(5); DECL16;
    for(uint64_t i=0;i<iters;i++){
        c0=svdot_s32(c0,a,b); c1=svdot_s32(c1,a,b); c2=svdot_s32(c2,a,b); c3=svdot_s32(c3,a,b);
        c4=svdot_s32(c4,a,b); c5=svdot_s32(c5,a,b); c6=svdot_s32(c6,a,b); c7=svdot_s32(c7,a,b);
        c8=svdot_s32(c8,a,b); c9=svdot_s32(c9,a,b); cA=svdot_s32(cA,a,b); cB=svdot_s32(cB,a,b);
        cC=svdot_s32(cC,a,b); cD=svdot_s32(cD,a,b); cE=svdot_s32(cE,a,b); cF=svdot_s32(cF,a,b);
    }
    return RED16;
}

static long stream_thread(const int8_t*buf,size_t blocks,int reps){
    svint8_t x=svdup_s8(7); DECL16;
    size_t nb=blocks&~(size_t)15;
    for(int r=0;r<reps;r++){ const int8_t*p=buf;
        for(size_t b=0;b<nb;b+=16,p+=16*64){
            c0=svdot_s32(c0,svld1_s8(svptrue_b8(),p+ 0*64),x); c1=svdot_s32(c1,svld1_s8(svptrue_b8(),p+ 1*64),x);
            c2=svdot_s32(c2,svld1_s8(svptrue_b8(),p+ 2*64),x); c3=svdot_s32(c3,svld1_s8(svptrue_b8(),p+ 3*64),x);
            c4=svdot_s32(c4,svld1_s8(svptrue_b8(),p+ 4*64),x); c5=svdot_s32(c5,svld1_s8(svptrue_b8(),p+ 5*64),x);
            c6=svdot_s32(c6,svld1_s8(svptrue_b8(),p+ 6*64),x); c7=svdot_s32(c7,svld1_s8(svptrue_b8(),p+ 7*64),x);
            c8=svdot_s32(c8,svld1_s8(svptrue_b8(),p+ 8*64),x); c9=svdot_s32(c9,svld1_s8(svptrue_b8(),p+ 9*64),x);
            cA=svdot_s32(cA,svld1_s8(svptrue_b8(),p+10*64),x); cB=svdot_s32(cB,svld1_s8(svptrue_b8(),p+11*64),x);
            cC=svdot_s32(cC,svld1_s8(svptrue_b8(),p+12*64),x); cD=svdot_s32(cD,svld1_s8(svptrue_b8(),p+13*64),x);
            cE=svdot_s32(cE,svld1_s8(svptrue_b8(),p+14*64),x); cF=svdot_s32(cF,svld1_s8(svptrue_b8(),p+15*64),x);
        } }
    return RED16;
}

int main(int argc,char**argv){
    double freq=(double)rdfreq();
    int maxt = argc>1?atoi(argv[1]):48;
    int threads[]={1,2,4,8,12,24,48};
    size_t PER=96ULL*1024*1024;                 /* per-thread stream buffer */
    const uint64_t CITERS=120ULL*1000*1000;     /* compute iters/thread */

    printf("A64FX int8 SDOT multi-core scaling (peak 512 GIOPS/core @2.0GHz)\n");
    printf("1 CMG = 12 cores; full node = 48 cores (4 CMG).\n\n");

    printf("[A] COMPUTE (register-resident, cache-blocked-GEMM ceiling)\n");
    printf("    %-4s %10s %12s %10s\n","thr","GIOPS","peak(thr*512)","%peak");
    volatile long sink=0;
    for(int i=0;i<7;i++){ int nt=threads[i]; if(nt>maxt) break;
        uint64_t t0=rdcyc();
        #pragma omp parallel num_threads(nt) reduction(+:sink)
        { sink+=compute_thread(CITERS); }
        uint64_t t1=rdcyc(); double sec=(double)(t1-t0)/freq;
        double gi=(double)nt*CITERS*16.0*OPS_PER_SDOT/sec/1e9;
        printf("    %-4d %10.1f %12d %9.1f%%\n",nt,gi,nt*512,gi/(nt*512.0)*100);
    }

    printf("\n[B] STREAM (each thread reduces own 96 MB, intensity 2 op/B)\n");
    printf("    %-4s %10s %10s %12s\n","thr","GB/s","GIOPS","GB/s/thr");
    for(int i=0;i<7;i++){ int nt=threads[i]; if(nt>maxt) break;
        /* per-thread buffers, first-touch local */
        int8_t **bufs=(int8_t**)calloc(nt,sizeof(int8_t*));
        #pragma omp parallel num_threads(nt)
        {
            int tid=omp_get_thread_num();
            int8_t*b=(int8_t*)aligned_alloc(256,PER);
            for(size_t j=0;j<PER;j++) b[j]=(int8_t)((j+tid)&0x3f);  /* first-touch */
            bufs[tid]=b;
        }
        int reps=3;
        uint64_t t0=rdcyc();
        #pragma omp parallel num_threads(nt) reduction(+:sink)
        { sink+=stream_thread(bufs[omp_get_thread_num()], PER/64, reps); }
        uint64_t t1=rdcyc(); double sec=(double)(t1-t0)/freq;
        double bytes=(double)PER*nt*reps;
        double gbps=bytes/sec/1e9;
        printf("    %-4d %10.1f %10.1f %12.1f\n",nt,gbps,2*gbps,gbps/nt);
        for(int t=0;t<nt;t++) free(bufs[t]); free(bufs);
    }
    (void)sink;
    return 0;
}
