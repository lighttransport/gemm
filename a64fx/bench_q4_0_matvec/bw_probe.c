/* bw_probe.c - true single-core & node HBM read bandwidth on A64FX.
 * Pure streaming loads (non-temporal svldnt1) with minimal compute, so the
 * result is the memory ceiling, not an issue/compute limit. Per-thread local
 * buffers (first-touch) so each thread hits its own CMG's HBM.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *           -D_GNU_SOURCE bw_probe.c -lm -o bw_probe
 * Run:   OMP_PROC_BIND=spread OMP_PLACES=cores ./bw_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <arm_sve.h>

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }

/* Sum a buffer with non-temporal loads, 8 independent accumulators. */
static long stream_read(const int8_t*buf,size_t bytes,int reps){
    svint8_t a0=svdup_s8(0),a1=svdup_s8(0),a2=svdup_s8(0),a3=svdup_s8(0);
    svint8_t a4=svdup_s8(0),a5=svdup_s8(0),a6=svdup_s8(0),a7=svdup_s8(0);
    svbool_t pg=svptrue_b8();
    size_t nb=(bytes/64)&~(size_t)7;
    for(int r=0;r<reps;r++){ const int8_t*p=buf;
        for(size_t b=0;b<nb;b+=8,p+=8*64){
            a0=svadd_s8_x(pg,a0,svldnt1_s8(pg,p+0*64)); a1=svadd_s8_x(pg,a1,svldnt1_s8(pg,p+1*64));
            a2=svadd_s8_x(pg,a2,svldnt1_s8(pg,p+2*64)); a3=svadd_s8_x(pg,a3,svldnt1_s8(pg,p+3*64));
            a4=svadd_s8_x(pg,a4,svldnt1_s8(pg,p+4*64)); a5=svadd_s8_x(pg,a5,svldnt1_s8(pg,p+5*64));
            a6=svadd_s8_x(pg,a6,svldnt1_s8(pg,p+6*64)); a7=svadd_s8_x(pg,a7,svldnt1_s8(pg,p+7*64));
        } }
    a0=svadd_s8_x(pg,svadd_s8_x(pg,svadd_s8_x(pg,a0,a1),svadd_s8_x(pg,a2,a3)),
                     svadd_s8_x(pg,svadd_s8_x(pg,a4,a5),svadd_s8_x(pg,a6,a7)));
    return (long)svaddv_s8(pg,a0);
}

int main(int argc,char**argv){
    double freq=(double)rdfreq();
    int maxt=argc>1?atoi(argv[1]):48;
    int threads[]={1,2,4,8,12,24,48};
    size_t PER=128ULL*1024*1024;   /* per-thread, >> L2 */
    int reps=4;
    volatile long sink=0;
    printf("A64FX HBM read BW probe (non-temporal streaming loads)\n");
    printf("per-thread buffer=%zu MB, reps=%d\n\n", PER>>20, reps);
    printf("  %-4s %10s %12s %10s\n","thr","GB/s","GB/s/thr","GB/s/CMG");
    for(int i=0;i<7;i++){ int nt=threads[i]; if(nt>maxt) break;
        int8_t**bufs=(int8_t**)calloc(nt,sizeof(int8_t*));
        #pragma omp parallel num_threads(nt)
        { int tid=omp_get_thread_num();
          int8_t*b=(int8_t*)aligned_alloc(256,PER);
          for(size_t j=0;j<PER;j++) b[j]=(int8_t)((j+tid)&0x7f);
          bufs[tid]=b; }
        uint64_t t0=rdcyc();
        #pragma omp parallel num_threads(nt) reduction(+:sink)
        { sink+=stream_read(bufs[omp_get_thread_num()],PER,reps); }
        uint64_t t1=rdcyc(); double sec=(double)(t1-t0)/freq;
        double gbps=(double)PER*nt*reps/sec/1e9;
        int cmgs = nt<=12?1:(nt+11)/12;
        printf("  %-4d %10.1f %12.1f %10.1f\n",nt,gbps,gbps/nt,gbps/cmgs);
        for(int t=0;t<nt;t++) free(bufs[t]); free(bufs);
    }
    (void)sink;
    return 0;
}
