/* Per-CMG-local matvec BW ceiling check (A64FX), 12B BF16 matrix sizes, null data.
 *
 * Replicates the transformer.h pool EXACTLY: 48 pthreads pinned via cmg=tid%4,
 * core=12+cmg*12+tid/4. Each thread first-touches its contiguous row-block of a
 * shared BF16 weight matrix (-> pages LOCAL to that thread's CMG, matching the
 * matvec's row-split), then streams it. Goal: confirm per-CMG-local placement
 * reaches the node's ~734 GB/s streaming ceiling (vs the ~60-97 GB/s seen with the
 * model's concentrated first-touch).
 *
 * Modes (PLACE env): local (default, per-thread first-touch), zero (thread 0 touches
 * all -> concentrated), interleave (set_mempolicy MPOL_INTERLEAVE).
 * COMPUTE env: 0 = pure ld/accumulate (memory ceiling); 1 = bf16->fp32 widen + fma.
 *
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast matvec_local_bench.c -lpthread -lm -o matvec_local_bench
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <arm_sve.h>

static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static int NT, REPS, COMPUTE, PLACE; /* PLACE: 0=local 1=zero 2=interleave 3=separate */
static uint16_t *W; static float *Xv;
static uint16_t *Wsep[64];   /* per-thread separate buffers (PLACE=separate) */
static size_t n_rows, K;
static pthread_barrier_t bar;
static double t0g, t1g;
static volatile double g_sink[64];

static void pin(int tid){
    int cmg=tid%4, local=tid/4; if(local>=12) local%=12;
    int core=12+cmg*12+local;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core,&s); sched_setaffinity(0,sizeof(s),&s);
}

static void *worker(void *arg){
    int tid=(int)(long)arg;
    pin(tid);
    size_t rows_per=n_rows/NT;
    size_t r0=(size_t)tid*rows_per, r1=(tid==NT-1)?n_rows:r0+rows_per;
    size_t myrows=r1-r0;
    if (PLACE==3){
        /* separate per-thread buffer, first-touched by owner (bw_probe-style) */
        Wsep[tid]=(uint16_t*)malloc(myrows*K*2);
        for(size_t r=0;r<myrows;r++){ uint16_t *row=Wsep[tid]+r*K; for(size_t k=0;k<K;k+=512) row[k]=(uint16_t)(r+k); }
    } else if (PLACE==4){
        /* shared buffer, mbind each thread's row-block to its CMG node (MPOL_BIND) */
#if defined(SYS_mbind)
        size_t pg=2*1024*1024;
        uintptr_t a=(uintptr_t)(W+r0*K)&~(pg-1);
        uintptr_t e=((uintptr_t)(W+r1*K)+pg-1)&~(pg-1);
        unsigned long nm=1UL<<(tid%4);
        syscall(SYS_mbind,(void*)a,(size_t)(e-a),2/*MPOL_BIND*/,&nm,8UL,2/*MPOL_MF_MOVE*/);
#endif
        for(size_t r=r0;r<r1;r++){ uint16_t *row=W+r*K; for(size_t k=0;k<K;k+=512) row[k]=(uint16_t)(r+k); }
    } else if (PLACE!=1 || tid==0){
        size_t a0 = (PLACE==1)?0:r0, a1=(PLACE==1)?n_rows:r1;
        for(size_t r=a0;r<a1;r++){ uint16_t *row=W+r*K; for(size_t k=0;k<K;k+=512) row[k]=(uint16_t)(r+k); }
    }
    pthread_barrier_wait(&bar);
    if(tid==0) t0g=wall();
    pthread_barrier_wait(&bar);
    /* stream this thread's rows */
    svbool_t pth=svptrue_b16(), pt=svptrue_b32();
    int vlh=svcnth(), vl=svcntw();
    double sink=0;
    for(int rep=0;rep<REPS;rep++){
        svfloat32_t fa=svdup_f32(0); svuint32_t ia=svdup_u32(0);
        for(size_t rr=r0;rr<r1;rr++){ const uint16_t *row = (PLACE==3) ? (Wsep[tid]+(rr-r0)*K) : (W+rr*K);
            if(COMPUTE){
                svuint16_t zero=svdup_u16(0);
                for(size_t k=0;k+vlh<=K;k+=vlh){
                    svuint16_t raw=svldnt1_u16(pth,row+k);
                    fa=svmla_f32_x(pt,fa,svreinterpret_f32_u16(svzip1_u16(zero,raw)),svld1_f32(pt,Xv+k));
                    fa=svmla_f32_x(pt,fa,svreinterpret_f32_u16(svzip2_u16(zero,raw)),svld1_f32(pt,Xv+k+vl));
                }
            } else {
                for(size_t k=0;k+vlh<=K;k+=vlh) ia=svadd_u32_x(pt,ia,svreinterpret_u32_u16(svldnt1_u16(pth,row+k)));
            }
        }
        sink += COMPUTE ? svaddv_f32(pt,fa) : (double)svaddv_u32(pt,ia);
    }
    g_sink[tid]=sink;
    pthread_barrier_wait(&bar);
    if(tid==0) t1g=wall();
    return NULL;
}

int main(int argc,char**argv){
    n_rows = argc>1?(size_t)atoll(argv[1]):153600;
    K      = argc>2?(size_t)atoll(argv[2]):3840;
    REPS   = argc>3?atoi(argv[3]):20;
    NT     = getenv("NT")?atoi(getenv("NT")):48;
    COMPUTE= getenv("COMPUTE")?atoi(getenv("COMPUTE")):0;
    const char*pl=getenv("PLACE"); PLACE = (pl&&!strcmp(pl,"zero"))?1:(pl&&!strcmp(pl,"interleave"))?2:(pl&&!strcmp(pl,"separate"))?3:(pl&&!strcmp(pl,"mbind"))?4:0;
#if defined(SYS_set_mempolicy)
    if(PLACE==2){ unsigned long nm=0xFFUL; syscall(SYS_set_mempolicy,3,&nm,8UL); }
#endif
    size_t bytes=n_rows*K*2;
    if(PLACE!=3) W=(uint16_t*)malloc(bytes); Xv=(float*)malloc(K*4);
    for(size_t k=0;k<K;k++) Xv[k]=0.01f*(k%7);
    if(PLACE!=3&&!W){ fprintf(stderr,"alloc failed\n"); return 1; }
    pthread_barrier_init(&bar,NULL,NT);
    pthread_t th[64];
    for(int t=1;t<NT;t++) pthread_create(&th[t],NULL,worker,(void*)(long)t);
    worker((void*)0L);
    for(int t=1;t<NT;t++) pthread_join(th[t],NULL);
    double dt=t1g-t0g, gb=(double)bytes*REPS/dt/1e9;
    const char*pn=PLACE==1?"zero(concentrated)":PLACE==2?"interleave":PLACE==3?"separate(per-CMG buffers)":PLACE==4?"mbind(per-block BIND)":"local(shared,first-touch)";
    printf("n_rows=%zu K=%zu (%.2f GB) reps=%d NT=%d COMPUTE=%d PLACE=%s\n",n_rows,K,bytes/1e9,REPS,NT,COMPUTE,pn);
    printf("  %.1f ms  %.1f GB/s  (sink=%.0f)\n", dt*1e3, gb, g_sink[0]);
    return 0;
}
