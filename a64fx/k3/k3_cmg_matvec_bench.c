#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <omp.h>
#include <arm_sve.h>
#include "ggml_dequant.h"
#define MPOL_BIND 2
#define MPOL_INTERLEAVE 3
static void *alloc_pol(int pol,unsigned long mask,size_t b){
    void*p=mmap(NULL,b,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED)return NULL;
    if(syscall(SYS_mbind,p,b,pol,&mask,64,0)!=0){perror("mbind");return NULL;}
    { uint16_t *h=(uint16_t*)p; size_t n=b/2;
      for(size_t i=0;i<n;i++) h[i]=(uint16_t)(0x3f80u ^ (i & 0x7fu)); }  /* ~1.0 bf16: never subnormal */
    return p;
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

/* one attention-sized projection: rows x cols bf16, 8-row tasks, 47 threads */
int main(int argc,char**argv){
    int cols=argc>1?atoi(argv[1]):7168, rows=argc>2?atoi(argv[2]):4224, NT=argc>3?atoi(argv[3]):47;
    rows=(rows/8/4)*8*4;                       /* divisible by 8 and by 4 CMGs */
    size_t nw=(size_t)rows*cols, bytes=nw*2;
    uint16_t *inter=alloc_pol(MPOL_INTERLEAVE,0xf0UL,bytes);
    uint16_t *part[4]; size_t rq=rows/4;        /* row-block per CMG */
    for(int i=0;i<4;i++) part[i]=alloc_pol(MPOL_BIND,1UL<<(4+i),rq*(size_t)cols*2);
    float *x=aligned_alloc(256,cols*4),*y=aligned_alloc(256,(size_t)rows*64*4);
    if(!inter||!part[3]||!x||!y){fprintf(stderr,"alloc failed\n");return 1;}
    for(int i=0;i<cols;i++)x[i]=1.0f/(i+1);
    double bi=1e9,bp=1e9;
    for(int rep=0;rep<5;rep++){
        double t=now();
        #pragma omp parallel num_threads(NT)
        { int id=omp_get_thread_num();cpu_set_t s;CPU_ZERO(&s);CPU_SET(12+id,&s);sched_setaffinity(0,sizeof s,&s);
          #pragma omp for schedule(static)
          for(int r=0;r<rows;r+=8){const uint16_t*p=inter+(size_t)r*cols;
            matvec_bf16_8row(y+(size_t)(r/8)*64,p,p+cols,p+2*cols,p+3*cols,p+4*cols,p+5*cols,p+6*cols,p+7*cols,x,cols);} }
        double d=now()-t; if(d<bi)bi=d;
        t=now();
        #pragma omp parallel num_threads(NT)
        { int id=omp_get_thread_num();cpu_set_t s;CPU_ZERO(&s);CPU_SET(12+id,&s);sched_setaffinity(0,sizeof s,&s);
          int cmg=id/12,lid=id%12,nl=(NT-cmg*12)<12?(NT-cmg*12):12;
          /* each thread touches ONLY its own CMG's row block */
          for(int r=lid*8;r<(int)rq;r+=nl*8){const uint16_t*p=part[cmg]+(size_t)r*cols;
            matvec_bf16_8row(y+(size_t)((cmg*rq+r)/8)*64,p,p+cols,p+2*cols,p+3*cols,p+4*cols,p+5*cols,p+6*cols,p+7*cols,x,cols);} }
        d=now()-t; if(d<bp)bp=d;
    }
    printf("cols=%d rows=%d threads=%d bytes=%.1f MB\n",cols,rows,NT,bytes/1e6);
    printf("  matvec, interleaved (k3 today) : %7.3f ms = %6.1f GB/s\n",bi*1e3,bytes/bi/1e9);
    printf("  matvec, CMG-local              : %7.3f ms = %6.1f GB/s  -> %.2fx\n",bp*1e3,bytes/bp/1e9,bi/bp);
    return 0;
}
