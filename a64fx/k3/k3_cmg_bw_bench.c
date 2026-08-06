#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <omp.h>
#include <arm_sve.h>

#define MPOL_BIND 2
static void *alloc_on(int node, size_t bytes){
    void *p = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    unsigned long mask = 1UL << node;
    if (syscall(SYS_mbind, p, bytes, MPOL_BIND, &mask, 64, 0) != 0){
        perror("mbind"); return NULL; }
    memset(p, 1, bytes);                 /* fault in under the binding */
    return p;
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

static float stream_read(const uint16_t *w, size_t n16){
    svbool_t pg=svptrue_b16(); svuint16_t a0=svdup_u16(0),a1=a0,a2=a0,a3=a0;
    int vl=(int)svcnth(); size_t i=0;
    for(;i+4*(size_t)vl<=n16;i+=4*vl){
        a0=sveor_u16_x(pg,a0,svld1_u16(pg,&w[i]));
        a1=sveor_u16_x(pg,a1,svld1_u16(pg,&w[i+vl]));
        a2=sveor_u16_x(pg,a2,svld1_u16(pg,&w[i+2*vl]));
        a3=sveor_u16_x(pg,a3,svld1_u16(pg,&w[i+3*vl]));
    }
    a0=sveor_u16_x(pg,sveor_u16_x(pg,a0,a1),sveor_u16_x(pg,a2,a3));
    return (float)svaddv_u16(pg,a0);
}
static int cmg_first_cpu(int node){ return 12 + (node-4)*12; }

int main(int argc,char**argv){
    size_t MB = argc>1?(size_t)atoi(argv[1]):512;
    int nthr   = argc>2?atoi(argv[2]):1;
    size_t bytes = MB<<20, n16 = bytes/2;
    void *buf[8];
    for(int n=4;n<8;n++){ buf[n]=alloc_on(n,bytes); if(!buf[n]){fprintf(stderr,"alloc node%d failed\n",n);return 1;} }
    printf("buffer=%zu MB threads=%d  (rows = CPU CMG, cols = memory CMG)\n", MB, nthr);
    printf("        ");
    for(int m=4;m<8;m++) printf("  mem%d   ",m);
    printf("\n");
    float sink=0;
    for(int c=4;c<8;c++){
        printf("cpu%d :", c);
        for(int m=4;m<8;m++){
            double best=1e9;
            for(int rep=0;rep<3;rep++){
                double t=now();
                #pragma omp parallel num_threads(nthr) reduction(+:sink)
                {
                    int id=omp_get_thread_num();
                    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cmg_first_cpu(c)+(id%12), &s);
                    sched_setaffinity(0,sizeof s,&s);
                    size_t chunk=n16/nthr, off=(size_t)id*chunk;
                    sink += stream_read((const uint16_t*)buf[m]+off, chunk);
                }
                double d=now()-t; if(d<best)best=d;
            }
            printf(" %7.1f", bytes/best/1e9);
        }
        printf("   GB/s\n");
    }
    printf("(sink %g)\n", sink);
    return 0;
}
