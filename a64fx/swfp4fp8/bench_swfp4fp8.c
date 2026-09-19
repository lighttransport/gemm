#define _GNU_SOURCE
#include "swfp4fp8.h"

#include <arm_sve.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

typedef struct { size_t n, k; const char *name; } shape_t;

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
}

static double median(double *v, int n) {
    qsort(v,(size_t)n,sizeof(*v),cmp_double);
    return v[n/2];
}

static void pin_thread(int tid) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(12+tid,&set);
    (void)pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
}

static double raw_read_ceiling(int threads, size_t bytes) {
    uint64_t *p=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED) return 0;
    size_t words=bytes/8;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(size_t i=0;i<words;++i){pin_thread(omp_get_thread_num());p[i]=mix32((uint32_t)i);}
    double times[7]; volatile uint64_t sink=0;
    for(int r=0;r<8;++r){
        uint64_t sum=0; double t0=now_sec();
        #pragma omp parallel num_threads(threads) reduction(+:sum)
        {
            int tid=omp_get_thread_num(),nt=omp_get_num_threads(); pin_thread(tid);
            size_t lo=words*(size_t)tid/(size_t)nt,hi=words*(size_t)(tid+1)/(size_t)nt;
            svbool_t pg=svptrue_b64();
            svuint64_t s0=svdup_u64(0),s1=svdup_u64(0),s2=svdup_u64(0),s3=svdup_u64(0);
            svuint64_t s4=svdup_u64(0),s5=svdup_u64(0),s6=svdup_u64(0),s7=svdup_u64(0);
            size_t vl=svcntd(),i=lo;
            for(;i+8*vl<=hi;i+=8*vl){
                s0=svadd_x(pg,s0,svld1(pg,p+i+0*vl));s1=svadd_x(pg,s1,svld1(pg,p+i+1*vl));
                s2=svadd_x(pg,s2,svld1(pg,p+i+2*vl));s3=svadd_x(pg,s3,svld1(pg,p+i+3*vl));
                s4=svadd_x(pg,s4,svld1(pg,p+i+4*vl));s5=svadd_x(pg,s5,svld1(pg,p+i+5*vl));
                s6=svadd_x(pg,s6,svld1(pg,p+i+6*vl));s7=svadd_x(pg,s7,svld1(pg,p+i+7*vl));
            }
            sum+=svaddv(pg,s0)+svaddv(pg,s1)+svaddv(pg,s2)+svaddv(pg,s3)+
                 svaddv(pg,s4)+svaddv(pg,s5)+svaddv(pg,s6)+svaddv(pg,s7);
            for(;i<hi;++i)sum+=p[i];
        }
        double dt=now_sec()-t0; sink=sum;
        if(r) times[r-1]=dt;
    }
    if(!sink) fprintf(stderr,"raw checksum unexpectedly zero\n");
    munmap(p,bytes); return (double)bytes/median(times,7)/1e9;
}

static int make_matrix(swfp4fp8_context *ctx, swfp4fp8_matrix **out,
                       swfp4fp8_format f, size_t n, size_t k) {
    int rc=0;
    if(f==SWFP4FP8_NVFP4_G16 || f==SWFP4FP8_MXFP4_G32){
        size_t cb=n*k/2,ng=k/(f==SWFP4FP8_NVFP4_G16?16:32);
        uint8_t *c=malloc(cb),*s=malloc(n*ng);
        if(!c||!s){free(c);free(s);return ENOMEM;}
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<cb;++i)c[i]=(uint8_t)mix32((uint32_t)i);
        memset(s,f==SWFP4FP8_NVFP4_G16?0x38:127,n*ng);
        if(f==SWFP4FP8_NVFP4_G16) rc=swfp4fp8_pack_nvfp4(ctx,out,n,k,c,s,0.125f,SWFP4FP8_LAYOUT_CANONICAL);
        else rc=swfp4fp8_pack_mxfp4(ctx,out,n,k,c,s);
        free(c);free(s);
    }else{
        size_t cb=n*k; uint8_t *c=malloc(cb);
        if(!c)return ENOMEM;
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<cb;++i){uint8_t q=(uint8_t)mix32((uint32_t)i);c[i]=((q&0x7f)==0x7f)?0x38:q;}
        if(f==SWFP4FP8_QPN8_TILE32){
            float *s=malloc((n/32)*sizeof(*s));if(!s){free(c);return ENOMEM;}
            for(size_t i=0;i<n/32;++i)s[i]=16.f;
            rc=swfp4fp8_pack_qpn8(ctx,out,n,k,c,s,SWFP4FP8_LAYOUT_CANONICAL);free(s);
        }else{
            size_t ns=((n+127)/128)*(k/128);uint8_t*s=malloc(ns);if(!s){free(c);return ENOMEM;}
            memset(s,123,ns);rc=swfp4fp8_pack_fp8_block128(ctx,out,n,k,c,s);free(s);
        }
        free(c);
    }
    return rc;
}

static void bench_one(swfp4fp8_context *ctx, swfp4fp8_format f,
                      const shape_t *sh, size_t m, int threads,
                      double ceiling, int lossy) {
    swfp4fp8_matrix *w=NULL;
    int rc=make_matrix(ctx,&w,f,sh->n,sh->k);
    if(rc){fprintf(stderr,"pack %s failed: %s\n",swfp4fp8_format_name(f),strerror(rc));return;}
    float *a=NULL,*c=NULL;
    if(posix_memalign((void**)&a,256,m*sh->k*sizeof(*a)) ||
       posix_memalign((void**)&c,256,m*sh->n*sizeof(*c))){fprintf(stderr,"allocation failed\n");exit(2);}
    for(size_t i=0;i<m*sh->k;++i)a[i]=((int)(mix32((uint32_t)i)%1024)-512)/4096.f;
    swfp4fp8_kernel kernel=lossy==1?SWFP4FP8_KERNEL_FP8_FTZ:
                            lossy==2?SWFP4FP8_KERNEL_FP4_SDOT:
                            lossy==3?SWFP4FP8_KERNEL_ROW:SWFP4FP8_KERNEL_AUTO;
    swfp4fp8_gemm_f32(ctx,w,a,sh->k,c,sh->n,m,kernel);
    double t[7];
    for(int r=0;r<7;++r){double t0=now_sec();swfp4fp8_gemm_f32(ctx,w,a,sh->k,c,sh->n,m,kernel);t[r]=now_sec()-t0;}
    double sec=median(t,7),bytes=(double)swfp4fp8_matrix_bytes(w),gbps=bytes/sec/1e9;
    double gmac=(double)m*sh->n*sh->k/sec/1e9;
    volatile float checksum=c[(m-1)*sh->n+(sh->n-1)];
    printf("%-13s %-12s M=%-3zu T=%-2d %-8s %9.3f ms %8.2f GB/s %6.1f%%R %9.2f Gmac/s chk=%g\n",
           swfp4fp8_format_name(f),sh->name,m,threads,swfp4fp8_kernel_name(kernel),
           sec*1e3,gbps,ceiling?100.0*gbps/ceiling:0.0,gmac,(double)checksum);
    free(a);free(c);swfp4fp8_matrix_destroy(w);
}

static void usage(const char *p) {
    fprintf(stderr,"usage: %s [--quick|--full|--scaling] [--threads N]\n",p);
}

int main(int argc,char **argv) {
    int threads=48,full=0,scaling=0;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--quick"))full=0;
        else if(!strcmp(argv[i],"--full"))full=1;
        else if(!strcmp(argv[i],"--scaling")){full=0;scaling=1;}
        else if(!strcmp(argv[i],"--threads")&&i+1<argc)threads=atoi(argv[++i]);
        else{usage(argv[0]);return 2;}
    }
    if(threads<1||threads>48){usage(argv[0]);return 2;}
    swfp4fp8_context *ctx=NULL;int rc=swfp4fp8_context_create(&ctx,threads,NULL);
    if(rc){fprintf(stderr,"context: %s\n",strerror(rc));return 2;}
    size_t raw_bytes=384ull<<20;
    double ceiling=raw_read_ceiling(threads,raw_bytes);
    printf("# A64FX SVE=%zu bits threads=%d raw_read=%.2f GB/s pool=%zu MiB\n",svcntb()*8,threads,ceiling,raw_bytes>>20);
    printf("# effective bandwidth counts one resident compressed matrix per call\n");
    const shape_t quick[]={{5120,1536,"qkv"},{3584,5120,"expert"},{8192,4096,"a64fx-wide"}};
    const shape_t all[]={{5120,1536,"qkv"},{5120,4352,"proj"},{8704,5120,"gate"},{4096,5120,"down"},{2048,5120,"small"},{62080,5120,"lm-head"},{3584,5120,"expert"},{8192,4096,"a64fx-wide"},{32768,1024,"a64fx-qb"},{4096,8192,"a64fx-bigk"}};
    const shape_t *sh=full?all:quick;size_t nsh=scaling?1:(full?sizeof(all)/sizeof(all[0]):sizeof(quick)/sizeof(quick[0]));
    const size_t mf4_full[]={1,2,4,8,16},mf8_full[]={1,2,4,8,16,32,64,96};
    const size_t mf4_quick[]={1,8,16},mf8_quick[]={1,8,16,32};
    const size_t mscale[]={1};
    for(int fi=0;fi<4;++fi) for(size_t si=0;si<nsh;++si){
        const size_t *ms;size_t nm;
        if(scaling){ms=mscale;nm=1;}
        else if(fi==0||fi==2){ms=full?mf4_full:mf4_quick;nm=full?5:3;}else{ms=full?mf8_full:mf8_quick;nm=full?8:4;}
        for(size_t mi=0;mi<nm;++mi)bench_one(ctx,(swfp4fp8_format)fi,&sh[si],ms[mi],threads,ceiling,0);
        if(si==0)bench_one(ctx,(swfp4fp8_format)fi,&sh[si],1,threads,ceiling,3);
        if((fi==0||fi==2)&&si==0)bench_one(ctx,(swfp4fp8_format)fi,&sh[si],1,threads,ceiling,2);
        if((fi==1||fi==3)&&si==0)bench_one(ctx,(swfp4fp8_format)fi,&sh[si],1,threads,ceiling,1);
    }
    swfp4fp8_context_destroy(ctx);return 0;
}
