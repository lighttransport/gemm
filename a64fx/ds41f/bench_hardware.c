#define _POSIX_C_SOURCE 200809L
#include <arm_sve.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char **argv)
{
    size_t distance=argc>1?strtoul(argv[1],NULL,10):0;
    size_t mib=argc>2?strtoul(argv[2],NULL,10):512;
    if(!mib||mib>4096)return 2;
    const size_t n=mib*1024*1024/sizeof(float);float *x;
    if(posix_memalign((void **)&x,256,n*sizeof(float)))return 1;
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;++i)x[i]=1;
    double best=1e9,checksum=0;
    for(int repeat=0;repeat<10;++repeat){double sum=0,start=now();
        #pragma omp parallel reduction(+:sum)
        {
            size_t begin=n*(size_t)omp_get_thread_num()/omp_get_num_threads();
            size_t end=n*(size_t)(omp_get_thread_num()+1)/omp_get_num_threads();
            svbool_t pg=svptrue_b32();svfloat32_t a=svdup_f32(0),b=a,c=a,d=a,e=a,f=a,g=a,h=a;
            size_t vl=svcntw(),i=begin;
            for(;i+8*vl<=end;i+=8*vl){
                if(distance&&i+distance+8*vl<=end){
                    __builtin_prefetch(x+i+distance,0,0);__builtin_prefetch(x+i+distance+vl,0,0);
                    __builtin_prefetch(x+i+distance+2*vl,0,0);__builtin_prefetch(x+i+distance+3*vl,0,0);}
                a=svadd_x(pg,a,svld1(pg,x+i));b=svadd_x(pg,b,svld1(pg,x+i+vl));
                c=svadd_x(pg,c,svld1(pg,x+i+2*vl));d=svadd_x(pg,d,svld1(pg,x+i+3*vl));
                e=svadd_x(pg,e,svld1(pg,x+i+4*vl));f=svadd_x(pg,f,svld1(pg,x+i+5*vl));
                g=svadd_x(pg,g,svld1(pg,x+i+6*vl));h=svadd_x(pg,h,svld1(pg,x+i+7*vl));}
            sum+=svaddv(pg,svadd_x(pg,svadd_x(pg,a,b),svadd_x(pg,c,d)));
            sum+=svaddv(pg,svadd_x(pg,svadd_x(pg,e,f),svadd_x(pg,g,h)));
            for(;i<end;++i)sum+=x[i];
        }
        double elapsed=now()-start;if(sum!=(double)n)return 1;
        if(elapsed<best)best=elapsed;checksum=sum;
    }
    printf("HBM_READ bytes=%zu prefetch_floats=%zu best_ms=%.3f GBs=%.3f nominal_1024_fraction=%.4f checksum=%.0f\n",n*4,distance,best*1e3,n*4/best/1e9,n*4/best/1e9/1024,checksum);fflush(stdout);
    free(x);
    const int iterations=1000000;double start=now(),result=0;
    #pragma omp parallel reduction(+:result)
    {
        svbool_t pg=svptrue_b32();svfloat32_t a=svdup_f32(1.000001f),b=svdup_f32(.000001f);
        svfloat32_t c0=svdup_f32(0),c1=svdup_f32(1),c2=svdup_f32(2),c3=svdup_f32(3);
        svfloat32_t c4=svdup_f32(4),c5=svdup_f32(5),c6=svdup_f32(6),c7=svdup_f32(7);
        svfloat32_t c8=svdup_f32(8),c9=svdup_f32(9),c10=svdup_f32(10),c11=svdup_f32(11);
        svfloat32_t c12=svdup_f32(12),c13=svdup_f32(13),c14=svdup_f32(14),c15=svdup_f32(15);
        svfloat32_t c16=svdup_f32(16),c17=svdup_f32(17),c18=svdup_f32(18),c19=svdup_f32(19);
        svfloat32_t c20=svdup_f32(20),c21=svdup_f32(21),c22=svdup_f32(22),c23=svdup_f32(23);
        for(int i=0;i<iterations;++i){
            c0=svmad_x(pg,c0,a,b);c1=svmad_x(pg,c1,a,b);c2=svmad_x(pg,c2,a,b);c3=svmad_x(pg,c3,a,b);
            c4=svmad_x(pg,c4,a,b);c5=svmad_x(pg,c5,a,b);c6=svmad_x(pg,c6,a,b);c7=svmad_x(pg,c7,a,b);
            c8=svmad_x(pg,c8,a,b);c9=svmad_x(pg,c9,a,b);c10=svmad_x(pg,c10,a,b);c11=svmad_x(pg,c11,a,b);
            c12=svmad_x(pg,c12,a,b);c13=svmad_x(pg,c13,a,b);c14=svmad_x(pg,c14,a,b);c15=svmad_x(pg,c15,a,b);
            c16=svmad_x(pg,c16,a,b);c17=svmad_x(pg,c17,a,b);c18=svmad_x(pg,c18,a,b);c19=svmad_x(pg,c19,a,b);
            c20=svmad_x(pg,c20,a,b);c21=svmad_x(pg,c21,a,b);c22=svmad_x(pg,c22,a,b);c23=svmad_x(pg,c23,a,b);
        }
        result+=svaddv(pg,c0)+svaddv(pg,c1)+svaddv(pg,c2)+svaddv(pg,c3)+svaddv(pg,c4)+svaddv(pg,c5)+svaddv(pg,c6)+svaddv(pg,c7);
        result+=svaddv(pg,c8)+svaddv(pg,c9)+svaddv(pg,c10)+svaddv(pg,c11)+svaddv(pg,c12)+svaddv(pg,c13)+svaddv(pg,c14)+svaddv(pg,c15);
        result+=svaddv(pg,c16)+svaddv(pg,c17)+svaddv(pg,c18)+svaddv(pg,c19)+svaddv(pg,c20)+svaddv(pg,c21)+svaddv(pg,c22)+svaddv(pg,c23);
    }
    double elapsed=now()-start,flops=(double)iterations*24*svcntw()*2*omp_get_max_threads();
    if(!isfinite(result))return 1;
    printf("FMA_CONTROL GFLOPs=%.3f nominal_6144_fraction=%.4f checksum=%g threads=%d\n",flops/elapsed/1e9,flops/elapsed/1e9/6144,result,omp_get_max_threads());
    return 0;
}
