#define _POSIX_C_SOURCE 200112L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern uint64_t fp4_ex_dequant_i16(const uint8_t*,int16_t*,size_t,size_t);
extern uint64_t fp4_ex_mul_i16(const uint8_t*,const uint8_t*,int16_t*,size_t,size_t);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void*aligned_alloc256(size_t n){void*p=0;return posix_memalign(&p,256,n)?0:p;}
static int dec(unsigned q){static const int v[8]={0,1,2,3,4,6,8,12};return(q&8)?-v[q&7]:v[q&7];}

int main(int argc,char**argv){
    size_t bytes=argc>1?strtoull(argv[1],0,0):32768;
    size_t reps=argc>2?strtoull(argv[2],0,0):4096;
    uint8_t*a=aligned_alloc256(bytes),*b=aligned_alloc256(bytes);
    int16_t*out=aligned_alloc256(bytes*2*sizeof(*out));
    if(!a||!b||!out)return 1;
    for(size_t i=0;i<bytes;++i){a[i]=(uint8_t)(i*37+11);b[i]=(uint8_t)(i*73+5);}
    uint64_t sum=fp4_ex_dequant_i16(a,out,bytes,1);
    for(size_t i=0;i<bytes;++i)if(out[2*i]!=dec(a[i]&15)||out[2*i+1]!=dec(a[i]>>4))return 2;
    double t=now();
    sum+=fp4_ex_dequant_i16(a,out,bytes,reps);double dd=now()-t;
    sum+=fp4_ex_mul_i16(a,b,out,bytes,1);
    for(size_t i=0;i<bytes;++i)if(out[2*i]!=dec(a[i]&15)*dec(b[i]&15)||out[2*i+1]!=dec(a[i]>>4)*dec(b[i]>>4))return 3;
    t=now();
    sum+=fp4_ex_mul_i16(a,b,out,bytes,reps);double dm=now()-t;
    double values=2.0*bytes*reps;
    printf("scalar EX-only FP4 bytes=%zu reps=%zu values=%.0f checksum=%llu\n",bytes,reps,values,(unsigned long long)sum);
    printf("dequant ns/value=%.3f Gvalue/s=%.3f\n",dd*1e9/values,values/dd/1e9);
    printf("mul     ns/value=%.3f Gproduct/s=%.3f effective-gflop/s=%.3f\n",dm*1e9/values,values/dm/1e9,2.0*values/dm/1e9);
    free(a);free(b);free(out);return 0;
}
