#define _POSIX_C_SOURCE 200112L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

extern uint64_t fp4_ex_dequant_i16(const uint8_t*,int16_t*,size_t,size_t);
extern uint64_t fp4_ex_mul_i16(const uint8_t*,const uint8_t*,int16_t*,size_t,size_t);
extern uint64_t fp4_ex_unpack_u8(const uint8_t*,int8_t*,size_t,size_t);
extern uint64_t fp4_ex_unpack_e2m1_i8(const uint8_t*,int8_t*,size_t,size_t);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void*aligned_alloc256(size_t n){void*p=0;return posix_memalign(&p,256,n)?0:p;}
static int dec(unsigned q){static const int v[8]={0,1,2,3,4,6,8,12};return(q&8)?-v[q&7]:v[q&7];}

int main(int argc,char**argv){
    size_t bytes=argc>1?strtoull(argv[1],0,0):32768;
    size_t reps=argc>2?strtoull(argv[2],0,0):4096;
    uint8_t*a=aligned_alloc256(bytes),*b=aligned_alloc256(bytes);
    int16_t*out=aligned_alloc256(bytes*2*sizeof(*out));int8_t*out8=aligned_alloc256(bytes*2);
    if(!a||!b||!out||!out8||bytes%8)return 1;
    for(size_t i=0;i<bytes;++i){a[i]=(uint8_t)(i*37+11);b[i]=(uint8_t)(i*73+5);
      if((a[i]&15)==8)a[i]&=0xf0;if((a[i]>>4)==8)a[i]&=0x0f;}
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
    fp4_ex_unpack_u8(a,out8,bytes,1);for(size_t i=0;i<bytes;i+=8)for(int j=0;j<8;++j)
      if(out8[2*i+j]!=(a[i+j]&15)||out8[2*i+8+j]!=(a[i+j]>>4))return 4;
    fp4_ex_unpack_e2m1_i8(a,out8,bytes,1);for(size_t i=0;i<bytes;i+=8)for(int j=0;j<8;++j)
      if(out8[2*i+j]!=dec(a[i+j]&15)||out8[2*i+8+j]!=dec(a[i+j]>>4))return 5;
    t=now();sum+=fp4_ex_unpack_u8(a,out8,bytes,reps);double du=now()-t;
    t=now();sum+=fp4_ex_unpack_e2m1_i8(a,out8,bytes,reps);double de=now()-t;
    printf("swar nibble-only ns/value=%.3f Gvalue/s=%.3f packed_GB/s=%.3f expanded_GB/s=%.3f\n",
      du*1e9/values,values/du/1e9,bytes*reps/du/1e9,values/du/1e9);
    printf("swar exact-e2m1 ns/value=%.3f Gvalue/s=%.3f packed_GB/s=%.3f expanded_GB/s=%.3f\n",
      de*1e9/values,values/de/1e9,bytes*reps/de/1e9,values/de/1e9);
    if(argc>3){size_t stream=strtoull(argv[3],0,0);uint8_t*si=aligned_alloc256(stream);int8_t*so=aligned_alloc256(stream*2);
      if(!si||!so||stream%96)return 6;for(size_t i=0;i<stream;++i){si[i]=(uint8_t)(i*37+11);
        if((si[i]&15)==8)si[i]&=0xf0;if((si[i]>>4)==8)si[i]&=0x0f;}
      for(int exact=0;exact<2;++exact){double dt[5];for(int r=0;r<5;++r){t=now();
#pragma omp parallel num_threads(12)
        {int id=omp_get_thread_num();size_t z=stream/12,off=z*(size_t)id;
         if(exact)fp4_ex_unpack_e2m1_i8(si+off,so+2*off,z,1);
         else fp4_ex_unpack_u8(si+off,so+2*off,z,1);}
        dt[r]=now()-t;}for(int i=1;i<5;++i){double x=dt[i];int j=i;while(j&&dt[j-1]>x){dt[j]=dt[j-1];--j;}dt[j]=x;}
        double d=dt[2];printf("12c %s stream_MiB=%.1f ms=%.3f packed_GB/s=%.2f expanded_GB/s=%.2f FP4_ceiling_GF=%.2f\n",
          exact?"exact-e2m1":"nibble-only",stream/1048576.,d*1e3,stream/d/1e9,2.0*stream/d/1e9,4.0*stream/d/1e9);}
      free(si);free(so);}
    free(a);free(b);free(out);free(out8);return 0;
}
