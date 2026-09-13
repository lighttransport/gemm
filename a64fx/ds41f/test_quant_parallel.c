#define _POSIX_C_SOURCE 200809L
#include "ds41f_int8.h"
#include "ds41f_kernels.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void)
{
    size_t sizes[]={32,512,1056,1280,2304,4096,5120,8192,32768};
    for(size_t t=0;t<9;++t){size_t n=sizes[t];float *x=malloc(n*4),*a=malloc((n+1)*4),*b=malloc((n+1)*4);if(!x||!a||!b)return 1;
        for(size_t i=0;i<n;++i)x[i]=sinf(i*.13f)*37;
        ds41f_int8_input ia,ib;ds41f_set_quant_parallel(0);a[n]=b[n]=12345;
        if(ds41f_act_quant(a,x,n)||ds41f_int8_prepare_input(&ia,a,n,32))return 1;
        ds41f_set_quant_parallel(1);if(ds41f_act_quant(b,x,n)||ds41f_int8_prepare_input(&ib,b,n,32))return 1;
        if(memcmp(a,b,(n+1)*4)||memcmp(ia.data,ib.data,n*2)||memcmp(ia.scale,ib.scale,n/32*4))return 1;
        memcpy(b,x,n*4);if(ds41f_act_quant(b,b,n)||memcmp(a,b,(n+1)*4))return 1;
        ds41f_int8_input_free(&ia);ds41f_int8_input_free(&ib);
        for(int mode=0;mode<2;++mode){ds41f_set_quant_parallel(mode);double start=now();for(int it=0;it<100;++it){if(ds41f_act_quant(a,x,n)||ds41f_int8_prepare_input(&ia,a,n,32))return 1;ds41f_int8_input_free(&ia);}
            printf("QUANT_PARALLEL mode=%d n=%zu fp8_plus_int8_us=%.3f\n",mode,n,(now()-start)*1e4);}
        x[n-1]=NAN;ds41f_set_quant_parallel(1);if(ds41f_act_quant(b,x,n)!=EDOM)return 1;
        free(x);free(a);free(b);
    }
    puts("QUANT_PARALLEL PASS sizes=9 bit_exact in_place canary nonfinite");return 0;
}
