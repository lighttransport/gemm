#define _POSIX_C_SOURCE 200809L
#include "ds41f_attention.h"
#include "ds41f_cache.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void)
{
    size_t sizes[]={0,1,7,8,9,127,512,1105};
    float q[4096],weights[32],key[128];uint8_t *rows=malloc(1105*356),mask[139];
    float *a=malloc(1106*4),*b=malloc(1106*4);if(!rows||!a||!b)return 1;
    for(size_t h=0;h<32;++h)weights[h]=(float)((int)(h%7)-3)*.0625f;
    for(size_t i=0;i<4096;++i)q[i]=sinf((float)i*.073f);
    for(size_t i=0;i<1105;++i){for(size_t j=0;j<128;++j)key[j]=cosf((float)(i+j)*.053f)*(i%11+1);
        if(ds41f_fp4_pack(rows+i*356+288,key,128,32,0))return 1;}
    for(size_t shape=0;shape<8;++shape)for(int pattern=0;pattern<3;++pattern){size_t n=sizes[shape];
        for(size_t i=0;i<139;++i)mask[i]=pattern==1?0:i%3!=0;
        a[n]=b[n]=123456;
        if(ds41f_index_scores(a,q,weights,rows,n,pattern?mask:NULL,0)||
           ds41f_index_scores(b,q,weights,rows,n,pattern?mask:NULL,1))return 1;
        if(memcmp(a,b,(n+1)*4)||a[n]!=123456){
            for(size_t i=0;i<n;++i)if(memcmp(a+i,b+i,4)){fprintf(stderr,"INDEX_HEADS fail n=%zu pattern=%d row=%zu %a %a\n",n,pattern,i,a[i],b[i]);break;}return 1;}}
    puts("INDEX_HEADS PASS cases=24 bit_exact masks tails canaries");
    for(int mode=0;mode<2;++mode){double sum=0,best=1e9;
        for(int it=0;it<31;++it){double t=now();if(ds41f_index_scores(b,q,weights,rows,1105,NULL,mode))return 1;
            t=now()-t;if(it){sum+=t;if(t<best)best=t;}}
        printf("INDEX_HEADS mode=%d mean_us=%.3f best_us=%.3f\n",mode,sum/30*1e6,best*1e6);}
    free(rows);free(a);free(b);return 0;
}
