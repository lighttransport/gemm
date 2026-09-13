#define _POSIX_C_SOURCE 200809L
#include "ds41f_ops.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint32_t seed=17;
static float sample(void){seed=1664525*seed+1013904223;return ((int)(seed>>16)-32768)/16384.f;}
static int check(size_t heads,size_t dim,size_t count,int pattern)
{
    size_t n=heads*dim,rows=count?count:1;
    float *q=malloc(n*4),*kv=malloc(rows*dim*4),*control=malloc(n*4),*out=malloc((n+2)*4),*sink=malloc(heads*4);
    int *ids=malloc(rows*sizeof(int));
    if(!q||!kv||!control||!out||!sink||!ids)return 1;
    for(size_t i=0;i<n;++i)q[i]=sample();
    for(size_t i=0;i<rows*dim;++i)kv[i]=sample();
    for(size_t h=0;h<heads;++h)sink[h]=(h%3==0?90.f:h%3==1?-90.f:sample());
    for(size_t i=0;i<count;++i)ids[i]=pattern==2?-1:pattern==1&&i%7==0?-1:(int)((i*17)%rows);
    if(ds41f_sparse_attention(control,q,kv,sink,ids,count,rows,heads,dim))return 1;
    for(int tile=1;tile<=6;++tile){if(tile==3||tile==5)continue;
        out[0]=out[n+1]=123456;
        if(ds41f_sparse_attention_tiled(out+1,q,kv,sink,ids,count,rows,heads,dim,tile))return 1;
        if(out[0]!=123456||out[n+1]!=123456)return 1;
        for(size_t i=0;i<n;++i)if(!isfinite(out[i+1])||memcmp(out+i+1,control+i,4)){
            fprintf(stderr,"tile mismatch h=%zu d=%zu n=%zu pattern=%d tile=%d i=%zu %a %a\n",heads,dim,count,pattern,tile,i,out[i+1],control[i]);return 1;}
    }
    if(ds41f_sparse_attention_tiled(out,q,kv,sink,ids,count,rows,heads,dim,3)!=EINVAL)return 1;
    if(count){ids[count-1]=(int)rows;if(ds41f_sparse_attention_tiled(out,q,kv,sink,ids,count,rows,heads,dim,4)!=ERANGE)return 1;}
    free(q);free(kv);free(control);free(out);free(sink);free(ids);return 0;
}
int main(void)
{
    size_t counts[]={0,1,2,3,127,128,129,511,512,639,640};
    for(size_t i=0;i<sizeof counts/sizeof *counts;++i)
        for(int pattern=0;pattern<3;++pattern)if(check(64,512,counts[i],pattern))return 1;
    for(size_t h=1;h<=17;++h)if(check(h,h%2?511:512,19,1))return 1;
    puts("SPARSE_TILES PASS 50 geometries, four tiles, bit-exact control, sink/masks/tails/canaries");
    size_t heads=64,dim=512,count=640,n=heads*dim;
    float *q=malloc(n*4),*kv=malloc(count*dim*4),*out=malloc(n*4),sink[64];int ids[640];
    if(!q||!kv||!out)return 1;
    for(size_t i=0;i<n;++i)q[i]=sample();
    for(size_t i=0;i<count*dim;++i)kv[i]=sample();
    for(size_t h=0;h<heads;++h)sink[h]=sample();
    for(size_t i=0;i<count;++i)ids[i]=(int)i;
    for(int tile=0;tile<=6;++tile){if(tile==3||tile==5)continue;
        double best=1e9,total=0;
        for(int it=0;it<31;++it){double t=now();
            int rc=tile?ds41f_sparse_attention_tiled(out,q,kv,sink,ids,count,count,heads,dim,tile):
                ds41f_sparse_attention(out,q,kv,sink,ids,count,count,heads,dim);
            if(rc)return 1;
            t=now()-t;if(it){total+=t;if(t<best)best=t;}}
        printf("SPARSE_TILE tile=%d best_us=%.3f mean_us=%.3f\n",tile,best*1e6,total/30*1e6);
    }
    free(q);free(kv);free(out);return 0;
}
