#define _POSIX_C_SOURCE 200809L
#include "ds41f_cache.h"
#include "ds41f_kernels.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"CACHE_SVE FAIL line=%d\n",__LINE__);return 1;}}while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void)
{
    float x[514],a[514],b[514];uint8_t ra[290],rb[290];size_t cases=0;
    for(int e4=0;e4<2;++e4)for(size_t group=16;group<=32;group*=2){
        for(unsigned bits=0;bits<65536;bits+=16){
            for(size_t i=0;i<512;++i)x[i+1]=ds41f_bf16_to_f32((uint16_t)(bits+i%16));
            memset(ra,123,sizeof ra);memset(rb,123,sizeof rb);size_t bytes=256+512/group;
            ds41f_set_cache_sve(0);int rc=ds41f_fp4_pack(ra+1,x+1,512,group,e4);
            ds41f_set_cache_sve(1);int got=ds41f_fp4_pack(rb+1,x+1,512,group,e4);
            REQUIRE(rc==got&&!memcmp(ra,rb,bytes+2));++cases;
        }
        for(unsigned scale=0;scale<256;++scale){size_t dim=group,bytes=dim/2+1;
            for(size_t i=0;i<dim/2;++i)ra[i+1]=(uint8_t)(i*17+8);
            ra[1+dim/2]=(uint8_t)scale;
            a[0]=a[dim+1]=b[0]=b[dim+1]=12345;
            ds41f_set_cache_sve(0);REQUIRE(!ds41f_fp4_unpack(a+1,ra+1,dim,group,e4));
            ds41f_set_cache_sve(1);REQUIRE(!ds41f_fp4_unpack(b+1,ra+1,dim,group,e4));
            for(size_t i=0;i<dim+2;++i)REQUIRE(!memcmp(a+i,b+i,4)||(isnan(a[i])&&isnan(b[i])));
            REQUIRE(bytes<=sizeof ra-2);++cases;
        }
    }
    for(int shape=0;shape<2;++shape){size_t dim=shape?512:128,group=shape?16:32,rows=shape?512:32,bytes=dim/2+dim/group;
        float *values=malloc(rows*dim*4),*out=malloc(rows*dim*4);uint8_t *packed=malloc(rows*bytes);
        REQUIRE(values&&out&&packed);
        for(size_t i=0;i<rows*dim;++i)values[i]=sinf((float)i*.071f)*.7f;
        for(int mode=0;mode<2;++mode){ds41f_set_cache_sve(mode);int loops=shape?500:2000;double start=now();
            for(int it=0;it<loops;++it){
                if(!shape)for(size_t row=0;row<rows;++row){
                    REQUIRE(!ds41f_fp4_pack(packed+row*bytes,values+row*dim,dim,group,shape));
                    REQUIRE(!ds41f_fp4_unpack(out+row*dim,packed+row*bytes,dim,group,shape));}
                else{
                    if(!it)for(size_t row=0;row<rows;++row)REQUIRE(!ds41f_fp4_pack(packed+row*bytes,values+row*dim,dim,group,shape));
                    #pragma omp parallel for schedule(static)
                    for(size_t row=0;row<rows;++row)ds41f_fp4_unpack(out+row*dim,packed+row*bytes,dim,group,shape);
                }
            }
            printf("CACHE_BENCH shape=%s sve=%d us=%.3f guard=%g\n",shape?"512_KV_rows":"32_index_queries",mode,(now()-start)*1e6/loops,out[7]);
        }
        free(values);free(out);free(packed);
    }
    printf("CACHE_SVE PASS cases=%zu BF16_domain grid_ties scale_domain signed_zero nonfinite tails canaries\n",cases);return 0;
}
